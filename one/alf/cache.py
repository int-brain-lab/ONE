"""Construct Parquet database from local file system.

NB: If using a remote Alyx instance it is advisable to generate the cache via the Alyx one_cache
management command, otherwise the resulting cache UUIDs will not match those on the database.

Examples
--------
>>> from one.api import One
>>> cache_dir = 'path/to/data'
>>> make_parquet_db(cache_dir)
>>> one = One(cache_dir=cache_dir)

"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import datetime
import uuid
from functools import partial
from pathlib import Path
import warnings
import logging
from copy import deepcopy

import pandas as pd
import numpy as np
from packaging import version
from iblutil.util import Bunch, ensure_list
from iblutil.io import parquet
from iblutil.io.hashfile import md5

from one.alf.spec import QC, is_uuid_string
from one.alf.io import iter_sessions
from one.alf.path import session_path_parts, get_alf_path

__all__ = [
    'make_parquet_db', 'load_tables', 'patch_tables', 'merge_tables',
    'remove_table_files', 'remove_missing_datasets', 'default_cache',
    'QC_TYPE', 'EMPTY_DATASETS_FRAME', 'EMPTY_SESSIONS_FRAME']
_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------------------------------

QC_TYPE = pd.CategoricalDtype(categories=[e.name for e in sorted(QC)], ordered=True)
"""pandas.api.types.CategoricalDtype: The cache table QC column data type."""

SESSIONS_COLUMNS = {
    'id': object,                 # str
    'lab': object,                # str
    'subject': object,            # str
    'date': object,               # datetime.date
    'number': np.uint16,          # int
    'task_protocol': object,      # str
    'projects': object            # str
}
"""dict: A map of sessions table fields and their data types."""

DATASETS_COLUMNS = {
    'eid': object,                # str
    'id': object,                 # str
    'rel_path': object,           # relative to the session path, includes the filename
    'file_size': 'UInt64',        # file size in bytes (nullable)
    'hash': object,               # sha1/md5, computed in load function
    'exists': bool,               # bool
    'qc': QC_TYPE                 # one.alf.spec.QC enumeration
}
"""dict: A map of datasets table fields and their data types."""

EMPTY_DATASETS_FRAME = (pd.DataFrame(columns=DATASETS_COLUMNS)
                        .astype(DATASETS_COLUMNS)
                        .set_index(['eid', 'id']))
"""pandas.DataFrame: An empty datasets dataframe with correct columns and dtypes."""

EMPTY_SESSIONS_FRAME = (pd.DataFrame(columns=SESSIONS_COLUMNS)
                        .astype(SESSIONS_COLUMNS)
                        .set_index('id'))
"""pandas.DataFrame: An empty sessions dataframe with correct columns and dtypes."""


# -------------------------------------------------------------------------------------------------
# Parsing util functions
# -------------------------------------------------------------------------------------------------

def _ses_str_id(session_path):
    """Returns a str id from a session path in the form '(lab/)subject/date/number'."""
    return Path(*filter(None, session_path_parts(session_path, assert_valid=True))).as_posix()


def _get_session_info(rel_ses_path):
    """Parse a relative session path.

    Parameters
    ----------
    rel_ses_path : _type_
        _description_

    Returns
    -------
    str
        Experiment ID expressed as a relative session posix path.
    str
        The lab name (empty str).
    datetime.date
        The session date.
    int
        The session number.
    str
        The task protocol (empty str).
    str
        The associated project (empty str).

    """
    lab, subject, s_date, num = session_path_parts(rel_ses_path, as_dict=False, assert_valid=True)
    eid = _ses_str_id(rel_ses_path)
    s_date = pd.to_datetime(s_date).date()
    return eid, lab or '', subject, s_date, int(num), '', ''


def _get_dataset_info(dset_path, ses_eid=None, compute_hash=False):
    """Create dataset record from local path.

    Parameters
    ----------
    dset_path : one.alf.ALFPath
        A full ALF path.
    ses_eid : str, UUID, optional
        A session uuid.
    compute_hash : bool, optional
        Whether to compute a file hash.

    Returns
    -------
    str, uuid.UUID
        The session uuid.
    str
        The dataset ID expressed as a posix path relative to the session.
    str
        The dataset posix path, relative to the session.
    int
        The dataset file size.
    str
        The file hash, or empty str if `compute_hash` is false.
    True
        Whether the file exists.
    str
        The QC value for the dataset ('NOT_SET').

    """
    rel_dset_path = get_alf_path(dset_path.relative_to_session())
    ses_eid = ses_eid or _ses_str_id(dset_path.session_path())
    file_size = dset_path.stat().st_size
    file_hash = md5(dset_path) if compute_hash else ''
    return ses_eid or pd.NA, rel_dset_path, rel_dset_path, file_size, file_hash, True, 'NOT_SET'


def _rel_path_to_uuid(df, id_key='rel_path', base_id=None, keep_old=False):
    base_id = base_id or uuid.uuid1()  # Base hash based on system by default
    toUUID = partial(uuid.uuid3, base_id)  # MD5 hash from base uuid and rel session path string
    if keep_old:
        df[f'{id_key}_'] = df[id_key].copy()
    df.loc[:, id_key] = df.groupby(id_key)[id_key].transform(lambda x: toUUID(x.name))
    return df


def _ids_to_uuid(df_ses, df_dsets):
    ns = uuid.uuid1()
    df_dsets = _rel_path_to_uuid(df_dsets, id_key='id', base_id=ns)
    df_ses = _rel_path_to_uuid(df_ses, id_key='id', base_id=ns, keep_old=True)
    # Copy new eids into datasets frame
    df_dsets['eid_'] = df_dsets['eid'].copy()
    df_dsets['eid'] = (df_ses
                       .set_index('id_')
                       .loc[df_dsets['eid'], 'id']
                       .values)
    # Check that the session int IDs in both frames match
    ses_id_set = df_ses.set_index('id_')['id']
    assert (df_dsets
            .set_index('eid_')['eid']
            .drop_duplicates()
            .equals(ses_id_set)), 'session int ID mismatch between frames'

    # Set index
    df_ses = df_ses.set_index('id').drop('id_', axis=1).sort_index()
    df_dsets = df_dsets.set_index(['eid', 'id']).drop('eid_', axis=1).sort_index()

    return df_ses, df_dsets


# -------------------------------------------------------------------------------------------------
# Main functions
# -------------------------------------------------------------------------------------------------

def _metadata(origin):
    """Metadata dictionary for Parquet files.

    Parameters
    ----------
    origin : str, pathlib.Path
        Path to full directory, or computer name / db name.

    """
    return {
        'date_created': datetime.datetime.now().isoformat(sep=' ', timespec='minutes'),
        'origin': str(origin),
    }


def _make_sessions_df(root_dir) -> pd.DataFrame:
    """Given a root directory, recursively finds all sessions and returns a sessions DataFrame.

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The folder to look for sessions.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame of session info.

    """
    rows = []
    for full_path in iter_sessions(root_dir):
        # Get the lab/Subjects/subject/date/number part of a file path
        rel_path = get_alf_path(full_path)
        # A dict of session info extracted from path
        ses_info = _get_session_info(rel_path)
        assert len(ses_info) == len(SESSIONS_COLUMNS)
        rows.append(ses_info)
    df = pd.DataFrame(rows, columns=SESSIONS_COLUMNS).astype(SESSIONS_COLUMNS)
    return df


def _make_datasets_df(root_dir, hash_files=False) -> pd.DataFrame:
    """Given a root directory, recursively finds all datasets and returns a datasets DataFrame.

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The folder to look for sessions.
    hash_files : bool
        If True, an MD5 is computed for each file and stored in the 'hash' column.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame of dataset info.

    """
    # Go through sessions and append datasets
    rows = []
    for session_path in iter_sessions(root_dir):
        for dset_path in session_path.iter_datasets(recursive=True):
            file_info = _get_dataset_info(dset_path, compute_hash=hash_files)
            assert len(file_info) == len(DATASETS_COLUMNS)
            rows.append(file_info)
    return pd.DataFrame(rows, columns=DATASETS_COLUMNS).astype(DATASETS_COLUMNS)


def default_cache(origin=''):
    """Returns an empty cache dictionary with the default tables.

    Parameters
    ----------
    origin : str, optional
        The origin of the cache (e.g. a computer name or database name).

    Returns
    -------
    Bunch
        A Bunch object containing the loaded cache tables and associated metadata.

    """
    table_meta = _metadata(origin)
    # The origin is now a set, however we leave _metadata as Alyx relies on this and sets
    # can't be serialized to JSON
    table_meta['origin'] = set(filter(None, [origin]))
    return Bunch({
            'datasets': EMPTY_DATASETS_FRAME.copy(),
            'sessions': EMPTY_SESSIONS_FRAME.copy(),
            '_meta': {
                'created_time': None,
                'loaded_time': None,
                'modified_time': None,
                'saved_time': None,
                'raw': {k: deepcopy(table_meta) for k in ('datasets', 'sessions')}}
        })


def make_parquet_db(root_dir, out_dir=None, hash_ids=True, hash_files=False, lab=None):
    """Given a data directory, index the ALF datasets and save the generated cache tables.

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The file directory to index.
    out_dir : str, pathlib.Path
        Optional output directory to save cache tables.  If None, the files are saved into the
        root directory.
    hash_ids : bool
        If True, experiment and dataset IDs will be UUIDs generated from the system and relative
        paths (required for use with ONE API).
    hash_files : bool
        If True, an MD5 hash is computed for each dataset and stored in the datasets table.
        This will substantially increase cache generation time.
    lab : str
        An optional lab name to associate with the data.  If the folder structure
        contains 'lab/Subjects', the lab name will be taken from the folder name.

    Returns
    -------
    pathlib.Path
        The full path of the saved sessions parquet table.
    pathlib.Path
        The full path of the saved datasets parquet table.

    """
    root_dir = Path(root_dir).resolve()

    # Make the data frames.
    df_ses = _make_sessions_df(root_dir)
    df_dsets = _make_datasets_df(root_dir, hash_files=hash_files)

    # Add integer id columns
    if hash_ids and len(df_ses) > 0:
        df_ses, df_dsets = _ids_to_uuid(df_ses, df_dsets)
        # For parquet all indices must be str
        df_ses.index = df_ses.index.map(str)
        df_dsets.index = df_dsets.index.map(lambda x: tuple(map(str, x)))

    if lab:  # Fill in lab name field
        assert not df_ses['lab'].any() or (df_ses['lab'] == 'lab').all(), 'lab name conflict'
        df_ses['lab'] = lab

    # Check any files were found
    if df_ses.empty or df_dsets.empty:
        warnings.warn(f'No {"sessions" if df_ses.empty else "datasets"} found', RuntimeWarning)

    # Output directory.
    out_dir = Path(out_dir or root_dir)
    assert out_dir.is_dir()
    assert out_dir.exists()

    # Parquet files to save.
    fn_ses = out_dir / 'sessions.pqt'
    fn_dsets = out_dir / 'datasets.pqt'

    # Parquet metadata.
    metadata = _metadata(root_dir)

    # Save the Parquet files.
    parquet.save(fn_ses, df_ses, metadata)
    parquet.save(fn_dsets, df_dsets, metadata)

    return fn_ses, fn_dsets


def cast_index_object(df: pd.DataFrame, dtype: type = uuid.UUID) -> pd.Index:
    """Cast the index object to the specified dtype.

    NB: The data frame index will remain as 'object', however the underlying object type will be
    modified.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with an index to cast.
    dtype : type, function
        The desired dtype or a mapping function.

    Returns
    -------
    pandas.DataFrame
        An updated data frame with a new index data type.

    """
    if isinstance(df.index, pd.MultiIndex):
        # df.index = df.index.map(lambda x: tuple(map(UUID, x)))
        levels = range(df.index.nlevels)
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(i).map(dtype, na_action='ignore') for i in levels],
            names=df.index.names
        )
    else:
        df.index = df.index.map(dtype, na_action='ignore')
    return df


def load_tables(tables_dir, glob_pattern='*.pqt'):
    """Load parquet cache files from a local directory.

    Parameters
    ----------
    tables_dir : str, pathlib.Path
        The directory location of the parquet files.
    glob_pattern : str
        A glob pattern to match the cache files.


    Returns
    -------
    Bunch
        A Bunch object containing the loaded cache tables and associated metadata.

    """
    caches = default_cache()
    meta = caches['_meta']
    INDEX_KEY = '.?id'
    for cache_file in Path(tables_dir).glob(glob_pattern):
        table = cache_file.stem
        # we need to keep this part fast enough for transient objects
        cache, meta['raw'][table] = parquet.load(cache_file)
        if 'date_created' not in meta['raw'][table]:
            _logger.warning(f"{cache_file} does not appear to be a valid table. Skipping")
            continue
        meta['loaded_time'] = datetime.datetime.now()

        # Set the appropriate index if none already set
        if isinstance(cache.index, pd.RangeIndex):
            idx_columns = sorted(cache.filter(regex=INDEX_KEY).columns)
            if len(idx_columns) == 0:
                raise KeyError('Failed to set index')
            cache.set_index(idx_columns, inplace=True)

        # Patch older tables
        cache = patch_tables(cache, meta['raw'][table].get('min_api_version'), table)

        # Cast indices to UUID
        # NB: Old caches may have pathstr indices
        if any(map(is_uuid_string, cache.index.get_level_values(0))):
            cache = cast_index_object(cache, uuid.UUID)

        # Check sorted
        # Sorting makes MultiIndex indexing O(N) -> O(1)
        if not cache.index.is_monotonic_increasing:
            cache.sort_index(inplace=True)

        # Ensure origin is a set (supports multiple origins)
        meta['raw'][table]['origin'] = set(
            filter(None, ensure_list(meta['raw'][table].get('origin', 'unknown'))))

        caches[table] = cache

    created = [datetime.datetime.fromisoformat(x['date_created'])
               for x in meta['raw'].values() if 'date_created' in x]
    if created:
        meta['created_time'] = min(created)
    return caches


def merge_tables(cache, strict=False, origin=None, **kwargs):
    """Update the cache tables with new records.

    Note: A copy of the tables in cache may be returned if the original tables are immutable.
    This can happen when tables are loaded from a parquet file.

    Parameters
    ----------
    dict
        A map of cache tables to update.
    strict : bool
        If not True, the columns don't need to match.  Extra columns in input tables are
        dropped and missing columns are added and filled with np.nan.
    origin : str
        The origin of the cache (e.g. a computer name or database name).
    kwargs
        pandas.DataFrame or pandas.Series to insert/update for each table.

    Returns
    -------
    datetime.datetime:
        A timestamp of when the cache was updated.

    Example
    -------
    >>> session, datasets = ses2records(self.get_details(eid, full=True))
    ... self._update_cache_from_records(sessions=session, datasets=datasets)

    Raises
    ------
    AssertionError
        When strict is True the input columns must exactly match those oo the cache table,
        including the order.
    KeyError
        One or more of the keyword arguments does not match a table in cache.

    """
    updated = None
    for table, records in kwargs.items():
        if records is None or records.empty:
            continue
        if table not in cache:
            raise KeyError(f'Table "{table}" not in cache')
        if isinstance(records, pd.Series):
            records = pd.DataFrame([records])
            records.index.set_names(cache[table].index.names, inplace=True)
        # Drop duplicate indices
        records = records[~records.index.duplicated(keep='first')]
        if not strict:
            # Deal with case where there are extra columns in the cache
            extra_columns = list(set(cache[table].columns) - set(records.columns))
            # Convert these columns to nullable, if required
            cache_columns = cache[table][extra_columns]
            cache[table][extra_columns] = cache_columns.convert_dtypes()
            column_ids = map(list(cache[table].columns).index, extra_columns)
            for col, n in sorted(zip(extra_columns, column_ids), key=lambda x: x[1]):
                dtype = cache[table][col].dtype
                nan = getattr(dtype, 'na_value', np.nan)
                val = records.get('exists', True) if col.startswith('exists_') else nan
                records.insert(n, col, val)
            # Drop any extra columns in the records that aren't in cache table
            to_drop = set(records.columns) - set(cache[table].columns)
            records = records.drop(to_drop, axis=1)
            records = records.reindex(columns=cache[table].columns)
        assert set(cache[table].columns) == set(records.columns)
        records = records.astype(cache[table].dtypes)
        # Update existing rows
        to_update = records.index.isin(cache[table].index)
        try:
            cache[table].loc[records.index[to_update], :] = records[to_update]
        except ValueError as e:
            if 'assignment destination is read-only' in str(e):
                # NB: nullable integer and categorical dtypes may be backed by immutable arrays
                # after loading from parquet and therefore must be copied before assignment
                cache[table] = cache[table].copy()
                cache[table].loc[records.index[to_update], :] = records[to_update]
            else:
                raise e  # pragma: no cover

        # Assign new rows
        to_assign = records[~to_update]
        frames = [cache[table], to_assign]
        # Concatenate and sort
        cache[table] = pd.concat(frames).sort_index()
        updated = datetime.datetime.now()
        # Update the table metadata with the origin
        if origin is not None:
            table_meta = cache['_meta']['raw'].get(table, {})
            if 'origin' not in table_meta:
                table_meta['origin'] = set(origin)
            else:
                table_meta['origin'].add(origin)
            cache['_meta']['raw'][table] = table_meta
    cache['_meta']['modified_time'] = updated
    return updated


def remove_missing_datasets(cache_dir, tables=None, remove_empty_sessions=True, dry=True):
    """Remove dataset files and session folders that are not in the provided cache.

    NB: This *does not* remove entries from the cache tables that are missing on disk.
    Non-ALF files are not removed. Empty sessions that exist in the sessions table are not removed.

    Parameters
    ----------
    cache_dir : str, pathlib.Path
    tables : dict[str, pandas.DataFrame], optional
        A dict with keys ('sessions', 'datasets'), containing the cache tables as DataFrames.
    remove_empty_sessions : bool
        Attempt to remove session folders that are empty and not in the sessions table.
    dry : bool
        If true, do not remove anything.

    Returns
    -------
    list
        A sorted list of paths to be removed.

    """
    cache_dir = Path(cache_dir)
    if tables is None:
        tables = load_tables(cache_dir)

    to_delete = set()
    from one.converters import session_record2path  # imported here due to circular imports
    gen_path = partial(session_record2path, root_dir=cache_dir)
    # map of session path to eid
    sessions = {gen_path(rec): eid for eid, rec in tables['sessions'].iterrows()}
    for session_path in iter_sessions(cache_dir):
        try:
            datasets = tables['datasets'].loc[sessions[session_path]]
        except KeyError:
            datasets = tables['datasets'].iloc[0:0, :]
        for dataset in session_path.iter_datasets():
            if dataset.relative_to_session().as_posix() not in datasets['rel_path']:
                to_delete.add(dataset)
        if session_path not in sessions and remove_empty_sessions:
            to_delete.add(session_path)

    if dry:
        print('The following session and datasets would be removed:', end='\n\t')
        print('\n\t'.join(sorted(map(str, to_delete))))
        return sorted(to_delete)

    # Delete datasets
    for path in to_delete:
        if path.is_file():
            _logger.debug(f'Removing {path}')
            path.unlink()
        else:
            # Recursively remove empty folders
            while path.parent != cache_dir and not next(path.rglob('*'), False):
                _logger.debug(f'Removing {path}')
                path.rmdir()
                path = path.parent

    return sorted(to_delete)


def remove_table_files(folder, tables=('sessions', 'datasets')):
    """Delete cache tables on disk.

    Parameters
    ----------
    folder : pathlib.Path
        The directory path containing cache tables to remove.
    tables : list of str
        A list of table names to remove, e.g. ['sessions', 'datasets'].
        NB: This will also delete the cache_info.json metadata file.

    Returns
    -------
    list of pathlib.Path
        A list of the removed files.

    """
    filenames = ('cache_info.json', *(f'{t}.pqt' for t in tables))
    removed = []
    for file in map(folder.joinpath, filenames):
        if file.exists():
            file.unlink()
            removed.append(file)
        else:
            _logger.warning('%s not found', file)
    return removed


def _cache_int2str(table: pd.DataFrame) -> pd.DataFrame:
    """Convert int ids to str ids for cache table.

    Parameters
    ----------
    table : pd.DataFrame
        A cache table (from One._cache).

    """
    # Convert integer uuids to str uuids
    if table.index.nlevels < 2 or not any(x.endswith('_0') for x in table.index.names):
        return table
    table = table.reset_index()
    int_cols = table.filter(regex=r'_\d{1}$').columns.sort_values()
    assert not len(int_cols) % 2, 'expected even number of columns ending in _0 or _1'
    names = sorted(set(c.rsplit('_', 1)[0] for c in int_cols.values))
    for i, name in zip(range(0, len(int_cols), 2), names):
        table[name] = parquet.np2str(table[int_cols[i:i + 2]])
    table = table.drop(int_cols, axis=1).set_index(names)
    return table


def patch_tables(table: pd.DataFrame, min_api_version=None, name=None) -> pd.DataFrame:
    """Reformat older cache tables to comply with this version of ONE.

    Currently this function will 1. convert integer UUIDs to string UUIDs; 2. rename the 'project'
    column to 'projects'; 3. add QC column; 4. drop session_path column.

    Parameters
    ----------
    table : pd.DataFrame
        A cache table (from One._cache).
    min_api_version : str
        The minimum API version supported by this cache table.
    name : {'dataset', 'session'} str
        The name of the table.

    """
    min_version = version.parse(min_api_version or '0.0.0')
    table = _cache_int2str(table)
    # Rename project column
    if min_version < version.Version('1.13.0') and 'project' in table.columns:
        table.rename(columns={'project': 'projects'}, inplace=True)
    if name == 'datasets' and min_version < version.Version('2.7.0') and 'qc' not in table.columns:
        qc = pd.Categorical.from_codes(np.zeros(len(table.index), dtype=int), dtype=QC_TYPE)
        table = table.assign(qc=qc)
    if name == 'datasets' and 'session_path' in table.columns:
        table = table.drop('session_path', axis=1)
    return table
