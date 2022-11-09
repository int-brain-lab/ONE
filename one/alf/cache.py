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

import pandas as pd
from iblutil.io import parquet
from iblutil.io.hashfile import md5

from one.alf.io import iter_sessions, iter_datasets
from one.alf.files import session_path_parts, get_alf_path
from one.converters import session_record2path

__all__ = ['make_parquet_db', 'remove_missing_datasets']
_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------------------------------

SESSIONS_COLUMNS = (
    'id',               # int64
    'lab',
    'subject',
    'date',             # datetime.date
    'number',           # int
    'task_protocol',
    'projects',
)

DATASETS_COLUMNS = (
    'id',               # int64
    'eid',              # int64
    'session_path',     # relative to the root
    'rel_path',         # relative to the session path, includes the filename
    'file_size',        # file size in bytes
    'hash',             # sha1/md5, computed in load function
    'exists',           # bool
)


# -------------------------------------------------------------------------------------------------
# Parsing util functions
# -------------------------------------------------------------------------------------------------

def _ses_str_id(session_path):
    """Returns a str id from a session path in the form '(lab/)subject/date/number'"""
    return Path(*filter(None, session_path_parts(session_path, assert_valid=True))).as_posix()


def _get_session_info(rel_ses_path):
    """Parse a relative session path."""
    out = session_path_parts(rel_ses_path, as_dict=True, assert_valid=True)
    out['id'] = _ses_str_id(rel_ses_path)
    out['date'] = pd.to_datetime(out['date']).date()
    out['number'] = int(out['number'])
    out['task_protocol'] = ''
    out['projects'] = ''
    return out


def _get_dataset_info(full_ses_path, rel_dset_path, ses_eid=None, compute_hash=False):
    rel_ses_path = get_alf_path(full_ses_path)
    full_dset_path = Path(full_ses_path, rel_dset_path).as_posix()
    file_size = Path(full_dset_path).stat().st_size
    ses_eid = ses_eid or _ses_str_id(rel_ses_path)
    return {
        'id': Path(rel_ses_path, rel_dset_path).as_posix(),
        'eid': str(ses_eid),
        'session_path': str(rel_ses_path),
        'rel_path': Path(rel_dset_path).as_posix(),
        'file_size': file_size,
        'hash': md5(full_dset_path) if compute_hash else None,
        'exists': True
    }


def _rel_path_to_uuid(df, id_key='rel_path', base_id=None, keep_old=False):
    base_id = base_id or uuid.uuid1()  # Base hash based on system by default
    toUUID = partial(uuid.uuid3, base_id)  # MD5 hash from base uuid and rel session path string
    if keep_old:
        df[f'{id_key}_'] = df[id_key].copy()
    df[id_key] = df[id_key].apply(lambda x: str(toUUID(x)))
    assert len(df[id_key].unique()) == len(df[id_key])  # WARNING This fails :(
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
    """
    Metadata dictionary for Parquet files.

    Parameters
    ----------
    origin : str, pathlib.Path
        Path to full directory, or computer name / db name
    """
    return {
        'date_created': datetime.datetime.now().isoformat(sep=' ', timespec='minutes'),
        'origin': str(origin),
    }


def _make_sessions_df(root_dir) -> pd.DataFrame:
    """
    Given a root directory, recursively finds all sessions and returns a sessions DataFrame

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The folder to look for sessions

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame of session info
    """
    rows = []
    for full_path in iter_sessions(root_dir):
        # Get the lab/Subjects/subject/date/number part of a file path
        rel_path = get_alf_path(full_path)
        # A dict of session info extracted from path
        ses_info = _get_session_info(rel_path)
        assert set(ses_info.keys()) <= set(SESSIONS_COLUMNS)
        rows.append(ses_info)
    df = pd.DataFrame(rows, columns=SESSIONS_COLUMNS)
    return df


def _make_datasets_df(root_dir, hash_files=False) -> pd.DataFrame:
    """
    Given a root directory, recursively finds all datasets and returns a datasets DataFrame

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The folder to look for sessions
    hash_files : bool
        If True, an MD5 is computed for each file and stored in the 'hash' column

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame of dataset info
    """
    df = pd.DataFrame([], columns=DATASETS_COLUMNS)
    # Go through sessions and append datasets
    for session_path in iter_sessions(root_dir):
        rows = []
        for rel_dset_path in iter_datasets(session_path):
            file_info = _get_dataset_info(session_path, rel_dset_path, compute_hash=hash_files)
            assert set(file_info.keys()) <= set(DATASETS_COLUMNS)
            rows.append(file_info)
        df = pd.concat((df, pd.DataFrame(rows, columns=DATASETS_COLUMNS)),
                       ignore_index=True, verify_integrity=True)
    return df


def make_parquet_db(root_dir, out_dir=None, hash_ids=True, hash_files=False, lab=None):
    """
    Given a data directory, index the ALF datasets and save the generated cache tables.

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The file directory to index.
    out_dir : str, pathlib.Path
        Optional output directory to save cache tables.  If None, the files are saved into the
        root directory.
    hash_ids : bool
        If True, experiment and dataset IDs will be UUIDs generated from the system and relative
        paths (required for use with ONE API)
    hash_files : bool
        If True, an MD5 hash is computed for each dataset and stored in the datasets table.
        This will substantially increase cache generation time.
    lab : str
        An optional lab name to associate with the data.  If the folder structure
        contains 'lab/Subjects', the lab name will be taken from the folder name.

    Returns
    -------
    pathlib.Path
        The full path of the saved sessions parquet table
    pathlib.Path
        The full path of the saved datasets parquet table
    """
    root_dir = Path(root_dir).resolve()

    # Make the dataframes.
    df_ses = _make_sessions_df(root_dir)
    df_dsets = _make_datasets_df(root_dir, hash_files=hash_files)

    # Add integer id columns
    if hash_ids and len(df_ses) > 0:
        df_ses, df_dsets = _ids_to_uuid(df_ses, df_dsets)

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


def remove_missing_datasets(cache_dir, tables=None, remove_empty_sessions=True, dry=True):
    """
    Remove dataset files and session folders that are not in the provided cache.

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
        tables = {}
        for name in ('datasets', 'sessions'):
            tables[name], _ = parquet.load(cache_dir / f'{name}.pqt')
    to_delete = []
    gen_path = partial(session_record2path, root_dir=cache_dir)
    sessions = sorted(map(lambda x: gen_path(x[1]), tables['sessions'].iterrows()))
    for session_path in iter_sessions(cache_dir):
        rel_session_path = session_path.relative_to(cache_dir).as_posix()
        datasets = tables['datasets'][tables['datasets']['session_path'] == rel_session_path]
        for dataset in iter_datasets(session_path):
            if dataset.as_posix() not in datasets['rel_path']:
                to_delete.append(session_path.joinpath(dataset))
        if session_path not in sessions and remove_empty_sessions:
            to_delete.append(session_path)

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
