"""Construct Parquet database from local file system.

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

import pandas as pd
from iblutil.io import parquet
from iblutil.io.hashfile import md5

from one.alf.io import iter_sessions
from one.alf.files import session_path_parts, get_alf_path
from one.alf.spec import is_valid

__all__ = ['make_parquet_db']

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
    'project',
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
    out['project'] = ''
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


def _rel_path_to_uuid(df, id_key='rel_path', base_id=None, drop_key=False):
    base_id = base_id or uuid.uuid1()  # Base hash based on system by default
    toUUID = partial(uuid.uuid3, base_id)  # MD5 hash from base uuid and rel session path string
    uuids = df[id_key].map(toUUID)
    assert len(uuids.unique()) == uuids.size  # WARNING This fails :(
    npuuid = parquet.uuid2np(uuids)
    df[f"{id_key}_0"] = npuuid[:, 0]
    df[f"{id_key}_1"] = npuuid[:, 1]
    if drop_key:
        df.drop(id_key, axis=1, inplace=True)


def _ids_to_int(df_ses, df_dsets, drop_id=False):
    ns = uuid.uuid1()
    _rel_path_to_uuid(df_dsets, id_key='id', base_id=ns, drop_key=drop_id)
    _rel_path_to_uuid(df_ses, id_key='id', base_id=ns, drop_key=False)
    # Copy int eids into datasets frame
    eid_cols = ['eid_0', 'eid_1']
    df_dsets[eid_cols] = (df_ses
                          .set_index('id')
                          .loc[df_dsets['eid'], ['id_0', 'id_1']]
                          .values)
    # Check that the session int IDs in both frames match
    ses_int_id_set = (df_ses
                      .set_index('id')[['id_0', 'id_1']]
                      .rename(columns=lambda x: f'e{x}'))
    assert (df_dsets
            .set_index('eid')[eid_cols]
            .drop_duplicates()
            .equals(ses_int_id_set)), 'session int ID mismatch between frames'
    # Drop original id fields
    if drop_id:
        df_ses.drop('id', axis=1, inplace=True)
        df_dsets.drop('eid', axis=1, inplace=True)


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
        for rel_dset_path in _iter_datasets(session_path):
            file_info = _get_dataset_info(session_path, rel_dset_path, compute_hash=hash_files)
            assert set(file_info.keys()) <= set(DATASETS_COLUMNS)
            rows.append(file_info)
        df = df.append(rows, ignore_index=True, verify_integrity=True)
    return df


def _iter_datasets(session_path):
    """Iterate over all files in a session, and yield relative dataset paths."""
    for p in sorted(Path(session_path).rglob('*.*')):
        if not p.is_dir() and is_valid(p.name):
            yield p.relative_to(session_path)


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
        paths.
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
        _ids_to_int(df_ses, df_dsets, drop_id=True)

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
