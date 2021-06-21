"""Construct Parquet database from local file system.
TODO Consolidate functions with ALF.files
TODO Deal graciously with empty tables
TODO Support Subjects folder as optional
TODO Allow lab passed as arg instead of folder structure
"""


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import datetime
import uuid
from functools import partial
from pathlib import Path
import re

import pandas as pd

from iblutil.io import parquet
from iblutil.io.hashfile import md5
from .folders import session_path


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
    'file_size',
    'hash',             # sha1/md5, computed in load function
    'exists',           # bool
)


def _compile(r):
    r = r.replace('/', r'\/')
    return re.compile(r)


def _pattern_to_regex(pattern):
    """Convert a path pattern with {...} into a regex."""
    return _compile(re.sub(r'{(\w+)}', r'(?P<\1>[a-zA-Z0-9\_\-\.]+)', pattern))


SESSION_PATTERN = "{lab}/Subjects/{subject}/{date}/{number}"
SESSION_REGEX = _pattern_to_regex('^%s/?$' % SESSION_PATTERN)

FILE_PATTERN = "^{lab}/Subjects/{subject}/{date}/{number}/alf/{filename}$"
FILE_REGEX = _pattern_to_regex(FILE_PATTERN)


def _metadata(origin):
    """
    Metadata dictionary for Parquet files.

    :param origin: path to full directory, or computer name / db name
    """
    return {
        'date_created': datetime.datetime.now().isoformat(sep=' ', timespec='minutes'),
        'origin': str(origin),
    }


# -------------------------------------------------------------------------------------------------
# Parsing util functions
# -------------------------------------------------------------------------------------------------

def _ses_eid(rel_ses_path):
    m = SESSION_REGEX.match(str(rel_ses_path))
    if not m:
        raise ValueError("The relative session path `%s` is invalid." % rel_ses_path)
    out = {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}
    return SESSION_PATTERN.format(**out)


def _parse_rel_ses_path(rel_ses_path):
    """Parse a relative session path."""
    m = SESSION_REGEX.match(str(rel_ses_path))
    if not m:
        raise ValueError("The relative session path `%s` is invalid." % rel_ses_path)
    out = {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}
    out['id'] = SESSION_PATTERN.format(**out)
    out['number'] = int(out['number'])
    out['date'] = pd.to_datetime(out['date']).date()
    out['task_protocol'] = ''
    out['project'] = ''
    return out


# def _parse_file_path(file_path):
#     """Parse a file path."""
#     m = FILE_REGEX.match(str(file_path))
#     if not m:
#         raise ValueError("The file path `%s` is invalid." % file_path)
#     return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _get_file_rel_path(file_path):
    """Get the lab/Subjects/subject/... part of a file path."""
    file_path = str(file_path).replace('\\', '/')
    # Find the relative part of the file path.
    i = file_path.index('/Subjects')
    if '/' not in file_path[:i]:
        return file_path
    i = file_path[:i].rindex('/') + 1
    return file_path[i:]


def _get_full_ses_path(file_path):
    return session_path(file_path)


# -------------------------------------------------------------------------------------------------
# Other util functions
# -------------------------------------------------------------------------------------------------

def _walk(root_dir):
    """Iterate over all files found within a root directory."""
    for p in sorted(Path(root_dir).rglob('*')):
        yield p


def _is_session_dir(path):
    """Return whether a path is a session directory.

    Example of a session dir: `/path/to/root/mainenlab/Subjects/ZM_1150/2019-05-07/001/`

    """
    return path.is_dir() and path.parent.parent.parent.name == 'Subjects'


def _is_file_in_session_dir(path):
    """Return whether a file path is within a session directory."""
    if path.name.startswith('.'):
        return False  # Ignore hidden files
    return not path.is_dir() and '/Subjects/' in str(path.parent.parent.parent).replace('\\', '/')


def _find_sessions(root_dir):
    """Iterate over all session directories found in a root directory."""
    for p in _walk(root_dir):
        if _is_session_dir(p):
            yield p


def _find_session_files(full_ses_path):
    """Iterate over all files in a session, and yield relative dataset paths."""
    for p in _walk(full_ses_path):
        if not (p.is_dir() or p.name.startswith('.')):  # Ignore folders and hidden files
            yield p.relative_to(full_ses_path)


def _get_dataset_info(full_ses_path, rel_dset_path, ses_eid=None, compute_hash=False):
    rel_ses_path = _get_file_rel_path(full_ses_path)
    full_dset_path = Path(full_ses_path, rel_dset_path).as_posix()
    file_size = Path(full_dset_path).stat().st_size
    ses_eid = ses_eid or _ses_eid(rel_ses_path)
    return {
        'id': Path(rel_ses_path, rel_dset_path).as_posix(),
        'eid': str(ses_eid),
        'session_path': str(rel_ses_path),
        'rel_path': Path(rel_dset_path).as_posix(),
        # 'dataset_type': '.'.join(str(rel_dset_path).split('/')[-1].split('.')[:-1]),
        'file_size': file_size,
        'hash': md5(full_dset_path) if compute_hash else None,
        'exists': True
    }


# -------------------------------------------------------------------------------------------------
# Main functions
# -------------------------------------------------------------------------------------------------

def _make_sessions_df(root_dir):
    rows = []
    for full_path in _find_sessions(root_dir):
        rel_path = _get_file_rel_path(full_path)
        ses_info = _parse_rel_ses_path(rel_path)
        assert set(ses_info.keys()) <= set(SESSIONS_COLUMNS)
        rows.append(ses_info)
    df = pd.DataFrame(rows, columns=SESSIONS_COLUMNS)
    return df


def _extend_datasets_df(df, root_dir, rel_ses_path, hash_files=False):
    rows = []
    for rel_dset_path in _find_session_files(root_dir / rel_ses_path):
        full_ses_path = root_dir / rel_ses_path
        file_info = _get_dataset_info(full_ses_path, rel_dset_path, compute_hash=hash_files)
        assert set(file_info.keys()) <= set(DATASETS_COLUMNS)
        rows.append(file_info)
    if df is None:
        df = pd.DataFrame(rows, columns=DATASETS_COLUMNS)
    else:
        df = df.append(rows, ignore_index=True, verify_integrity=True)
    return df


def _make_datasets_df(root_dir, hash_files=False):
    df = None
    # Go through all found sessions.
    for full_path in _find_sessions(root_dir):
        rel_ses_path = _get_file_rel_path(full_path)
        # Append the datasets of each session.
        df = _extend_datasets_df(df, root_dir, rel_ses_path, hash_files=hash_files)
    return df


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


def make_parquet_db(root_dir, out_dir=None, hash_ids=True, hash_files=False):
    root_dir = Path(root_dir).resolve()

    # Make the dataframes.
    df_ses = _make_sessions_df(root_dir)
    df_dsets = _make_datasets_df(root_dir, hash_files=hash_files)

    # Add integer id columns
    if hash_ids and len(df_ses) > 0:
        _ids_to_int(df_ses, df_dsets, drop_id=True)

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
