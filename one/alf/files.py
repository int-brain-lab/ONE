"""
Module for identifying and parsing ALF file names.

An ALF file has the following components (those in brackets are optional):
    (_namespace_)object.attribute(_timescale)(.extra.parts).ext

Note the following:
    Object attributes may not contain an underscore unless followed by 'times' or 'intervals'.
    A namespace must not contain extra underscores (i.e. `name_space` and `__namespace__` are not
    valid).
    ALF files must always have an extension.

For more information, see the following documentation:
    https://int-brain-lab.github.io/ONE/alf_intro.html

"""
from collections import OrderedDict
from datetime import datetime
from typing import Union, Optional
from pathlib import Path
import logging

from . import spec
from .spec import SESSION_SPEC, COLLECTION_SPEC, FILE_SPEC, REL_PATH_SPEC

_logger = logging.getLogger(__name__)


def rel_path_parts(rel_path, as_dict=False, assert_valid=True):
    """Parse a relative path into the relevant parts.

    A relative path follows the pattern
        (collection/)(#revision#/)_namespace_object.attribute_timescale.extra.extension

    Parameters
    ----------
    rel_path : str, pathlib.Path
        A relative path string.
    as_dict : bool
        If true, an OrderedDict of parts are returned with the keys ('lab', 'subject', 'date',
        'number'), otherwise a tuple of values are returned.
    assert_valid : bool
        If true a ValueError is raised when the session cannot be parsed, otherwise an empty
        dict of tuple of Nones is returned.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.
    """
    return _path_parts(rel_path, REL_PATH_SPEC, True, as_dict, assert_valid)


def session_path_parts(session_path, as_dict=False, assert_valid=True):
    """Parse a session path into the relevant parts.

    Return keys:
        - lab
        - subject
        - date
        - number

    Parameters
    ----------
    session_path : str, pathlib.Path
        A session path string.
    as_dict : bool
        If true, an OrderedDict of parts are returned with the keys ('lab', 'subject', 'date',
        'number'), otherwise a tuple of values are returned.
    assert_valid : bool
        If true a ValueError is raised when the session cannot be parsed, otherwise an empty
        dict of tuple of Nones is returned.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.

    Raises
    ------
    ValueError
        Invalid ALF session path (assert_valid is True).
    """
    return _path_parts(session_path, SESSION_SPEC, False, as_dict, assert_valid)


def _path_parts(path, spec_str, match=True, as_dict=False, assert_valid=True):
    """Given a ALF and a spec string, parse into parts.

    Parameters
    ----------
    path : str, pathlib.Path
        An ALF path or dataset.
    match : bool
        If True, string must match exactly, otherwise search for expression within path.
    as_dict : bool
        When true a dict of matches is returned.
    assert_valid : bool
        When true an exception is raised when the filename cannot be parsed.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.

    Raises
    ------
    ValueError
        Invalid ALF path (assert_valid is True).
    """
    if hasattr(path, 'as_posix'):
        path = path.as_posix()
    pattern = spec.regex(spec_str)
    empty = OrderedDict.fromkeys(pattern.groupindex.keys())
    parsed = (pattern.match if match else pattern.search)(path)
    if parsed:  # py3.8
        parsed_dict = parsed.groupdict()
        return OrderedDict(parsed_dict) if as_dict else tuple(parsed_dict.values())
    elif assert_valid:
        raise ValueError(f'Invalid ALF: "{path}"')
    else:
        return empty if as_dict else tuple(empty.values())


def filename_parts(filename, as_dict=False, assert_valid=True) -> Union[dict, tuple]:
    """
    Return the parsed elements of a given ALF filename.

    Parameters
    ----------
    filename : str
        The name of the file.
    as_dict : bool
        When true a dict of matches is returned.
    assert_valid : bool
        When true an exception is raised when the filename cannot be parsed.

    Returns
    -------
    namespace : str
        The _namespace_ or None if not present.
    object : str
        ALF object.
    attribute : str
        The ALF attribute.
    timescale : str
        The ALF _timescale or None if not present.
    extra : str
        Any extra parts to the filename, or None if not present.
    extension : str
        The file extension.

    Examples
    --------
    >>> filename_parts('_namespace_obj.times_timescale.extra.foo.ext')
    ('namespace', 'obj', 'times', 'timescale', 'extra.foo', 'ext')
    >>> filename_parts('spikes.clusters.npy', as_dict=True)
    {'namespace': None,
     'object': 'spikes',
     'attribute': 'clusters',
     'timescale': None,
     'extra': None,
     'extension': 'npy'}
    >>> filename_parts('spikes.times_ephysClock.npy')
    (None, 'spikes', 'times', 'ephysClock', None, 'npy')
    >>> filename_parts('_iblmic_audioSpectrogram.frequencies.npy')
    ('iblmic', 'audioSpectrogram', 'frequencies', None, None, 'npy')
    >>> filename_parts('_spikeglx_ephysData_g0_t0.imec.wiring.json')
    ('spikeglx', 'ephysData_g0_t0', 'imec', None, 'wiring', 'json')
    >>> filename_parts('_spikeglx_ephysData_g0_t0.imec0.lf.bin')
    ('spikeglx', 'ephysData_g0_t0', 'imec0', None, 'lf', 'bin')
    >>> filename_parts('_ibl_trials.goCue_times_bpod.csv')
    ('ibl', 'trials', 'goCue_times', 'bpod', None, 'csv')

    Raises
    ------
    ValueError
        Invalid ALF dataset (assert_valid is True).
    """
    return _path_parts(filename, FILE_SPEC, True, as_dict, assert_valid)


def full_path_parts(path, as_dict=False, assert_valid=True) -> Union[dict, tuple]:
    """Parse all filename and folder parts.

    Parameters
    ----------
    path : str, pathlib.Path.
        The ALF path
    as_dict : bool
        When true a dict of matches is returned.
    assert_valid : bool
        When true an exception is raised when the filename cannot be parsed.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.

    Examples
    --------
    >>> full_path_parts(
    ...    'lab/Subjects/subject/2020-01-01/001/collection/#revision#/'
    ...    '_namespace_obj.times_timescale.extra.foo.ext')
    ('lab', 'subject', '2020-01-01', '001', 'collection', 'revision',
    'namespace', 'obj', 'times','timescale', 'extra.foo', 'ext')
    >>> full_path_parts('spikes.clusters.npy', as_dict=True)
    {'lab': None,
     'subject': None,
     'date': None,
     'number': None,
     'collection': None,
     'revision': None,
     'namespace': None,
     'object': 'spikes',
     'attribute': 'clusters',
     'timescale': None,
     'extra': None,
     'extension': 'npy'}

    Raises
    ------
    ValueError
        Invalid ALF path (assert_valid is True).
    """
    path = Path(path)
    # NB We try to determine whether we have a folder or filename path.  Filenames contain at
    # least two periods, however it is currently permitted to have any number of periods in a
    # collection, making the ALF path ambiguous.
    if sum(x == '.' for x in path.name) < 2:  # folder only
        folders = folder_parts(path, as_dict, assert_valid)
        dataset = filename_parts('', as_dict, assert_valid=False)
    elif '/' not in path.as_posix():  # filename only
        folders = folder_parts('', as_dict, assert_valid=False)
        dataset = filename_parts(path.name, as_dict, assert_valid)
    else:  # full filepath
        folders = folder_parts(path.parent, as_dict, assert_valid)
        dataset = filename_parts(path.name, as_dict, assert_valid)
    if as_dict:
        return OrderedDict(**folders, **dataset)
    else:
        return folders + dataset


def folder_parts(folder_path, as_dict=False, assert_valid=True) -> Union[dict, tuple]:
    """Parse all folder parts, including session, collection and revision.

    Parameters
    ----------
    folder_path : str, pathlib.Path
        The ALF folder path.
    as_dict : bool
        When true a dict of matches is returned.
    assert_valid : bool
        When true an exception is raised when the filename cannot be parsed.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.

    Examples
    --------
    >>> folder_parts('lab/Subjects/subject/2020-01-01/001/collection/#revision#')
    ('lab', 'subject', '2020-01-01', '001', 'collection', 'revision')
    >>> folder_parts(Path('lab/Subjects/subject/2020-01-01/001'), as_dict=True)
    {'lab': 'lab',
     'subject': 'subject',
     'date': '2020-01-01',
     'number': '001',
     'collection': None,
     'revision': None}

    Raises
    ------
    ValueError
        Invalid ALF path (assert_valid is True).
    """
    if hasattr(folder_path, 'as_posix'):
        folder_path = folder_path.as_posix()
    if folder_path and folder_path[-1] != '/':  # Slash required for regex pattern
        folder_path = folder_path + '/'
    spec_str = f'{SESSION_SPEC}/{COLLECTION_SPEC}'
    return _path_parts(folder_path, spec_str, False, as_dict, assert_valid)


def _isdatetime(s: str) -> bool:
    """Returns True if input is valid ISO date string."""
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_session_path(path: Union[str, Path]) -> Optional[Path]:
    """
    Returns the session path from any filepath if the date/number pattern is found,
    including the root directory.

    Returns
    -------
    pathlib.Path
        The session path part of the input path or None if path invalid.

    Examples
    --------
    >>> get_session_path('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001')
    Path('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001')

    >>> get_session_path('C:\\Data\\subject\\2020-01-01\\1\\trials.intervals.npy')
    Path('C:/Data/subject/2020-01-01/1')
    """
    if path is None:
        return
    if isinstance(path, str):
        path = Path(path)
    sess = None
    for i, p in enumerate(path.parts):
        if p.isdigit() and _isdatetime(path.parts[i - 1]):
            sess = Path().joinpath(*path.parts[:i + 1])

    return sess


def get_alf_path(path: Union[str, Path]) -> str:
    """Returns the ALF part of a path or filename.
    Attempts to return the first valid part of the path, first searching for a session path,
    then relative path (collection/revision/filename), then just the filename.  If all invalid,
    None is returned.

    Parameters
    ----------
    path : str, pathlib.Path
        A path to parse.

    Returns
    -------
    str
        A string containing the full ALF path, session path, relative path or filename.

    Examples
    --------
    >>> get_alf_path('etc/etc/lab/Subjects/subj/2021-01-21/001')
    'lab/Subjects/subj/2021-01-21/001/collection/file.attr.ext'

    >>> get_alf_path('etc/etc/subj/2021-01-21/001/collection/file.attr.ext')
    'subj/2021-01-21/001/collection/file.attr.ext'

    >>> get_alf_path('collection/file.attr.ext')
    'collection/file.attr.ext'
    """
    if not isinstance(path, str):
        path = Path(path).as_posix()
    path = path.strip('/')

    # Check if session path
    match_session = spec.regex(SESSION_SPEC).search(path)
    if match_session:
        return path[match_session.start():]

    # Check if filename / relative path (i.e. collection + filename)
    parts = path.rsplit('/', 1)
    match_filename = spec.regex(FILE_SPEC).match(parts[-1])
    if match_filename:
        return path if spec.regex(f'{COLLECTION_SPEC}{FILE_SPEC}').match(path) else parts[-1]


def add_uuid_string(file_path, uuid):
    """
    Add a UUID to the filename of an ALF path.

    Adds a UUID to an ALF filename as an extra part, e.g.
    'obj.attr.ext' -> 'obj.attr.a976e418-c8b8-4d24-be47-d05120b18341.ext'.

    Parameters
    ----------
    file_path : str, pathlib.Path, pathlib.PurePath
        An ALF path to add the UUID to.
    uuid : str, uuid.UUID
        The UUID to add.

    Returns
    -------
    pathlib.Path, pathlib.PurePath
        A new Path or PurePath object with a UUID in the filename.

    Examples
    --------
    >>> add_uuid_string('/path/to/trials.intervals.npy', 'a976e418-c8b8-4d24-be47-d05120b18341')
    Path('/path/to/trials.intervals.a976e418-c8b8-4d24-be47-d05120b18341.npy')

    Raises
    ------
    ValueError
        `uuid` must be a valid hyphen-separated hexadecimal UUID.

    See Also
    --------
    one.alf.files.remove_uuid_string
    one.alf.spec.is_uuid
    """
    if isinstance(uuid, str) and not spec.is_uuid_string(uuid):
        raise ValueError('Should provide a valid UUID v4')
    uuid = str(uuid)
    # NB: Only instantiate as Path if not already a Path, otherwise we risk changing the class
    if isinstance(file_path, str):
        file_path = Path(file_path)
    name_parts = file_path.stem.split('.')
    if uuid == name_parts[-1]:
        _logger.warning(f'UUID already found in file name: {file_path.name}: IGNORE')
        return file_path
    return file_path.parent.joinpath(f"{'.'.join(name_parts)}.{uuid}{file_path.suffix}")


def remove_uuid_string(file_path):
    """
    Remove UUID from a filename of an ALF path.

    Parameters
    ----------
    file_path : str, pathlib.Path, pathlib.PurePath
        An ALF path to add the UUID to.

    Returns
    -------
    pathlib.Path, pathlib.PurePath
        A new Path or PurePath object without a UUID in the filename.

    Examples
    --------
    >>> add_uuid_string('/path/to/trials.intervals.a976e418-c8b8-4d24-be47-d05120b18341.npy')
    Path('/path/to/trials.intervals.npy')

    >>> add_uuid_string('/path/to/trials.intervals.npy')
    Path('/path/to/trials.intervals.npy')

    See Also
    --------
    one.alf.files.add_uuid_string
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    name_parts = file_path.stem.split('.')

    if spec.is_uuid_string(name_parts[-1]):
        file_path = file_path.with_name('.'.join(name_parts[:-1]) + file_path.suffix)
    return file_path
