"""Module for identifying and parsing ALF file names.

An ALF file has the following components (those in brackets are optional):
    (_namespace_)object.attribute(_timescale)(.extra.parts).ext

Note the following:
    Object attributes may not contain an underscore unless followed by 'times' or 'intervals'.
    A namespace must not contain extra underscores (i.e. `name_space` and `__namespace__` are not
    valid).
    ALF files must always have an extension.

For more information, see the following documentation:
    https://int-brain-lab.github.io/ONE/alf_intro.html


ALFPath differences
-------------------
ALFPath.iter_datasets returns full paths (close the pathlib.Path.iterdir), whereas
alf.io.iter_datasets returns relative paths as POSIX strings (TODO).

ALFPath.parse_* methods return a dict by default, whereas parse_* functions return
tuples by default. Additionally, the parse_* functions raise ALFInvalid errors by
default if the path can't be parsed.  ALFPath.parse_* methods have no validation
option.

ALFPath properties return empty str instead of None if ALF part isn't present..
"""
import os
import pathlib
from collections import OrderedDict
from datetime import datetime
from typing import Union, Optional, Iterable
import logging

from iblutil.util import Listable

from .exceptions import ALFInvalid
from . import spec
from .spec import SESSION_SPEC, COLLECTION_SPEC, FILE_SPEC, REL_PATH_SPEC

_logger = logging.getLogger(__name__)
__all__ = [
    'ALFPath', 'PureALFPath', 'WindowsALFPath', 'PosixALFPath',
    'PureWindowsALFPath', 'PurePosixALFPath'
]


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
        If true an ALFInvalid is raised when the session cannot be parsed, otherwise an empty
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
        If true an ALFInvalid is raised when the session cannot be parsed, otherwise an empty
        dict of tuple of Nones is returned.

    Returns
    -------
    OrderedDict, tuple
        A dict if as_dict is true, or a tuple of parsed values.

    Raises
    ------
    ALFInvalid
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
    ALFInvalid
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
        raise ALFInvalid(path)
    else:
        return empty if as_dict else tuple(empty.values())


def filename_parts(filename, as_dict=False, assert_valid=True) -> Union[dict, tuple]:
    """Return the parsed elements of a given ALF filename.

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
    ALFInvalid
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
    ALFInvalid
        Invalid ALF path (assert_valid is True).

    """
    path = pathlib.Path(path)
    # NB We try to determine whether we have a folder or filename path.  Filenames contain at
    # least two periods, however it is currently permitted to have any number of periods in a
    # collection, making the ALF path ambiguous.
    if sum(x == '.' for x in path.name) < 2:  # folder only
        folders = folder_parts(path, as_dict, assert_valid)
        if assert_valid:
            # Edge case: ensure is indeed folder by checking that name is in parts
            invalid_file = path.name not in (folders.values() if as_dict else folders)
            is_revision = f'#{folders["revision"] if as_dict else folders[-1]}#' == path.name
            if not is_revision and invalid_file:
                raise ALFInvalid(path)
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
    ALFInvalid
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


def get_session_path(path: Union[str, pathlib.Path]) -> Optional[pathlib.Path]:
    """Return full session path from any file path if the date/number pattern is found.

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
        path = pathlib.Path(path)
    for i, p in enumerate(path.parts):
        if p.isdigit() and _isdatetime(path.parts[i - 1]):
            return path.__class__().joinpath(*path.parts[:i + 1])


def get_alf_path(path: Union[str, pathlib.Path]) -> str:
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
        path = pathlib.Path(path).as_posix()
    path = path.strip('/')

    # Check if session path
    if match_session := spec.regex(SESSION_SPEC).search(path):
        return path[match_session.start():]

    # Check if filename / relative path (i.e. collection + filename)
    parts = path.rsplit('/', 1)
    if spec.regex(FILE_SPEC).match(parts[-1]):
        return path if spec.regex(f'{COLLECTION_SPEC}{FILE_SPEC}').match(path) else parts[-1]


def add_uuid_string(file_path, uuid):
    """Add a UUID to the filename of an ALF path.

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
    one.alf.path.ALFPath.with_uuid
    one.alf.path.remove_uuid_string
    one.alf.spec.is_uuid

    """
    if isinstance(uuid, str) and not spec.is_uuid_string(uuid):
        raise ValueError('Should provide a valid UUID v4')
    uuid = str(uuid)
    # NB: Only instantiate as Path if not already a Path, otherwise we risk changing the class
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    name_parts = file_path.stem.split('.')
    if spec.is_uuid(name_parts[-1]):
        *name_parts, old_uuid = name_parts
        if old_uuid == uuid:
            _logger.warning(f'UUID already found in file name: {file_path.name}: IGNORE')
            return file_path
        else:
            _logger.debug('Replacing %s with %s in %s', old_uuid, uuid, file_path)
    return file_path.parent.joinpath(f"{'.'.join(name_parts)}.{uuid}{file_path.suffix}")


def remove_uuid_string(file_path):
    """Remove UUID from a filename of an ALF path.

    Parameters
    ----------
    file_path : str, pathlib.Path, pathlib.PurePath
        An ALF path to add the UUID to.

    Returns
    -------
    ALFPath, PureALFPath, pathlib.Path, pathlib.PurePath
        A new Path or PurePath object without a UUID in the filename.

    Examples
    --------
    >>> add_uuid_string('/path/to/trials.intervals.a976e418-c8b8-4d24-be47-d05120b18341.npy')
    Path('/path/to/trials.intervals.npy')

    >>> add_uuid_string('/path/to/trials.intervals.npy')
    Path('/path/to/trials.intervals.npy')

    See Also
    --------
    one.alf.path.ALFPath.without_uuid
    one.alf.path.add_uuid_string

    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    name_parts = file_path.stem.split('.')

    if spec.is_uuid_string(name_parts[-1]):
        file_path = file_path.with_name('.'.join(name_parts[:-1]) + file_path.suffix)
    return file_path


def padded_sequence(file_path):
    """Ensures a file path contains a zero-padded experiment sequence folder.

    Parameters
    ----------
    file_path : str, pathlib.Path, pathlib.PurePath
        A session or file path to convert.

    Returns
    -------
    ALFPath, PureALFPath
        The same path but with the experiment sequence folder zero-padded.  If a PurePath was
        passed, a PurePath will be returned, otherwise a Path object is returned.

    Examples
    --------
    >>> file_path = '/iblrigdata/subject/2023-01-01/1/_ibl_experiment.description.yaml'
    >>> padded_sequence(file_path)
    pathlib.Path('/iblrigdata/subject/2023-01-01/001/_ibl_experiment.description.yaml')

    Supports folders and will not affect already padded paths

    >>> session_path = pathlib.PurePosixPath('subject/2023-01-01/001')
    >>> padded_sequence(file_path)
    pathlib.PurePosixPath('subject/2023-01-01/001')

    """
    file_path = ensure_alf_path(file_path)
    if (session_path := get_session_path(file_path)) is None:
        raise ValueError('path must include a valid ALF session path, e.g. subject/YYYY-MM-DD/N')
    idx = len(file_path.parts) - len(session_path.parts)
    sequence = str(int(session_path.parts[-1])).zfill(3)  # zero-pad if necessary
    return file_path.parents[idx].joinpath(sequence, file_path.relative_to(session_path))


def without_revision(file_path):
    """Return file path without a revision folder.

    Parameters
    ----------
    file_path : str, pathlib.Path
        A valid ALF dataset path.

    Returns
    -------
    pathlib.Path
        The input file path without a revision folder.

    Examples
    --------
    >>> without_revision('/lab/Subjects/subject/2023-01-01/001/collection/#revision#/obj.attr.ext')
    Path('/lab/Subjects/subject/2023-01-01/001/collection/obj.attr.ext')

    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    *_, collection, revision = folder_parts(file_path.parent)
    return get_session_path(file_path).joinpath(*filter(None, (collection, file_path.name)))


class PureALFPath(pathlib.PurePath):  # py3.12 supports direct subclassing
    """Base class for manipulating Alyx file (ALF) paths without I/O.

    Similar to a pathlib PurePath object but with methods for validating, parsing, and replacing
    ALF path parts.

    Parameters
    ----------
    args : str, pathlib.PurePath
        One or more pathlike objects to combine into an ALF path object.

    """

    def __new__(cls, *args):
        """Construct a ALFPurePath from one or several strings and or existing PurePath objects.

        The strings and path objects are combined so as to yield a canonicalized path, which is
        incorporated into the new PurePath object.
        """
        if cls is PureALFPath:
            cls = PureWindowsALFPath if os.name == 'nt' else PurePosixALFPath
        return super().__new__(cls, *args)

    def is_dataset(self):
        """Determine if path is an ALF dataset, rather than a folder.

        Returns
        -------
        bool
            True if filename is ALF dataset.

        """
        return spec.is_valid(self.name)

    def is_valid_alf(path) -> bool:
        """Check if path is a valid ALF path.

        This returns true if the input path matches any part of the ALF path specification.
        This method can be used as a static method with any pathlike input, or as an instance
        method.  This will validate both directory paths and file paths.

        Parameters
        ----------
        path : str, pathlib.PurePath
            A path to check the validity of.

        Returns
        -------
        bool
            True if the path is recognized as a valid ALF path.

        Examples
        --------
        >>> ALFPath('/home/foo/2020-01-01/001').is_valid_alf()
        True

        >>> ALFPath('/home/foo/2020-01-01/001/alf/spikes.times.npy').is_valid_alf()
        True

        >>> ALFPath.is_valid_alf('_ibl_wheel.timestamps.npy')
        True

        >>> ALFPath.is_valid_alf('foo.bar')
        False

        See Also
        --------
        PureALFPath.is_dataset - Test whether file name is valid as well as directory path.
        full_path_parts - Validates path and returns the parsed ALF path parts.

        """
        try:
            return any(full_path_parts(path))
        except ALFInvalid:
            return False

    def is_session_path(path) -> bool:
        """Check if path is a valid ALF session path.

        This returns true if the input path matches the ALF session path specification.
        This method can be used as a static method with any pathlike input, or as an instance
        method.

        Parameters
        ----------
        path : str, pathlib.PurePath
            A session path to check the validity of.

        Returns
        -------
        bool
            True if the path is recognized as a valid ALF session path.

        Examples
        --------
        >>> ALFPath('/home/foo/2020-01-01/001').is_session_path()
        True

        >>> ALFPath('/home/foo/2020-01-01/001/alf/spikes.times.npy').is_session_path()
        False

        >>> ALFPath.is_session_path('_ibl_wheel.timestamps.npy')
        False

        >>> ALFPath.is_valid_alf('lab/Subjects/foo/2020-01-01/001')
        True

        See Also
        --------
        PureALFPath.is_valid_alf - Test whether path is generally valid a valid ALF path.
        PureALFPath.session_path_parts - Returns parsed session path parts as tuple of str.

        """
        return spec.is_session_path(path)

    def session_path(self):
        """Extract the full session path.

        Returns the session path from the filepath if the date/number pattern is found,
        including the root directory.

        Returns
        -------
        PureALFPath
            The session path part of the input path or None if path invalid.

        Examples
        --------
        >>> ALFPath('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001').session_path()
        ALFPath('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001')

        >>> ALFPath('C:\\Data\\subject\\2020-01-01\\1\\trials.intervals.npy').session_path()
        ALFPath('C:/Data/subject/2020-01-01/1')

        """
        return get_session_path(self)

    def session_path_short(self, include_lab=False) -> str:
        """Return only the ALF session path as a posix str.

        Params
        ------
        include_lab : bool
            If true, the lab/subject/date/number is returned, otherwise the lab part is dropped.

        Returns
        -------
        str
            The session path part of the input path or None if path invalid.

        Examples
        --------
        >>> ALFPath('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001').session_path_short()
        'subject/2020-01-01/001'

        >>> alfpath = ALFPath('/mnt/sd0/Data/lab/Subjects/subject/2020-01-01/001')
        >>> alfpath.session_path_short(include_lab=True)
        'lab/subject/2020-01-01/001'

        >>> ALFPath('C:\\Data\\subject\\2020-01-01\\1\\trials.intervals.npy').session_path_short()
        'subject/2020-01-01/1'

        """
        idx = 0 if include_lab else 1
        if any(parts := self.session_parts[idx:]):
            return '/'.join(parts)

    def without_lab(self) -> 'PureALFPath':
        """Return path without the <lab>/Subjects/ part.

        If the <lab>/Subjects pattern is not found, the same path is returned.

        Returns
        -------
        PureALFPath
            The same path without the <lab>/Subjects part.

        """
        p = self.as_posix()
        if m := spec.regex('{lab}/Subjects/').search(p):
            return self.__class__(p[:m.start()], p[m.end():])
        else:
            return self

    def relative_to_lab(self) -> 'PureALFPath':
        """Return path relative to <lab>/Subjects/ part.

        Returns
        -------
        PureALFPath
            The same path, relative to the <lab>/Subjects/ part.

        Raises
        ------
        ValueError
            The path doesn't contain a <lab>/Subjects/ pattern.

        """
        p = self.as_posix()
        if m := spec.regex('{lab}/Subjects/').search(p):
            return self.__class__(p[m.end():])
        else:
            raise ValueError(f'{self} does not contain <lab>/Subjects pattern')

    def relative_to_session(self):
        """Return path relative to session part.

        Returns
        -------
        PureALFPath
            The same path, relative to the <lab>/Subjects/<subject>/<date>/<number> part.

        Raises
        ------
        ValueError
            The path doesn't contain a <lab>/Subjects/ pattern.

        """
        if (session_path := self.session_path()):
            return self.relative_to(session_path)
        else:
            raise ValueError(f'{self} does not contain session path pattern')

    def parse_alf_path(self, as_dict=True):
        """Parse all filename and folder parts.

        Parameters
        ----------
        as_dict : bool
            When true a dict of matches is returned.

        Returns
        -------
        OrderedDict, tuple
            A dict if as_dict is true, or a tuple of parsed values.

        Examples
        --------
        >>> alfpath = PureALFPath(
        ...     'lab/Subjects/subject/2020-01-01/001/collection/#revision#/'
        ...     '_namespace_obj.times_timescale.extra.foo.ext')
        >>> alfpath.parse_alf_path()
        {'lab': 'lab',
        'subject': 'subject',
        'date': '2020-01-01',
        'number': '001',
        'collection': 'collection',
        'revision': 'revision',
        'namespace': 'namespace',
        'object': 'obj',
        'attribute': 'times',
        'timescale': 'timescale',
        'extra': 'extra.foo',
        'extension': 'ext'}

        >>> PureALFPath('_namespace_obj.times_timescale.extra.foo.ext').parse_alf_path()
        (None, None, None, None, None, None, 'namespace',
        'obj', 'times','timescale', 'extra.foo', 'ext')

        """
        return full_path_parts(self, assert_valid=False, as_dict=as_dict)

    def parse_alf_name(self, as_dict=True):
        """Return the parsed elements of a given ALF filename.

        Parameters
        ----------
        as_dict : bool
            When true a dict of matches is returned.

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
        >>> alfpath = PureALFPath(
        ...     'lab/Subjects/subject/2020-01-01/001/collection/#revision#/'
        ...     '_namespace_obj.times_timescale.extra.foo.ext')
        >>> alfpath.parse_alf_name()
        {'namespace': 'namespace',
        'object': 'obj',
        'attribute': 'times',
        'timescale': 'timescale',
        'extra': 'extra.foo',
        'extension': 'ext'}

        >>> PureALFPath('spikes.clusters.npy', as_dict=False)
        (None, 'spikes', 'clusters', None, None, npy)

        """
        return filename_parts(self.name, assert_valid=False, as_dict=as_dict)

    @property
    def dataset_name_parts(self):
        """tuple of str: the dataset name parts, with empty strings for missing parts."""
        return tuple(p or '' for p in self.parse_alf_name(as_dict=False))

    @property
    def session_parts(self):
        """tuple of str: the session path parts, with empty strings for missing parts."""
        return tuple(p or '' for p in session_path_parts(self, assert_valid=False))

    @property
    def alf_parts(self):
        """tuple of str: the full ALF path parts, with empty strings for missing parts."""
        return tuple(p or '' for p in self.parse_alf_path(as_dict=False))

    @property
    def namespace(self):
        """str: The namespace part of the ALF name, or and empty str if not present."""
        return self.dataset_name_parts[0]

    @property
    def object(self):
        """str: The object part of the ALF name, or and empty str if not present."""
        return self.dataset_name_parts[1]

    @property
    def attribute(self):
        """str: The attribute part of the ALF name, or and empty str if not present."""
        return self.dataset_name_parts[2]

    @property
    def timescale(self):
        """str: The timescale part of the ALF name, or and empty str if not present."""
        return self.dataset_name_parts[3]

    @property
    def extra(self):
        """str: The extra part of the ALF name, or and empty str if not present."""
        return self.dataset_name_parts[4]

    def with_object(self, obj):
        """Return a new path with the ALF object changed.

        Parameters
        ----------
        obj : str
            An ALF object name part to use.

        Returns
        -------
        PureALFPath
            The same file path but with the object part replaced with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(str(self))
        ns_obj, rest = self.name.split('.', 1)
        ns, _ = spec.regex(FILE_SPEC.split('\\.')[0]).match(ns_obj).groups()
        ns = f'_{ns}_' if ns else ''
        return self.with_name(f'{ns}{obj}.{rest}')

    def with_namespace(self, ns):
        """Return a new path with the ALF namespace added or changed.

        Parameters
        ----------
        namespace : str
            An ALF namespace part to use.

        Returns
        -------
        PureALFPath
            The same file path but with the namespace part added/replaced with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(self)
        ns_obj, rest = self.name.split('.', 1)
        _, obj = spec.regex(FILE_SPEC.split('\\.')[0]).match(ns_obj).groups()
        ns = f'_{ns}_' if ns else ''
        return self.with_name(f'{ns}{obj}.{rest}')

    def with_attribute(self, attr):
        """Return a new path with the ALF attribute changed.

        Parameters
        ----------
        attribute : str
            An ALF attribute part to use.

        Returns
        -------
        PureALFPath
            The same file path but with the attribute part replaced with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(self)
        ns_obj, attr_ts, rest = self.name.split('.', 2)
        _, ts = spec.regex('{attribute}(?:_{timescale})?').match(attr_ts).groups()
        ts = f'_{ts}' if ts else ''
        return self.with_name(f'{ns_obj}.{attr}{ts}.{rest}')

    def with_timescale(self, timescale):
        """Return a new path with the ALF timescale added or changed.

        Parameters
        ----------
        timescale : str
            An ALF timescale part to use.

        Returns
        -------
        PureALFPath
            The same file path but with the timescale part added/replaced with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(self)
        ns_obj, attr_ts, rest = self.name.split('.', 2)
        attr, _ = spec.regex('{attribute}(?:_{timescale})?').match(attr_ts).groups()
        ts = f'_{timescale}' if timescale else ''
        return self.with_name(f'{ns_obj}.{attr}{ts}.{rest}')

    def with_extra(self, extra, append=False):
        """Return a new path with extra ALF parts added or changed.

        Parameters
        ----------
        extra : str, list of str
            Extra ALF parts to add/replace.
        append : bool
            When false (default) any existing extra parts are replaced instead of added to.

        Returns
        -------
        PureALFPath
            The same file path but with the extra part(s) replaced or appended to with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(self)
        parts = self.stem.split('.', 2)
        if isinstance(extra, str):
            extra = extra.strip('.').split('.')
        if (prev := parts.pop() if len(parts) > 2 else None) and append:
            extra = (prev, *extra)
        obj_attr = '.'.join(parts)
        if extra := '.'.join(filter(None, extra)):
            return self.with_stem(f'{obj_attr}.{extra}')
        else:
            return self.with_stem(obj_attr)

    def with_extension(self, ext):
        """Return a new path with the ALF extension (suffix) changed.

        Note that unlike PurePath's `with_suffix` method, this asserts that the filename is a valid
        ALF dataset and the `ext` argument should be without the period.

        Parameters
        ----------
        ext : str
            An ALF extension part to use (sans period).

        Returns
        -------
        PureALFPath
            The same file path but with the extension part replaced with the input.

        Raises
        ------
        ALFInvalid
            The path is not a valid ALF dataset (e.g. doesn't have a three-part filename, or
            contains invalid characters).

        """
        if not self.is_dataset():
            raise ALFInvalid(str(self))
        return self.with_suffix(f'.{ext}')

    def with_padded_sequence(path):
        """Ensures a file path contains a zero-padded experiment sequence folder.

        Parameters
        ----------
        path : str pathlib.PurePath
            A session or file path to convert.

        Returns
        -------
        ALFPath, PureALFPath
            The same path but with the experiment sequence folder zero-padded.  If a PurePath was
            passed, a PurePath will be returned, otherwise a Path object is returned.

        Examples
        --------
        Supports calling as static function

        >>> file_path = '/iblrigdata/subject/2023-01-01/1/_ibl_experiment.description.yaml'
        >>> ALFPath.with_padded_sequence(file_path)
        ALFPath('/iblrigdata/subject/2023-01-01/001/_ibl_experiment.description.yaml')

        Supports folders and will not affect already padded paths

        >>> ALFPath('subject/2023-01-01/001').with_padded_sequence(file_path)
        ALFPath('subject/2023-01-01/001')

        """
        return padded_sequence(path)

    def with_revision(self, revision):
        """Return a new path with the ALF revision part added/changed.

        Parameters
        ----------
        revision : str
            An ALF revision part to use (NB: do not include the pound sign '#').

        Returns
        -------
        PureALFPath
            The same file path but with the revision part added or replaced with the input.

        Examples
        --------
        If not in the ALF path, one will be added

        >>> ALFPath('/subject/2023-01-01/1/alf/obj.attr.ext').with_revision('revision')
        ALFPath('/subject/2023-01-01/1/alf/#xxx#/obj.attr.ext')

        If a revision is already in the ALF path it will be replaced

        >>> ALFPath('/subject/2023-01-01/1/alf/#revision#/obj.attr.ext').with_revision('xxx')
        ALFPath('/subject/2023-01-01/1/alf/#xxx#/obj.attr.ext')

        Raises
        ------
        ALFInvalid
            The ALF path is not valid or is relative to the session path.  The path must include
            the session parts otherwise the path is too ambiguous to determine validity.
        ALFInvalid
            The revision provided does not match the ALF specification pattern.

        See Also
        --------
        PureALFPath.without_revision

        """
        # Validate the revision input
        revision, = _path_parts(revision, '^{revision}$', match=True, assert_valid=True)
        if PureALFPath.is_dataset(self):
            return self.without_revision().parent / f'#{revision}#' / self.name
        else:
            return self.without_revision() / f'#{revision}#'

    def without_revision(self):
        """Return a new path with the ALF revision part removed.

        Returns
        -------
        PureALFPath
            The same file path but with the revision part removed.

        Examples
        --------
        If not in the ALF path, no change occurs

        >>> ALFPath('/subject/2023-01-01/1/alf/obj.attr.ext').with_revision('revision')
        ALFPath('/subject/2023-01-01/1/alf/obj.attr.ext')

        If a revision is in the ALF path it will be removed

        >>> ALFPath('/subject/2023-01-01/1/alf/#revision#/obj.attr.ext').without_revision()
        ALFPath('/subject/2023-01-01/1/alf/obj.attr.ext')

        Raises
        ------
        ALFInvalid
            The ALF path is not valid or is relative to the session path.  The path must include
            the session parts otherwise the path is too ambiguous to determine validity.

        See Also
        --------
        PureALFPath.with_revision

        """
        if PureALFPath.is_dataset(self):
            # Is a file path (rather than folder path)
            return without_revision(self)
        if not self.is_valid_alf():
            raise ALFInvalid(f'{self} not a valid ALF path or is relative to session')
        elif spec.regex('^#{revision}#$').match(self.name):
            # Includes revision
            return self.parent
        else:
            # Does not include revision
            return self

    def with_uuid(self, uuid):
        """Return a new path with the ALF UUID part added/changed.

        Parameters
        ----------
        uuid : str, uuid.UUID
            The UUID to add.

        Returns
        -------
        PureALFPath
            A new ALFPath object with a UUID in the filename.

        Examples
        --------
        >>> uuid = 'a976e418-c8b8-4d24-be47-d05120b18341'
        >>> ALFPath('/path/to/trials.intervals.npy').with_uuid(uuid)
        ALFPath('/path/to/trials.intervals.a976e418-c8b8-4d24-be47-d05120b18341.npy')

        Raises
        ------
        ValueError
            `uuid` must be a valid hyphen-separated hexadecimal UUID.
        ALFInvalid
            Path is not a valid ALF file path.

        """
        if not self.is_dataset():
            raise ALFInvalid(f'{self} is not a valid ALF dataset file path')
        return add_uuid_string(self, uuid)

    def without_uuid(self):
        """Return a new path with the ALF UUID part removed.

        Returns
        -------
        PureALFPath
            A new ALFPath object with a UUID removed from the filename, if present.

        Examples
        --------
        >>> alfpath = ALFPath('/path/to/trials.intervals.a976e418-c8b8-4d24-be47-d05120b18341.npy')
        >>> alfpath.without_uuid(uuid)
        ALFPath('/path/to/trials.intervals.npy')

        >>> ALFPath('/path/to/trials.intervals.npy').without_uuid(uuid)
        ALFPath('/path/to/trials.intervals.npy')

        """
        return remove_uuid_string(self) if self.is_dataset() else self


class ALFPath(PureALFPath):
    """Base class for manipulating Alyx file (ALF) paths with system calls.

    Similar to a pathlib Path object but with methods for validating, parsing, and replacing ALF
    path parts. This class also contains methods that work on system files.

    Parameters
    ----------
    args : str, pathlib.PurePath
        One or more pathlike objects to combine into an ALF path object.

    """

    def __new__(cls, *args):
        """Construct a ALFPurePath from one or several strings and or existing PurePath objects.

        The strings and path objects are combined so as to yield a canonicalized path, which is
        incorporated into the new PurePath object.
        """
        return super().__new__(WindowsALFPath if os.name == 'nt' else PosixALFPath, *args)

    def is_dataset(self) -> bool:
        """Determine if path is an ALF dataset, rather than a folder.

        Unlike pathlib and PureALFPath methods, this will return False if the path exists but
        is a folder, otherwise this simply tests the path name, whether it exists or not.

        Returns
        -------
        bool
            True if filename is ALF dataset.

        """
        return not self.is_dir() and spec.is_valid(self.name)

    def is_session_path(self) -> bool:
        """Check if path is a valid ALF session path.

        This returns true if the input path matches the ALF session path specification.
        This method can be used as a static method with any pathlike input, or as an instance
        method.

        Unlike the PureALFPath method, this will return false if the path matches but is in fact
        a file on disk.

        Parameters
        ----------
        path : str, pathlib.PurePath
            A session path to check the validity of.

        Returns
        -------
        bool
            True if the path is recognized as a valid ALF session path.

        Examples
        --------
        >>> ALFPath('/home/foo/2020-01-01/001').is_session_path()
        True

        >>> ALFPath('/home/foo/2020-01-01/001/alf/spikes.times.npy').is_session_path()
        False

        >>> ALFPath.is_session_path('_ibl_wheel.timestamps.npy')
        False

        >>> ALFPath.is_valid_alf('lab/Subjects/foo/2020-01-01/001')
        True

        See Also
        --------
        PureALFPath.is_valid_alf - Test whether path is generally valid a valid ALF path.
        PureALFPath.session_path_parts - Returns parsed session path parts as tuple of str.

        """
        return not self.is_file() and spec.is_session_path(self)

    def is_valid_alf(path) -> bool:
        """Check if path is a valid ALF path.

        This returns true if the input path matches any part of the ALF path specification.
        This method can be used as a static method with any pathlike input, or as an instance
        method.  This will validate both directory paths and file paths.

        Unlike the PureALFPath method, this one will return false if the path matches a dataset
        file pattern but is actually a folder on disk, or if the path matches as a file but is
        is a folder on disk.

        Parameters
        ----------
        path : str, pathlib.PurePath
            A path to check the validity of.

        Returns
        -------
        bool
            True if the path is recognized as a valid ALF path.

        Examples
        --------
        >>> ALFPath('/home/foo/2020-01-01/001').is_valid_alf()
        True

        >>> ALFPath('/home/foo/2020-01-01/001/alf/spikes.times.npy').is_valid_alf()
        True

        >>> ALFPath.is_valid_alf('_ibl_wheel.timestamps.npy')
        True

        >>> ALFPath.is_valid_alf('foo.bar')
        False

        See Also
        --------
        PureALFPath.is_dataset - Test whether file name is valid as well as directory path.
        full_path_parts - Validates path and returns the parsed ALF path parts.

        """
        try:
            parsed = full_path_parts(path, as_dict=True)
        except ALFInvalid:
            return False
        is_dataset = parsed['object'] is not None
        if isinstance(path, str):
            path = ALFPath(path)
        if hasattr(path, 'is_file') and path.is_file():
            return is_dataset
        elif hasattr(path, 'is_dir') and path.is_dir():
            return not is_dataset
        return True

    def iter_datasets(self, recursive=False):
        """Iterate over all files in path, and yield relative dataset paths.

        Parameters
        ----------
        recursive : bool
            If true, yield datasets in subdirectories.

        Yields
        ------
        ALFPath
            The next valid dataset path in lexicographical order.

        See Also
        --------
        one.alf.io.iter_datasets - Equivalent function that can take any pathlike input and returns
         paths relative to the input path.

        """
        glob = self.rglob if recursive else self.glob
        for p in sorted(glob('*.*.*')):
            if not p.is_dir() and p.is_dataset:
                yield p


class PureWindowsALFPath(pathlib.PureWindowsPath, PureALFPath):
    """PureALFPath subclass for Windows systems."""

    pass


class PurePosixALFPath(pathlib.PurePosixPath, PureALFPath):
    """PureALFPath subclass for non-Windows systems."""

    pass


class WindowsALFPath(pathlib.WindowsPath, ALFPath):
    """ALFPath subclass for Windows systems."""

    pass


class PosixALFPath(pathlib.PosixPath, ALFPath):
    """ALFPath subclass for non-Windows systems."""

    pass


def ensure_alf_path(path) -> Listable(PureALFPath):
    """Ensure path is a PureALFPath instance.

    Ensures the path entered is cast to a PureALFPath instance. If input class is PureALFPath or
    pathlib.PurePath, a PureALFPath instance is returned, otherwise an ALFPath instance is
    returned.

    Parameters
    ----------
    path : str, pathlib.PurePath, ALFPath, iterable
        One or more path-like objects.

    Returns
    -------
    ALFPath, PureALFPath, list of ALFPath, list of PureALFPath
        One or more ALFPath objects.

    Raises
    ------
    TypeError
        Unexpected path instance; input must be a str or pathlib.PurePath instance, or an
        iterable thereof.

    """
    if isinstance(path, PureALFPath):
        # Already an ALFPath instance
        return path
    if isinstance(path, pathlib.PurePath):
        # Cast pathlib instance to equivalent ALFPath
        if isinstance(path, pathlib.Path):
            return ALFPath(path)
        elif isinstance(path, pathlib.PurePosixPath):
            return PurePosixALFPath(path)
        elif isinstance(path, pathlib.PureWindowsPath):
            return PureWindowsALFPath(path)
        else:
            return PureALFPath(path)
    if isinstance(path, str):
        # Cast str to ALFPath
        return ALFPath(path)
    if isinstance(path, Iterable):
        # Cast list, generator, tuple, etc. to list of ALFPath
        return list(map(ensure_alf_path, path))
    raise TypeError(f'expected os.PathLike type, got {type(path)} instead')
