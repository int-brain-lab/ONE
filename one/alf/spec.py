"""The complete ALF specification descriptors and validators."""
import re
import textwrap
from uuid import UUID
from typing import Union

from iblutil.util import flatten

"""dict: The ALF part names and their definitions."""
SPEC_DESCRIPTION = {
    'lab': 'The name of the lab where the data were collected (optional).',
    'Subjects': 'An optional directory to indicate that the experiment data are divided by '
                'subject.  If organizing by lab, this directory is required.',
    'subject': 'The subject name, typically an arbitrary label',
    'date': 'The date on which the experiment session took place, in ISO format, i.e. yyyy-mm-dd',
    'number': 'The sequential session number of the day, optionally zero-padded to be three '
              'numbers, e.g. 001, 002, etc.',
    'collection': 'An optional folder to group data by modality, device, etc.  This is necessary '
                  'when a session contains multiple measurements of the same type, from example '
                  'spike times from multiple probes.  Label examples include "probe00", '
                  '"raw_video_data".',
    'revision': 'An optional folder to organize data by version.  The version label is arbitrary, '
                'however the folder must start and end with pound signs, e.g. "#v1.0.0#". '
                'Unlike collections, if a specified revision is not found, the previous revision '
                'will be returned.  The revisions are ordered lexicographically.',
    'namespace': 'An option filename prefix for data that are not not expected to be a community '
                 'standard, for example task specific events.  The namespace may also be used to '
                 'indicate data unique to a given piece of hardware or software, and is '
                 'identified by underscores, e.g. "_iblrig_", "_phy_".',
    'object': 'Every file describing a given object has the same number of rows (i.e. the 1st '
              'dimension of an npy file, number of frames in a video file, etc).  You can '
              'therefore think of the files for an object as together defining a table, with '
              'column headings given by the attribute in the file names, and values given by the '
              'file contents.  Object names should be in Haskell case and pluralized, '
              'e.g. "wheelMoves", "sparseNoise", "trials".\nEncoding of relations between objects '
              'can be achieved by a simplified relational model.  If the attribute name of one '
              'file matches the object name of a second, then the first file is guaranteed to '
              'contain integers referring to the rows of the second. For example, '
              '"spikes.clusters.npy" would contain integer references to the rows of '
              '"clusters.brain_location.json" and "clusters.probes.npy"; and '
              '"clusters.probes.npy" would contain integer references to "probes.insertion.json". '
              '\nBe careful of plurals ("clusters.probe.npy" would not correspond to '
              '"probes.insertion.json") and remember we count arrays starting from 0.',
    'attribute': 'Together with the object, the attribute represents the type of data in the '
                 'file, for example "times", "amplitudes", "clusters".  The names should be in '
                 'Haskell case, however the following three attributes may be separated by an '
                 'underscore, e.g. "stimOn_times".\nThe attribute "times" is reserved for '
                 'discrete event times and comprises a numerical array containing times of the '
                 'events in seconds, relative to a universal timescale common to all files.\n'
                 'The attribute "intervals" should have two columns, indicating the start and end '
                 'times of each interval relative to the universal timescale.\n'
                 'Continuous timeseries are represented by the "timestamps" attribute.  The file '
                 'may contain a vector of times in universal seconds if unevenly sampled, or two '
                 'rows each representing a synchronization point, the first column giving the '
                 'sample number (counting from 0), and the second column giving the '
                 'corresponding time in universal seconds.  The times corresponding to all '
                 'samples are then found by linear interpolation.  NB: the "timestamps" file is '
                 'an exception to the rule that all files representing a continuous timeseries '
                 'object must have one row per sample, as it will often have substantially less.',
    'timescale': 'If you want to represent times relative to another (non-universal) timescale, '
                 'a timescale can be appended after an underscore e.g. '
                 '"spikes.times_ephysClock.npy", "trials.intervals_nidaq", '
                 '"wheel.timestamps_bpod.csv".',
    'extra': 'File names could have as many optional parts as you like: '
             '"object.attribute.x1.x2.[…].xN.extension".  The extra name parts play no formal '
             'role, but can serve several additional purposes. For example, it could be a UUID or '
             'file hash for archiving purposes.  If there are multiple files with the same '
             'object, attribute, and extensions but different extra parts, these should be '
             'treated as files to be concatenated, for example to allow multiple-part tif files '
             'as produced by scanimage to be encoded in ALF. The concatenation would happen in '
             'hierarchical lexicographical order: i.e. by lexicographic order of x1, '
             'then x2, etc.',
    'extension': 'ALF can deal with any sort of file, as long as it has a concept of a number of '
                 'rows (or primary dimension). The type of file is recognized by its extension. \n'
                 'Preferred choices:\n\n.npy: numpy array file. This is recommended over flat '
                 'binary since datatype and shape is stored in the file.  If you have an array of '
                 '3 or more dimensions, the first dimension counts as the number of rows.\n\n'
                 '.tsv: tab-delimited text file. This is recommended over comma-separated files'
                 'since text fields often have commas in. All rows should have the same number '
                 'of columns. The first row contains tab-separated names for each column.\n\n'
                 '.bin: flat binary file. It’s better to use .npy for storing binary data but '
                 'some recording systems save in flat binary.  Rather than convert them, '
                 'you can ALFize a flat binary file by adding a metadata file, which specifies '
                 'the number of columns (as the size of the "columns" array) and the binary '
                 'datatype as a top-level key "dtype", using numpy naming conventions.'
}
"""dict: The ALF part names and their definitions."""

# ========================================================== #
# The following are the specifications and patterns for ALFs #
# ========================================================== #

SESSION_SPEC = '({lab}/Subjects/)?{subject}/{date}/{number}'
"""str: The session specification pattern"""

COLLECTION_SPEC = r'({collection}/)?(#{revision}#/)?'
"""str: The collection and revision specification pattern"""

FILE_SPEC = r'_?{namespace}?_?{object}\.{attribute}(?:_{timescale})?(?:\.{extra})*\.{extension}$'
"""str: The filename specification pattern"""

REL_PATH_SPEC = f'{COLLECTION_SPEC}{FILE_SPEC}'
"""str: The collection, revision and filename specification pattern"""

FULL_SPEC = f'{SESSION_SPEC}/{REL_PATH_SPEC}'
"""str: The full ALF path specification pattern"""

_DEFAULT = (
    ('lab', r'\w+'),
    ('subject', r'[\w.-]+'),
    ('date', r'\d{4}-\d{2}-\d{2}'),
    ('number', r'\d{1,3}'),
    ('collection', r'[\w./-]+'),
    ('revision', r'[\w.-]+'),  # brackets
    # to include underscores: r'(?P<namespace>(?:^_)\w+(?:_))?'
    ('namespace', '(?<=_)[a-zA-Z0-9]+'),  # brackets
    ('object', r'\w+'),
    # to treat _times and _intervals as timescale: (?P<attribute>[a-zA-Z]+)_?
    # (?:_[a-z]+_)? allows attribute level namespaces (deprecated)
    ('attribute', r'(?:_[a-z]+_)?[a-zA-Z0-9]+(?:_times(?=[_.])|_intervals(?=[_.]))?'),  # brackets
    ('timescale', r'\w+'),  # brackets
    ('extra', r'[.\w-]+'),  # brackets
    ('extension', r'\w+')
)


def path_pattern() -> str:
    """Returns a template string representing the where the ALF parts lie in an ALF path.
    Brackets denote optional parts.  This is used for documentation purposes only.
    """
    return ''.join(filter(lambda c: c not in '{}?*\\$', FULL_SPEC))


def describe(part=None, width=99):
    """Print a description of an ALF part.
    Prints the path pattern along with a description of the given ALF part (or all parts if None).

    Parameters
    ----------
    part : str
        ALF part to describe.  One from `SPEC_DESCRIPTION.keys()`.  If None, all parts are
        described.
    width : int
        The max line length.

    Returns
    -------
    None

    Examples
    -------
    >>> describe()
    >>> describe('collection')
    >>> describe('extension', width=120)
    """
    full_spec = path_pattern()
    print(full_spec)
    if part:
        if part not in SPEC_DESCRIPTION.keys():
            all_parts = '"' + '", "'.join(SPEC_DESCRIPTION.keys()) + '"'
            raise ValueError(f'Unknown ALF part "{part}", should be one of {all_parts}')
        parts = [part]
        span = re.search(part, full_spec).span()
        ' ' * len(full_spec)
        print(' ' * span[0] + '^' * (span[1] - span[0]) + ' ' * (len(full_spec) - span[1]))
    else:
        parts = SPEC_DESCRIPTION.keys()
    for part in parts:
        print('\n' + part.upper())
        # Split by max width
        lines = flatten(textwrap.wrap(ln, width, replace_whitespace=False)
                        for ln in SPEC_DESCRIPTION[part].splitlines())
        [print(ln) for ln in lines]


def _dromedary(string) -> str:
    """
    Convert a string to camel case.  Acronyms/initialisms are preserved.

    Parameters
    ----------
    string : str
        To be converted to camel case

    Returns
    -------
    str
        The string in camel case

    Examples
    --------
    >>> _dromedary('Hello world') == 'helloWorld'
    >>> _dromedary('motion_energy') == 'motionEnergy'
    >>> _dromedary('passive_RFM') == 'passive RFM'
    >>> _dromedary('FooBarBaz') == 'fooBarBaz'

    See Also
    --------
    readableALF
    """
    def _capitalize(x):
        return x if x.isupper() else x.capitalize()
    if not string:  # short circuit on None and ''
        return string
    first, *other = re.split(r'[_\s]', string)
    if len(other) == 0:
        # Already camel/Pascal case, ensure first letter lower case
        return first[0].lower() + first[1:]
    # Convert to camel case, preserving all-uppercase elements
    first = first if first.isupper() else first.casefold()
    return ''.join([first, *map(_capitalize, other)])


def _named(pattern, name):
    """Wraps a regex pattern in a named capture group"""
    return f'(?P<{name}>{pattern})'


def regex(spec: str = FULL_SPEC, **kwargs) -> re.Pattern:
    """
    Construct a regular expression pattern for parsing or validating an ALF

    Parameters
    ----------
    spec : str
        The spec string to construct the regular expression from
    kwargs : dict[str]
        Optional patterns to replace the defaults

    Returns
    -------
    re.Pattern
        A regular expression Pattern object

    Examples
    --------
    Regex for a filename

    >>> pattern = regex(spec=FILE_SPEC)

    Regex for a complete path (including root)

    >>> pattern = '.*' + regex(spec=FULL_SPEC).pattern

    Regex pattern for specific object name

    >>> pattern = regex(object='trials')
    """
    fields = dict(_DEFAULT)
    if not fields.keys() >= kwargs.keys():
        unknown = next(k for k in kwargs.keys() if k not in fields.keys())
        raise KeyError(f'Unknown field "{unknown}"')
    fields.update({k: v for k, v in kwargs.items() if v is not None})
    spec_str = spec.format(**{k: _named(fields[k], k) for k in re.findall(r'(?<={)\w+', spec)})
    return re.compile(spec_str)


def is_valid(filename):
    """
    Returns a True for a given file name if it is an ALF file, otherwise returns False

    Parameters
    ----------
    filename : str
        The name of the file to evaluate

    Returns
    -------
    bool
        True if filename is valid ALF

    Examples
    --------
    >>> is_valid('trials.feedbackType.npy')
    True
    >>> is_valid('_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.metadata.json')
    True
    >>> is_valid('spike_train.npy')
    False
    >>> is_valid('channels._phy_ids.csv')  # WARNING: attribute level namespaces are deprecated
    True
    """
    return regex(FILE_SPEC).match(filename) is not None


def is_session_path(path_object):
    """
    Checks if the syntax corresponds to a session path.  Note that there is no physical check
    about existence nor contents

    Parameters
    ----------
    path_object : str, pathlib.Path
        The path object to validate

    Returns
    -------
    bool
        True if session path a valid ALF session path
    """
    session_spec = re.compile(regex(SESSION_SPEC).pattern + '$')
    if hasattr(path_object, 'as_posix'):
        path_object = path_object.as_posix()
    path_object = path_object.strip('/')
    return session_spec.search(path_object) is not None


def is_uuid_string(string: str) -> bool:
    """
    Bool test for randomly generated hexadecimal uuid validity.
    NB: unlike is_uuid, is_uuid_string checks that uuid is correctly hyphen separated
    """
    return isinstance(string, str) and is_uuid(string, (3, 4, 5)) and str(UUID(string)) == string


def is_uuid(uuid: Union[str, int, bytes, UUID], versions=(4,)) -> bool:
    """Bool test for randomly generated hexadecimal uuid validity.
    Unlike `is_uuid_string`, this function accepts UUID objects
    """
    if not isinstance(uuid, (UUID, str, bytes, int)):
        return False
    elif not isinstance(uuid, UUID):
        try:
            uuid = UUID(uuid) if isinstance(uuid, str) else UUID(**{type(uuid).__name__: uuid})
        except ValueError:
            return False
    return isinstance(uuid, UUID) and uuid.version in versions


def to_alf(object, attribute, extension, namespace=None, timescale=None, extra=None):
    """
    Given a set of ALF file parts, return a valid ALF file name.  Essential periods and
    underscores are added by the function.

    Parameters
    ----------
    object : str
        The ALF object name
    attribute : str
        The ALF object attribute name
    extension : str
        The file extension
    namespace : str
        An optional namespace
    timescale : str, tuple
        An optional timescale
    extra : str, tuple
        One or more optional extra ALF attributes

    Returns
    -------
    str
        A file name string built from the ALF parts

    Examples
    --------
    >>> to_alf('spikes', 'times', 'ssv')
    'spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl')
    '_ibl_spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl', timescale='ephysClock')
    '_ibl_spikes.times_ephysClock.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl', timescale=('ephys clock', 'minutes'))
    '_ibl_spikes.times_ephysClock_minutes.ssv'
    >>> to_alf('spikes', 'times', 'npy', namespace='ibl', timescale='ephysClock', extra='raw')
    '_ibl_spikes.times_ephysClock.raw.npy'
    >>> to_alf('wheel', 'timestamps', 'npy', 'ibl', 'bpod', ('raw', 'v12'))
    '_ibl_wheel.timestamps_bpod.raw.v12.npy'
    """
    # Validate inputs
    if not extension:
        raise TypeError('An extension must be provided')
    elif extension.startswith('.'):
        extension = extension[1:]
    if re.search('_(?!times$|intervals)', attribute):
        raise ValueError('Object attributes must not contain underscores')
    if any(pt is not None and '.' in pt for pt in
           (object, attribute, namespace, extension, timescale)):
        raise ValueError('ALF parts must not contain a period (`.`)')
    if '_' in (namespace or ''):
        raise ValueError('Namespace must not contain extra underscores')
    if object[0] == '_':
        raise ValueError('Objects must not contain underscores; use namespace arg instead')
    # Ensure parts are camel case (converts whitespace and snake case)
    if timescale:
        timescale = filter(None, [timescale] if isinstance(timescale, str) else timescale)
        timescale = '_'.join(map(_dromedary, timescale))
    object = _dromedary(object)

    # Optional extras may be provided as string or tuple of strings
    if not extra:
        extra = ()
    elif isinstance(extra, str):
        extra = extra.split('.')

    # Construct ALF file
    parts = (('_%s_' % namespace if namespace else '') + object,
             attribute + ('_%s' % timescale if timescale else ''),
             *extra,
             extension)
    return '.'.join(parts)


def readableALF(name: str, capitalize: bool = False) -> str:
    """Convert camel case string to space separated string.

    Given an ALF object name or attribute, return a string where the camel case words are space
    separated.  Acronyms/initialisms are preserved.

    Parameters
    ----------
    name : str
        The ALF part to format (e.g. object name or attribute).
    capitalize : bool
        If true, return with the first letter capitalized.

    Returns
    -------
    str
        The name formatted for display, with spaces and capitalization.

    Examples
    --------
    >>> readableALF('sparseNoise') == 'sparse noise'
    >>> readableALF('someROIDataset') == 'some ROI dataset'
    >>> readableALF('someROIDataset', capitalize=True) == 'Some ROI dataset'

    See Also
    --------
    _dromedary
    """
    words = []
    i = 0
    matches = re.finditer(r'[A-Z](?=[a-z0-9])|(?<=[a-z0-9])[A-Z]', name)
    for j in map(re.Match.start, matches):
        words.append(name[i:j])
        i = j
    words.append(name[i:])
    display_str = ' '.join(map(lambda s: s if s.isupper() else s.lower(), words))
    return display_str[0].upper() + display_str[1:] if capitalize else display_str
