"""
Module for identifying and parsing ALF file names.

An ALF file has the following components (those in brackets are optional):
    (_namespace_)object.attribute(_timescale)(.extra.parts).ext

Note the following:
    Object attributes may not contain an underscore unless followed by 'times' or 'intervals'.
    A namespace must not contain extra underscores (i.e. `name_space` and `__namespace__` are not
    valid)
    ALF files must always have an extension

For more information, see the following documentation:
    https://int-brain-lab.github.io/iblenv/one_docs/one_reference.html#alf

Created on Tue Sep 11 18:06:21 2018

@author: Miles
"""
import re
import os
from fnmatch import fnmatch

# to include underscores: r'(?P<namespace>(?:^_)\w+(?:_))?'
# to treat _times and _intervals as timescale: (?P<attribute>[a-zA-Z]+)_?
ALF_EXP = re.compile(
    r'^_?(?P<namespace>(?<=_)[a-zA-Z0-9]+)?_?'
    r'(?P<object>\w+)\.'
    r'(?P<attribute>[a-zA-Z0-9]+(?:_times(?=[_\b.])|_intervals(?=[_\b.]))?)_?'
    r'(?P<timescale>(?:_?)\w+)*\.?'
    r'(?P<extra>[.\w-]+)*\.'
    r'(?P<extension>\w+$)')

"""The following are the specifications and patterns for ALFs"""
SESSION_SPEC = '{lab}/(Subjects/)?{subject}/{date}/{number}'
COLLECTION_SPEC = r'{collection}(/#{revision}#)?'
FILE_SPEC = r'_?{namespace}?_?{object}\.{attribute}_?{timescale}*\.?{extra}*\.{extension}$'
FULL_SPEC = f'{SESSION_SPEC}/{COLLECTION_SPEC}/{FILE_SPEC}'
_DEFAULT = (
    ('lab', r'\w+'),
    ('subject', r'[\w-]+'),
    ('date', r'\d{4}-\d{2}-\d{2}'),
    ('number', r'\d{1,3}'),
    ('collection', r'[\w/]+'),
    ('revision', r'[\w-]+'),  # brackets
    ('namespace', '(?<=_)[a-zA-Z0-9]+'),  # brackets
    ('object', r'\w+'),
    ('attribute', r'[a-zA-Z0-9]+(?:_times(?=[_\b.])|_intervals(?=[_\b.]))?'),  # brackets
    ('timescale', r'(?:_?)\w+'),  # brackets
    ('extra', r'[.\w-]+'),  # brackets
    ('extension', r'\w+')
)


def _named(pattern, name):
    """Wraps a regex pattern in a named capture group"""
    return f'(?P<{name}>{pattern})'


def regex(spec: str = FULL_SPEC, **kwargs) -> str:
    """
    Construct a regular expression pattern for parsing or validating an ALF

    Examples:
        # Regex for a filename
        pattern = regex(spec=FILE_SPEC)

        # Regex for a complete path (including root)
        pattern = '.*' + regex(spec=FULL_SPEC)

        # Regex pattern for specific object name
        pattern = regex(object='trials)

    :param spec: The spec string to construct the regular expression from
    :param kwargs: Optional patterns to replace the defaults
    :return: A regular expression pattern string
    """
    fields = dict(_DEFAULT)
    if not fields.keys() >= kwargs.keys():
        unknown = next(k for k in kwargs.keys() if k not in fields.keys())
        raise KeyError(f'Unknown field "{unknown}"')
    fields.update({k: v for k, v in kwargs.items() if v is not None})
    return spec.format(**{k: _named(fields[k], k) for k in re.findall(r'(?<={)\w+', spec)})


def is_valid(filename):
    """
    Returns a True for a given file name if it is an ALF file, otherwise returns False

    Examples:
        >>> is_valid('trials.feedbackType.npy')
        True
        >>> is_valid('_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.metadata.json')
        True
        >>> is_valid('spike_train.npy')
        False
        >>> is_valid('channels._phy_ids.csv')
        False

    Args:
        filename (str): The name of the file

    Returns:
        bool
    """
    return ALF_EXP.match(filename) is not None


def alf_parts(filename, as_dict=False):
    """
    Return the parsed elements of a given ALF filename.

    Examples:
        >>> alf_parts('_namespace_obj.times_timescale.extra.foo.ext')
        ('namespace', 'obj', 'times', 'timescale', 'extra.foo', 'ext')
        >>> alf_parts('spikes.clusters.npy', as_dict=True)
        {'namespace': None,
         'object': 'spikes',
         'attribute': 'clusters',
         'timescale': None,
         'extra': None,
         'extension': 'npy'}
        >>> alf_parts('spikes.times_ephysClock.npy')
        (None, 'spikes', 'times', 'ephysClock', None, 'npy')
        >>> alf_parts('_iblmic_audioSpectrogram.frequencies.npy')
        ('iblmic', 'audioSpectrogram', 'frequencies', None, None, 'npy')
        >>> alf_parts('_spikeglx_ephysData_g0_t0.imec.wiring.json')
        ('spikeglx', 'ephysData_g0_t0', 'imec', None, 'wiring', 'json')
        >>> alf_parts('_spikeglx_ephysData_g0_t0.imec0.lf.bin')
        ('spikeglx', 'ephysData_g0_t0', 'imec0', None, 'lf', 'bin')
        >>> alf_parts('_ibl_trials.goCue_times_bpod.csv')
        ('ibl', 'trials', 'goCue_times', 'bpod', None, 'csv')

    Args:
        filename (str): The name of the file
        as_dict (bool): when True a dict of matches is returned

    Returns:
        namespace (str): The _namespace_ or None if not present
        object (str): ALF object
        attribute (str): The ALF attribute
        timescale (str): The ALF _timescale or None if not present
        extra (str): Any extra parts to the filename, or None if not present
        extension (str): The file extension
    """
    m = ALF_EXP.match(filename)
    if not m:
        raise ValueError('Invalid ALF filename')
    return m.groupdict() if as_dict else m.groups()


def _dromedary(string) -> str:
    """
    Convert a string to camel case.  Acronyms/initialisms are preserved.

    Examples:
        _dromedary('Hello world') == 'helloWorld'
        _dromedary('motion_energy') == 'motionEnergy'
        _dromedary('passive_RFM') == 'passive RFM'
        _dromedary('FooBarBaz') == 'fooBarBaz'

    :param string: To be converted to camel case
    :return: The string in camel case
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


def to_alf(object, attribute, extension, namespace=None, timescale=None, extra=None):
    """
    Given a set of ALF file parts, return a valid ALF file name.  Essential periods and
    underscores are added by the function.

    Args:
        object (str): The ALF object name
        attribute (str): The ALF object attribute name
        extension (str): The file extension
        namespace (str): An optional namespace
        timescale (str): An optional timescale
        extra (str, tuple): One or more optional extra ALF attributes

    Returns:
        filename (str): a file name string built from the ALF parts

    Examples:
    >>> to_alf('spikes', 'times', 'ssv')
    'spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl')
    '_ibl_spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl', timescale='ephysClock')
    '_ibl_spikes.times_ephysClock.ssv'
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
    # Ensure parts are camel case (converts whitespace and snake case)
    object, timescale = map(_dromedary, (object, timescale))

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


def filter_by(alf_path, **kwargs):
    """
    Given a path and optional filters, returns all ALF files and their associated parts. The
    filters constitute a logical AND.

    Args:
        alf_path (str): A Path to a directory containing ALF files
        object (str): filter by a given object (e.g. 'spikes')
        attribute (str): filter by a given attribute (e.g. 'intervals')
        extension (str): filter by extension (e.g. 'npy')
        namespace (str): filter by a given namespace (e.g. 'ibl') or None for files without one
        timescale (str): filter by a given timescale (e.g. 'bpod') or None for files without one
        extra (str, list): filter by extra parameters (e.g. 'raw') or None for files without extra
                           parts. NB: Wild cards not permitted here.

    Returns:
        alf_files (list): list of ALF files and tuples of their parts
        attributes (list of dicts): list of parsed file parts

    Examples:
        # Filter files with universal timescale
        filter_by(alf_path, timescale=None)

        # Filter files by a given ALF object
        filter_by(alf_path, object='wheel')

        # Filter using wildcard, e.g. 'wheel' and 'wheelMoves' ALF objects
        filter_by(alf_path, object='wh*')

        # Filter all intervals that are in bpod time
        filter_by(alf_path, attribute='intervals', timescale='bpod')
    """
    alf_files = [f for f in os.listdir(alf_path) if is_valid(f)]
    attributes = [alf_parts(f, as_dict=True) for f in alf_files]

    if kwargs:
        # Validate keyword arguments against regex group names
        invalid = kwargs.keys() - ALF_EXP.groupindex.keys()
        if invalid:
            raise TypeError("%s() got an unexpected keyword argument '%s'"
                            % (__name__, set(invalid).pop()))

        # Ensure 'extra' input is a list; if str split on dot
        if 'extra' in kwargs and isinstance(kwargs['extra'], str):
            kwargs['extra'] = kwargs['extra'].split('.')

        # Iterate over ALF files
        for file, attr in zip(alf_files.copy(), attributes.copy()):
            for k, v in kwargs.items():  # Iterate over attributes
                if v is None or attr[k] is None:
                    # If either is None, both should be None to match
                    match = v is attr[k]
                elif k == 'extra':
                    # Check all provided extra fields match those in ALF
                    match = all(elem in attr[k].split('.') for elem in v if elem)
                else:
                    # Check given attribute matches, allowing wildcards
                    match = fnmatch(attr[k], v)

                if not match:  # Remove file from list and move on to next file
                    alf_files.remove(file)
                    attributes.remove(attr)
                    break

    return alf_files, [tuple(attr.values()) for attr in attributes]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
