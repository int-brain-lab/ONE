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
    https://int-brain-lab.github.io/iblenv/one_docs/one_reference.html#alf  # FIXME Change link

Created on Tue Sep 11 18:06:21 2018

@author: Miles
"""
import re
import os
from fnmatch import fnmatch
from collections import OrderedDict

from .spec import is_valid, COLLECTION_SPEC, FILE_SPEC, regex

# FIXME Replace ALF_EXP with FILE_SPEC
# to include underscores: r'(?P<namespace>(?:^_)\w+(?:_))?'
# to treat _times and _intervals as timescale: (?P<attribute>[a-zA-Z]+)_?
ALF_EXP = re.compile(
    r'^_?(?P<namespace>(?<=_)[a-zA-Z0-9]+)?_?'
    r'(?P<object>\w+)\.'
    r'(?P<attribute>[a-zA-Z0-9]+(?:_times(?=[_\b.])|_intervals(?=[_\b.]))?)_?'
    r'(?P<timescale>(?:_?)\w+)*\.?'
    r'(?P<extra>[.\w-]+)*\.'
    r'(?P<extension>\w+$)')


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


def rel_path_parts(rel_path, as_dict=False, assert_valid=True):
    # TODO Can be simplified if we make collections optional in spec regex
    empty_collection = OrderedDict.fromkeys(('collection', 'revision'))
    if hasattr(rel_path, 'as_posix'):
        rel_path = rel_path.as_posix()
    par_path = rel_path.rsplit('/', 1)
    if len(par_path) == 1:
        m = filename_parts(par_path[0], as_dict=as_dict, assert_valid=assert_valid)
        if any(m.values() if as_dict else m):
            return OrderedDict(**empty_collection, **m) if as_dict else (None, None, *m)
        else:
            empty = re.compile(regex(f'{COLLECTION_SPEC}{FILE_SPEC}')).groupindex.keys()
            return OrderedDict.fromkeys(empty) if as_dict else tuple([None] * len(empty))
    else:
        folders, filname = par_path
        rel_parts = re.match(regex(f'^{COLLECTION_SPEC}$'), folders + '/')
        file_parts = filename_parts(filname, as_dict=as_dict, assert_valid=assert_valid)
        if rel_parts:
            return OrderedDict(**rel_parts.groupdict(), **file_parts)\
                if as_dict else (*rel_parts.groupdict().values(), *file_parts)
        elif assert_valid:
            raise ValueError('Invalid collection/revision')
        return OrderedDict({**empty_collection, **file_parts})\
            if as_dict else (None, None, *file_parts)


def session_path_parts(session_path: str) -> dict:
    pass


def filename_parts(filename, as_dict=False, assert_valid=True):
    pattern = re.compile(regex(FILE_SPEC))
    empty = OrderedDict.fromkeys(pattern.groupindex.keys())
    m = pattern.match(str(filename))
    if m:  # py3.8
        return OrderedDict(m.groupdict()) if as_dict else m.groups()
    elif assert_valid:
        raise ValueError('Invalid ALF filename')
    else:
        return empty if as_dict else empty.values()


def path_parts(file_path: str) -> dict:
    pass


def folder_parts(folder_path: str) -> dict:
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
