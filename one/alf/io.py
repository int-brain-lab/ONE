"""I/O functions for ALyx Files.

Provides support for time-series reading and interpolation as per the specifications
For a full overview of the scope of the format, see:

https://int-brain-lab.github.io/ONE/alf_intro.html
"""

import json
import copy
import logging
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from iblutil.util import Bunch
from iblutil.io import parquet
from iblutil.io import jsonable
from .exceptions import ALFObjectNotFound
from . import files, spec
from .spec import FILE_SPEC

_logger = logging.getLogger(__name__)


class AlfBunch(Bunch):
    """A dict-like object that supports dot indexing and conversion to DataFrame"""

    @property
    def check_dimensions(self):
        """int: 0 for consistent dimensions, 1 for inconsistent dimensions"""
        return check_dimensions(self)

    def append(self, b, inplace=False):
        """
        Appends one bunch to another, key by key

        Parameters
        ----------
        b : Bunch, dict
            A Bunch of data to append
        inplace : bool
            If true, the data are appended in place, otherwise a copy is returned

        Returns
        -------
        ALFBunch, None
            An ALFBunch with the data appended, or None if inplace is True
        """
        # default is to return a copy
        if inplace:
            a = self
        else:
            a = AlfBunch(copy.deepcopy(self))
        # handles empty bunches for convenience if looping
        if b == {}:
            return a
        if a == {}:
            return AlfBunch(b)
        # right now supports only strictly matching keys. Will implement other cases as needed
        if set(a.keys()) != set(b.keys()):
            raise NotImplementedError('Append bunches only works with strictly matching keys'
                                      'For more complex merges, convert to pandas dataframe.')
        # do the merge; only concatenate lists and np arrays right now
        for k in a:
            if isinstance(a[k], np.ndarray):
                a[k] = np.concatenate((a[k], b[k]), axis=0)
            elif isinstance(a[k], list):
                a[k].extend(b[k])
            else:
                _logger.warning(f'bunch key "{k}" is a {a[k].__class__}. I don\'t know how to'
                                f' handle that. Use pandas for advanced features')
        if a.check_dimensions != 0:
            print_sizes = '\n'.join(f'{v.shape},\t{k}' for k, v in a.items())
            _logger.warning(f'Inconsistent dimensions for object: \n{print_sizes}')

        return a

    def to_df(self) -> pd.DataFrame:
        """Return DataFrame with data keys as columns"""
        return dataframe(self)

    @staticmethod
    def from_df(df) -> 'AlfBunch':
        data = dict(zip(df.columns, df.values.T))
        split_keys = sorted(x for x in data.keys() if re.match(r'.+?_[01]$', x))
        for x1, x2 in zip(*[iter(split_keys)] * 2):
            data[x1[:-2]] = np.c_[data.pop(x1), data.pop(x2)]
        return AlfBunch(data)


def dataframe(adict):
    """
    Converts an Bunch conforming to size conventions into a pandas DataFrame.
    For 2-D arrays, stops at 10 columns per attribute.

    Parameters
    ----------
    adict : dict, Bunch
        A dict-like object of data to convert to DataFrame

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame of data
    """
    if check_dimensions(adict) != 0:
        raise ValueError('Can only convert to DataFrame objects with consistent size')
    # easy case where there are only vectors
    if all([len(adict[k].shape) == 1 for k in adict]):
        return pd.DataFrame(adict)
    # pandas has trouble with 2d data, chop it off with a limit of 10 columns per dataset
    df = pd.DataFrame()
    for k in adict.keys():
        if adict[k].ndim == 1:
            df[k] = adict[k]
        elif adict[k].ndim == 2 and adict[k].shape[1] == 1:
            df[k] = adict[k][:, 0]
        elif adict[k].ndim == 2:
            for i in np.arange(adict[k].shape[1]):
                df[f"{k}_{i}"] = adict[k][:, i]
                if i == 9:
                    break
        else:
            _logger.warning(f'{k} attribute is 3D or more and won\'t convert to dataframe')
            continue
    return df


def _find_metadata(file_alf) -> Path:
    """
    File path for an existing meta-data file for an alf_file

    Parameters
    ----------
    file_alf : str, pathlib.Path
        A path of existing ALF

    Returns
    -------
    pathlib.Path
        Path of meta-data file if exists
    """
    file_alf = Path(file_alf)
    ns, obj = file_alf.name.split('.')[:2]
    return next(file_alf.parent.glob(f'{ns}.{obj}*.metadata*.json'), None)


def read_ts(filename):
    """
    Load time-series from ALF format

    Parameters
    ----------
    filename : str, pathlib.Path
        An ALF path whose values to load

    Returns
    -------
    numpy.ndarray
        An array of timestamps belonging to the ALF path object
    numpy.ndarray
         An array of values in filename

    Examples
    --------
    >>> t, d = read_ts(filename)
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    # alf format is object.attribute.extension, for example '_ibl_wheel.position.npy'
    _, obj, attr, *_, ext = files.filename_parts(filename.parts[-1])

    try:
        # looking for matching object with attribute timestamps: '_ibl_wheel.timestamps.npy'
        (time_file,), _ = filter_by(filename.parent, object=obj, attribute='times*', extension=ext)
        assert time_file
    except (ValueError, AssertionError):
        name = spec.to_alf(obj, attr, ext)
        raise FileNotFoundError(name + ' not found! No time-scale for ' + str(filename))

    ts = np.load(filename.parent / time_file)
    val = np.load(filename)
    # Ensure timestamps
    return ts2vec(ts, val.shape[0]), _ensure_flat(val)


def _ensure_flat(arr):
    """
    Given a single column array, returns a flat vector.  Other shapes are returned unchanged.

    Parameters
    ----------
    arr : numpy.array
        An array with shape (n, 1)

    Returns
    -------
    numpy.ndarray
        A vector with shape (n,)
    """
    return arr.flatten() if arr.ndim == 2 and arr.shape[1] == 1 else arr


def ts2vec(ts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Interpolate a continuous timeseries of the shape (2, 2)

    Parameters
    ----------
    ts : numpy.array
        a 2x2 numpy array of the form (sample, ts)
    n_samples : int
        Number of samples; i.e. the size of the resulting vector

    Returns
    -------
    numpy.ndarray
        A vector of interpolated timestamps
    """
    if len(ts.shape) == 1:
        return ts
    elif ts.ndim == 2 and ts.shape[1] == 1:
        return ts.flatten()  # Deal with MATLAB single column array
    if ts.ndim > 2 or ts.shape[1] != 2:
        raise ValueError('Array shape should be (2, 2)')
    # Linearly interpolate the times
    x = np.arange(n_samples)
    return np.interp(x, ts[:, 0], ts[:, 1])


def check_dimensions(dico):
    """
    Test for consistency of dimensions as per ALF specs in a dictionary.

    Alf broadcasting rules: only accepts consistent dimensions for a given axis
    a dimension is consistent with another if it's empty, 1, or equal to the other arrays
    dims [a, 1],  [1, b] and [a, b] are all consistent, [c, 1] is not

    Parameters
    ----------
    dico : ALFBunch, dict
        Dictionary containing data

    Returns
    -------
    int
        Status 0 for consistent dimensions, 1 for inconsistent dimensions
    """
    supported = (np.ndarray, pd.DataFrame)  # Data types that have a shape attribute
    shapes = [dico[lab].shape for lab in dico
              if isinstance(dico[lab], supported) and not lab.startswith('timestamps')]
    first_shapes = [sh[0] for sh in shapes]
    # Continuous timeseries are permitted to be a (2, 2)
    timeseries = [k for k, v in dico.items()
                  if k.startswith('timestamps') and isinstance(v, np.ndarray)]
    if any(timeseries):
        for key in timeseries:
            if dico[key].ndim == 1 or (dico[key].ndim == 2 and dico[key].shape[1] == 1):
                # Should be vector with same length as other attributes
                first_shapes.append(dico[key].shape[0])
            elif dico[key].ndim > 1 and dico[key].shape != (2, 2):
                return 1  # ts not a (2, 2) arr or a vector

    ok = len(first_shapes) == 0 or set(first_shapes).issubset({max(first_shapes), 1})
    return int(ok is False)


def load_file_content(fil):
    """
    Returns content of files. Designed for very generic file formats:
    so far supported contents are `json`, `npy`, `csv`, `(h)tsv`, `ssv`, `jsonable`

    Parameters
    ----------
    fil : str, pathlib.Path
        File to read

    Returns
    -------
    any
        Array/json/pandas dataframe depending on format
    """
    if not fil:
        return
    fil = Path(fil)
    if fil.stat().st_size == 0:
        return
    if fil.suffix == '.csv':
        return pd.read_csv(fil).squeeze('columns')
    if fil.suffix == '.json':
        try:
            with open(fil) as _fil:
                return json.loads(_fil.read())
        except Exception as e:
            _logger.error(e)
            return None
    if fil.suffix == '.jsonable':
        return jsonable.read(fil)
    if fil.suffix == '.npy':
        return _ensure_flat(np.load(file=fil, allow_pickle=True))
    if fil.suffix == '.pqt':
        return parquet.load(fil)[0]
    if fil.suffix == '.ssv':
        return pd.read_csv(fil, delimiter=' ').squeeze('columns')
    if fil.suffix in ('.tsv', '.htsv'):
        return pd.read_csv(fil, delimiter='\t').squeeze('columns')
    if fil.suffix in ('.yml', '.yaml'):
        with open(fil, 'r') as _fil:
            return yaml.safe_load(_fil)
    return Path(fil)


def _ls(alfpath, object=None, **kwargs) -> (list, tuple):
    """
    Given a path, an object and a filter, returns all files and associated attributes

    Parameters
    ----------
    alfpath : str, pathlib.Path
        The folder to list
    object : str, list
        An ALF object name to filter by
    wildcards : bool
        If true uses unix shell style pattern matching, otherwise uses regular expressions
    kwargs : dict
        Other ALF parts to filter, including namespace, attribute, etc.

    Returns
    -------
    list
        A list of ALF paths
    tuple
        A tuple of ALF attributes corresponding to the file paths

    Raises
    ------
    ALFObjectNotFound
        No matching ALF object was found in the alfpath directory
    """
    alfpath = Path(alfpath)
    if not alfpath.exists():
        files_alf = attributes = None
    elif alfpath.is_dir():
        if object is None:
            # List all ALF files
            files_alf, attributes = filter_by(alfpath, **kwargs)
        else:
            files_alf, attributes = filter_by(alfpath, object=object, **kwargs)
    else:
        object = files.filename_parts(alfpath.name)[1]
        alfpath = alfpath.parent
        files_alf, attributes = filter_by(alfpath, object=object, **kwargs)

    # raise error if no files found
    if not files_alf:
        err_str = f'object "{object}" ' if object else 'ALF files'
        raise ALFObjectNotFound(f'No {err_str} found in {alfpath}')

    return [alfpath.joinpath(f) for f in files_alf], attributes


def iter_sessions(root_dir):
    """
    Recursively iterate over session paths in a given directory.

    Parameters
    ----------
    root_dir : str, pathlib.Path
        The folder to look for sessions.

    Yields
    -------
    pathlib.Path
        The next session path in lexicographical order.
    """
    if spec.is_session_path(root_dir):
        yield root_dir
    for path in sorted(Path(root_dir).rglob('*')):
        if path.is_dir() and spec.is_session_path(path):
            yield path


def iter_datasets(session_path):
    """
    Iterate over all files in a session, and yield relative dataset paths.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The folder to look for datasets.

    Yields
    -------
    pathlib.Path
        The next dataset path (relative to the session path) in lexicographical order.
    """
    for p in sorted(Path(session_path).rglob('*.*')):
        if not p.is_dir() and spec.is_valid(p.name):
            yield p.relative_to(session_path)


def exists(alfpath, object, attributes=None, **kwargs) -> bool:
    """
    Test if ALF object and optionally specific attributes exist in the given path

    Parameters
    ----------
    alfpath : str, pathlib.Path
        The folder to look into
    object : str
        ALF object name
    attributes : str, list
        Wanted attributes
    wildcards : bool
        If true uses unix shell style pattern matching, otherwise uses regular expressions
    kwargs : dict
        Other ALF parts to filter by

    Returns
    -------
    bool
        For multiple attributes, returns True only if all attributes are found
    """

    # if the object is not found, return False
    try:
        _, attributes_found = _ls(alfpath, object, **kwargs)
    except (FileNotFoundError, ALFObjectNotFound):
        return False

    # if object found and no attribute provided, True
    if not attributes:
        return True

    # if attributes provided, test if all are found
    if isinstance(attributes, str):
        attributes = [attributes]
    attributes_found = set(part[2] for part in attributes_found)
    return set(attributes).issubset(attributes_found)


def load_object(alfpath, object=None, short_keys=False, **kwargs):
    """Reads all files sharing the same object name.

    For example, if the file provided to the function is `spikes.times`, the function will
    load `spikes.times`, `spikes.clusters`, `spikes.depths`, `spike.amps` in a dictionary
    whose keys will be `times`, `clusters`, `depths`, `amps`

    Full Reference here: https://int-brain-lab.github.io/ONE/alf_intro.html

    Simplified example: _namespace_object.attribute_timescale.part1.part2.extension

    Parameters
    ----------
    alfpath : str, pathlib.Path, list
        Any ALF path pertaining to the object OR directory containing ALFs OR list of paths
    object : str, list, None
        The ALF object(s) to filter by.  If a directory is provided and object is None, all valid
        ALF files returned
    short_keys : bool
        By default, the output dictionary keys will be compounds of attributes, timescale and
        any eventual parts separated by a dot. Use True to shorten the keys to the attribute
        and timescale
    wildcards : bool
        If true uses unix shell style pattern matching, otherwise uses regular expressions
    kwargs : dict
        Other ALF parts to filter by

    Returns
    -------
    AlfBunch
        An ALFBunch (dict-like) of all attributes pertaining to the object

    Examples
    --------
        Load 'spikes' object

        >>> spikes = load_object('full/path/to/my/alffolder/', 'spikes')

        Load 'trials' object under the 'ibl' namespace

        >>> trials = load_object('/subject/2021-01-01/001', 'trials', namespace='ibl')
    """
    if isinstance(alfpath, (Path, str)):
        if Path(alfpath).is_dir() and object is None:
            raise ValueError('If a directory is provided, the object name should be provided too')
        files_alf, parts = _ls(alfpath, object, **kwargs)
    else:  # A list of paths allows us to load an object from different revisions
        files_alf = alfpath
        parts = [files.filename_parts(x.name) for x in files_alf]
        assert len(set(p[1] for p in parts)) == 1
        object = next(x[1] for x in parts)
    # Take attribute and timescale from parts list
    attributes = [p[2] if not p[3] else '_'.join(p[2:4]) for p in parts]
    if not short_keys:  # Include extra parts in the keys
        attributes = [attr + ('.' + p[4] if p[4] else '') for attr, p in zip(attributes, parts)]
        # TODO List duplicates; raise ALFError
    assert len(set(attributes)) == len(attributes), (
        f'multiple object {object} with the same attribute in {alfpath}, restrict parts/namespace')
    out = AlfBunch({})

    # load content for each file
    for fil, att in zip(files_alf, attributes):
        # if there is a corresponding metadata file, read it:
        meta_data_file = _find_metadata(fil)
        # if this is the actual meta-data file, skip and it will be read later
        if meta_data_file == fil:
            continue
        out[att] = load_file_content(fil)
        if meta_data_file:
            meta = load_file_content(meta_data_file)
            # the columns keyword splits array along the last dimension
            if 'columns' in meta.keys():
                out.update({v: out[att][::, k] for k, v in enumerate(meta['columns'])})
                out.pop(att)
                meta.pop('columns')
            # if there is other stuff in the dictionary, save it, otherwise disregard
            if meta:
                out[att + 'metadata'] = meta
    if 'table' in out.keys():  # Merge 'table' dataframe into bunch
        out.update(AlfBunch.from_df(out.pop('table')))
    status = out.check_dimensions
    timeseries = [k for k in out.keys() if 'timestamps' in k]
    if any(timeseries) and len(out.keys()) > len(timeseries) and status == 0:
        # Get length of one of the other arrays
        ignore = ('timestamps', 'meta')
        n_samples = next(v for k, v in out.items() if not any(x in k for x in ignore)).shape[0]
        for key in timeseries:
            # Expand timeseries if necessary
            out[key] = ts2vec(out[key], n_samples)
    if status != 0:
        print_sizes = '\n'.join(f'{v.shape},\t{k}' for k, v in out.items())
        _logger.warning(f'Inconsistent dimensions for object: {object} \n{print_sizes}')
    return out


def save_object_npy(alfpath, dico, object, parts=None, namespace=None, timescale=None) -> list:
    """
    Saves a dictionary in `ALF format`_ using object as object name and dictionary keys as
    attribute names. Dimensions have to be consistent.

    Simplified ALF example: _namespace_object.attribute.part1.part2.extension

    Parameters
    ----------
    alfpath : str, pathlib.Path
        Path of the folder to save data to
    dico : dict
        Dictionary to save to npy; keys correspond to ALF attributes
    object : str
        Name of the object to save
    parts : str, list, None
        Extra parts to the ALF name
    namespace : str, None
        The optional namespace of the object
    timescale : str, None
        The optional timescale of the object

    Returns
    -------
    list
        List of written files

    Examples
    --------
    >>> spikes = {'times': np.arange(50), 'depths': np.random.random(50)}
    >>> files = save_object_npy('/path/to/my/alffolder/', spikes, 'spikes')

    .. _ALF format:
        https://int-brain-lab.github.io/ONE/alf_intro.html
    """
    alfpath = Path(alfpath)
    status = check_dimensions(dico)
    if status != 0:
        raise ValueError('Dimensions are not consistent to save all arrays in ALF format: ' +
                         str([(k, v.shape) for k, v in dico.items()]))
    out_files = []
    for k, v in dico.items():
        out_file = alfpath / spec.to_alf(object, k, 'npy',
                                         extra=parts, namespace=namespace, timescale=timescale)
        np.save(out_file, v)
        out_files.append(out_file)
    return out_files


def save_metadata(file_alf, dico) -> None:
    """Writes a meta data file matching a current ALF file object.

    For example given an alf file `clusters.ccfLocation.ssv` this will write a dictionary in JSON
    format in `clusters.ccfLocation.metadata.json`

    Reserved keywords:
        - **columns**: column names for binary tables.
        - **row**: row names for binary tables.
        - **unit**

    Parameters
    ----------
    file_alf : str, pathlib.Path
        Full path to the alf object
    dico : dict, ALFBunch
        Dictionary containing meta-data
    """
    assert spec.is_valid(file_alf.parts[-1]), 'ALF filename not valid'
    file_meta_data = file_alf.parent / (file_alf.stem + '.metadata.json')
    with open(file_meta_data, 'w+') as fid:
        fid.write(json.dumps(dico, indent=1))


def remove_uuid_file(file_path, dry=False) -> Path:
    """
    Renames a file without the UUID and returns the new pathlib.Path object
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    name_parts = file_path.name.split('.')
    if not spec.is_uuid_string(name_parts[-2]):
        return file_path
    name_parts.pop(-2)
    new_path = file_path.parent.joinpath('.'.join(name_parts))
    if not dry and file_path.exists():
        file_path.replace(new_path)
    return new_path


def remove_uuid_recursive(folder, dry=False) -> None:
    """
    Within a folder, recursive renaming of all files to remove UUID
    """
    for fn in Path(folder).rglob('*.*'):
        print(remove_uuid_file(fn, dry=dry))


def next_num_folder(session_date_folder: Union[str, Path]) -> str:
    """Return the next number for a session given a session_date_folder"""
    session_date_folder = Path(session_date_folder)
    if not session_date_folder.exists():
        return '001'
    session_nums = [
        int(x.name) for x in session_date_folder.iterdir()
        if x.is_dir() and not x.name.startswith('.') and x.name.isdigit()
    ]
    out = f'{max(session_nums or [0]) + 1:03d}'
    assert len(out) == 3, 'ALF spec does not support session numbers > 999'
    return out


def remove_empty_folders(folder: Union[str, Path]) -> None:
    """Will iteratively remove any children empty folders"""
    all_folders = sorted(x for x in Path(folder).rglob('*') if x.is_dir())
    for f in reversed(all_folders):  # Reversed sorted ensures we remove deepest first
        try:
            f.rmdir()
        except Exception:
            continue


def filter_by(alf_path, wildcards=True, **kwargs):
    """
    Given a path and optional filters, returns all ALF files and their associated parts. The
    filters constitute a logical AND.  For all but `extra`, if a list is provided, one or more
    elements must match (a logical OR).

    Parameters
    ----------
    alf_path : str, pathlib.Path
        A path to a folder containing ALF datasets
    wildcards : bool
        If true, kwargs are matched as unix-style patterns, otherwise as regular expressions
    object : str, list
        Filter by a given object (e.g. 'spikes')
    attribute : str, list
        Filter by a given attribute (e.g. 'intervals')
    extension : str, list
        Filter by extension (e.g. 'npy')
    namespace : str, list
        Filter by a given namespace (e.g. 'ibl') or None for files without one
    timescale : str, list
        Filter by a given timescale (e.g. 'bpod') or None for files without one
    extra : str, list
        Filter by extra parameters (e.g. 'raw') or None for files without extra parts
        NB: Wild cards not permitted here.

    Returns
    -------
    alf_files : str
        A Path to a directory containing ALF files
    attributes : list of dicts
        A list of parsed file parts

    Examples
    --------
    Filter files with universal timescale

    >>> filter_by(alf_path, timescale=None)

    Filter files by a given ALF object

    >>> filter_by(alf_path, object='wheel')

    Filter using wildcard, e.g. 'wheel' and 'wheelMoves' ALF objects

    >>> filter_by(alf_path, object='wh*')

    Filter all intervals that are in bpod time

    >>> filter_by(alf_path, attribute='intervals', timescale='bpod')

    Filter all files containing either 'intervals' OR 'timestamps' attributes

    >>> filter_by(alf_path, attribute=['intervals', 'timestamps'])

    Filter all files using a regular expression

    >>> filter_by(alf_path, object='^wheel.*', wildcards=False)
    >>> filter_by(alf_path, object=['^wheel$', '.*Moves'], wildcards=False)
    """
    alf_files = [f for f in os.listdir(alf_path) if spec.is_valid(f)]
    attributes = [files.filename_parts(f, as_dict=True) for f in alf_files]

    if kwargs:
        # Validate keyword arguments against regex group names
        invalid = kwargs.keys() - spec.regex(FILE_SPEC).groupindex.keys()
        if invalid:
            raise TypeError('%s() got an unexpected keyword argument "%s"'
                            % (__name__, set(invalid).pop()))

        # # Ensure 'extra' input is a list; if str split on dot
        if 'extra' in kwargs and isinstance(kwargs['extra'], str):
            kwargs['extra'] = kwargs['extra'].split('.')

        def _match(part, pattern, split=None):
            if pattern is None or part is None:
                # If either is None, both should be None to match
                return pattern is part
            elif split:
                # Check all provided extra fields match those in ALF
                return all(elem in part.split(split) for elem in pattern if elem)
            elif not isinstance(pattern, str):
                if wildcards:
                    return any(_match(part, x, split) for x in pattern)
                else:
                    return re.match('|'.join(pattern), part) is not None
            else:
                # Check given attribute matches, allowing wildcards
                return fnmatch(part, pattern) if wildcards else re.match(pattern, part) is not None

        # Iterate over ALF files
        for file, attr in zip(alf_files.copy(), attributes.copy()):
            for k, v in kwargs.items():  # Iterate over attributes
                match = _match(attr[k], v, '.' if k == 'extra' else None)

                if not match:  # Remove file from list and move on to next file
                    alf_files.remove(file)
                    attributes.remove(attr)
                    break

    return alf_files, [tuple(attr.values()) for attr in attributes]
