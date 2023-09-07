"""A module for inter-converting experiment identifiers.

There are multiple ways to uniquely identify an experiment:
    - eid (str) : An experiment UUID as a string
    - np (int64) : An experiment UUID encoded as 2 int64s
    - path (Path) : A pathlib ALF path of the form `<lab>/Subjects/<subject>/<date>/<number>`
    - ref (str) : An experiment reference string of the form `yyyy-mm-dd_n_subject`
    - url (str) : A remote http session path of the form `<lab>/Subjects/<subject>/<date>/<number>`
"""
import re
import functools
import datetime
from uuid import UUID
from inspect import unwrap
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from typing import Optional, Union, Mapping, List, Iterable as Iter

import pandas as pd
from iblutil.util import Bunch

from one.alf.spec import is_session_path, is_uuid_string
from one.alf.files import get_session_path, add_uuid_string, session_path_parts, remove_uuid_string
from .util import Listable, ensure_list


def recurse(func):
    """Decorator to call decorated function recursively if first arg is non-string iterable.

    Allows decorated methods to accept both single values, and lists/tuples of values.  When
    given the latter, a list is returned.  This decorator is intended to work on class methods,
    therefore the first arg is assumed to be the object.  Maps and pandas objects are not
    iterated over.

    Parameters
    ----------
    func : function
        A method to decorate.

    Returns
    -------
    function
        The decorated method.
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        if len(args) <= 1:
            return func(*args, **kwargs)
        obj, first = args[:2]
        exclude = (str, Mapping, pd.Series, pd.DataFrame)
        if isinstance(first, Iter) and not isinstance(first, exclude):
            return [func(obj, item, *args[2:], **kwargs) for item in first]
        else:
            return func(obj, first, *args[2:], **kwargs)
    return wrapper_decorator


def parse_values(func):
    """Convert str values in reference dict to appropriate type.

    Example
    -------
    >>> parse_values(lambda x: x)({'date': '2020-01-01', 'sequence': '001'}, parse=True)
    {'date': datetime.date(2020, 1, 1), 'sequence': 1}
    """
    def parse_ref(ref):
        if ref:
            if isinstance(ref['date'], str):
                if len(ref['date']) == 10:
                    ref['date'] = datetime.date.fromisoformat(ref['date'])
                else:
                    ref['date'] = datetime.datetime.fromisoformat(ref['date']).date()
            ref['sequence'] = int(ref['sequence'])
        return ref

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parse = kwargs.pop('parse', True)
        ref = func(*args, **kwargs)
        if not parse or isinstance(ref, str):
            return ref
        elif isinstance(ref, list):
            return list(map(parse_ref, ref))
        else:
            return parse_ref(ref)
    return wrapper_decorator


class ConversionMixin:
    """A mixin providing methods to interconvert experiment identifiers"""

    def __init__(self):
        self._cache = None
        self._par = None

    @recurse
    def to_eid(self,
               id: Listable(Union[str, Path, UUID, dict]) = None,
               cache_dir: Optional[Union[str, Path]] = None) -> Listable(str):
        """Given any kind of experiment identifier, return a corresponding eid string.

        NB: Currently does not support integer IDs.

        Parameters
        ----------
        id : str, pathlib.Path, UUID, dict, tuple, list
            An experiment identifier
        cache_dir : pathlib.Path, str
            An optional cache directory path for intermittent conversion to path

        Returns
        -------
        str, None
            An experiment ID string or None if session not in cache

        Raises
        ------
        ValueError
            Input ID invalid
        """
        # TODO Could add np2str here
        # if isinstance(id, (list, tuple)):  # Recurse
        #     return [self.to_eid(i, cache_dir) for i in id]
        if id is None:
            return
        elif isinstance(id, UUID):
            return str(id)
        elif self.is_exp_ref(id):
            return self.ref2eid(id)
        elif isinstance(id, dict):
            assert {'subject', 'number', 'lab'}.issubset(id)
            root = Path(cache_dir or self.cache_dir)
            id = root.joinpath(
                id['lab'],
                'Subjects', id['subject'],
                str(id.get('date') or id['start_time'][:10]),
                ('%03d' % id['number']))

        if isinstance(id, Path):
            return self.path2eid(id)
        elif isinstance(id, str):
            if is_session_path(id) or get_session_path(id):
                return self.path2eid(id)
            if len(id) > 36:
                id = id[-36:]
            if not is_uuid_string(id):
                raise ValueError('Invalid experiment ID')
            else:
                return id
        else:
            raise ValueError('Unrecognized experiment ID')

    @recurse
    def eid2path(self, eid: str) -> Optional[Listable(Path)]:
        """
        From an experiment id or a list of experiment ids, gets the local cache path

        Parameters
        ----------
        eid : str, uuid.UUID
            Experiment ID (UUID) or list of UUIDs

        Returns
        -------
        pathlib.Path
            A session path
        """
        # If not valid return None
        if not is_uuid_string(eid):
            raise ValueError(eid + " is not a valid eID/UUID string")
        if self._cache['sessions'].size == 0:
            return

        # load path from cache
        try:
            ses = self._cache['sessions'].loc[eid].squeeze()
            assert isinstance(ses, pd.Series), 'Duplicate eids in sessions table'
            return session_record2path(ses.to_dict(), self.cache_dir)
        except KeyError:
            return

    @recurse
    def path2eid(self, path_obj):
        """
        From a local path, gets the experiment id

        Parameters
        ----------
        path_obj : pathlib.Path, str
            Local path or list of local paths

        Returns
        -------
        eid, list
            Experiment ID (eid) string or list of eids
        """
        # else ensure the path ends with mouse,date, number
        session_path = get_session_path(path_obj)
        sessions = self._cache['sessions']

        # if path does not have a date and a number, or cache is empty return None
        if session_path is None or sessions.size == 0:
            return None

        # reduce session records from cache
        toDate = datetime.date.fromisoformat
        subject, date, number = session_path.parts[-3:]
        for col, val in zip(('subject', 'date', 'number'), (subject, toDate(date), int(number))):
            sessions = sessions[sessions[col] == val]
            if sessions.size == 0:
                return

        assert len(sessions) == 1

        eid, = sessions.index.values
        return eid

    @recurse
    def path2record(self, path) -> pd.Series:
        """Convert a file or session path to a dataset or session cache record.

        NB: Assumes <lab>/Subjects/<subject>/<date>/<number> pattern

        Parameters
        ----------
        path : str, pathlib.Path
            Local path or HTTP URL

        Returns
        -------
        pandas.Series
            A cache file record
        """
        is_session = is_session_path(path)
        rec = self._cache['sessions' if is_session else 'datasets']
        if rec.empty:
            return
        # if (rec := self._cache['datasets']).empty:  # py 3.8
        #     return

        if is_session_path(path):
            lab, subject, date, number = session_path_parts(path)
            rec = rec[
                (rec['lab'] == lab) & (rec['subject'] == subject) &
                (rec['number'] == int(number)) &
                (rec['date'] == datetime.date.fromisoformat(date))
            ]
            return None if rec.empty else rec.squeeze()

        # Deal with file path
        if isinstance(path, str) and path.startswith('http'):
            # Remove the UUID from path
            path = urlsplit(path).path.strip('/')
            path = remove_uuid_string(PurePosixPath(path))
            session_path = get_session_path(path).as_posix()
        else:
            # No way of knowing root session path parts without cache tables
            eid = self.path2eid(path)
            session_series = self.list_datasets(eid, details=True).session_path
            if not eid or session_series.empty:
                return
            session_path, *_ = session_series

        rec = rec[rec['session_path'] == session_path]
        rec = rec[rec['rel_path'].apply(lambda x: path.as_posix().endswith(x))]
        assert len(rec) < 2, 'Multiple records found'
        return None if rec.empty else rec.squeeze()

    @recurse
    def path2url(self, filepath):
        """
        Given a local file path, constructs the URL of the remote file.

        Parameters
        ----------
        filepath : str, pathlib.Path
            A local file path

        Returns
        -------
        str
            A remote URL string
        """
        record = self.path2record(filepath)
        if record is None:
            return
        return unwrap(self.record2url)(record)

    def record2url(self, record):
        """Convert a session or dataset record to a remote URL

        NB: Requires online instance

        Parameters
        ----------
        record : pd.Series, pd.DataFrame
            A datasets or sessions cache record.  If DataFrame, iterate over and returns list.

        Returns
        -------
        str, list
            A dataset URL or list if input is DataFrame
        """
        webclient = getattr(self, '_web_client', False)
        assert webclient, 'No Web client found for instance'
        # FIXME Should be OneAlyx converter only
        if isinstance(record, pd.DataFrame):
            return [self.record2url(r) for _, r in record.iterrows()]
        elif isinstance(record, pd.Series):
            is_session_record = 'rel_path' not in record
            if is_session_record:
                # NB: This assumes the root path is in the webclient URL
                session_spec = '{lab}/Subjects/{subject}/{date}/{number:03d}'
                url = record.get('session_path') or session_spec.format(**record)
                return webclient.rel_path2url(url)
            uuid = ensure_list(record.name)[-1]  # may be (eid, did) or simply did
        else:
            raise TypeError(
                f'record must be pandas.DataFrame or pandas.Series, got {type(record)} instead')

        session_path, rel_path = record[['session_path', 'rel_path']].to_numpy().flatten()
        url = PurePosixPath(session_path, rel_path)
        return webclient.rel_path2url(add_uuid_string(url, uuid).as_posix())

    def record2path(self, dataset) -> Optional[Path]:
        """
        Given a set of dataset records, checks the corresponding exists flag in the cache
        correctly reflects the files system

        Parameters
        ----------
        dataset : pd.DataFrame, pd.Series
            A datasets dataframe slice

        Returns
        -------
        pathlib.Path
            File path for the record
        """
        assert isinstance(dataset, pd.Series) or len(dataset) == 1
        session_path, rel_path = dataset[['session_path', 'rel_path']].to_numpy().flatten()
        file = Path(self.cache_dir, session_path, rel_path)
        return file  # files[0] if len(datasets) == 1 else files

    @recurse
    def eid2ref(self, eid: Union[str, Iter], as_dict=True, parse=True) \
            -> Union[str, Mapping, List]:
        """
        Get human-readable session ref from path.

        Parameters
        ----------
        eid : str, uuid.UUID
            The experiment uuid to find reference for.
        as_dict : bool
            If false a string is returned in the form 'subject_sequence_yyyy-mm-dd'.
        parse : bool
            If true, the reference date and sequence are parsed from strings to their respective
            data types.

        Returns
        -------
        dict, str, list
            One or more objects with keys ('subject', 'date', 'sequence'), or strings with the
            form yyyy-mm-dd_n_subject.

        Examples
        --------
        >>> eid = '4e0b3320-47b7-416e-b842-c34dc9004cf8'
        >>> one.eid2ref(eid)
        {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}
        >>> one.eid2ref(eid, parse=False)
        {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}
        >>> one.eid2ref(eid, as_dict=False)
        '2018-07-13_1_flowers'
        >>> one.eid2ref(eid, as_dict=False, parse=False)
        '2018-07-13_001_flowers'
        >>> one.eid2ref([eid, '7dc3c44b-225f-4083-be3d-07b8562885f4'])
        [{'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1},
         {'subject': 'KS005', 'date': datetime.date(2019, 4, 11), 'sequence': 1}]
        """
        d = self.get_details(eid)
        if parse:
            ref = {'subject': d['subject'], 'date': d['date'], 'sequence': d['number']}
            format_str = '{date:%Y-%m-%d}_{sequence:d}_{subject:s}'
        else:
            ref = {
                'subject': d['subject'], 'date': str(d['date']), 'sequence': '%03d' % d['number']
            }
            format_str = '{date:s}_{sequence:s}_{subject:s}'
        return Bunch(ref) if as_dict else format_str.format(**ref)

    @recurse
    def ref2eid(self, ref: Union[Mapping, str, Iter]) -> Union[str, List]:
        """
        Returns experiment uuid, given one or more experiment references.

        Parameters
        ----------
        ref : str, dict, list
            One or more objects with keys ('subject', 'date', 'sequence'), or strings with
            the form yyyy-mm-dd_n_subject.

        Returns
        -------
        str, list
            One or more experiment uuid strings.

        Examples
        --------
        >>> base = 'https://test.alyx.internationalbrainlab.org'
        >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
        Connected to...
        >>> ref = {'date': datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
        >>> one.ref2eid(ref)
        '4e0b3320-47b7-416e-b842-c34dc9004cf8'
        >>> one.ref2eid(['2018-07-13_1_flowers', '2019-04-11_1_KS005'])
        ['4e0b3320-47b7-416e-b842-c34dc9004cf8',
         '7dc3c44b-225f-4083-be3d-07b8562885f4']
        """
        ref = self.ref2dict(ref, parse=False)  # Ensure dict
        session = self.search(
            subject=ref['subject'],
            date_range=str(ref['date']),
            number=ref['sequence'])
        assert len(session) == 1, 'session not found'
        return session[0]

    @recurse
    def ref2path(self, ref):
        """
        Convert one or more experiment references to session path(s).

        Parameters
        ----------
        ref : str, dict, list
            One or more objects with keys ('subject', 'date', 'sequence'), or strings with
            the form yyyy-mm-dd_n_subject.

        Returns
        -------
        pathlib.Path
            Path object(s) for the experiment session(s).

        Examples
        --------
        >>> base = 'https://test.alyx.internationalbrainlab.org'
        >>> one = ONE(username='test_user', password='TapetesBloc18', base_url=base)
        Connected to...
        >>> ref = {'subject': 'flowers', 'date': datetime(2018, 7, 13).date(), 'sequence': 1}
        >>> one.ref2path(ref)
        WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001')
        >>> one.ref2path(['2018-07-13_1_flowers', '2019-04-11_1_KS005'])
        [WindowsPath('E:/FlatIron/zadorlab/Subjects/flowers/2018-07-13/001'),
         WindowsPath('E:/FlatIron/cortexlab/Subjects/KS005/2019-04-11/001')]
        """
        eid2path = unwrap(self.eid2path)
        ref2eid = unwrap(self.ref2eid)
        return eid2path(self, ref2eid(self, ref))

    @staticmethod
    @parse_values
    def path2ref(path_str: Union[str, Path, Iter], as_dict=True) -> Union[Bunch, List]:
        """
        Returns a human readable experiment reference, given a session path.
        The path need not exist.

        Parameters
        ----------
        path_str : str
            A path to a given session.
        as_dict : bool
            If True a Bunch is returned, otherwise a string.

        Returns
        -------
        dict, str, list
            One or more objects with keys ('subject', 'date', 'sequence').

        Examples
        --------
        >>> path_str = Path('E:/FlatIron/Subjects/zadorlab/flowers/2018-07-13/001')
        >>> path2ref(path_str)
        {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}
        >>> path2ref(path_str, parse=False)
        {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}
        >>> path_str2 = Path('E:/FlatIron/Subjects/churchlandlab/CSHL046/2020-06-20/002')
        >>> path2ref([path_str, path_str2])
        [{'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1},
         {'subject': 'CSHL046', 'date': datetime.date(2020, 6, 20), 'sequence': 2}]
        """
        if isinstance(path_str, (list, tuple)):
            return [unwrap(ConversionMixin.path2ref)(x) for x in path_str]
        pattern = r'(?P<subject>[\w-]+)([\\/])(?P<date>\d{4}-\d{2}-\d{2})(\2)(?P<sequence>\d{3})'
        match = re.search(pattern, str(path_str))
        if match:
            ref = match.groupdict()
            return Bunch(ref) if as_dict else '{date:s}_{sequence:s}_{subject:s}'.format(**ref)

    @staticmethod
    def is_exp_ref(ref: Union[str, Mapping, Iter]) -> Union[bool, List[bool]]:
        """
        Returns True is ref is a valid experiment reference.

        Parameters
        ----------
        ref : str, dict, list
            One or more objects with keys ('subject', 'date', 'sequence'), or strings with
            the form yyyy-mm-dd_n_subject.

        Returns
        -------
        bool, list of bool
            True if ref is valid.

        Examples
        --------
        >>> ref = {'date': datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
        >>> is_exp_ref(ref)
        True
        >>> is_exp_ref('2018-07-13_001_flowers')
        True
        >>> is_exp_ref('invalid_ref')
        False
        """
        if isinstance(ref, (list, tuple)):
            return [ConversionMixin.is_exp_ref(x) for x in ref]
        if isinstance(ref, (Bunch, dict)):
            if not {'subject', 'date', 'sequence'}.issubset(ref):
                return False
            ref = '{date}_{sequence}_{subject}'.format(**ref)
        elif not isinstance(ref, str):
            return False
        return re.compile(r'\d{4}(-\d{2}){2}_(\d{1,3})_\w+').match(ref) is not None

    @staticmethod
    @parse_values
    def ref2dict(ref: Union[str, Mapping, Iter]) -> Union[Bunch, List]:
        """
        Returns a Bunch (dict-like) from a reference string (or list thereof).

        Parameters
        ----------
        ref : str, list
            One or more experiment reference strings.

        Returns
        -------
        iblutil.util.Bunch
            A Bunch in with keys ('subject', 'sequence', 'date').

        Examples
        --------
        >>> ref2dict('2018-07-13_1_flowers')
        {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
        >>> ref2dict('2018-07-13_001_flowers', parse=False)
        {'date': '2018-07-13', 'sequence': '001', 'subject': 'flowers'}
        >>> ref2dict(['2018-07-13_1_flowers', '2020-01-23_002_ibl_witten_01'])
        [{'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'},
         {'date': datetime.date(2020, 1, 23), 'sequence': 2, 'subject': 'ibl_witten_01'}]
        """
        if isinstance(ref, (list, tuple)):
            return [ConversionMixin.ref2dict(x) for x in ref]
        if isinstance(ref, (Bunch, dict)):
            return Bunch(ref)  # Short circuit
        ref = dict(zip(['date', 'sequence', 'subject'], ref.split('_', 2)))
        return Bunch(ref)

    @staticmethod
    def dict2ref(ref_dict) -> Union[str, List]:
        """
        Convert an experiment reference dict to a string in the format yyyy-mm-dd_n_subject.

        Parameters
        ----------
        ref_dict : dict, Bunch, list, tuple
            A map with the keys ('subject', 'date', 'sequence').

        Returns
        -------
        str, list:
            An experiment reference string, or list thereof.
        """
        if isinstance(ref_dict, (list, tuple)):
            return [ConversionMixin.dict2ref(x) for x in ref_dict]
        if not ref_dict:
            return
        if 'sequence' not in ref_dict and 'number' in ref_dict:
            ref_dict = ref_dict.copy()
            ref_dict['sequence'] = ref_dict.pop('number')
        if 'date' not in ref_dict and 'start_time' in ref_dict:
            ref_dict = ref_dict.copy()
            if isinstance(ref_dict['start_time'], str):
                ref_dict['date'] = ref_dict['start_time'][:10]
            else:
                ref_dict['date'] = ref_dict['start_time'].date()
        parsed = any(not isinstance(k, str) for k in ref_dict.values())
        format_str = ('{date:%Y-%m-%d}_{sequence:d}_{subject:s}'
                      if parsed
                      else '{date:s}_{sequence:s}_{subject:s}')
        return format_str.format(**ref_dict)


def one_path_from_dataset(dset, one_cache):
    """
    Returns local one file path from a dset record or a list of dsets records from REST.
    Unlike `to_eid`, this function does not require ONE, and the dataset may not exist.

    Parameters
    ----------
    dset : dict, list
        Dataset dictionary or list of dictionaries from Alyx rest endpoint.
    one_cache : str, pathlib.Path, pathlib.PurePath
        The local ONE data cache directory.

    Returns
    -------
    pathlib.Path
        The local path for a given dataset.
    """
    return path_from_dataset(dset, root_path=one_cache, uuid=False)


def path_from_dataset(dset, root_path=PurePosixPath('/'), repository=None, uuid=False):
    """
    Returns the local file path from a dset record from a REST query.
    Unlike `to_eid`, this function does not require ONE, and the dataset may not exist.

    Parameters
    ----------
    dset : dict, list
        Dataset dictionary or list of dictionaries from Alyx rest endpoint.
    root_path : str, pathlib.Path, pathlib.PurePath
        The prefix path such as the ONE download directory or remote http server root.
    repository : str, None
        Which data repository to use from the file_records list, defaults to first online
        repository.
    uuid : bool
        If True, the file path will contain the dataset UUID.

    Returns
    -------
    pathlib.Path, list
        File path or list of paths.
    """
    if isinstance(dset, list):
        return [path_from_dataset(d) for d in dset]
    if repository:
        fr = next((fr for fr in dset['file_records'] if fr['data_repository'] == repository))
    else:
        fr = next((fr for fr in dset['file_records'] if fr['data_url']))
    uuid = dset['url'][-36:] if uuid else None
    return path_from_filerecord(fr, root_path=root_path, uuid=uuid)


def path_from_filerecord(fr, root_path=PurePosixPath('/'), uuid=None):
    """
    Returns a data file Path constructed from an Alyx file record.  The Path type returned
    depends on the type of root_path: If root_path is a string a Path object is returned,
    otherwise if the root_path is a PurePath, the same path type is returned.

    Parameters
    ----------
    fr : dict
        An Alyx file record dict.
    root_path : str, pathlib.Path
        An optional root path.
    uuid : str, uuid.UUID
        An optional dataset UUID to add to the file name.

    Returns
    -------
    pathlib.Path
        A filepath as a pathlib object.
    """
    if isinstance(fr, list):
        return [path_from_filerecord(f) for f in fr]
    repo_path = (p := fr['data_repository_path'])[p[0] == '/':]  # Remove slash at start, if any
    file_path = PurePosixPath(repo_path, fr['relative_path'])
    if root_path:
        # NB: By checking for string we won't cast any PurePaths
        if isinstance(root_path, str):
            root_path = Path(root_path)
        file_path = root_path / file_path
    if uuid:
        file_path = add_uuid_string(file_path, uuid)
    return file_path


def session_record2path(session, root_dir=None):
    """
    Convert a session record into a path.

    If a lab key is present, the path will be in the form
    root_dir/lab/Subjects/subject/yyyy-mm-dd/nnn, otherwise root_dir/subject/yyyy-mm-dd/nnn.

    Parameters
    ----------
    session : Mapping
        A session record with keys ('subject', 'date', 'number'[, 'lab']).
    root_dir : str, pathlib.Path, pathlib.PurePath
        A root directory to prepend.

    Returns
    -------
    pathlib.Path, Pathlib.PurePath
        A constructed path of the session.

    Examples
    --------
    >>> session_record2path({'subject': 'ALK01', 'date': '2020-01-01', 'number': 1})
    PurePosixPath('ALK01/2020-01-01/001')

    >>> record = {'date': datetime.datetime.fromisoformat('2020-01-01').date(),
    ...           'number': '001', 'lab': 'foo', 'subject': 'ALK01'}
    >>> session_record2path(record, Path('/home/user'))
    Path('/home/user/foo/Subjects/ALK01/2020-01-01/001')
    """
    rel_path = PurePosixPath(
        session.get('lab') if session.get('lab') else '',
        'Subjects' if session.get('lab') else '',
        session['subject'], str(session['date']), str(session['number']).zfill(3)
    )
    if not root_dir:
        return rel_path
    elif isinstance(root_dir, str):
        root_dir = Path(root_dir)
    return Path(root_dir).joinpath(rel_path)
