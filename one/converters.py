"""
A module for inter-converting experiment identifiers.

There are multiple ways to uniquely identify an experiment:
    - eid (str) : An experiment UUID as a string
    - np (int64) : An experiment UUID encoded as 2 int64s
    - path (Path) : A pathlib ALF path of the form <lab>/Subjects/<subject>/<date>/<number>
    - ref (str) : An experiment reference string of the form yyyy-mm-dd_n_subject
    - url (str) : An remote http session path of the form <lab>/Subjects/<subject>/<date>/<number>
"""
import functools
import datetime
from inspect import getmembers, isfunction, unwrap
from pathlib import Path, PurePosixPath
from typing import Optional, Union, Sequence, Mapping, List, Iterable as Iter

import numpy as np
import pandas as pd

import one.alf.io as alfio
import one.alf.files
from .lib.brainbox.io import parquet
from .lib.brainbox.core import Bunch

def Listable(t): return Union[t, Sequence[t]]  # noqa


def recurse(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        first, *args = args
        if isinstance(first, Iter) and not isinstance(first, (str, Mapping)):
            return [func(item, *args, **kwargs) for item in first]
        else:
            return func(first, *args, **kwargs)
    return wrapper_decorator


def parse_values(func):
    """Convert str values in reference dict to appropriate type"""
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        parse = kwargs.pop('parse', True)
        ref = func(*args, **kwargs)
        if parse:
            if isinstance(ref['date'], str):
                if len(ref['date']) == 10:
                    ref['date'] = datetime.date.fromisoformat(ref['date'])
                else:
                    ref['date'] = datetime.datetime.fromisoformat(ref['date']).date()
            ref['sequence'] = int(ref['sequence'])
        return ref
    return wrapper_decorator


class ConversionMixin:

    def __init__(self):
        self._cache = None
        self._par = None

    # def to_eid(self, id):
    #     pass

    def eid2path(self, eid: str) -> Optional[Listable(Path)]:
        """
        From an experiment id or a list of experiment ids, gets the local cache path
        :param eid: eid (UUID) or list of UUIDs
        :return: eid or list of eids
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            path_list = []
            for p in eid:
                path_list.append(self.eid2path(p))
            return path_list
        # If not valid return None
        if not alfio.is_uuid_string(eid):
            raise ValueError(eid + " is not a valid eID/UUID string")
        if self._cache['sessions'].size == 0:
            return

        # load path from cache
        if self._index_type() is int:
            # ids = np.array(self._cache['sessions'].index.tolist())
            # ids = self._cache['sessions'].reset_index()[['eid_0', 'eid_1']].to_numpy()
            # ic = find_first_2d(ids, parquet.str2np(eid))
            # if ic is not None:
            #     ses = self._cache['sessions'].iloc[ic]
            eid = parquet.str2np(eid).tolist()
        try:
            ses = self._cache['sessions'].loc[eid]
            assert len(ses) == 1, 'Duplicate eids in sessions table'
            ses, = ses.to_dict('records')
            return Path(self._cache_dir).joinpath(
                ses['lab'], 'Subjects', ses['subject'],
                str(ses['date']), str(ses['number']).zfill(3))
        except KeyError:
            return

    def path2eid(self, path_obj):
        """
        From a local path, gets the experiment id
        :param path_obj: local path or list of local paths
        :return: eid or list of eids
        """
        # else ensure the path ends with mouse,date, number
        session_path = alfio.get_session_path(path_obj)
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
        if isinstance(eid, tuple):
            eid = parquet.np2str(np.array(eid))
        return eid

    def path2record(self, filepath):
        """
        TODO Return Series instead of DataFrame
        NB: Assumes <lab>/Subjects/<subject>/<date>/<number> pattern
        :param filepath: File path or http URL
        :return:
        """
        if isinstance(filepath, str) and filepath.startswith('http'):
            # Remove the UUID from path
            filepath = alfio.remove_uuid_file(PurePosixPath(filepath), dry=True)
        session_path = '/'.join(alfio.get_session_path(filepath).parts[-5:])
        if (rec := self._cache['datasets']).empty:
            return
        rec = rec[rec['session_path'] == session_path]
        rec = rec[rec['rel_path'].apply(lambda x: filepath.as_posix().endswith(x))]
        assert len(rec) < 2, 'Multiple records found'
        return None if rec.empty else rec

    def path2url(self, filepath):
        """
        Given a local file path, constructs the URL of the remote file.
        :param filepath: A local file path
        :return: A URL string
        """
        record = self.path2record(filepath)
        if record is None:
            return
        return unwrap(self.record2url)(self, record)

    def record2url(self, dataset):
        assert self._web_client
        # FIXME Should be OneAlyx converter only
        # TODO Document
        # for i, rec in dataset.iterrows():
        if isinstance(dataset, pd.Series):
            uuid, = parquet.np2str(np.array([dataset.name]))
        else:
            assert len(dataset) == 1
            uuid, = parquet.np2str(dataset.reset_index()[['id_0', 'id_1']])
        session_path, rel_path = dataset[['session_path', 'rel_path']].to_numpy().flatten()
        url = PurePosixPath(session_path, rel_path)
        return self._web_client.rel_path2url(alfio.add_uuid_string(url, uuid).as_posix())

    def record2path(self, dataset) -> Optional[Path]:
        """
        Given a set of dataset records, checks the corresponding exists flag in the cache
        correctly reflects the files system
        :param dataset: A datasets dataframe slice
        :return: File path for the record
        """
        assert len(dataset) == 1
        session_path, rel_path = dataset[['session_path', 'rel_path']].to_numpy().flatten()
        file = Path(self._cache_dir, session_path, rel_path)
        return file  # files[0] if len(datasets) == 1 else files

    def eid2ref(self, eid):
        pass

    @recurse
    def ref2eid(self, ref: Union[Mapping, str, Iter]) -> Union[str, List]:
        """
        Returns experiment uuid, given one or more experiment references
        :param ref: One or more objects with keys ('subject', 'date', 'sequence'), or strings with
        the form yyyy-mm-dd_n_subject
        :param one: An instance of ONE
        :return: an experiment uuid string

        Examples:
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
            subjects=ref['subject'],
            date_range=str(ref['date']),
            number=ref['sequence'])
        assert len(session) == 1, 'session not found'
        return session[0]

    def ref2path(self, ref):
        pass

    def path2pid(self, pid):
        pass

    @staticmethod
    @recurse
    @parse_values
    def ref2dict(ref: Union[str, Mapping, Iter]) -> Union[Bunch, List]:
        """
        Returns a Bunch (dict-like) from a reference string (or list thereof)
        :param ref: One or more objects with keys ('subject', 'date', 'sequence')
        :return: A Bunch in with keys ('subject', 'sequence', 'date')

        Examples:
        >>> ref2dict('2018-07-13_1_flowers')
        {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
        >>> ref2dict('2018-07-13_001_flowers', parse=False)
        {'date': '2018-07-13', 'sequence': '001', 'subject': 'flowers'}
        >>> ref2dict(['2018-07-13_1_flowers', '2020-01-23_002_ibl_witten_01'])
        [{'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'},
         {'date': datetime.date(2020, 1, 23), 'sequence': 2, 'subject': 'ibl_witten_01'}]
        """
        if isinstance(ref, (Bunch, dict)):
            return Bunch(ref)  # Short circuit
        ref = dict(zip(['date', 'sequence', 'subject'], ref.split('_', 2)))
        return Bunch(ref)

    def to(self, eid, type):
        if type == 'path':
            return self.eid2path(eid)
        elif type == 'ref':
            return self.ref_from_eid(eid)
        else:
            raise ValueError(f'Unsupported type "{type}"')


def deprecate(func):
    """Print deprecation warning about decorated function"""
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        import warnings
        warnings.warn(f'Use "{func.__name__}" instead', DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper_decorator


# Add deprecated legacy methods
from_funcs = getmembers(ConversionMixin,
                        lambda x: isfunction(x) and '2' in x.__name__)
for name, fn in from_funcs:
    setattr(ConversionMixin, name, recurse(fn))  # Add recursion decorator
    attr = '{1}_from_{0}'.format(*name.split('2'))
    setattr(ConversionMixin, attr, deprecate(recurse(fn)))  # Add from function alias
