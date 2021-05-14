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
from typing import Optional, Union, Sequence, Mapping, Iterable

import numpy as np
import pandas as pd

import one.alf.io as alfio
import one.alf.files
from .lib.brainbox.io import parquet

def Listable(t): return Union[t, Sequence[t]]  # noqa


def recurse(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        first, *args = args
        if isinstance(first, Iterable) and not isinstance(first, (str, Mapping)):
            return [func(item, *args, **kwargs) for item in first]
        else:
            return func(first, *args, **kwargs)
    return wrapper_decorator


class ConversionMixin:

    def __init__(self):
        self._cache = None
        self._par = None

    # def to_eid(self, id):
    #     pass

    def path_from_eid(self, eid: str) -> Optional[Listable(Path)]:
        """
        From an experiment id or a list of experiment ids, gets the local cache path
        :param eid: eid (UUID) or list of UUIDs
        :return: eid or list of eids
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            path_list = []
            for p in eid:
                path_list.append(self.path_from_eid(p))
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

    def eid_from_path(self, path_obj):
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

    def record_from_path(self, filepath):
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

    def url_from_path(self, filepath):
        """
        Given a local file path, constructs the URL of the remote file.
        :param filepath: A local file path
        :return: A URL string
        """
        record = self.record_from_path(filepath)
        if record is None:
            return
        return unwrap(self.url_from_record)(self, record)

    def url_from_record(self, dataset):
        # FIXME Should be OneAlyx converter only
        # for i, rec in dataset.iterrows():
        if isinstance(dataset, pd.Series):
            uuid, = parquet.np2str(np.array([dataset.name]))
        else:
            assert len(dataset) == 1
            uuid, = parquet.np2str(dataset.reset_index()[['id_0', 'id_1']])
        session_path, rel_path = dataset[['session_path', 'rel_path']].to_numpy().flatten()
        url = PurePosixPath(session_path, rel_path)
        return self._web_client.rel_path_to_url(alfio.add_uuid_string(url, uuid).as_posix())

    def path_from_record(self, dataset) -> Optional[Path]:
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

    def ref_from_eid(self, eid):
        pass

    def eid_from_ref(self, ref):
        pass

    def path_from_ref(self, ref):
        pass

    def pid_to_path(self, pid):
        pass

    def to(self, eid, type):
        if type == 'path':
            return self.path_from_eid(eid)
        elif type == 'ref':
            return self.ref_from_eid(eid)
        else:
            raise ValueError(f'Unsupported type "{type}"')


from_funcs = getmembers(ConversionMixin,
                        lambda x: isfunction(x) and '_from_' in x.__name__)
for name, fn in from_funcs:
    setattr(ConversionMixin, name, recurse(fn))  # Add recursion decorator
    attr = '{0}2{1}'.format(*name.split('_from_'))
    from_fn = getattr(ConversionMixin, '{1}_from_{0}'.format(*name.split('_from_')), None)
    if from_fn:
        setattr(ConversionMixin, attr, recurse(from_fn))  # Add 2 function alias
