"""
TODO Document
TODO Add query_type; handle param in decorator
TODO Include exp_ref parsing
TODO Add Offline property?
TODO Add sig to ONE Light uuids
TODO Save changes to cache
TODO Fix update cache in AlyxONE - save parquet table
TODO save parquet in update_filesystem
TODO Fix one.search exist_only
TODO Update cache fixtures with new test alyx cache


Points of discussion:
    - Module structure: oneibl is too restrictive, naming module `one` means obj should have
    different name
    - How to deal with factory?
    - Does remote query mean REST query (current) or re-downloading the cache?
        - Option 1: 'remote' means REST query, 'auto' means refresh cache if old, 'local' means
          use current cache, 'refresh' re-download cache
        - Option 2: 'remote' means clobber datasets, 'auto' means download missing,
          'local' means load local datasets only, 'refresh' means re-download cache then use auto
    - NB: Wildcards will behave differently between REST and pandas
    - Currently downloading cache is lazy - no cache files are downloaded until a load method is
      called
    - How to deal with load? Just remove it? Keep it in OneAlyx as legacy? How to release ONE2.0
    - Dealing with lists must be consistent.  Three options:
        - two methods each, e.g. load_dataset and load_datasets (con: a lot of overhead)
        - allow list inputs, recursive calls (con: function logic slightly more complex)
        - no list inputs; rely on list comprehensions (con: makes accessing meta data complex)
    - Suggestion to store params file in cache dir, with master params location in the usual place.
      Master params file could store map of URLs to caches + the default location.
      This would allow for multiple database cache directories and solve tests issue.
    - Do we need the cache dir to be a param for every function?
    - Need to check performance of 1. (re)setting index, 2. converting object array to 2D int array
    - Conceivably you could have a subclass for Figshare, etc., not just Alyx
"""
import abc
import concurrent.futures
import warnings
import logging
import os
import fnmatch
import re
from urllib.error import HTTPError
from datetime import datetime, timedelta
from functools import wraps
from inspect import unwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence, Union, Optional, List, Dict
from uuid import UUID

import tqdm
import pandas as pd
import numpy as np

import one.params
import one.webclient as wc
import one.alf.io as alfio
from .alf.files import is_valid, alf_parts, COLLECTION_SPEC, FILE_SPEC, regex as alf_regex
# from ibllib.misc.exp_ref import is_exp_ref
from .alf.exceptions import \
    ALFMultipleObjectsFound, ALFObjectNotFound, ALFMultipleCollectionsFound
from one.lib.io import hashfile
from pprint import pprint
from one.lib.brainbox.io import parquet
from one.lib.brainbox.core import Bunch
from one.lib.brainbox.numerical import ismember2d, find_first_2d
from one.converters import ConversionMixin

_logger = logging.getLogger('ibllib')  # TODO Refactor log


def Listable(t): return Union[t, Sequence[t]]  # noqa


NTHREADS = 4  # number of download threads

_ENDPOINTS = {  # keynames are possible input arguments and values are actual endpoints
    'data': 'dataset-types',
    'dataset': 'datasets',
    'datasets': 'datasets',
    'dataset-types': 'dataset-types',
    'dataset_types': 'dataset-types',
    'dataset-type': 'dataset-types',
    'dataset_type': 'dataset-types',
    'dtypes': 'dataset-types',
    'dtype': 'dataset-types',
    'users': 'users',
    'user': 'users',
    'subject': 'subjects',
    'subjects': 'subjects',
    'labs': 'labs',
    'lab': 'labs'}


def _ses2pandas(ses, dtypes=None):
    """
    :param ses: session dictionary from rest endpoint
    :param dtypes: list of dataset types
    :return:
    """
    # selection: get relevant dtypes only if there is an url associated
    rec = list(filter(lambda x: x['url'], ses['data_dataset_session_related']))
    if dtypes == ['__all__'] or dtypes == '__all__':
        dtypes = None
    if dtypes is not None:
        rec = list(filter(lambda x: x['dataset_type'] in dtypes, rec))
    include = ['id', 'hash', 'dataset_type', 'name', 'file_size', 'collection']
    uuid_fields = ['id', 'eid']
    join = {'subject': ses['subject'], 'lab': ses['lab'], 'eid': ses['url'][-36:],
            'start_time': np.datetime64(ses['start_time']), 'number': ses['number'],
            'task_protocol': ses['task_protocol']}
    col = parquet.rec2col(rec, include=include, uuid_fields=uuid_fields, join=join,
                          types={'file_size': np.double}).to_df()
    return col


def parse_id(method):
    """
    Ensures the input experiment identifier is an experiment UUID string
    :param method: An ONE method whose second arg is an experiment id
    :return: A wrapper function that parses the id to the expected string
    TODO Move to converters.py
    """

    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        id = self.to_eid(id)
        return method(self, id, *args, **kwargs)

    return wrapper


class One(ConversionMixin):
    search_terms = (
        'dataset', 'date_range', 'laboratory', 'number', 'project', 'subject', 'task_protocol'
    )

    def __init__(self, cache_dir=None):
        # get parameters override if inputs provided
        super().__init__()
        if not getattr(self, '_cache_dir', None):  # May already be set by subclass
            self._cache_dir = cache_dir or one.params.get_cache_dir()
        self.cache_expiry = timedelta(hours=24)
        # init the cache file
        self._load_cache()

    def _load_cache(self, cache_dir=None, **kwargs):
        self._cache = Bunch({'expired': False, 'created_time': None})
        INDEX_KEY = 'id'
        for table in ('sessions', 'datasets'):
            cache_file = Path(cache_dir or self._cache_dir).joinpath(table + '.pqt')
            if cache_file.exists():
                # we need to keep this part fast enough for transient objects
                cache, info = parquet.load(cache_file)
                created = datetime.fromisoformat(info['date_created'])
                if self._cache['created_time']:
                    self._cache['created_time'] = min([self._cache['created_time'], created])
                else:
                    self._cache['created_time'] = created
                self._cache['loaded_time'] = datetime.now()
                self._cache['expired'] |= datetime.now() - created > self.cache_expiry
                if self._cache['expired']:
                    t_str = (f'{self.cache_expiry.days} days(s)'
                             if self.cache_expiry.days >= 1
                             else f'{self.cache_expiry.seconds / (60 * 2)} hour(s)')
                    _logger.warning(f'{table.title()} cache over {t_str} old')
            else:
                self._cache['expired'] = True
                self._cache[table] = pd.DataFrame()
                continue

            # Set the appropriate index if none already set
            if isinstance(cache.index, pd.RangeIndex):
                num_index = [f'{INDEX_KEY}_{n}' for n in range(2)]
                try:
                    int_eids = cache[num_index].any(axis=None)
                except KeyError:
                    int_eids = False
                cache.set_index(num_index if int_eids else INDEX_KEY, inplace=True)

            # Check sorted
            is_sorted = (cache.index.is_lexsorted()
                         if isinstance(cache.index, pd.MultiIndex)
                         else True)
            # Sorting makes MultiIndex indexing O(N) -> O(1)
            if table == 'datasets' and not is_sorted:
                cache.sort_index(inplace=True)

            self._cache[table] = cache
        return self._cache.get('loaded_time', None)

    def refresh_cache(self, mode='auto'):
        """Check and reload cache tables

        :param mode:
        :return: Loaded time
        """
        if mode == 'local':  # TODO maybe rename mode
            pass
        elif mode == 'auto':
            if datetime.now() - self._cache['loaded_time'] > self.cache_expiry:
                _logger.info('Cache expired, refreshing')
                self._load_cache()
        elif mode == 'refresh':
            _logger.debug('Forcing reload of cache')
            self._load_cache(clobber=True)
        else:
            raise ValueError(f'Unknown refresh type "{mode}"')
        return self._cache.get('loaded_time', None)

    def download_datasets(self, dsets, **kwargs) -> List[Path]:
        """
        TODO Support slice, dicts and URLs?
        Download several datasets given a slice of the datasets table
        :param dsets: list of dataset dictionaries from an Alyx REST query OR list of URL strings
        :return: local file path list
        """
        out_files = []
        if hasattr(dsets, 'iterrows'):
            dsets = map(lambda x: x[1], dsets.iterrows())
        # FIXME Thread timeout?
        with concurrent.futures.ThreadPoolExecutor(max_workers=NTHREADS) as executor:
            # TODO Subclass can just call web client method directly, no need to pass hash, etc.
            futures = [executor.submit(self.download_dataset, dset, file_size=dset['file_size'],
                                       hash=dset['hash'], **kwargs) for dset in dsets]
            concurrent.futures.wait(futures)
            for future in futures:
                out_files.append(future.result())
        return out_files

    def download_dataset(self, dset, cache_dir=None, **kwargs) -> Path:
        """
        Download a dataset from an alyx REST dictionary
        :param dset: single dataset dictionary from an Alyx REST query OR URL string
        :param cache_dir (optional): root directory to save the data in (home/downloads by default)
        :return: local file path
        """
        pass

    def search(self, details=False, exists_only=False, **kwargs):
        """
        Applies a filter to the sessions (eid) table and returns a list of json dictionaries
         corresponding to sessions.

        For a list of search terms, use the methods

         one.search_terms

        :param dataset: list of datasets
        :type dataset: list of str

        :param date_range: list of 2 strings or list of 2 dates that define the range (inclusive)
        :type date_range: list, str, timestamp

        :param details: default False, returns also the session details as per the REST response
        :type details: bool

        :param lab: a str or list of lab names
        :type lab: list or str

        :param number: number of session to be returned; will take the first n sessions found
        :type number: list, str or int

        :param subjects: a list of subjects nickname
        :type subjects: list or str

        :param task_protocol: task protocol name (can be partial, i.e. any task protocol
                              containing that str will be found)
        :type task_protocol: list or str

        :param project: project name (can be partial, i.e. any task protocol containing
                        that str will be found)
        :type project: list or str

        :return: list of eids, if details is True, also returns a list of dictionaries,
         each entry corresponding to a matching session
        :rtype: list, list


        """

        def validate_input(inarg):
            """Ensure input is a list"""
            return [inarg] if isinstance(inarg, str) or not isinstance(inarg, Iterable) else inarg

        def all_present(x, dsets, exists=None):
            """Returns true if all datasets present in Series"""

            return all(any(x.str.contains(y)) for y in dsets)

        def autocomplete(term):
            """
            Validate search term and return complete name, e.g. autocomplete('subj') == 'subject'
            """
            full_key = (x for x in self.search_terms if x.lower().startswith(term))
            key_ = next(full_key, None)
            if not key_:
                raise ValueError(f'Invalid search term "{term}"\n'
                                 'Note: remote search terms may differ')
            elif next(full_key, None):
                raise ValueError(f'Ambiguous search term "{term}"')
            return key_

        # Iterate over search filters, reducing the sessions table
        sessions = self._cache['sessions']

        # Ensure sessions filtered in a particular order, with datasets last
        search_order = ('date_range', 'number', 'dataset')
        sort_fcn = lambda itm: -1 if itm[0] not in search_order else search_order.index(itm[0])
        queries = {autocomplete(k): v for k, v in kwargs.items()}  # Validate and get full name
        for key, value in sorted(queries.items(), key=sort_fcn):
            # key = autocomplete(key)  # Validate and get full name
            # No matches; short circuit
            if sessions.size == 0:
                return []
            # String fields
            elif key in ('subject', 'task_protocol', 'laboratory', 'project'):
                query = '|'.join(validate_input(value))
                mask = sessions['lab' if key == 'laboratory' else key].str.contains(query)
                sessions = sessions[mask.astype(bool, copy=False)]
            elif key == 'date_range':
                start, end = _validate_date_range(value)
                session_date = pd.to_datetime(sessions['date'])
                sessions = sessions[(session_date >= start) & (session_date <= end)]
            elif key == 'number':
                query = validate_input(value)
                sessions = sessions[sessions[key].isin(query)]
            # Dataset check is biggest so this should be done last
            elif key == 'dataset':
                index = ['eid_0', 'eid_1'] if self._index_type('datasets') is int else 'eid'
                query = validate_input(value)
                datasets = self._cache['datasets']
                isin, _ = ismember2d(datasets[['eid_0', 'eid_1']].values,
                                     np.array(sessions.index.values.tolist()))
                if exists_only:
                    # For each session check any dataset both contains query and exists
                    # FIXME This doesn't work
                    mask = (
                        datasets[isin]
                            .groupby(index, sort=False)
                            .apply(lambda x: all_present(x['rel_path'], query) & x['exists'])
                    )
                else:
                    # For each session check any dataset contains query
                    mask = (
                        datasets[isin]
                            .groupby(index, sort=False)['rel_path']
                            .aggregate(lambda x: all_present(x, query))
                    )
                # eids of matching dataset records
                idx = mask[mask].index

                # Reduce sessions table by datasets mask
                sessions = sessions.loc[idx]

        # Return results
        if sessions.size == 0:
            return []
        eids = sessions.index.to_list()
        if self._index_type() is int:
            eids = parquet.np2str(np.array(eids))

        if details:
            return eids, sessions.reset_index().iloc[:, 2:].to_dict('records', Bunch)
        else:
            return eids

    def to_eid(self,
               id: Listable(Union[str, Path, UUID, dict]) = None,
               cache_dir: Optional[Union[str, Path]] = None) -> Listable(str):
        # TODO Could add np2str here
        if isinstance(id, (list, tuple)):  # Recurse
            return [self.to_eid(i, cache_dir) for i in id]
        if isinstance(id, UUID):
            return str(id)
        # elif is_exp_ref(id):
        #     return ref2eid(id, one=self)
        elif isinstance(id, dict):
            assert {'subject', 'number', 'start_time', 'lab'}.issubset(id)
            root = Path(cache_dir or self._cache_dir)
            id = root.joinpath(
                id['lab'],
                'Subjects', id['subject'],
                id['start_time'][:10],
                ('%03d' % id['number']))

        if alfio.is_session_path(id):
            return self.eid_from_path(id)
        elif isinstance(id, str):
            if len(id) > 36:
                id = id[-36:]
            if not alfio.is_uuid_string(id):
                raise ValueError('Invalid experiment ID')
            else:
                return id
        else:
            raise ValueError('Unrecognized experiment ID')

    def _update_filesystem(self, datasets, offline=True, update_exists=True, clobber=False):
        """Update the local filesystem for the given datasets
        Given a set of datasets, check whether records correctly reflect the filesystem.  If
        TODO This needs changing; overlaod for downloading?
        TODO change name to check_files, check_present, present_datasets, check_local_files?
         check_filesystem?

        :param datasets: A list or DataFrame of dataset records
        :param offline: If false and Web client present, downloads the missing datasets from a
        remote repository
        :param update_exists: If true, the cache is updated to reflect the filesystem
        :param clobber: If true and not offline, datasets are re-downloaded regardless of local
        filesystem
        :return: A list of file paths for the datasets (None elements for non-existent datasets)
        """
        if offline or not getattr(self, '_web_client', None):
            files = []
            if not isinstance(datasets, pd.DataFrame):
                # Cast from Series or set of dicts (i.e. from REST datasets endpoint)
                datasets = pd.DataFrame(list(datasets))
            for i, rec in datasets.iterrows():
                file = Path(self._cache_dir, *rec[['session_path', 'rel_path']])
                if file.exists():
                    # TODO Factor out; hash & file size also checked in _download_file;
                    #  see _update_cache - we need to save changed cache
                    files.append(file)
                    new_hash = hashfile.md5(file)
                    new_size = file.stat().st_size
                    hash_mismatch = rec['hash'] and new_hash != rec['hash']
                    size_mismatch = rec['file_size'] and new_size != rec['file_size']
                    if hash_mismatch or size_mismatch:
                        # the local file hash doesn't match the dataset table cached hash
                        # datasets.at[i, ['hash', 'file_size']] = new_hash, new_size
                        # Raise warning if size changed or hash changed and wasn't empty
                        if size_mismatch or (hash_mismatch and rec['hash']):
                            _logger.warning(f'local md5 or size mismatch')
                else:
                    files.append(None)
                if rec['exists'] != file.exists():
                    datasets.at[i, 'exists'] = not rec['exists']
                    if update_exists:
                        self._cache['datasets'].at[i, 'exists'] = rec['exists']
        else:
            # TODO deal with clobber and exists here?
            files = self.download_datasets(datasets, update_exists=update_exists, clobber=clobber)
        return files

    def _index_type(self, table='sessions'):
        idx_0 = self._cache[table].index.values[0]
        if len(self._cache[table].index.names) == 2 and all(isinstance(x, int) for x in idx_0):
            return int
        elif len(self._cache[table].index.names) == 1 and isinstance(idx_0, str):
            return str
        else:
            raise IndexError

    @parse_id
    def get_details(self, eid: Union[str, Path, UUID], full: bool = False):
        int_ids = self._index_type() is int
        if int_ids:
            eid = parquet.str2np(eid).tolist()
        det = self._cache['sessions'].loc[eid]
        if full:
            # to_drop = 'eid' if int_ids else ['eid_0', 'eid_1']
            # det = det.drop(to_drop, axis=1)
            det = self._cache['datasets'].join(det, on=det.index.names, how='right')
        return det

    def list_subjects(self, query_type='auto') -> List[str]:
        """
        List all subjects in database
        :return: Sorted list of subject names
        """
        self.refresh_cache(query_type)
        return self._cache['sessions']['subject'].sort_values().unique()

    def list_datasets(self, eid=None, sorted=False, query_type='auto') -> Union[
        np.ndarray, pd.DataFrame]:
        """
        Given one or more eids, return the datasets for those sessions.  If no eid is provided,
        a list of all unique datasets is returned

        TODO Return datasets sorted by session date, relpath

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :return: Slice of datasets table or numpy array if eid is None
        """
        if not eid:
            return self._cache['datasets']['rel_path'].unique()
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if self._index_type() is int:
            eid_num = parquet.str2np(eid)
            index = ['eid_0', 'eid_1']
            isin, _ = ismember2d(self._cache['datasets'][index].to_numpy(), eid_num)
            datasets = self._cache['datasets'][isin]
        else:
            session_match = self._cache['datasets']['eid'].isin(eid)
            datasets = self._cache['datasets'][session_match]
        return datasets

    @parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = 'alf',
                    query_type: str = 'auto',
                    **kwargs) -> Union[alfio.AlfBunch, List[Path]]:
        """
        Load all attributes of an ALF object from a Session ID and an object name.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param obj: The ALF object to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        Supports asterisks as wildcards.
        :param download_only: When true the data are downloaded and the file paths are returned
        :param kwargs: Optional filters for the ALF objects, including namespace and timescale
        :return: An ALF bunch or if download_only is True, a list of Paths objects

        Examples:
        load_object(eid, '*moves')
        load_object(eid, 'trials')
        load_object(eid, 'spikes', collection='*probe01')
        """
        datasets = self.list_datasets(eid)

        if len(datasets) == 0:
            raise ALFObjectNotFound(f'ALF object "{obj}" not found in cache')

        expression = alf_regex(f'{COLLECTION_SPEC}/{FILE_SPEC}', object=obj, collection=collection)
        REGEX = True
        if not REGEX:
            obj.replace('*', '.*')
        table = datasets['rel_path'].str.extract(expression)
        match = ~table[['collection', 'object']].isna().all(axis=1)

        # Validate result before loading
        if table['object'][match].unique().size > 1:
            raise ALFMultipleObjectsFound('The following matching objects were found: ' +
                                          ', '.join(table['object'][match].unique()))
        elif not match.any():
            raise ALFObjectNotFound(f'ALF object "{obj}" not found on Alyx')
        if table['collection'][match].unique().size > 1:
            raise ALFMultipleCollectionsFound('Matching object belongs to multiple collections:' +
                                              ', '.join(table['collection'][match].unique()))

        datasets = datasets[match]

        # parquet.np2str(np.array(datasets.index.values.tolist()))
        # For those that don't exist, download them
        # return alfio.load_object(path, table[match]['object'].values[0])
        download_only = kwargs.pop('download_only', False)
        files = self._update_filesystem(datasets, offline=query_type == 'local')
        pprint(files)
        files = [x for x in files if x]
        if not files:
            raise ALFObjectNotFound(f'ALF object "{obj}" not found on Alyx')

        if download_only:
            return files

        # self._check_exists(datasets[~datasets['exists']])
        return alfio.load_object(files[0].parent, table[match]['object'].values[0], **kwargs)

    @parse_id
    def load_session_dataset(self,
                             eid: Union[str, Path, UUID],
                             dataset: str,
                             collection: Optional[str] = 'alf',
                             revision: Optional[str] = None,
                             **kwargs) -> Any:
        datasets = self.list_datasets(eid)

        if len(datasets) == 0:
            raise ALFObjectNotFound(f'ALF dataset "{dataset}" not found in cache')

        # Split path into
        # TODO This could maybe be an ALF function
        expression = alf_regex(COLLECTION_SPEC, revision=revision, collection=collection)
        table = datasets['rel_path'].str.rsplit('/', 1, expand=True)
        match = table[1] == dataset
        # Check collection and revision matches
        table = table[0].str.extract(expression)
        match &= ~table['collection'].isna() & (~table['revision'].isna() if revision else True)
        if not match.any():
            raise ALFObjectNotFound('Dataset not found')
        elif sum(match) != 1:
            raise ALFMultipleCollectionsFound('Multiple datasets returned')

        download_only = kwargs.pop('download_only', False)
        # Check files exist / download remote files
        file, = self._update_filesystem(datasets[match], **kwargs)

        if not file:
            raise ALFObjectNotFound('Dataset not found')
        elif download_only:
            return file
        return alfio.load_file_content(file)

    def load_dataset_from_id(self,
                             dset_id: Union[str, UUID],
                             download_only: bool = False,
                             details: bool = False,
                             **kwargs) -> Any:
        if isinstance(dset_id, str):
            dset_id = parquet.str2np(dset_id)
        elif isinstance(dset_id, UUID):
            dset_id = parquet.uuid2np([dset_id])
        # else:
        #     dset_id = np.asarray(dset_id)
        if self._index_type('datasets') is int:
            try:
                dataset = self._cache['datasets'].loc[dset_id.tolist()]
                assert len(dataset) == 1
                dataset = dataset.iloc[0]
            except KeyError:
                raise ALFObjectNotFound('Dataset not found')
            except AssertionError:
                raise ALFMultipleObjectsFound('Duplicate dataset IDs')
        else:
            ids = self._cache['datasets'][['id_0', 'id_1']].to_numpy()
            try:
                dataset = self._cache['datasets'].iloc[find_first_2d(ids, dset_id)]
                assert len(dataset) == 1
            except TypeError:
                raise ALFObjectNotFound('Dataset not found')
            except AssertionError:
                raise ALFMultipleObjectsFound('Duplicate dataset IDs')

        filepath, = self._update_filesystem(dataset)
        if not filepath:
            raise ALFObjectNotFound('Dataset not found')
        output = filepath if download_only else alfio.load_file_content(filepath)
        if details:
            return output, dataset
        else:
            return output

    # @parse_id
    # def load(self, eid, datasets=None, dclass_output=False, cache_dir=None,
    #          download_only=False, clobber=False, offline=False, keep_uuid=False):
    #     """
    #     From a Session ID and dataset types, queries Alyx database, downloads the data
    #     from Globus, and loads into numpy array.
    #
    #     :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
    #      database. Could be a full Alyx URL:
    #      'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
    #      '698361f6-b7d0-447d-a25d-42afdef7a0da'. Can also be a list of the above for multiple eids.
    #     :type eid: str
    #     :param datasets: [None]: Alyx dataset types to be returned.
    #     :type datasets: list
    #     :param dclass_output: [False]: forces the output as dataclass to provide context.
    #     :type dclass_output: bool
    #      If None or an empty dataset_type is specified, the output will be a dictionary by default.
    #     :param cache_dir: temporarily overrides the cache_dir from the parameter file
    #     :type cache_dir: str
    #     :param download_only: do not attempt to load data in memory, just download the files
    #     :type download_only: bool
    #     :param clobber: force downloading even if files exists locally
    #     :type clobber: bool
    #     :param keep_uuid: keeps the UUID at the end of the filename (defaults to False)
    #     :type keep_uuid: bool
    #
    #     :return: List of numpy arrays matching the size of dataset_types parameter, OR
    #      a dataclass containing arrays and context data.
    #     :rtype: list, dict, dataclass SessionDataInfo
    #     """
    #     # if no dataset_type is provided:
    #     # a) force the output to be a dictionary that provides context to the data
    #     # b) download all types that have a data url specified whithin the alf folder
    #     datasets = [datasets] if isinstance(datasets, str) else datasets
    #
    #
    #     if not datasets or datasets == 'all':
    #         dclass_output = True
    #     if offline:
    #         dc = self._make_dataclass_offline(eid_str, datasets, **kwargs)
    #     else:
    #         dc = self._make_dataclass(eid_str, datasets, **kwargs)
    #     # load the files content in variables if requested
    #     if not download_only:
    #         for ind, fil in enumerate(dc.local_path):
    #             dc.data[ind] = alfio.load_file_content(fil)
    #     # parse output arguments
    #     if dclass_output:
    #         return dc
    #     # if required, parse the output as a list that matches dataset_types requested
    #     list_out = []
    #     for dt in datasets:
    #         if dt not in dc.dataset_type:
    #             _logger.warning('dataset ' + dt + ' not found for session: ' + eid_str)
    #             list_out.append(None)
    #             continue
    #         for i, x, in enumerate(dc.dataset_type):
    #             if dt == x:
    #                 if dc.data[i] is not None:
    #                     list_out.append(dc.data[i])
    #                 else:
    #                     list_out.append(dc.local_path[i])
    #     return list_out

    @abc.abstractmethod
    def list(self, **kwargs):
        pass

    @staticmethod
    def setup(**kwargs):
        """
        Interactive command tool that populates parameter file for ONE IBL.
        """
        one.params.setup(**kwargs)


def ONE(mode='auto', **kwargs):
    """ONE API factory
    Determine which class to instantiate depending on parameters passed.
    """
    if kwargs.pop('offline', False):
        _logger.warning('the offline kwarg will probably be removed. '
                        'ONE is now offline by default anyway')
        warnings.warn('"offline" param will be removed; use mode="local"', DeprecationWarning)

    if (any(x in kwargs for x in ('base_url', 'username', 'password')) or
            not kwargs.get('cache_dir', False)):
        return OneAlyx(mode=mode, **kwargs)

    # TODO This feels hacky
    # If cache dir was provided and corresponds to one configured with an Alyx client, use OneAlyx
    try:
        one.params._check_cache_conflict(kwargs.get('cache_dir'))
        return One(**kwargs)
    except AssertionError:
        # Cache dir corresponds to a Alyx repo, call OneAlyx
        return OneAlyx(mode=mode, **kwargs)


class OneAlyx(One):
    def __init__(self, username=None, password=None, base_url=None, mode='auto', **kwargs):
        # Load Alyx Web client
        self._web_client = wc.AlyxClient(username=username,
                                         password=password,
                                         base_url=base_url,
                                         silent=kwargs.pop('silent', False))
        self.mode = mode
        # get parameters override if inputs provided
        super(OneAlyx, self).__init__(**kwargs)

    def _load_cache(self, cache_dir=None, clobber=False):
        if not clobber:
            super(OneAlyx, self)._load_cache(self._cache_dir)  # Load any present cache
            if (self._cache and not self._cache['expired']) or self.mode == 'local':
                return

        # Determine whether a newer cache is available
        cache_info = self.alyx.get('cache/info', expires=True)
        remote_created = datetime.fromisoformat(cache_info['date_created'])
        if (remote_created - self._cache['created_time']) < timedelta(minutes=1):
            _logger.info('No newer cache available')
            return

        # Download the remote cache files
        _logger.info('Downloading remote caches...')
        try:
            files = self.alyx.download_cache_tables()
            assert any(files)
            super(OneAlyx, self)._load_cache(self._cache_dir)  # Reload cache after download
        except HTTPError:
            _logger.error(f'Failed to load the remote cache file')

    @property
    def alyx(self):
        return self._web_client

    @property
    def _cache_dir(self):
        return self._web_client.cache_dir

    def help(self, dataset_type=None):
        # TODO Move the AlyxClient; add to rest examples
        if not dataset_type:
            return self.alyx.rest('dataset-types', 'list')
        if isinstance(dataset_type, list):
            for dt in dataset_type:
                self.help(dataset_type=dt)
                return
        if not isinstance(dataset_type, str):
            print('No dataset_type provided or wrong type. Should be str')
            return
        out = self.alyx.rest('dataset-types', 'read', dataset_type)
        print(out['description'])

    def list(self, eid: Optional[Union[str, Path, UUID]] = None, details=False
             ) -> Union[List, Dict[str, str]]:
        """
        From a Session ID, queries Alyx database for datasets related to a session.

        :param eid: Experiment session uuid str
        :type eid: str

        :param details: If false returns a list of path, otherwise returns the REST dictionary
        :type eid: bool

        :return: list of strings or dict of lists if details is True
        :rtype:  list, dict
        """
        if not eid:
            return [x['name'] for x in self.alyx.rest('dataset-types', 'list')]

        # Session specific list
        dsets = self.alyx.rest('datasets', 'list', session=eid, exists=True)
        if not details:
            dsets = sorted([Path(dset['collection']).joinpath(dset['name']) for dset in dsets])
        return dsets

    @parse_id
    def load(self, eid, dataset_types=None, dclass_output=False, dry_run=False, cache_dir=None,
             download_only=False, clobber=False, offline=False, keep_uuid=False):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array.

        :param eid: Experiment ID, for IBL this is the UUID of the Session as per Alyx
         database. Could be a full Alyx URL:
         'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da' or only the UUID:
         '698361f6-b7d0-447d-a25d-42afdef7a0da'. Can also be a list of the above for multiple eids.
        :type eid: str
        :param dataset_types: [None]: Alyx dataset types to be returned.
        :type dataset_types: list
        :param dclass_output: [False]: forces the output as dataclass to provide context.
        :type dclass_output: bool
         If None or an empty dataset_type is specified, the output will be a dictionary by default.
        :param cache_dir: temporarly overrides the cache_dir from the parameter file
        :type cache_dir: str
        :param download_only: do not attempt to load data in memory, just download the files
        :type download_only: bool
        :param clobber: force downloading even if files exists locally
        :type clobber: bool
        :param keep_uuid: keeps the UUID at the end of the filename (defaults to False)
        :type keep_uuid: bool

        :return: List of numpy arrays matching the size of dataset_types parameter, OR
         a dataclass containing arrays and context data.
        :rtype: list, dict, dataclass SessionDataInfo
        """
        # this is a wrapping function to keep signature and docstring accessible for IDE's
        return self._load_recursive(eid, dataset_types=dataset_types, dclass_output=dclass_output,
                                    dry_run=dry_run, cache_dir=cache_dir, keep_uuid=keep_uuid,
                                    download_only=download_only, clobber=clobber, offline=offline)

    @parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = None,
                     download_only: bool = False) -> Any:
        """
        Load a single dataset from a Session ID and a dataset type.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param dataset: The ALF dataset to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        For IBL this is the relative path of the file from the session root.
        Supports asterisks as wildcards.
        :param download_only: When true the data are downloaded and the file path is returned
        :return: dataset or a Path object if download_only is true

        Examples:
            intervals = one.load_dataset(eid, '_ibl_trials.intervals.npy')
            intervals = one.load_dataset(eid, '*trials.intervals*')
            filepath = one.load_dataset(eid '_ibl_trials.intervals.npy', download_only=True)
            spikes = one.load_dataset(eid 'spikes.times.npy', collection='alf/probe01')
        """
        search_str = 'name__regex,' + dataset.replace('.', r'\.').replace('*', '.*')
        if collection:
            search_str += ',collection__regex,' + collection.replace('*', '.*')
        results = self.alyx.rest('datasets', 'list', session=eid, django=search_str, exists=True)

        # Get filenames of returned ALF files
        collection_set = {x['collection'] for x in results}
        if len(collection_set) > 1:
            raise ALFMultipleCollectionsFound('Matching dataset belongs to multiple collections:' +
                                              ', '.join(collection_set))
        if len(results) > 1:
            raise ALFMultipleObjectsFound('The following matching datasets were found: ' +
                                          ', '.join(x['name'] for x in results))
        if len(results) == 0:
            raise ALFObjectNotFound(f'Dataset "{dataset}" not found on Alyx')

        filename = self.download_dataset(results[0])
        assert filename is not None, 'failed to download dataset'

        return filename if download_only else alfio.load_file_content(filename)

    @parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = 'alf',
                    download_only: bool = False,
                    query_type: str = 'auto',
                    clobber: bool = False,
                    **kwargs) -> Union[alfio.AlfBunch, List[Path]]:
        """
        Load all attributes of an ALF object from a Session ID and an object name.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param obj: The ALF object to load.  Supports asterisks as wildcards.
        :param collection: The collection to which the object belongs, e.g. 'alf/probe01'.
        Supports asterisks as wildcards.
        :param download_only: When true the data are downloaded and the file paths are returned
        :param query_type: Query cache ('local') or Alyx database ('remote')
        :param kwargs: Optional filters for the ALF objects, including namespace and timescale
        :return: An ALF bunch or if download_only is True, a list of Paths objects

        Examples:
        load_object(eid, '*moves')
        load_object(eid, 'trials')
        load_object(eid, 'spikes', collection='*probe01')
        """
        # TODO
        if query_type != 'remote' or (query_type == 'auto' and self.offline == True):
            load_object_offline = unwrap(super().load_object)  # Skip parse_id decorator
            return load_object_offline(self, eid, obj,
                                       collection=collection, download_only=download_only,
                                       query_type=query_type, **kwargs)
        # Filter server-side by collection and dataset name
        search_str = 'name__regex,' + obj.replace('*', '.*')
        if collection and collection != 'all':
            search_str += ',collection__regex,' + collection.replace('*', '.*')
        results = self.alyx.rest('datasets', 'list', exists=True, session=eid, django=search_str)
        pattern = re.compile(fnmatch.translate(obj))

        # Further refine by matching object part of ALF datasets
        def match(r):
            return is_valid(r['name']) and pattern.match(alf_parts(r['name'])[1])

        # Get filenames of returned ALF files
        returned_obj = {alf_parts(x['name'])[1] for x in results if match(x)}

        # Validate result before loading
        if len(returned_obj) > 1:
            raise ALFMultipleObjectsFound('The following matching objects were found: ' +
                                          ', '.join(returned_obj))
        elif len(returned_obj) == 0:
            raise ALFObjectNotFound(f'ALF object "{obj}" not found on Alyx')
        collection_set = {x['collection'] for x in results if match(x)}
        if len(collection_set) > 1:
            raise ALFMultipleCollectionsFound('Matching object belongs to multiple collections:' +
                                              ', '.join(collection_set))

        # Download and optionally load the datasets
        out_files = self._update_filesystem((x for x in results if match(x)), clobber=clobber)
        # out_files = self.download_datasets(x for x in results if match(x))
        assert not any(x is None for x in out_files), 'failed to download object'
        if download_only:
            return out_files
        else:
            return alfio.load_object(out_files[0].parent, obj, **kwargs)

    def _load_recursive(self, eid, **kwargs):
        """
        From a Session ID and dataset types, queries Alyx database, downloads the data
        from Globus, and loads into numpy array. Supports multiple sessions
        """
        if isinstance(eid, str):
            return self._load(eid, **kwargs)
        if isinstance(eid, list):
            # dataclass output requested
            if kwargs.get('dclass_output', False):
                for i, e in enumerate(eid):
                    if i == 0:
                        out = self._load(e, **kwargs)
                    else:
                        out.append(self._load(e, **kwargs))
            else:  # list output requested
                out = []
                for e in eid:
                    out.append(self._load(e, **kwargs)[0])
            return out

    def to_eid(self,
               id: Listable(Union[str, Path, UUID, dict]) = None,
               cache_dir: Optional[Union[str, Path]] = None) -> Listable(str):
        if isinstance(id, (list, tuple)):  # Recurse
            return [self.to_eid(i, cache_dir) for i in id]
        if isinstance(id, UUID):
            return str(id)
        # elif is_exp_ref(id):
        #     return ref2eid(id, one=self)
        elif isinstance(id, dict):
            assert {'subject', 'number', 'start_time', 'lab'}.issubset(id)
            root = Path(self._cache_dir)
            id = root.joinpath(
                id['lab'],
                'Subjects', id['subject'],
                id['start_time'][:10],
                ('%03d' % id['number']))

        if alfio.is_session_path(id):
            return self.eid_from_path(id)
        elif isinstance(id, str):
            if len(id) > 36:
                id = id[-36:]
            if not alfio.is_uuid_string(id):
                raise ValueError('Invalid experiment ID')
            else:
                return id
        else:
            raise ValueError('Unrecognized experiment ID')

    def pid2eid(self, pid: str, query_type='auto') -> (str, str):
        """
        Given an Alyx probe UUID string, returns the session id string and the probe label
        (i.e. the ALF collection)

        :param pid: A probe UUID
        :param query_type: Query mode, options include 'auto', 'remote' and 'refresh'
        :return: (experiment ID, probe label)
        """
        if query_type != 'remote':
            self.refresh_cache(query_type)
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplemented('Converting probe IDs required remote connection')
        rec = self.alyx.rest('insertions', 'read', id=pid)
        return rec['session'], rec['name']

    def _ls(self, table=None, verbose=False):
        """
        Queries the database for a list of 'users' and/or 'dataset-types' and/or 'subjects' fields

        :param table: the table (s) to query among: 'dataset-types','users'
         and 'subjects'; if empty or None assumes all tables
        :type table: str
        :param verbose: [False] prints the list in the current window
        :type verbose: bool

        :return: list of names to query, list of full raw output in json serialized format
        :rtype: list, list
        """
        assert (isinstance(table, str))
        table_field_names = {
            'dataset-types': 'name',
            'datasets': 'name',
            'users': 'username',
            'subjects': 'nickname',
            'labs': 'name'}
        if not table or table not in list(set(_ENDPOINTS.keys())):
            raise KeyError("The attribute/endpoint: " + table + " doesn't exist \n" +
                           "possible values are " + str(set(_ENDPOINTS.values())))

        field_name = table_field_names[_ENDPOINTS[table]]
        full_out = self.alyx.get('/' + _ENDPOINTS[table])
        list_out = [f[field_name] for f in full_out]
        if verbose:
            pprint(list_out)
        return list_out, full_out

    def autocomplete(self, term):
        """ TODO Move to super class
        Validate search term and return complete name, e.g. autocomplete('subj') == 'subject'
        """
        full_key = (x for x in self.search_terms if x.lower().startswith(term))
        key_ = next(full_key, None)
        if not key_:
            if term.lower() in ('dtype', 'dtypes', 'dataset_types', 'dataset_type'):
                _logger.warning('Searching by dataset type is deprecated')
                return 'dataset_type'
            raise ValueError(f'Invalid search term "{term}"')
        elif next(full_key, None):
            raise ValueError(f'Ambiguous search term "{term}"')
        return key_

    # def search(self, dataset_types=None, users=None, subjects=None, date_range=None,
    #            lab=None, number=None, task_protocol=None, details=False):
    def search(self, details=False, limit=None, query_type='auto', **kwargs):
        """
        Applies a filter to the sessions (eid) table and returns a list of json dictionaries
         corresponding to sessions.

        For a list of search terms, use the methods

        >>> one.search_terms

        :param dataset_types: list of dataset_types
        :type dataset_types: list of str

        :param date_range: list of 2 strings or list of 2 dates that define the range
        :type date_range: list

        :param details: default False, returns also the session details as per the REST response
        :type details: bool

        :param lab: a str or list of lab names
        :type lab: list or str

        :param limit: default None, limits results (if pagination enabled on server)
        :type limit: int List of possible search terms

        :param location: a str or list of lab location (as per Alyx definition) name
                         Note: this corresponds to the specific rig, not the lab geographical
                         location per se
        :type location: str

        :param number: number of session to be returned; will take the first n sessions found
        :type number: str or int

        :param performance_lte / performance_gte: search only for sessions whose performance is
        less equal or greater equal than a pre-defined threshold as a percentage (0-100)
        :type performance_gte: float

        :param subjects: a list of subjects nickname
        :type subjects: list or str

        :param task_protocol: a str or list of task protocol name (can be partial, i.e.
                              any task protocol containing that str will be found)
        :type task_protocol: str

        :param users: a list of users
        :type users: list or str

        :return: list of eids, if details is True, also returns a list of json dictionaries,
         each entry corresponding to a matching session
        :rtype: list, list


        """
        if query_type != 'remote':
            return super(OneAlyx, self).search(details=details, **kwargs)

        # small function to make sure string inputs are interpreted as lists
        def validate_input(inarg):
            if isinstance(inarg, str):
                return [inarg]
            elif isinstance(inarg, int):
                return [str(inarg)]
            else:
                return inarg

        # loop over input arguments and build the url
        url = '/sessions?'
        for key, value in kwargs.items():
            field = self.autocomplete(key)  # Validate and get full name
            # check that the input matches one of the defined filters
            if field == 'date_range':
                query = _validate_date_range(value)
                url += f'&{field}=' + ','.join(x.date().isoformat() for x in query)
            elif field == 'dataset_type':  # legacy
                url += f'&dataset_type=' + ','.join(validate_input(value))
            elif field == 'dataset':
                url += (f'&django=data_dataset_session_related__dataset_type__name__icontains,' +
                        ','.join(validate_input(value)))
            else:  # TODO Overload search terms (users, etc.)
                url += f'&{field}=' + ','.join(validate_input(value))
        # the REST pagination argument has to be the last one
        if limit:
            url += f'&limit={limit}'
        # implements the loading itself
        ses = self.alyx.get(url)
        if len(ses) > 2500:
            eids = [s['url'] for s in tqdm.tqdm(ses)]  # flattens session info
        else:
            eids = [s['url'] for s in ses]
        eids = [e.split('/')[-1] for e in eids]  # remove url to make it portable

        if details:
            for s in ses:
                if all([s.get('lab'), s.get('subject'), s.get('start_time')]):
                    s['local_path'] = str(Path(self._cache_dir, s['lab'], 'Subjects',
                                               s['subject'], s['start_time'][:10],
                                               str(s['number']).zfill(3)))
                else:
                    s['local_path'] = None
            return eids, ses
        else:
            return eids

    def download_dataset(self, dset, cache_dir=None, update_cache=True, **kwargs):
        """
        Download a dataset from an alyx REST dictionary
        :param dset: single dataset dictionary from an Alyx REST query OR URL string
        :param cache_dir: root directory to save the data in (home/downloads by default)
        :return: local file path
        """
        if isinstance(dset, str):
            url = dset
            id = self.record_from_path(url).index
        else:
            if 'file_records' not in dset:  # Convert dataset Series to alyx dataset dict
                url = self.url_from_record(dset)
                id = dset.index
            else:
                url = next((fr['data_url'] for fr in dset['file_records']
                            if fr['data_url'] and fr['exists']), None)
                id = dset['id']

        if not url:
            # str_dset = Path(dset['collection']).joinpath(dset['name'])
            # str_dset = dset['rel_path']
            _logger.warning(f"Dataset not found")
            if update_cache:
                if isinstance(id, str) and self._index_type('datasets') is int:
                    id = parquet.str2np(id)
                elif self._index_type('datasets') is str and not isinstance(id, str):
                    id = parquet.np2str(id)
                self._cache['datasets'].at[id, 'exists'] = False
            return
        target_dir = Path(cache_dir or self._cache_dir, alfio.get_alf_path(url)).parent
        return self._download_file(url=url, target_dir=target_dir, **kwargs)

    def _tag_mismatched_file_record(self, url):
        fr = self.alyx.rest('files', 'list', django=f'dataset,{Path(url).name.split(".")[-2]},'
                                                    f'data_repository__globus_is_personal,False')
        if len(fr) > 0:
            json_field = fr[0]['json']
            if json_field is None:
                json_field = {'mismatch_hash': True}
            else:
                json_field.update({'mismatch_hash': True})
            self.alyx.rest('files', 'partial_update', id=fr[0]['url'][-36:],
                           data={'json': json_field})

    def _download_file(self, url, target_dir,
                       clobber=False, offline=False, keep_uuid=False, file_size=None, hash=None):
        """
        Downloads a single file from an HTTP webserver
        :param url:
        :param clobber: (bool: False) overwrites local dataset if any
        :param offline:
        :param keep_uuid:
        :param file_size:
        :param hash:
        :return:
        """
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        local_path = target_dir / os.path.basename(url)
        if not keep_uuid:
            local_path = alfio.remove_uuid_file(local_path, dry=True)
        if Path(local_path).exists():
            # the local file hash doesn't match the dataset table cached hash
            hash_mismatch = hash and hashfile.md5(Path(local_path)) != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if hash_mismatch or file_size_mismatch:
                clobber = True
                _logger.warning(f'local md5 or size mismatch, re-downloading {local_path}')
        # if there is no cached file, download
        else:
            clobber = True
        if clobber and not offline:
            local_path, md5 = self.alyx.download_file(
                url, cache_dir=str(target_dir), clobber=clobber, return_md5=True)
            # post download, if there is a mismatch between Alyx and the newly downloaded file size
            # or hash flag the offending file record in Alyx for database maintenance
            hash_mismatch = hash and md5 != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if hash_mismatch or file_size_mismatch:
                self._tag_mismatched_file_record(url)
                # TODO Update cache here
        if keep_uuid:
            return local_path
        else:
            return alfio.remove_uuid_file(local_path)

    @staticmethod
    def setup(**kwargs):
        """
        TODO Interactive command tool that sets up cache for ONE.
        """
        root_dir = input('Select a directory from which to build cache')
        if root_dir:
            import alf.onepqt
            print('Building ONE cache from filesystem...')
            alf.onepqt.make_parquet_db(root_dir, **kwargs)

    def path_from_eid(self, eid: str, use_cache: bool = True) -> Listable(Path):
        """
        From an experiment id or a list of experiment ids, gets the local cache path
        :param eid: eid (UUID) or list of UUIDs
        :param use_cache: if set to False, will force database connection
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
            print(eid, " is not a valid eID/UUID string")
            return

        # first try avoid hitting the database
        if self._cache['sessions'].size > 0 and use_cache:
            cache_path = super().path_from_eid(eid)
            if cache_path:
                return cache_path

        # if it wasn't successful, query Alyx
        ses = self.alyx.rest('sessions', 'list', django=f'pk,{eid}')
        if len(ses) == 0:
            return None
        else:
            return Path(self._cache_dir).joinpath(
                ses[0]['lab'], 'Subjects', ses[0]['subject'], ses[0]['start_time'][:10],
                str(ses[0]['number']).zfill(3))

    def eid_from_path(self, path_obj: Union[str, Path], use_cache: bool = True) -> Listable(Path):
        """
        From a local path, gets the experiment id
        :param path_obj: local path or list of local paths
        :param use_cache: if set to False, will force database connection
        :return: eid or list of eids
        """
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            path_obj = [Path(x) for x in path_obj]
            eid_list = []
            for p in path_obj:
                eid_list.append(self.eid_from_path(p))
            return eid_list
        # else ensure the path ends with mouse,date, number
        path_obj = Path(path_obj)
        session_path = alfio.get_session_path(path_obj)
        # if path does not have a date and a number return None
        if session_path is None:
            return None

        # try the cached info to possibly avoid hitting database
        cache_eid = super().eid_from_path(path_obj)
        if cache_eid:
            return cache_eid

        # if not search for subj, date, number XXX: hits the DB
        uuid = self.search(subjects=session_path.parts[-3],
                           date_range=session_path.parts[-2],
                           number=session_path.parts[-1])

        # Return the uuid if any
        return uuid[0] if uuid else None

    def url_from_path(self, filepath, query_type='local'):
        """
        Given a local file path, returns the URL of the remote file.
        :param filepath: A local file path
        :return: A URL string
        """
        if query_type not in ('local', 'remote'):
            raise ValueError(f'Unknown query_type "{query_type}"')
        if query_type == 'local':
            return super(OneAlyx, self).url_from_path(filepath)
        eid = self.eid_from_path(filepath)
        try:
            dataset, = self.alyx.rest('datasets', 'list', session=eid, name=Path(filepath).name)
            return next(
                r['data_url'] for r in dataset['file_records'] if r['data_url'] and r['exists'])
        except (ValueError, StopIteration):
            raise ALFObjectNotFound(f'File record for {filepath} not found on Alyx')

    @parse_id
    def datasets_from_type(self, eid, dataset_type, full=False):
        """
        Get list of datasets belonging to a given dataset type for a given session
        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param dataset_type: A dataset type, e.g. camera.times
        :param full: If True, a dictionary of details is returned for each dataset
        :return: A list of datasets belonging to that session's dataset type
        """
        restriction = f'session__id,{eid},dataset_type__name,{dataset_type}'
        datasets = self.alyx.rest('datasets', 'list', django=restriction)
        return datasets if full else [d['name'] for d in datasets]

    def get_details(self, eid: str, full: bool = False):
        """ Returns details of eid like from one.search, optional return full
        session details.
        """
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            details_list = []
            for p in eid:
                details_list.append(self.get_details(p, full=full))
            return details_list
        # If not valid return None
        if not alfio.is_uuid_string(eid):
            print(eid, " is not a valid eID/UUID string")
            return
        # load all details
        dets = self.alyx.rest("sessions", "read", eid)
        if full:
            return dets
        # If it's not full return the normal output like from a one.search
        det_fields = ["subject", "start_time", "number", "lab", "project",
                      "url", "task_protocol", "local_path"]
        out = {k: v for k, v in dets.items() if k in det_fields}
        out.update({'local_path': self.path_from_eid(eid)})
        return out

    def _update_cache(self, ses, dataset_types):
        """
        TODO move to One
        :param ses: session details dictionary as per Alyx response
        :param dataset_types:
        :return: is_updated (bool): if the cache was updated or not
        """
        save = False
        pqt_dsets = _ses2pandas(ses, dtypes=dataset_types)
        # if the dataframe is empty, return
        if pqt_dsets.size == 0:
            return
        # if the cache is empty create the cache variable
        elif self._cache.size == 0:
            self._cache = pqt_dsets
            save = True
        # the cache is not empty and there are datasets in the query
        else:
            isin, icache = ismember2d(pqt_dsets[['id_0', 'id_1']].to_numpy(),
                                      self._cache[['id_0', 'id_1']].to_numpy())
            # check if the hash / filesize fields have changed on patching
            heq = (self._cache['hash'].iloc[icache].to_numpy() ==
                   pqt_dsets['hash'].iloc[isin].to_numpy())
            feq = np.isclose(self._cache['file_size'].iloc[icache].to_numpy(),
                             pqt_dsets['file_size'].iloc[isin].to_numpy(),
                             rtol=0, atol=0, equal_nan=True)
            eq = np.logical_and(heq, feq)
            # update new hash / filesizes
            if not np.all(eq):
                self._cache.iloc[icache, 4:6] = pqt_dsets.iloc[np.where(isin)[0], 4:6].to_numpy()
                save = True
            # append datasets that haven't been found
            if not np.all(isin):
                self._cache = self._cache.append(pqt_dsets.iloc[np.where(~isin)[0]])
                self._cache = self._cache.reindex()
                save = True
        if save:
            # before saving makes sure pandas did not cast uuids in float
            typs = [t for t, k in zip(self._cache.dtypes, self._cache.keys()) if 'id_' in k]
            assert (all(map(lambda t: t == np.int64, typs)))
            # if this gets too big, look into saving only when destroying the ONE object
            parquet.save(self._cache_file, self._cache)


def _validate_date_range(date_range):
    """
    Validates and arrange date range in a 2 elements list

    Examples:
        _validate_date_range('2020-01-01')  # On this day
        _validate_date_range(datetime.date(2020, 1, 1))
        _validate_date_range(np.array(['2022-01-30', '2022-01-30'], dtype='datetime64[D]'))
        _validate_date_range(pd.Timestamp(2020, 1, 1))
        _validate_date_range(np.datetime64(2021, 3, 11))
        _validate_date_range(['2020-01-01'])  # from date
        _validate_date_range(['2020-01-01', None])  # from date
        _validate_date_range([None, '2020-01-01'])  # up to date
    """
    if date_range is None:
        return

    # Ensure we have exactly two values
    if isinstance(date_range, str) or not isinstance(date_range, Iterable):
        # date_range = (date_range, pd.Timestamp(date_range) + pd.Timedelta(days=1))
        dt = pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        date_range = (date_range, pd.Timestamp(date_range) + dt)
    elif len(date_range) == 1:
        date_range = [date_range[0], pd.Timestamp.max]
    elif len(date_range) != 2:
        raise ValueError

    # For comparisons, ensure both values are pd.Timestamp (datetime, date and datetime64
    # objects will be converted)
    start, end = date_range
    start = start or pd.Timestamp.min  # Convert None to lowest possible date
    end = end or pd.Timestamp.max  # Convert None to highest possible date

    # Convert to timestamp
    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)

    return start, end
