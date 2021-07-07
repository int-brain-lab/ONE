"""Classes for searching, listing and (down)loading ALyx Files
TODO Document
TODO Add sig to ONE Light uuids
TODO Save changes to cache
TODO Fix update cache in AlyxONE - save parquet table
TODO save parquet in update_filesystem
TODO edit alf exceptions constructor with default message template?

Points of discussion:
    - Module structure: oneibl is too restrictive, naming module `one` means obj should have
    different name
    - Download datasets timeout
    - NB: Wildcards will behave differently between REST and pandas - should regex or unix be
    default?
    - What should the collection default be? 'alf'? None? An ONE param?
    - Load has been removed altogether
    - Remove clobber method from load functions?
    - Remove exists_only from search method?
    - Support for pids?
    - Dealing with lists must be consistent.  Three options:
        - two methods each, e.g. load_dataset and load_datasets (con: a lot of overhead)
        - allow list inputs, recursive calls (con: function logic slightly more complex)
        - no list inputs; rely on list comprehensions (con: makes accessing meta data complex)
    - Need to check performance of 1. (re)setting index, 2. converting object array to 2D int array
    - NB: Sessions table date ordered.  Indexing by eid is therefore O(N) but not done in code.
    Datasets table has sorted index.
    - Conceivably you could have a subclass for Figshare, etc., not just Alyx
    - We have mode and query_type, stick to just one?
"""
import concurrent.futures
import warnings
import logging
import os
# import fnmatch
import re
from datetime import datetime, timedelta
from functools import lru_cache, reduce
from inspect import unwrap
from pathlib import Path
from typing import Any, Union, Optional, List
from uuid import UUID

import pandas as pd
import numpy as np
import requests.exceptions
from iblutil.io import parquet, hashfile
from iblutil.util import Bunch
from iblutil.numerical import ismember2d

import one.params
import one.webclient as wc
import one.alf.io as alfio
import one.alf.exceptions as alferr
from .alf.cache import make_parquet_db
from .alf.files import rel_path_parts
from .alf.spec import COLLECTION_SPEC, regex as alf_regex
from one.converters import ConversionMixin
import one.util as util

_logger = logging.getLogger(__name__)

N_THREADS = 4  # number of download threads


class One(ConversionMixin):
    _search_terms = (
        'dataset', 'date_range', 'laboratory', 'number', 'project', 'subject', 'task_protocol'
    )

    def __init__(self, cache_dir=None, mode='auto'):
        """An API for searching and loading data on a local filesystem

        Parameters
        ----------
        cache_dir : str, Path
            Path to the data files.  If Alyx parameters have been set up for this location,
            an OneAlyx instance is returned.  If data_dir and base_url are None, the default
            location is used.
        mode : str
            Query mode, options include 'auto' (reload cache daily), 'local' (offline) and
            'refresh' (always reload cache tables).  Most methods have a `query_type` parameter
            that can override the class mode.
        """
        # get parameters override if inputs provided
        super().__init__()
        if not getattr(self, '_cache_dir', None):  # May already be set by subclass
            self._cache_dir = cache_dir or one.params.get_cache_dir()
        self.cache_expiry = timedelta(hours=24)
        self.mode = mode
        self.regex = True  # Flag indicating whether to use regex or wildcards
        # init the cache file
        self._cache = Bunch({'_meta': {
            'expired': False,
            'created_time': None,
            'loaded_time': None,
            'raw': {}  # map of original table metadata
        }})
        self._load_cache()

    def __repr__(self):
        return f'One ({"off" if self.offline else "on"}line, {self._cache_dir})'

    @property
    def offline(self):
        return self.mode == 'local' or not getattr(self, '_web_client', False)

    @util.refresh
    def search_terms(self, query_type=None):
        return self._search_terms

    def _load_cache(self, cache_dir=None, **kwargs):
        meta = self._cache['_meta']
        INDEX_KEY = 'id'
        for cache_file in Path(cache_dir or self._cache_dir).glob('*.pqt'):
            table = cache_file.stem
            # we need to keep this part fast enough for transient objects
            cache, meta['raw'][table] = parquet.load(cache_file)
            if 'date_created' not in meta['raw'][table]:
                _logger.warning(f"{cache_file} does not appear to be a valid table. Skipping")
                continue
            created = datetime.fromisoformat(meta['raw'][table]['date_created'])
            meta['created_time'] = min([meta['created_time'] or datetime.max, created])
            meta['loaded_time'] = datetime.now()
            meta['expired'] |= datetime.now() - created > self.cache_expiry

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

        if len(self._cache) == 1:
            # No tables present
            meta['expired'] = True
            self._cache.update({'datasets': pd.DataFrame(), 'sessions': pd.DataFrame()})
        self._cache['_meta'] = meta
        return self._cache['_meta']['loaded_time']

    def refresh_cache(self, mode='auto'):
        """Check and reload cache tables

        :param mode:
        :return: Loaded time
        """
        if mode in ('local', 'remote'):  # TODO maybe rename mode
            pass
        elif mode == 'auto':
            if datetime.now() - self._cache['_meta']['loaded_time'] >= self.cache_expiry:
                _logger.info('Cache expired, refreshing')
                self._load_cache()
        elif mode == 'refresh':
            _logger.debug('Forcing reload of cache')
            self._load_cache(clobber=True)
        else:
            raise ValueError(f'Unknown refresh type "{mode}"')
        return self._cache['_meta']['loaded_time']

    def _download_datasets(self, dsets, **kwargs) -> List[Path]:
        """
        Download several datasets given a set of datasets
        :param dsets: list of dataset dictionaries from an Alyx REST query OR list of URL strings
        :return: local file path list
        """
        out_files = []
        if hasattr(dsets, 'iterrows'):
            dsets = list(map(lambda x: x[1], dsets.iterrows()))
        timeout = reduce(lambda x, y: x + y.get('file_size', 0), dsets, 0) / 625000  # 5 Mb/s
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            # TODO Subclass can just call web client method directly, no need to pass hash, etc.
            futures = [executor.submit(self._download_dataset, dset, file_size=dset['file_size'],
                                       hash=dset['hash'], **kwargs) for dset in dsets]
            concurrent.futures.wait(futures, timeout=np.ceil(timeout) + 10)
            for future in futures:
                out_files.append(future.result())
        return out_files

    def _download_dataset(self, dset, cache_dir=None, **kwargs) -> Path:
        """
        Download a dataset from an alyx REST dictionary
        :param dset: single dataset dictionary from an Alyx REST query OR URL string
        :param cache_dir (optional): root directory to save the data in (home/downloads by default)
        :return: local file path
        """
        pass  # pragma: no cover

    def search(self, details=False, exists_only=False, query_type='auto', **kwargs):
        """
        Searches sessions matching the given criteria and returns a list of matching eids

        For a list of search terms, use the methods

         one.search_terms()

        For all of the search parameters, a single value or list may be provided.  For dataset,
        the sessions returned will contain all listed datasets.  For the other parameters,
        the session must contain at least one of the entries

        :param dataset: list of dataset names. Returns sessions containing all these datasets.
        A dataset matches if it contains the search string e.g. 'wheel.position' matches
        '_ibl_wheel.position.npy'
        :param date_range: list of 2 strings or list of 2 dates that define the range (inclusive)
        :param lab: a str or list of lab names, returns sessions from any of these labs
        :param number: number of session to be returned, i.e. number in sequence for a given date
        :param subjects: a list of subjects nickname, returns sessions for any of these subjects
        :param task_protocol: task protocol name (can be partial, i.e. any task protocol
                              containing that str will be found)
        :param project: project name (can be partial, i.e. any task protocol containing
                        that str will be found)

        :param details: if true also returns a dict of dataset details
        :param query_type: Query cache ('local') or Alyx database ('remote')

        :return: list of eids, if details is True, also returns a list of dictionaries,
         each entry corresponding to a matching session
        """

        def all_present(x, dsets, exists=True):
            """Returns true if all datasets present in Series"""
            return all(any(x.str.contains(y) & exists) for y in dsets)

        # Iterate over search filters, reducing the sessions table
        sessions = self._cache['sessions']

        # Ensure sessions filtered in a particular order, with datasets last
        search_order = ('date_range', 'number', 'dataset')

        def sort_fcn(itm):
            return -1 if itm[0] not in search_order else search_order.index(itm[0])

        # Validate and get full name for queries
        search_terms = self.search_terms(query_type='local')
        queries = {util.autocomplete(k, search_terms): v for k, v in kwargs.items()}
        for key, value in sorted(queries.items(), key=sort_fcn):
            # key = util.autocomplete(key)  # Validate and get full name
            # No matches; short circuit
            if sessions.size == 0:
                return ([], None) if details else []
            # String fields
            elif key in ('subject', 'task_protocol', 'laboratory', 'project'):
                query = '|'.join(util.ensure_list(value))
                mask = sessions['lab' if key == 'laboratory' else key].str.contains(query)
                sessions = sessions[mask.astype(bool, copy=False)]
            elif key == 'date_range':
                start, end = util.validate_date_range(value)
                session_date = pd.to_datetime(sessions['date'])
                sessions = sessions[(session_date >= start) & (session_date <= end)]
            elif key == 'number':
                query = util.ensure_list(value)
                sessions = sessions[sessions[key].isin(map(int, query))]
            # Dataset check is biggest so this should be done last
            elif key == 'dataset':
                index = ['eid_0', 'eid_1'] if self._index_type('datasets') is int else 'eid'
                query = util.ensure_list(value)
                datasets = self._cache['datasets']
                if self._index_type() is int:
                    isin, _ = ismember2d(datasets[['eid_0', 'eid_1']].values,
                                         np.array(sessions.index.values.tolist()))
                else:
                    isin = datasets['eid'].isin(sessions.index.values)
                # For each session check any dataset both contains query and exists
                mask = (
                    (datasets[isin]
                        .groupby(index, sort=False)
                        .apply(lambda x: all_present(x['rel_path'], query, x['exists'])))
                )
                # eids of matching dataset records
                idx = mask[mask].index

                # Reduce sessions table by datasets mask
                sessions = sessions.loc[idx]

        # Return results
        if sessions.size == 0:
            return ([], None) if details else []
        eids = sessions.index.to_list()
        if self._index_type() is int:
            eids = parquet.np2str(np.array(eids))

        if details:
            return eids, sessions.reset_index().iloc[:, 2:].to_dict('records', Bunch)
        else:
            return eids

    def _update_filesystem(self, datasets, offline=None, update_exists=True, clobber=False):
        """Update the local filesystem for the given datasets
        Given a set of datasets, check whether records correctly reflect the filesystem.
        Called by load methods, this returns a list of file paths to load and return.
        TODO This needs changing; overlaod for downloading?
        TODO change name to check_files, check_present, present_datasets, check_local_files?
         check_filesystem?
         This changes datasets frame, calls _update_cache(sessions=None, datasets=None) to
         update and save tables.  Download_datasets can also call this function.

        :param datasets: A list or DataFrame of dataset records
        :param offline: If false and Web client present, downloads the missing datasets from a
        remote repository
        :param update_exists: If true, the cache is updated to reflect the filesystem
        :param clobber: If true and not offline, datasets are re-downloaded regardless of local
        filesystem
        :return: A list of file paths for the datasets (None elements for non-existent datasets)
        """
        if offline or self.offline:
            files = []
            if isinstance(datasets, pd.Series):
                datasets = pd.DataFrame([datasets])
            elif not isinstance(datasets, pd.DataFrame):
                # Cast set of dicts (i.e. from REST datasets endpoint)
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
                    # TODO clobber and tag mismatched
                    if hash_mismatch or size_mismatch:
                        # the local file hash doesn't match the dataset table cached hash
                        # datasets.at[i, ['hash', 'file_size']] = new_hash, new_size
                        # Raise warning if size changed or hash changed and wasn't empty
                        if size_mismatch or (hash_mismatch and rec['hash']):
                            _logger.warning('local md5 or size mismatch')
                else:
                    files.append(None)
                if rec['exists'] != file.exists():
                    datasets.at[i, 'exists'] = not rec['exists']
                    if update_exists:
                        self._cache['datasets'].loc[i, 'exists'] = rec['exists']
        else:
            # TODO deal with clobber and exists here?
            files = self._download_datasets(datasets, update_cache=update_exists, clobber=clobber)
        return files

    def _index_type(self, table='sessions'):
        idx_0 = self._cache[table].index.values[0]
        if len(self._cache[table].index.names) == 2 and all(isinstance(x, int) for x in idx_0):
            return int
        elif len(self._cache[table].index.names) == 1 and isinstance(idx_0, str):
            return str
        else:
            raise IndexError

    @util.refresh
    @util.parse_id
    def get_details(self, eid: Union[str, Path, UUID], full: bool = False):
        # Int ids return DataFrame, making str eid a list ensures Series not returned
        int_ids = self._index_type() is int
        idx = parquet.str2np(eid).tolist() if int_ids else [eid]
        try:
            det = self._cache['sessions'].loc[idx]
            assert len(det) == 1
        except KeyError:
            raise alferr.ALFObjectNotFound(eid)
        except AssertionError:
            raise alferr.ALFMultipleObjectsFound(f'Multiple sessions in cache for eid {eid}')
        if not full:
            return det.iloc[0]
        # to_drop = 'eid' if int_ids else ['eid_0', 'eid_1']
        # det = det.drop(to_drop, axis=1)
        column = ['eid_0', 'eid_1'] if int_ids else 'eid'
        return self._cache['datasets'].join(det, on=column, how='right')

    @util.refresh
    def list_subjects(self) -> List[str]:
        """
        List all subjects in database
        :return: Sorted list of subject names
        """
        return self._cache['sessions']['subject'].sort_values().unique()

    @util.refresh
    def list_datasets(self, eid=None, collection=None,
                      details=False, query_type=None) -> Union[np.ndarray, pd.DataFrame]:

        """
        Given an eid, return the datasets for those sessions.  If no eid is provided,
        a list of all datasets is returned.  When details is false, a sorted array of unique
        datasets is returned (their relative paths).

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param details: When true, a pandas DataFrame is returned, otherwise a numpy array of
        relative paths (collection/revision/filename) - see one.alf.spec.describe for details.
        :param collection: Filter by a given collection string
        :return: Slice of datasets table or numpy array if details is False
        """
        datasets = self._cache['datasets']
        if not eid:
            datasets = self._filter_by_collection(datasets, collection)
            return datasets.copy() if details else datasets['rel_path'].unique()
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if not eid:
            return datasets.iloc[0:0]  # Return empty
        if self._index_type() is int:
            eid_num = parquet.str2np(eid)
            index = ['eid_0', 'eid_1']
            isin, _ = ismember2d(datasets[index].to_numpy(), eid_num)
            datasets = datasets[isin]
        else:
            session_match = datasets['eid'] == eid
            datasets = datasets[session_match]

        datasets = self._filter_by_collection(datasets, collection)
        # Return only the relative path
        return datasets if details else datasets['rel_path'].sort_values().values

    @util.refresh
    def list_collections(self, eid=None, details=False) -> Union[np.ndarray, dict]:
        """
        List the collections for a given experiment.  If no experiment id is given,
        all collections are returned.

        Parameters
        ----------
        eid : [str, UUID, Path, dict]
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path
        details : bool
            If true a dict of pandas datasets tables is returned with collections as keys,
            otherwise a numpy array of unique collections

        Returns
        -------
            A numpy array of unique collections or dict of datasets tables
        """
        datasets = self.list_datasets(eid, details=True).copy()
        datasets['collection'] = datasets.rel_path.apply(
            lambda x: rel_path_parts(x, assert_valid=False)[0] or ''
        )
        if details:
            return {k: table.drop('collection', axis=1)
                    for k, table in datasets.groupby('collection')}
        else:
            return np.sort(datasets['collection'].unique())

    @util.refresh
    def list_revisions(self, eid=None, dataset=None, collection=None, details=False):
        """
        List the revisions for a given experiment.  If no experiment id is given,
        all collections are returned.

        Parameters
        ----------
        eid : [str, UUID, Path, dict]
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path
        details : bool
            If true a dict of pandas datasets tables is returned with collections as keys,
            otherwise a numpy array of unique collections

        Returns
        -------
            A numpy array of unique collections or dict of datasets tables
        """
        datasets = self.list_datasets(eid, collection, details=True).copy()
        if dataset:
            match = datasets.rel_path.apply(lambda x: x.split('/')[-1]).str.match(dataset)
            datasets = datasets[match]
        datasets['revision'] = datasets.rel_path.apply(
            lambda x: (rel_path_parts(x, assert_valid=False)[1] or '').strip('#')
        )
        if details:
            return {k: table.drop('revision', axis=1)
                    for k, table in datasets.groupby('revision')}
        else:
            return np.sort(datasets['revision'].unique())

    @staticmethod
    def _filter_by_collection(datasets: pd.DataFrame, collection: str) -> pd.DataFrame:
        """
        Return a pandas datasets table containing records that match a given collection
        :param datasets: A datasets cache table
        :param collection: The collection name to match
        :return: A slice of the input table, where all datasets match a given collection
        """
        if collection is None:
            return datasets
        expression = alf_regex(f'^{COLLECTION_SPEC}$', collection=collection)
        table = datasets['rel_path'].str.rsplit('/', 1, expand=True)
        if table.columns.stop == 1:
            match = np.ones(len(datasets)) == (collection == '')
        else:
            # Check collection matches
            table = (table[0] + '/').str.extract(expression)
            match = ~table['collection'].isna()
        return datasets[match]

    @util.refresh
    @util.parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = None,
                    revision: Optional[str] = None,
                    query_type: str = 'auto',
                    download_only: bool = False,
                    **kwargs) -> Union[alfio.AlfBunch, List[Path]]:
        """
        TODO Add more examples;
        Load all attributes of an ALF object from a Session ID and an object name.  Any datasets
        with matching object name will be loaded.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param obj: The ALF object to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        Supports asterisks as wildcards.
        :param revision: The last revision before a given string (typically an ISO date).  If
        None, the default dataset is returned (usually the most recent revision).
        :param query_type: Query cache ('local') or Alyx database ('remote')
        :param download_only: When true the data are downloaded and the file paths are returned
        :param kwargs: Additional filters for datasets, including namespace and timescale
        :return: An ALF bunch or if download_only is True, a list of Paths objects

        Examples:
            load_object(eid, 'moves')
            load_object(eid, 'trials')
            load_object(eid, 'spikes', collection='.*probe01')
            load_object(eid, 'spikes', namespace='ibl')
            load_object(eid, 'spikes', timescale='ephysClock')
        """
        datasets = self.list_datasets(eid, details=True, query_type=query_type)

        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(obj)

        REGEX = True
        if not REGEX:
            import fnmatch
            obj = re.compile(fnmatch.translate(obj))

        # expression = alf_regex(f'{COLLECTION_SPEC}/{FILE_SPEC}',
        #                        object=obj, collection=collection, revision=revision)
        # table = datasets['rel_path'].str.extract(expression)
        # match = ~table[['collection', 'object', 'revision']].isna().all(axis=1)

        dataset = {'object': obj, **kwargs}
        datasets = util.filter_datasets(datasets, dataset, collection, revision,
                                        assert_unique=False)

        # Validate result before loading
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(obj)
        parts = [rel_path_parts(x) for x in datasets.rel_path]
        unique_objects = set(x[3] or '' for x in parts)
        unique_collections = set(x[0] or '' for x in parts)
        if len(unique_objects) > 1:
            raise alferr.ALFMultipleObjectsFound('"' + '", "'.join(unique_objects) + '"')
        if len(unique_collections) > 1:
            raise alferr.ALFMultipleCollectionsFound('"' + '", "'.join(unique_collections) + '"')

        # parquet.np2str(np.array(datasets.index.values.tolist()))
        # For those that don't exist, download them
        # return alfio.load_object(path, table[match]['object'].values[0])
        offline = None if query_type == 'auto' else self.mode == 'local'
        files = self._update_filesystem(datasets, offline=offline)
        files = [x for x in files if x]
        if not files:
            raise alferr.ALFObjectNotFound(f'ALF object "{obj}" not found on disk')

        if download_only:
            return files

        return alfio.load_object(files, **kwargs)

    @util.refresh
    @util.parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = None,
                     revision: Optional[str] = None,
                     query_type: str = 'auto',
                     download_only: bool = False,
                     **kwargs) -> Any:
        """
        Load a dataset for a given session id and dataset name

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path.
        :param dataset: The ALF dataset to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        For IBL this is the relative path of the file from the session root.
        Supports asterisks as wildcards.
        :param revision: The last revision before a given string (typically an ISO date).  If
        None, the default dataset is returned (usually the most recent revision).
        :param query_type: Query cache ('local') or Alyx database ('remote')
        :param download_only: When true the data are downloaded and the file paths are returned
        :param kwargs:
        :return:
        """
        datasets = self.list_datasets(eid, details=True, query_type=query_type)

        datasets = util.filter_datasets(datasets, dataset, collection, revision)
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(f'Dataset "{dataset}" not found')

        # Check files exist / download remote files
        file, = self._update_filesystem(datasets, **kwargs)

        if not file:
            raise alferr.ALFObjectNotFound('Dataset not found')
        elif download_only:
            return file
        return alfio.load_file_content(file)

    @util.refresh
    @util.parse_id
    def load_datasets(self,
                      eid: Union[str, Path, UUID],
                      datasets: List[str],
                      collections: Optional[str] = None,
                      revisions: Optional[str] = None,
                      query_type: str = 'auto',
                      assert_present=True,
                      download_only: bool = False,
                      **kwargs) -> Any:
        """
        Load datasets for a given session id.  Returns two lists the length of datasets.  The
        first is the data (or file paths if download_data is false), the second is a list of
        meta data Bunches.  If assert_present is false, missing data will be returned as None.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path.
        :param datasets: The ALF datasets to load.  Supports asterisks as wildcards.
        :param collections:  The collection(s) to which the object(s) belong, e.g. 'alf/probe01'.
        For IBL this is the relative path of the file from the session root.
        Supports asterisks as wildcards.
        :param revisions: The last revision before a given string (typically an ISO date).  If
        None, the default dataset is returned (usually the most recent revision).
        :param query_type: Query cache ('local') or Alyx database ('remote')
        :param assert_present: If true, missing datasets raises and error, otherwise None is
        returned
        :param download_only: When true the data are downloaded and the file paths are returned
        :param kwargs:

        :return: Returns a list of data (or file paths) the length of datasets, and a list of
        meta data Bunches. If assert_present is False, missing data will be None
        """

        def _verify_specifiers(specifiers):
            """Ensure specifiers lists matching datasets length"""
            out = []
            for spec in specifiers:
                if not spec or isinstance(spec, str):
                    out.append([spec] * len(datasets))
                elif len(spec) != len(datasets):
                    raise ValueError(
                        'Collection and revision specifiers must match number of datasets')
                else:
                    out.append(spec)
            return out

        if isinstance(datasets, str):
            raise TypeError('`datasets` must be a non-string iterable')
        # Check input args
        collections, revisions = _verify_specifiers([collections, revisions])

        # Short circuit
        all_datasets = self.list_datasets(eid, details=True, query_type=query_type)
        if len(all_datasets) == 0:
            if assert_present:
                raise alferr.ALFObjectNotFound(f'No datasets found for session {eid}')
            else:
                _logger.warning(f'No datasets found for session {eid}')
                return None, all_datasets
        if len(datasets) == 0:
            return None, all_datasets.iloc[0:0]  # Return empty

        # Filter and load missing
        slices = [util.filter_datasets(all_datasets, x, y, z)
                  for x, y, z in zip(datasets, collections, revisions)]
        present = [len(x) == 1 for x in slices]
        present_datasets = pd.concat(slices)

        if not all(present):
            missing_list = ', '.join(x for x, y in zip(datasets, present) if not y)
            # FIXME include collection and revision also
            message = f'The following datasets are not in the cache: {missing_list}'
            if assert_present:
                raise alferr.ALFObjectNotFound(message)
            else:
                _logger.warning(message)

        # Check files exist / download remote files
        files = self._update_filesystem(present_datasets, **kwargs)

        if any(x is None for x in files):
            missing_list = ', '.join(x for x, y in zip(present_datasets.rel_path, files) if not y)
            message = f'The following datasets were not downloaded: {missing_list}'
            raise alferr.ALFObjectNotFound(message) if assert_present else _logger.warning(message)

        # Make list of metadata Bunches out of the table
        records = (present_datasets
                   .reset_index()
                   .drop(['eid_0', 'eid_1'], axis=1)
                   .to_dict('records', Bunch))

        # Ensure result same length as input datasets list
        files = [None if not here else files.pop(0) for here in present]
        records = [None if not here else records.pop(0) for here in files]
        if download_only:
            return files, records
        return [alfio.load_file_content(x) for x in files], records

    @util.refresh
    def load_dataset_from_id(self,
                             dset_id: Union[str, UUID],
                             download_only: bool = False,
                             details: bool = False,
                             **kwargs) -> Any:
        int_idx = self._index_type('datasets') is int
        if isinstance(dset_id, str) and int_idx:
            dset_id = parquet.str2np(dset_id)
        elif isinstance(dset_id, UUID):
            dset_id = parquet.uuid2np([dset_id]) if int_idx else str(dset_id)
        try:
            if int_idx:
                dataset = self._cache['datasets'].loc[dset_id.tolist()].iloc[0]
            else:
                dataset = self._cache['datasets'].loc[[dset_id]]
            assert isinstance(dataset, pd.Series) or len(dataset) == 1
        except KeyError:
            raise alferr.ALFObjectNotFound('Dataset not found')
        except AssertionError:
            raise alferr.ALFMultipleObjectsFound('Duplicate dataset IDs')

        filepath, = self._update_filesystem(dataset)
        if not filepath:
            raise alferr.ALFObjectNotFound('Dataset not found')
        output = filepath if download_only else alfio.load_file_content(filepath)
        if details:
            return output, dataset
        else:
            return output

    @staticmethod
    def setup(cache_dir, **kwargs):
        """
        Interactive command tool that populates parameter file for ONE IBL.
        FIXME See subclass
        """
        make_parquet_db(cache_dir, **kwargs)
        return One(cache_dir, mode='local')


@lru_cache(maxsize=1)
def ONE(*, mode='auto', **kwargs):
    """ONE API factory
    Determine which class to instantiate depending on parameters passed.

    Parameters
    ----------
    mode : str
        Query mode, options include 'auto', 'local' (offline) and 'remote' (online only).  Most
        methods have a `query_type` parameter that can override the class mode.
    cache_dir : str, Path
        Path to the data files.  If Alyx parameters have been set up for this location,
        an OneAlyx instance is returned.  If data_dir and base_url are None, the default
        location is used.
    base_url : str
        An Alyx database URL.  The URL must start with 'http'.
    username : str
        An Alyx database login username.
    password : str
        An Alyx database password.
    cache_rest : str
        If not in 'local' mode, this determines which http request types to cache.  Default is
        'GET'.  Use None to deactivate cache (not recommended).

    Returns
    -------
        An One instance if mode is 'local', otherwise an OneAlyx instance.
    """
    if kwargs.pop('offline', False):
        _logger.warning('the offline kwarg will probably be removed. '
                        'ONE is now offline by default anyway')
        warnings.warn('"offline" param will be removed; use mode="local"', DeprecationWarning)
        mode = 'local'

    if (any(x in kwargs for x in ('base_url', 'username', 'password')) or
            not kwargs.get('cache_dir', False)):
        return OneAlyx(mode=mode, **kwargs)

    # TODO This feels hacky
    # If cache dir was provided and corresponds to one configured with an Alyx client, use OneAlyx
    try:
        one.params._check_cache_conflict(kwargs.get('cache_dir'))
        return One(mode=mode, **kwargs)
    except AssertionError:
        # Cache dir corresponds to a Alyx repo, call OneAlyx
        return OneAlyx(mode=mode, **kwargs)


class OneAlyx(One):
    def __init__(self, username=None, password=None, base_url=None, cache_dir=None,
                 mode='auto', **kwargs):
        """An API for searching and loading data through the Alyx database

        Parameters
        ----------
        mode : str
            Query mode, options include 'auto', 'local' (offline) and 'remote' (online only).  Most
            methods have a `query_type` parameter that can override the class mode.
        cache_dir : str, Path
            Path to the data files.  If Alyx parameters have been set up for this location,
            an OneAlyx instance is returned.  If data_dir and base_url are None, the default
            location is used.
        base_url : str
            An Alyx database URL.  The URL must start with 'http'.
        username : str
            An Alyx database login username.
        password : str
            An Alyx database password.
        cache_rest : str
            If not in 'local' mode, this determines which http request types to cache.  Default is
            'GET'.  Use None to deactivate cache (not recommended).
        """

        # Load Alyx Web client
        self._web_client = wc.AlyxClient(username=username,
                                         password=password,
                                         base_url=base_url,
                                         cache_dir=cache_dir,
                                         **kwargs)
        self._search_endpoint = 'sessions'
        # get parameters override if inputs provided
        super(OneAlyx, self).__init__(mode=mode, cache_dir=cache_dir)

    def __repr__(self):
        return f'One ({"off" if self.offline else "on"}line, {self.alyx.base_url})'

    def _load_cache(self, cache_dir=None, clobber=False):
        cache_meta = self._cache['_meta']
        if not clobber:
            super(OneAlyx, self)._load_cache(self._cache_dir)  # Load any present cache
            if (self._cache and not cache_meta['expired']) or self.mode == 'local':
                return

        # Warn user if expired
        if (
            cache_meta['expired'] and
            cache_meta.get('created_time', False) and
            not self.alyx.silent
        ):
            age = datetime.now() - cache_meta['created_time']
            t_str = (f'{age.days} days(s)'
                     if age.days >= 1
                     else f'{np.floor(age.seconds / (60 * 2))} hour(s)')
            _logger.info(f'cache over {t_str} old')

        try:
            # Determine whether a newer cache is available
            cache_info = self.alyx.get('cache/info', expires=True)
            remote_created = datetime.fromisoformat(cache_info['date_created'])
            local_created = cache_meta.get('created_time', None)
            if local_created and (remote_created - local_created) < timedelta(minutes=1):
                _logger.info('No newer cache available')
                return

            # Download the remote cache files
            _logger.info('Downloading remote caches...')
            files = self.alyx.download_cache_tables()
            assert any(files)
            super(OneAlyx, self)._load_cache(self._cache_dir)  # Reload cache after download
        except requests.exceptions.HTTPError:
            _logger.error('Failed to load the remote cache file')
            self.mode = 'remote'
        except (ConnectionError, requests.exceptions.ConnectionError):
            _logger.error('Failed to connect to Alyx')
            self.mode = 'local'

    @property
    def alyx(self):
        return self._web_client

    @property
    def _cache_dir(self):
        return self._web_client.cache_dir

    @util.refresh
    def search_terms(self, query_type=None):
        """
        Returns a list of search terms to be passed as kwargs to the search method

        Parameters
        ----------
        query_type : str
            If 'remote', the search terms are largely determined by the REST endpoint used

        Returns
        -------
            Tuple of search strings
        """
        if (query_type or self.mode) != 'remote':
            return self._search_terms
        # Return search terms from REST schema
        fields = self.alyx.rest_schemes[self._search_endpoint]['list']['fields']
        excl = ('lab',)  # 'laboratory' already in search terms
        return tuple({*self._search_terms, *(x['name'] for x in fields if x['name'] not in excl)})

    def describe_dataset(self, dataset_type=None):
        # TODO Move to AlyxClient?; add to rest examples; rename to describe?
        if not dataset_type:
            return self.alyx.rest('dataset-types', 'list')
        try:
            assert isinstance(dataset_type, str) and not alfio.is_uuid_string(dataset_type)
            _logger.disabled = True
            out = self.alyx.rest('dataset-types', 'read', dataset_type)
        except (AssertionError, requests.exceptions.HTTPError):
            # Try to get dataset type from dataset name
            out = self.alyx.rest('dataset-types', 'read', self.dataset2type(dataset_type))
        finally:
            _logger.disabled = False
        print(out['description'])
        return out

    @util.refresh
    def list_datasets(self, eid=None, collection=None,
                      details=False, query_type=None) -> Union[np.ndarray, pd.DataFrame]:
        if (query_type or self.mode) != 'remote':
            return super().list_datasets(collection=collection, details=details,
                                         query_type=query_type)
        elif not eid:
            warnings.warn('Unable to list all remote datasets')
            return super().list_datasets(collection=collection, details=details,
                                         query_type=query_type)
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if not eid:
            return self._cache['datasets'].iloc[0:0]  # Return empty
        _, datasets = util.ses2records(self.alyx.rest('sessions', 'read', id=eid))
        # Return only the relative path
        return datasets if details else datasets['rel_path'].sort_values().values

    @util.refresh
    @util.parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: str = None,
                     revision: str = None,
                     query_type: str = None,
                     download_only: bool = False) -> Any:
        """
        Load a single dataset from a Session ID and a dataset type.

        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path.
        :param dataset: The ALF dataset to load.  Supports asterisks as wildcards.
        :param collection:  The collection to which the object belongs, e.g. 'alf/probe01'.
        For IBL this is the relative path of the file from the session root.
        Supports asterisks as wildcards.
        :param revision: The last revision before a given string (typically an ISO date).  If
        None, the default dataset is returned (usually the most recent revision).
        :param query_type: Query cache ('local') or Alyx database ('remote')
        :param download_only: When true the data are downloaded and the file path is returned.
        :return: dataset or a Path object if download_only is true.

        TODO This method may be removed once default dataset is added to Alyx serializer

        Examples:
            intervals = one.load_dataset(eid, '_ibl_trials.intervals.npy')
            intervals = one.load_dataset(eid, '*trials.intervals*')
            filepath = one.load_dataset(eid '_ibl_trials.intervals.npy', download_only=True)
            spike_times = one.load_dataset(eid 'spikes.times.npy', collection='alf/probe01')
            old_spikes = one.load_dataset(eid, ''spikes.times.npy',
                                          collection='alf/probe01', revision='2020-08-31')
        """
        query_type = query_type or self.mode
        if query_type != 'remote':
            load_dataset_offline = unwrap(super().load_dataset)  # Skip parse_id decorator
            return load_dataset_offline(self, eid, dataset,
                                        collection=collection,
                                        revision=revision,
                                        download_only=download_only,
                                        query_type=query_type)
        search_str = 'name__regex,' + dataset.replace('.', r'\.').replace('*', '.*')
        if collection:
            search_str += ',collection__regex,' + collection.replace('*', '.*')
        results = self.alyx.rest('datasets', 'list', session=eid, django=search_str, exists=True)

        # Get filenames of returned ALF files
        collection_set = {x['collection'] for x in results}
        if len(collection_set) > 1:
            raise alferr.ALFMultipleCollectionsFound(
                'Matching dataset belongs to multiple collections:' + ', '.join(collection_set))
        dataset_set = {x['name'] for x in results}
        if len(dataset_set) > 1:
            raise alferr.ALFMultipleObjectsFound('The following matching datasets were found: ' +
                                                 ', '.join(x['name'] for x in results))
        # Deal with revisions
        if len(results) > 1:
            if revision is None:
                # Take default dataset
                results = [x for x in results if x['default_dataset']]
                assert len(results) == 1, 'Number of default revisions != 1'
            else:
                idx = util.index_last_before([x['revision'] or '' for x in results], revision)
                results = [] if idx is None else [results[idx]]
        if len(results) == 0:
            raise alferr.ALFObjectNotFound(f'Dataset "{dataset}" not found on Alyx')

        filename = self._download_dataset(results[0])
        assert filename is not None, 'failed to download dataset'
        return filename if download_only else alfio.load_file_content(filename)

    @util.refresh
    def load_collection(self, eid, collection):
        raise NotImplementedError()

    @util.refresh
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
            raise NotImplementedError('Converting probe IDs required remote connection')
        rec = self.alyx.rest('insertions', 'read', id=pid)
        return rec['session'], rec['name']

    def search(self, details=False, query_type=None, **kwargs):
        """
        Searches sessions matching the given criteria and returns a list of matching eids

        For a list of remote database search terms, use the method

         one.search_terms(query_type='remote')

        For all of the search parameters, a single value or list may be provided.  For dataset,
        the sessions returned will contain all listed datasets.  For the other parameters,
        the session must contain at least one of the entries

        :param dataset: list of dataset names. Returns sessions containing all these datasets.
        A dataset matches if it contains the search string e.g. 'wheel.position' matches
        '_ibl_wheel.position.npy'
        :param date_range: list of 2 strings or list of 2 dates that define the range (inclusive)
        :param lab: a str or list of lab names, returns sessions from any of these labs
        :param number: number of session to be returned, i.e. number in sequence for a given date
        :param subjects: a list of subjects nickname, returns sessions for any of these subjects
        :param task_protocol: task protocol name (can be partial, i.e. any task protocol
                              containing that str will be found)
        :param project: project name (can be partial, i.e. any task protocol containing
                        that str will be found)
        :param performance_lte / performance_gte: search only for sessions whose performance is
                less equal or greater equal than a pre-defined threshold as a percentage (0-100)
        :param users: a list of users
        :param location: a str or list of lab location (as per Alyx definition) name
                         Note: this corresponds to the specific rig, not the lab geographical
                         location per se
        :param dataset_types: list of dataset_types
        :param limit: the number of results to fetch in one go (if pagination enabled on server)

        :param details: if true also returns a dict of dataset details
        :param query_type: Query cache ('local') or Alyx database ('remote')

        :return: list of eids and, if details is True, also returns a list of dictionaries,
         each entry corresponding to a matching session
        """
        query_type = query_type or self.mode
        if query_type != 'remote':
            return super(OneAlyx, self).search(details=details, query_type=query_type, **kwargs)

        # loop over input arguments and build the url
        search_terms = self.search_terms(query_type=query_type)
        params = {'django': kwargs.pop('django', '')}
        for key, value in sorted(kwargs.items()):
            field = util.autocomplete(key, search_terms)  # Validate and get full name
            # check that the input matches one of the defined filters
            if field == 'date_range':
                params[field] = [x.date().isoformat() for x in util.validate_date_range(value)]
            elif field == 'dataset':
                query = ('data_dataset_session_related__dataset_type__name__icontains,' +
                         ','.join(util.ensure_list(value)))
                params['django'] += (',' if params['django'] else '') + query
            elif field == 'laboratory':
                params['lab'] = value
            else:
                params[field] = value

        # Make GET request
        ses = self.alyx.rest(self._search_endpoint, 'list', **params)
        # LazyId only transforms records when indexed
        eids = util.LazyId(ses)
        return (eids, ses) if details else eids

    def _download_dataset(self, dset, cache_dir=None, update_cache=True, **kwargs):
        """
        Download a dataset from an alyx REST dictionary
        :param dset: single dataset dictionary from an Alyx REST query OR URL string
        :param cache_dir: root directory to save the data in (home/downloads by default)
        :param update_cache: if true, the cache is updated when filesystem discrepancies are
        encountered
        :return: local file path
        """
        if isinstance(dset, str) and dset.startswith('http'):
            url = dset
        elif isinstance(dset, (str, Path)):
            url = self.path2url(dset)
            if not url:
                _logger.warning('Dataset not found in cache')
                return
        else:
            if 'data_url' in dset:  # data_dataset_session_related dict
                url = dset['data_url']
                did = dset['id']
            elif 'file_records' not in dset:  # Convert dataset Series to alyx dataset dict
                url = self.record2url(dset)
                did = dset.index
            else:  # from datasets endpoint
                url = next((fr['data_url'] for fr in dset['file_records']
                            if fr['data_url'] and fr['exists']), None)
                did = dset['url'][-36:]

        if not url:
            _logger.warning("Dataset not found")
            if update_cache:
                if isinstance(did, str) and self._index_type('datasets') is int:
                    did = parquet.str2np(did)
                elif self._index_type('datasets') is str and not isinstance(did, str):
                    did = parquet.np2str(did)
                self._cache['datasets'].at[did, 'exists'] = False
            return
        target_dir = Path(cache_dir or self._cache_dir, alfio.get_alf_path(url)).parent
        return self._download_file(url=url, target_dir=target_dir, **kwargs)

    def _tag_mismatched_file_record(self, url):
        fr = self.alyx.rest('files', 'list',
                            django=f'dataset,{Path(url).name.split(".")[-2]},'
                                   f'data_repository__globus_is_personal,False',
                            no_cache=True)
        if len(fr) > 0:
            json_field = fr[0]['json']
            if json_field is None:
                json_field = {'mismatch_hash': True}
            else:
                json_field.update({'mismatch_hash': True})
            self.alyx.rest('files', 'partial_update',
                           id=fr[0]['url'][-36:], data={'json': json_field})

    def _download_file(self, url, target_dir,
                       clobber=False, offline=None, keep_uuid=False, file_size=None, hash=None):
        """
        Downloads a single file from an HTTP webserver.  The webserver in question is set by the
        AlyxClient object.
        :param url: an absolute or relative URL for a remote dataset
        :param clobber: (bool: False) overwrites local dataset if any
        :param offline:
        :param keep_uuid:
        :param file_size:
        :param hash:
        :return:
        """
        if offline is None:
            offline = self.mode == 'local'
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        local_path = target_dir / os.path.basename(url)
        if not keep_uuid:
            local_path = alfio.remove_uuid_file(local_path, dry=True)
        if Path(local_path).exists():
            # the local file hash doesn't match the dataset table cached hash
            hash_mismatch = hash and hashfile.md5(Path(local_path)) != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if (hash_mismatch or file_size_mismatch) and not offline:
                clobber = True
                if not self.alyx.silent:
                    _logger.warning(f'local md5 or size mismatch, re-downloading {local_path}')
        # if there is no cached file, download
        else:
            clobber = True
        if clobber and not offline:
            local_path, md5 = self.alyx.download_file(
                url, cache_dir=str(target_dir), clobber=clobber, return_md5=True)
            # TODO If 404 update JSON on Alyx for data record
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
            print('Building ONE cache from filesystem...')
            from one.alf import onepqt
            onepqt.make_parquet_db(root_dir, **kwargs)
            return One(cache_dir=root_dir)

    @util.refresh
    @util.parse_id
    def eid2path(self, eid: str, query_type=None) -> util.Listable(Path):
        """
        From an experiment id gets the local cache path
        :param eid: Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
        :param query_type: if set to 'remote', will force database connection
        :return: eid or list of eids
        """
        # first try avoid hitting the database
        mode = query_type or self.mode
        if mode != 'remote':
            cache_path = super().eid2path(eid)
            if cache_path or mode == 'local':
                return cache_path

        # if it wasn't successful, query Alyx
        ses = self.alyx.rest('sessions', 'list', django=f'pk,{eid}')
        if len(ses) == 0:
            return None
        else:
            return Path(self._cache_dir).joinpath(
                ses[0]['lab'], 'Subjects', ses[0]['subject'], ses[0]['start_time'][:10],
                str(ses[0]['number']).zfill(3))

    @util.refresh
    def path2eid(self, path_obj: Union[str, Path], query_type=None) -> util.Listable(Path):
        """
        From a local path, gets the experiment id
        :param path_obj: local path or list of local paths
        :param query_type: if set to 'remote', will force database connection
        :return: eid or list of eids
        """
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            path_obj = [Path(x) for x in path_obj]
            eid_list = []
            unwrapped = unwrap(self.path2eid)
            for p in path_obj:
                eid_list.append(unwrapped(self, p))
            return eid_list
        # else ensure the path ends with mouse,date, number
        path_obj = Path(path_obj)

        # try the cached info to possibly avoid hitting database
        mode = query_type or self.mode
        if mode != 'remote':
            cache_eid = super().path2eid(path_obj)
            if cache_eid or mode == 'local':
                return cache_eid

        session_path = alfio.get_session_path(path_obj)
        # if path does not have a date and a number return None
        if session_path is None:
            return None

        # if not search for subj, date, number XXX: hits the DB
        search = unwrap(self.search)
        uuid = search(subject=session_path.parts[-3],
                      date_range=session_path.parts[-2],
                      number=session_path.parts[-1],
                      query_type='remote')

        # Return the uuid if any
        return uuid[0] if uuid else None

    @util.refresh
    def path2url(self, filepath, query_type='auto'):
        """
        Given a local file path, returns the URL of the remote file.
        :param filepath: A local file path
        :param query_type: if set to 'remote', will force database connection
        :return: A URL string
        """
        if query_type != 'remote':
            return super(OneAlyx, self).path2url(filepath)
        eid = self.path2eid(filepath)
        try:
            dataset, = self.alyx.rest('datasets', 'list', session=eid, name=Path(filepath).name)
            return next(
                r['data_url'] for r in dataset['file_records'] if r['data_url'] and r['exists'])
        except (ValueError, StopIteration):
            raise alferr.ALFObjectNotFound(f'File record for {filepath} not found on Alyx')

    @util.parse_id
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

    def dataset2type(self, dset):
        """Return dataset type from dataset"""
        # Ensure dset is a str uuid
        if isinstance(dset, str) and not alfio.is_uuid_string(dset):
            dset = self._dataset_name2id(dset)
        if isinstance(dset, np.ndarray):
            dset = parquet.np2str(dset)[0]
        if isinstance(dset, tuple) and all(isinstance(x, int) for x in dset):
            dset = parquet.np2str(np.array(dset))
        if not alfio.is_uuid_string(dset):
            raise ValueError('Unrecognized name or UUID')
        return self.alyx.rest('datasets', 'read', id=dset)['dataset_type']

    def describe_revision(self, revision):
        raise NotImplementedError('Requires changes to revisions endpoint')
        rec = self.alyx.rest('revisions', 'list', name=revision)  # py 3.8
        # if rec := self.alyx.rest('revisions', 'list', name=revision):  # py 3.8
        if rec:
            print(rec[0]['description'])
        else:
            print(f'Revision "{revision}" not found')

    def _dataset_name2id(self, dset_name, eid=None):
        # TODO finish function
        datasets = self.list_datasets(eid) if eid else self._cache['datasets']
        # Get ID of fist matching dset
        for idx, rel_path in datasets['rel_path'].items():
            if dset_name in rel_path:
                return idx

    @util.refresh
    @util.parse_id
    def get_details(self, eid: str, full: bool = False, query_type=None):
        """ Returns details of eid like from one.search, optional return full
        session details.
        """
        if (query_type or self.mode) == 'local':
            return super().get_details(eid, full=full)
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
        out.update({'local_path': self.eid2path(eid)})
        return out

    # def _update_cache(self, ses, dataset_types):
    #     """
    #     TODO move to One; currently unused
    #     :param ses: session details dictionary as per Alyx response
    #     :param dataset_types:
    #     :return: is_updated (bool): if the cache was updated or not
    #     """
    #     save = False
    #     pqt_dsets = _ses2pandas(ses, dtypes=dataset_types)
    #     # if the dataframe is empty, return
    #     if pqt_dsets.size == 0:
    #         return
    #     # if the cache is empty create the cache variable
    #     elif self._cache.size == 0:
    #         self._cache = pqt_dsets
    #         save = True
    #     # the cache is not empty and there are datasets in the query
    #     else:
    #         isin, icache = ismember2d(pqt_dsets[['id_0', 'id_1']].to_numpy(),
    #                                   self._cache[['id_0', 'id_1']].to_numpy())
    #         # check if the hash / filesize fields have changed on patching
    #         heq = (self._cache['hash'].iloc[icache].to_numpy() ==
    #                pqt_dsets['hash'].iloc[isin].to_numpy())
    #         feq = np.isclose(self._cache['file_size'].iloc[icache].to_numpy(),
    #                          pqt_dsets['file_size'].iloc[isin].to_numpy(),
    #                          rtol=0, atol=0, equal_nan=True)
    #         eq = np.logical_and(heq, feq)
    #         # update new hash / filesizes
    #         if not np.all(eq):
    #             self._cache.iloc[icache, 4:6] = pqt_dsets.iloc[np.where(isin)[0], 4:6].to_numpy()
    #             save = True
    #         # append datasets that haven't been found
    #         if not np.all(isin):
    #             self._cache = self._cache.append(pqt_dsets.iloc[np.where(~isin)[0]])
    #             self._cache = self._cache.reindex()
    #             save = True
    #     if save:
    #         # before saving makes sure pandas did not cast uuids in float
    #         typs = [t for t, k in zip(self._cache.dtypes, self._cache.keys()) if 'id_' in k]
    #         assert (all(map(lambda t: t == np.int64, typs)))
    #         # if this gets too big, look into saving only when destroying the ONE object
    #         parquet.save(self._cache_file, self._cache)
