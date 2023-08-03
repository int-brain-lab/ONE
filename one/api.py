"""Classes for searching, listing and (down)loading ALyx Files."""
import collections.abc
import urllib.parse
import warnings
import logging
from datetime import datetime, timedelta
from functools import lru_cache, partial
from inspect import unwrap
from pathlib import Path, PurePosixPath
from typing import Any, Union, Optional, List
from uuid import UUID
from urllib.error import URLError
import time
import threading

import pandas as pd
import numpy as np
import requests.exceptions
import packaging.version

from iblutil.io import parquet, hashfile
from iblutil.util import Bunch, flatten

import one.params
import one.webclient as wc
import one.alf.io as alfio
import one.alf.files as alfiles
import one.alf.exceptions as alferr
from .alf.cache import make_parquet_db
from .alf.spec import is_uuid_string
from . import __version__
from one.converters import ConversionMixin, session_record2path
import one.util as util

_logger = logging.getLogger(__name__)
__all__ = ['ONE', 'One', 'OneAlyx']
"""int: The number of download threads"""
N_THREADS = 4


class One(ConversionMixin):
    """An API for searching and loading data on a local filesystem"""
    _search_terms = (
        'dataset', 'date_range', 'laboratory', 'number', 'projects', 'subject', 'task_protocol'
    )

    def __init__(self, cache_dir=None, mode='auto', wildcards=True, tables_dir=None):
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
        wildcards : bool
            If true, use unix shell style matching instead of regular expressions.
        tables_dir : str, pathlib.Path
            An optional location of the cache tables.  If None, the tables are assumed to be in the
            cache_dir.
        """
        # get parameters override if inputs provided
        super().__init__()
        if not getattr(self, 'cache_dir', None):  # May already be set by subclass
            self.cache_dir = cache_dir or one.params.get_cache_dir()
        self._tables_dir = tables_dir or self.cache_dir
        self.cache_expiry = timedelta(hours=24)
        self.mode = mode
        self.wildcards = wildcards  # Flag indicating whether to use regex or wildcards
        self.record_loaded = False
        # init the cache file
        self._reset_cache()
        self.load_cache()

    def __repr__(self):
        return f'One ({"off" if self.offline else "on"}line, {self.cache_dir})'

    @property
    def offline(self):
        """bool: True if mode is local or no Web client set"""
        return self.mode == 'local' or not getattr(self, '_web_client', False)

    @util.refresh
    def search_terms(self, query_type=None) -> tuple:
        """List the search term keyword args for use in the search method"""
        return self._search_terms

    def _reset_cache(self):
        """Replace the cache object with a Bunch that contains the right fields"""
        self._cache = Bunch({'_meta': {
            'expired': False,
            'created_time': None,
            'loaded_time': None,
            'modified_time': None,
            'saved_time': None,
            'raw': {}  # map of original table metadata
        }})

    def load_cache(self, tables_dir=None, **kwargs):
        """
        Load parquet cache files from a local directory.

        Parameters
        ----------
        tables_dir : str, pathlib.Path
            An optional directory location of the parquet files, defaults to One._tables_dir.
        """
        self._reset_cache()
        meta = self._cache['_meta']
        INDEX_KEY = '.?id'
        self._tables_dir = Path(tables_dir or self._tables_dir or self.cache_dir)
        for cache_file in self._tables_dir.glob('*.pqt'):
            table = cache_file.stem
            # we need to keep this part fast enough for transient objects
            cache, meta['raw'][table] = parquet.load(cache_file)
            if 'date_created' not in meta['raw'][table]:
                _logger.warning(f"{cache_file} does not appear to be a valid table. Skipping")
                continue
            meta['loaded_time'] = datetime.now()

            # Set the appropriate index if none already set
            if isinstance(cache.index, pd.RangeIndex):
                idx_columns = cache.filter(regex=INDEX_KEY).columns.tolist()
                if len(idx_columns) == 0:
                    raise KeyError('Failed to set index')
                cache.set_index(idx_columns, inplace=True)

            # Patch older tables
            cache = util.patch_cache(cache, meta['raw'][table].get('min_api_version'))

            # Check sorted
            # Sorting makes MultiIndex indexing O(N) -> O(1)
            if not cache.index.is_monotonic_increasing:
                cache.sort_index(inplace=True)

            self._cache[table] = cache

        if len(self._cache) == 1:
            # No tables present
            meta['expired'] = True
            meta['raw'] = {}
            self._cache.update({'datasets': pd.DataFrame(), 'sessions': pd.DataFrame()})
            if self.offline:  # In online mode, the cache tables should be downloaded later
                warnings.warn(f'No cache tables found in {self._tables_dir}')
        created = [datetime.fromisoformat(x['date_created'])
                   for x in meta['raw'].values() if 'date_created' in x]
        if created:
            meta['created_time'] = min(created)
            meta['expired'] |= datetime.now() - meta['created_time'] > self.cache_expiry
        self._cache['_meta'] = meta
        return self._cache['_meta']['loaded_time']

    def save_cache(self, save_dir=None, force=False):
        """Save One._cache attribute into parquet tables if recently modified.

        Parameters
        ----------
        save_dir : str, pathlib.Path
            The directory path into which the tables are saved.  Defaults to cache directory.
        force : bool
            If True, the cache is saved regardless of modification time.
        """
        threading.Thread(target=lambda: self._save_cache(save_dir=save_dir, force=force)).start()

    def _save_cache(self, save_dir=None, force=False):
        """
        Checks if another process is writing to file, if so waits before saving.

        Parameters
        ----------
        save_dir : str, pathlib.Path
            The directory path into which the tables are saved.  Defaults to cache directory.
        force : bool
            If True, the cache is saved regardless of modification time.
        """
        TIMEOUT = 30  # Delete lock file this many seconds after creation/modification or waiting
        lock_file = Path(self.cache_dir).joinpath('.cache.lock')
        save_dir = Path(save_dir or self.cache_dir)
        meta = self._cache['_meta']
        modified = meta.get('modified_time') or datetime.min
        update_time = max(meta.get(x) or datetime.min for x in ('loaded_time', 'saved_time'))
        if modified < update_time and not force:
            return  # Not recently modified; return

        # Check if in use by another process
        while lock_file.exists():
            if time.time() - lock_file.stat().st_ctime > TIMEOUT:
                lock_file.unlink(missing_ok=True)
            else:
                time.sleep(.1)

        _logger.info('Saving cache tables...')
        lock_file.touch()
        try:
            for table in filter(lambda x: not x[0] == '_', self._cache.keys()):
                metadata = meta['raw'][table]
                metadata['date_modified'] = modified.isoformat(sep=' ', timespec='minutes')
                filename = save_dir.joinpath(f'{table}.pqt')
                parquet.save(filename, self._cache[table], metadata)
                _logger.debug(f'Saved {filename}')
            meta['saved_time'] = datetime.now()
        finally:
            lock_file.unlink()

    def refresh_cache(self, mode='auto'):
        """Check and reload cache tables

        Parameters
        ----------
        mode : {'local', 'refresh', 'auto', 'remote'}
            Options are 'local' (don't reload); 'refresh' (reload); 'auto' (reload if expired);
            'remote' (don't reload)

        Returns
        -------
        datetime.datetime
            Loaded timestamp
        """
        # NB: Currently modified table will be lost if called with 'refresh';
        # May be instances where modified cache is saved then immediately replaced with a new
        # remote cache. Also it's too slow :(
        # self.save_cache()  # Save cache if modified
        if mode in {'local', 'remote'}:
            pass
        elif mode == 'auto':
            if datetime.now() - self._cache['_meta']['loaded_time'] >= self.cache_expiry:
                _logger.info('Cache expired, refreshing')
                self.load_cache()
        elif mode == 'refresh':
            _logger.debug('Forcing reload of cache')
            self.load_cache(clobber=True)
        else:
            raise ValueError(f'Unknown refresh type "{mode}"')
        return self._cache['_meta']['loaded_time']

    def _update_cache_from_records(self, strict=False, **kwargs):
        """
        Update the cache tables with new records.

        Parameters
        ----------
        strict : bool
            If not True, the columns don't need to match.  Extra columns in input tables are
            dropped and missing columns are added and filled with np.nan.
        **kwargs
            pandas.DataFrame or pandas.Series to insert/update for each table

        Returns
        -------
        datetime.datetime:
            A timestamp of when the cache was updated

        Example
        -------
        >>> session, datasets = util.ses2records(self.get_details(eid, full=True))
        ... self._update_cache_from_records(sessions=session, datasets=datasets)

        Raises
        ------
        AssertionError
            When strict is True the input columns must exactly match those oo the cache table,
            including the order.
        KeyError
            One or more of the keyword arguments does not match a table in One._cache
        """
        updated = None
        for table, records in kwargs.items():
            if records is None or records.empty:
                continue
            if table not in self._cache:
                raise KeyError(f'Table "{table}" not in cache')
            if isinstance(records, pd.Series):
                records = pd.DataFrame([records])
            if not strict:
                # Deal with case where there are extra columns in the cache
                extra_columns = set(self._cache[table].columns) - set(records.columns)
                for col in extra_columns:
                    n = list(self._cache[table].columns).index(col)
                    records.insert(n, col, np.nan)
                # Drop any extra columns in the records that aren't in cache table
                to_drop = set(records.columns) - set(self._cache[table].columns)
                records.drop(to_drop, axis=1, inplace=True)
                records = records.reindex(columns=self._cache[table].columns)
            assert all(self._cache[table].columns == records.columns)
            # Update existing rows
            to_update = records.index.isin(self._cache[table].index)
            self._cache[table].loc[records.index[to_update], :] = records[to_update]
            # Assign new rows
            to_assign = records[~to_update]
            if isinstance(self._cache[table].index, pd.MultiIndex) and not to_assign.empty:
                # Concatenate and sort (no other way for non-unique index within MultiIndex)
                self._cache[table] = pd.concat([self._cache[table], to_assign]).sort_index()
            else:
                for index, record in to_assign.iterrows():
                    self._cache[table].loc[index, :] = record[self._cache[table].columns].values
            updated = datetime.now()
        self._cache['_meta']['modified_time'] = updated
        return updated

    def save_loaded_ids(self, sessions_only=False, clear_list=True):
        """
        Save list of UUIDs corresponding to datasets or sessions where datasets were loaded.

        Parameters
        ----------
        sessions_only : bool
            If true, save list of experiment IDs, otherwise the full list of dataset IDs.
        clear_list : bool
            If true, clear the current list of loaded dataset IDs after saving.

        Returns
        -------
        list of str
            List of UUIDs.
        pathlib.Path
            The file path of the saved list.
        """
        if '_loaded_datasets' not in self._cache or self._cache['_loaded_datasets'].size == 0:
            warnings.warn('No datasets loaded; check "record_datasets" attribute is True')
            return [], None
        if sessions_only:
            name = 'session_uuid'
            idx = self._cache['datasets'].index.isin(self._cache['_loaded_datasets'], 'id')
            ids = self._cache['datasets'][idx].index.unique('eid').values
        else:
            name = 'dataset_uuid'
            ids = self._cache['_loaded_datasets']

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S%z")
        filename = Path(self._tables_dir or self.cache_dir) / f'{timestamp}_loaded_{name}s.csv'
        pd.DataFrame(ids, columns=[name]).to_csv(filename, index=False)
        if clear_list:
            self._cache['_loaded_datasets'] = np.array([])
        return ids, filename

    def _download_datasets(self, dsets, **kwargs) -> List[Path]:
        """
        Download several datasets given a set of datasets.

        NB: This will not skip files that are already present.  Use check_filesystem instead.

        Parameters
        ----------
        dsets : list
            List of dataset dictionaries from an Alyx REST query OR URL strings

        Returns
        -------
        list of pathlib.Path
            A local file path list
        """
        # Looking to entirely remove method
        pass

    def _download_dataset(self, dset, cache_dir=None, **kwargs) -> Path:
        """
        Download a dataset from an Alyx REST dictionary

        Parameters
        ----------
        dset : pandas.Series, dict, str
            A single dataset dictionary from an Alyx REST query OR URL string
        cache_dir : str, pathlib.Path
            The root directory to save the data in (home/downloads by default)

        Returns
        -------
        pathlib.Path
            The local file path
        """
        pass  # pragma: no cover

    def search(self, details=False, query_type=None, **kwargs):
        """
        Searches sessions matching the given criteria and returns a list of matching eids

        For a list of search terms, use the method

            one.search_terms()

        For all of the search parameters, a single value or list may be provided.  For `dataset`,
        the sessions returned will contain all listed datasets.  For the other parameters,
        the session must contain at least one of the entries.

        For all but `date_range` and `number`, any field that contains the search string is
        returned.  Wildcards are not permitted, however if wildcards property is True, regular
        expressions may be used (see notes and examples).

        Parameters
        ----------
        dataset : str, list
            One or more dataset names. Returns sessions containing all these datasets.
            A dataset matches if it contains the search string e.g. 'wheel.position' matches
            '_ibl_wheel.position.npy'.
        date_range : str, list, datetime.datetime, datetime.date, pandas.timestamp
            A single date to search or a list of 2 dates that define the range (inclusive).  To
            define only the upper or lower date bound, set the other element to None.
        lab : str
            A str or list of lab names, returns sessions from any of these labs.
        number : str, int
            Number of session to be returned, i.e. number in sequence for a given date.
        subject : str, list
            A list of subject nicknames, returns sessions for any of these subjects.
        task_protocol : str
            The task protocol name (can be partial, i.e. any task protocol containing that str
            will be found).
        projects : str, list
            The project name(s) (can be partial, i.e. any project containing that str
            will be found).
        details : bool
            If true also returns a dict of dataset details.
        query_type : str, None
            Query cache ('local') or Alyx database ('remote').

        Returns
        -------
        list
            A list of eids.
        (list)
            (If details is True) a list of dictionaries, each entry corresponding to a matching
            session.

        Examples
        --------
        Search for sessions with 'training' in the task protocol.

        >>> eids = one.search(task='training')

        Search for sessions by subject 'MFD_04'.

        >>> eids = one.search(subject='MFD_04')

        Do an exact search for sessions by subject 'FD_04'.

        >>> assert one.wildcards is True, 'the wildcards flag must be True for regex expressions'
        >>> eids = one.search(subject='^FD_04$')

        Search for sessions on a given date, in a given lab, containing trials and spike data.

        >>> eids = one.search(date='2023-01-01', lab='churchlandlab', dataset=['trials', 'spikes'])

        Notes
        -----
        - In default and local mode, most queries are case-sensitive partial matches. When lists
          are provided, the search is a logical OR, except for `datasets`, which is a logical AND.
        - All search terms are true for a session to be returned, i.e. subject matches AND project
          matches, etc.
        - In remote mode most queries are case-insensitive partial matches.
        - In default and local mode, when the one.wildcards flag is True (default), queries are
          interpreted as regular expressions. To turn this off set one.wildcards to False.
        - In remote mode regular expressions are only supported using the `django` argument.
        """

        def all_present(x, dsets, exists=True):
            """Returns true if all datasets present in Series"""
            return all(any(x.str.contains(y, regex=self.wildcards) & exists) for y in dsets)

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
            elif key in ('subject', 'task_protocol', 'laboratory', 'projects'):
                query = '|'.join(util.ensure_list(value))
                key = 'lab' if key == 'laboratory' else key
                mask = sessions[key].str.contains(query, regex=self.wildcards)
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
                query = util.ensure_list(value)
                datasets = self._cache['datasets']
                has_dset = sessions.index.isin(datasets.index.get_level_values('eid'))
                datasets = datasets.loc[(sessions.index.values[has_dset], ), :]
                # For each session check any dataset both contains query and exists
                mask = (
                    (datasets
                        .groupby('eid', sort=False)
                        .apply(lambda x: all_present(x['rel_path'], query, x['exists'])))
                )
                # eids of matching dataset records
                idx = mask[mask].index

                # Reduce sessions table by datasets mask
                sessions = sessions.loc[idx]

        # Return results
        if sessions.size == 0:
            return ([], None) if details else []
        sessions = sessions.sort_values(['date', 'subject', 'number'], ascending=False)
        eids = sessions.index.to_list()

        if details:
            return eids, sessions.reset_index(drop=True).to_dict('records', Bunch)
        else:
            return eids

    def _check_filesystem(self, datasets, offline=None, update_exists=True):
        """Update the local filesystem for the given datasets.

        Given a set of datasets, check whether records correctly reflect the filesystem.
        Called by load methods, this returns a list of file paths to load and return.
        TODO This needs changing; overload for downloading?
         This changes datasets frame, calls _update_cache(sessions=None, datasets=None) to
         update and save tables.  Download_datasets can also call this function.

        Parameters
        ----------
        datasets : pandas.Series, pandas.DataFrame, list of dicts
            A list or DataFrame of dataset records
        offline : bool, None
            If false and Web client present, downloads the missing datasets from a remote
            repository
        update_exists : bool
            If true, the cache is updated to reflect the filesystem

        Returns
        -------
        A list of file paths for the datasets (None elements for non-existent datasets)
        """
        if isinstance(datasets, pd.Series):
            datasets = pd.DataFrame([datasets])
        elif not isinstance(datasets, pd.DataFrame):
            # Cast set of dicts (i.e. from REST datasets endpoint)
            datasets = util.datasets2records(list(datasets))
        indices_to_download = []  # indices of datasets that need (re)downloading
        files = []  # file path list to return
        # If the session_path field is missing from the datasets table, fetch from sessions table
        if 'session_path' not in datasets.columns:
            if 'eid' not in datasets.index.names:
                # Get slice of full frame with eid in index
                _dsets = self._cache['datasets'][
                    self._cache['datasets'].index.get_level_values(1).isin(datasets.index)
                ]
                idx = _dsets.index.get_level_values(1)
            else:
                _dsets = datasets
                idx = pd.IndexSlice[:, _dsets.index.get_level_values(1)]
            # Ugly but works over unique sessions, which should be quicker
            session_path = (self._cache['sessions']
                            .loc[_dsets.index.get_level_values(0).unique()]
                            .apply(session_record2path, axis=1))
            datasets.loc[idx, 'session_path'] = \
                pd.Series(_dsets.index.get_level_values(0)).map(session_path).values

        # First go through datasets and check if file exists and hash matches
        for i, rec in datasets.iterrows():
            file = Path(self.cache_dir, *rec[['session_path', 'rel_path']])
            if file.exists():
                # Check if there's a hash mismatch
                # If so, add this index to list of datasets that need downloading
                if rec['hash'] is not None:
                    if hashfile.md5(file) != rec['hash']:
                        _logger.warning('local md5 mismatch on dataset: %s',
                                        PurePosixPath(rec.session_path, rec.rel_path))
                        indices_to_download.append(i)
                elif rec['file_size'] and file.stat().st_size != rec['file_size']:
                    _logger.warning('local file size mismatch on dataset: %s',
                                    PurePosixPath(rec.session_path, rec.rel_path))
                    indices_to_download.append(i)
                files.append(file)  # File exists so add to file list
            else:
                # File doesn't exist so add None to output file list
                files.append(None)
                # Add this index to list of datasets that need downloading
                indices_to_download.append(i)
            if rec['exists'] != file.exists():
                with warnings.catch_warnings():
                    # Suppress future warning: exist column should always be present
                    msg = '.*indexing on a MultiIndex with a nested sequence of labels.*'
                    warnings.filterwarnings('ignore', message=msg)
                    datasets.at[i, 'exists'] = not rec['exists']
                    if update_exists:
                        _logger.debug('Updating exists field')
                        if isinstance(i, tuple):
                            self._cache['datasets'].loc[i, 'exists'] = not rec['exists']
                        else:  # eid index level missing in datasets input
                            i = pd.IndexSlice[:, i]
                            self._cache['datasets'].loc[i, 'exists'] = not rec['exists']
                        self._cache['_meta']['modified_time'] = datetime.now()

        # If online and we have datasets to download, call download_datasets with these datasets
        if not (offline or self.offline) and indices_to_download:
            dsets_to_download = datasets.loc[indices_to_download]
            # Returns list of local file paths and set to variable
            new_files = self._download_datasets(dsets_to_download, update_cache=update_exists)
            # Add each downloaded file to the output list of files
            for i, file in zip(indices_to_download, new_files):
                files[datasets.index.get_loc(i)] = file

        if self.record_loaded:
            loaded = np.fromiter(map(bool, files), bool)
            loaded_ids = np.array(datasets.index.to_list())[loaded]
            if '_loaded_datasets' not in self._cache:
                self._cache['_loaded_datasets'] = np.unique(loaded_ids)
            else:
                loaded_set = np.hstack([self._cache['_loaded_datasets'], loaded_ids])
                self._cache['_loaded_datasets'] = np.unique(loaded_set, axis=0)

        # Return full list of file paths
        return files

    @util.refresh
    @util.parse_id
    def get_details(self, eid: Union[str, Path, UUID], full: bool = False):
        """Return session details for a given session ID

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        full : bool
            If True, returns a DataFrame of session and dataset info

        Returns
        -------
        pd.Series, pd.DataFrame
            A session record or full DataFrame with dataset information if full is True
        """
        # Int ids return DataFrame, making str eid a list ensures Series not returned
        try:
            det = self._cache['sessions'].loc[[eid]]
            assert len(det) == 1
        except KeyError:
            raise alferr.ALFObjectNotFound(eid)
        except AssertionError:
            raise alferr.ALFMultipleObjectsFound(f'Multiple sessions in cache for eid {eid}')
        if not full:
            return det.iloc[0]
        # .reset_index('eid', drop=True)
        return self._cache['datasets'].join(det, on='eid', how='right')

    @util.refresh
    def list_subjects(self) -> List[str]:
        """
        List all subjects in database

        Returns
        -------
        list
            Sorted list of subject names
        """
        return self._cache['sessions']['subject'].sort_values().unique().tolist()

    @util.refresh
    def list_datasets(self, eid=None, filename=None, collection=None, revision=None,
                      details=False, query_type=None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Given an eid, return the datasets for those sessions.  If no eid is provided,
        a list of all datasets is returned.  When details is false, a sorted array of unique
        datasets is returned (their relative paths).

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        filename : str, dict, list
            Filters datasets and returns only the ones matching the filename.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        collection : str, list
            The collection to which the object belongs, e.g. 'alf/probe01'.
            This is the relative path of the file from the session root.
            Supports asterisks as wildcards.
        revision : str
            Filters datasets and returns only the ones matching the revision.
            Supports asterisks as wildcards.
        details : bool
            When true, a pandas DataFrame is returned, otherwise a numpy array of
            relative paths (collection/revision/filename) - see one.alf.spec.describe for details.
        query_type : str
            Query cache ('local') or Alyx database ('remote').

        Returns
        -------
        np.ndarray, pd.DataFrame
            Slice of datasets table or numpy array if details is False.

        Examples
        --------
        List all unique datasets in ONE cache

        >>> datasets = one.list_datasets()

        List all datasets for a given experiment

        >>> datasets = one.list_datasets(eid)

        List all datasets for an experiment that match a collection name

        >>> probe_datasets = one.list_datasets(eid, collection='*probe*')

        List datasets for an experiment that have 'wheel' in the filename

        >>> datasets = one.list_datasets(eid, filename='*wheel*')

        List datasets for an experiment that are part of a 'wheel' or 'trial(s)' object

        >>> datasets = one.list_datasets(eid, {'object': ['wheel', 'trial?']})
        """
        datasets = self._cache['datasets']
        filter_args = dict(collection=collection, filename=filename, wildcards=self.wildcards,
                           revision=revision, revision_last_before=False, assert_unique=False)
        if not eid:
            datasets = util.filter_datasets(datasets, **filter_args)
            return datasets.copy() if details else datasets['rel_path'].unique().tolist()
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if not eid:
            return datasets.iloc[0:0]  # Return empty
        try:
            datasets = datasets.loc[(eid,), :]
        except KeyError:
            return datasets.iloc[0:0]  # Return empty

        datasets = util.filter_datasets(datasets, **filter_args)
        # Return only the relative path
        return datasets if details else datasets['rel_path'].sort_values().values.tolist()

    @util.refresh
    def list_collections(self, eid=None, filename=None, collection=None, revision=None,
                         details=False, query_type=None) -> Union[np.ndarray, dict]:
        """
        List the collections for a given experiment.  If no experiment ID is given,
        all collections are returned.

        Parameters
        ----------
        eid : [str, UUID, Path, dict]
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path
        filename : str, dict, list
            Filters datasets and returns only the collections containing matching datasets.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        collection : str, list
            Filter by a given pattern. Supports asterisks as wildcards.
        revision : str
            Filters collections and returns only the ones with the matching revision.
            Supports asterisks as wildcards
        details : bool
            If true a dict of pandas datasets tables is returned with collections as keys,
            otherwise a numpy array of unique collections
        query_type : str
            Query cache ('local') or Alyx database ('remote')

        Returns
        -------
        list, dict
            A list of unique collections or dict of datasets tables

        Examples
        --------
        List all unique collections in ONE cache

        >>> collections = one.list_collections()

        List all collections for a given experiment

        >>> collections = one.list_collections(eid)

        List all collections for a given experiment and revision

        >>> revised = one.list_collections(eid, revision='2020-01-01')

        List all collections that have 'probe' in the name.

        >>> collections = one.list_collections(eid, collection='*probe*')

        List collections for an experiment that have datasets with 'wheel' in the name

        >>> collections = one.list_collections(eid, filename='*wheel*')

        List collections for an experiment that contain numpy datasets

        >>> collections = one.list_collections(eid, {'extension': 'npy'})
        """
        filter_kwargs = dict(eid=eid, collection=collection, filename=filename,
                             revision=revision, query_type=query_type)
        datasets = self.list_datasets(details=True, **filter_kwargs).copy()

        datasets['collection'] = datasets.rel_path.apply(
            lambda x: alfiles.rel_path_parts(x, assert_valid=False)[0] or ''
        )
        if details:
            return {k: table.drop('collection', axis=1)
                    for k, table in datasets.groupby('collection')}
        else:
            return datasets['collection'].unique().tolist()

    @util.refresh
    def list_revisions(self, eid=None, filename=None, collection=None, revision=None,
                       details=False, query_type=None):
        """
        List the revisions for a given experiment.  If no experiment id is given,
        all collections are returned.

        Parameters
        ----------
        eid : str, UUID, Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path
        filename : str, dict, list
            Filters datasets and returns only the revisions containing matching datasets.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        collection : str, list
            Filter by a given collection. Supports asterisks as wildcards.
        revision : str, list
            Filter by a given pattern. Supports asterisks as wildcards.
        details : bool
            If true a dict of pandas datasets tables is returned with collections as keys,
            otherwise a numpy array of unique collections
        query_type : str
            Query cache ('local') or Alyx database ('remote')

        Returns
        -------
        list, dict
            A list of unique collections or dict of datasets tables

        Examples
        --------
        List all revisions in ONE cache

        >>> revisions = one.list_revisions()

        List all revisions for a given experiment

        >>> revisions = one.list_revisions(eid)

        List all revisions for a given experiment that contain the trials object

        >>> revisions = one.list_revisions(eid, filename={'object': 'trials'})

        List all revisions for a given experiment that start with 2020 or 2021

        >>> revisions = one.list_revisions(eid, revision=['202[01]*'])

        """
        datasets = self.list_datasets(eid=eid, details=True, query_type=query_type).copy()

        # Call filter util ourselves with the revision_last_before set to False
        kwargs = dict(collection=collection, filename=filename, revision=revision,
                      revision_last_before=False, wildcards=self.wildcards, assert_unique=False)
        datasets = util.filter_datasets(datasets, **kwargs)
        datasets['revision'] = datasets.rel_path.apply(
            lambda x: (alfiles.rel_path_parts(x, assert_valid=False)[1] or '').strip('#')
        )
        if details:
            return {k: table.drop('revision', axis=1)
                    for k, table in datasets.groupby('revision')}
        else:
            return datasets['revision'].unique().tolist()

    @util.refresh
    @util.parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = None,
                    revision: Optional[str] = None,
                    query_type: Optional[str] = None,
                    download_only: bool = False,
                    **kwargs) -> Union[alfio.AlfBunch, List[Path]]:
        """
        Load all attributes of an ALF object from a Session ID and an object name.  Any datasets
        with matching object name will be loaded.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        obj : str
            The ALF object to load.  Supports asterisks as wildcards.
        collection : str
            The collection to which the object belongs, e.g. 'alf/probe01'.
            This is the relative path of the file from the session root.
            Supports asterisks as wildcards.
        revision : str
            The dataset revision (typically an ISO date).  If no exact match, the previous
            revision (ordered lexicographically) is returned.  If None, the default revision is
            returned (usually the most recent revision).  Regular expressions/wildcards not
            permitted.
        query_type : str
            Query cache ('local') or Alyx database ('remote')
        download_only : bool
            When true the data are downloaded and the file path is returned. NB: The order of the
            file path list is undefined.
        **kwargs
            Additional filters for datasets, including namespace and timescale. For full list
            see the one.alf.spec.describe function.

        Returns
        -------
        one.alf.io.AlfBunch, list
            An ALF bunch or if download_only is True, a list of Paths objects

        Examples
        --------
        >>> load_object(eid, 'moves')
        >>> load_object(eid, 'trials')
        >>> load_object(eid, 'spikes', collection='*probe01')  # wildcards is True
        >>> load_object(eid, 'spikes', collection='.*probe01')  # wildcards is False
        >>> load_object(eid, 'spikes', namespace='ibl')
        >>> load_object(eid, 'spikes', timescale='ephysClock')

        Load specific attributes:

        >>> load_object(eid, 'spikes', attribute=['times*', 'clusters'])
        """
        query_type = query_type or self.mode
        datasets = self.list_datasets(eid, details=True, query_type=query_type)

        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(obj)

        dataset = {'object': obj, **kwargs}
        datasets = util.filter_datasets(datasets, dataset, collection, revision,
                                        assert_unique=False, wildcards=self.wildcards)

        # Validate result before loading
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(obj)
        parts = [alfiles.rel_path_parts(x) for x in datasets.rel_path]
        unique_objects = set(x[3] or '' for x in parts)
        unique_collections = set(x[0] or '' for x in parts)
        if len(unique_objects) > 1:
            raise alferr.ALFMultipleObjectsFound(*unique_objects)
        if len(unique_collections) > 1:
            raise alferr.ALFMultipleCollectionsFound(*unique_collections)

        # For those that don't exist, download them
        offline = None if query_type == 'auto' else self.mode == 'local'
        files = self._check_filesystem(datasets, offline=offline)
        files = [x for x in files if x]
        if not files:
            raise alferr.ALFObjectNotFound(f'ALF object "{obj}" not found on disk')

        if download_only:
            return files

        return alfio.load_object(files, wildcards=self.wildcards, **kwargs)

    @util.refresh
    @util.parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = None,
                     revision: Optional[str] = None,
                     query_type: Optional[str] = None,
                     download_only: bool = False,
                     **kwargs) -> Any:
        """
        Load a single dataset for a given session id and dataset name.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        dataset : str, dict
            The ALF dataset to load.  May be a string or dict of ALF parts.  Supports asterisks as
            wildcards.
        collection : str
            The collection to which the object belongs, e.g. 'alf/probe01'.
            This is the relative path of the file from the session root.
            Supports asterisks as wildcards.
        revision : str
            The dataset revision (typically an ISO date).  If no exact match, the previous
            revision (ordered lexicographically) is returned.  If None, the default revision is
            returned (usually the most recent revision).  Regular expressions/wildcards not
            permitted.
        query_type : str
            Query cache ('local') or Alyx database ('remote')
        download_only : bool
            When true the data are downloaded and the file path is returned.

        Returns
        -------
        np.ndarray, pathlib.Path
            Dataset or a Path object if download_only is true.

        Examples
        --------
        >>> intervals = one.load_dataset(eid, '_ibl_trials.intervals.npy')

        Load dataset without specifying extension

        >>> intervals = one.load_dataset(eid, 'trials.intervals')  # wildcard mode only
        >>> intervals = one.load_dataset(eid, '.*trials.intervals.*')  # regex mode only
        >>> intervals = one.load_dataset(eid, dict(object='trials', attribute='intervals'))
        >>> filepath = one.load_dataset(eid, '_ibl_trials.intervals.npy', download_only=True)
        >>> spike_times = one.load_dataset(eid, 'spikes.times.npy', collection='alf/probe01')
        >>> old_spikes = one.load_dataset(eid, 'spikes.times.npy',
        ...                               collection='alf/probe01', revision='2020-08-31')
        >>> old_spikes = one.load_dataset(eid, 'alf/probe01/#2020-08-31#/spikes.times.npy')

        Raises
        ------
        ValueError
            When a relative paths is provided (e.g. 'collection/#revision#/object.attribute.ext'),
            the collection and revision keyword arguments must be None.
        one.alf.exceptions.ALFObjectNotFound
            The dataset was not found in the cache or on disk.
        one.alf.exceptions.ALFMultipleCollectionsFound
            The dataset provided exists in multiple collections or matched multiple different
            files. Provide a specific collection to load, and make sure any wildcard/regular
            expressions are specific enough.

        Warnings
        --------
        UserWarning
            When a relative paths is provided (e.g. 'collection/#revision#/object.attribute.ext'),
            wildcards/regular expressions must not be used. To use wildcards, pass the collection
            and revision as separate keyword arguments.
        """
        datasets = self.list_datasets(eid, details=True, query_type=query_type or self.mode)
        # If only two parts and wildcards are on, append ext wildcard
        if self.wildcards and isinstance(dataset, str) and len(dataset.split('.')) == 2:
            dataset += '.*'
            _logger.debug('Appending extension wildcard: ' + dataset)

        assert_unique = ('/' if isinstance(dataset, str) else 'collection') not in dataset
        # Check if wildcard was used (this is not an exhaustive check)
        if not assert_unique and isinstance(dataset, str) and '*' in dataset:
            warnings.warn('Wildcards should not be used with relative path as input.')
        if not assert_unique and (collection is not None or revision is not None):
            raise ValueError(
                'collection and revision kwargs must be None when dataset is a relative path')
        datasets = util.filter_datasets(datasets, dataset, collection, revision,
                                        wildcards=self.wildcards, assert_unique=assert_unique)
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(f'Dataset "{dataset}" not found')

        # Check files exist / download remote files
        file, = self._check_filesystem(datasets, **kwargs)

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
                      query_type: Optional[str] = None,
                      assert_present=True,
                      download_only: bool = False,
                      **kwargs) -> Any:
        """
        Load datasets for a given session id.  Returns two lists the length of datasets.  The
        first is the data (or file paths if download_data is false), the second is a list of
        meta data Bunches.  If assert_present is false, missing data will be returned as None.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        datasets : list of strings
            The ALF datasets to load.  May be a string or dict of ALF parts.  Supports asterisks
            as wildcards.
        collections : str, list
            The collection(s) to which the object(s) belong, e.g. 'alf/probe01'.
            This is the relative path of the file from the session root.
            Supports asterisks as wildcards.
        revisions : str, list
            The dataset revision (typically an ISO date).  If no exact match, the previous
            revision (ordered lexicographically) is returned.  If None, the default revision is
            returned (usually the most recent revision).  Regular expressions/wildcards not
            permitted.
        query_type : str
            Query cache ('local') or Alyx database ('remote')
        assert_present : bool
            If true, missing datasets raises and error, otherwise None is returned
        download_only : bool
            When true the data are downloaded and the file path is returned.

        Returns
        -------
        list
            A list of data (or file paths) the length of datasets
        list
            A list of meta data Bunches. If assert_present is False, missing data will be None

        Notes
        -----
        - There are three ways the datasets may be formatted: the object.attribute; the file name
          (including namespace and extension); the ALF components as a dict; the dataset path,
          relative to the session path, e.g. collection/object.attribute.ext.
        - When relative paths are provided (e.g. 'collection/#revision#/object.attribute.ext'),
          wildcards/regular expressions must not be used. To use wildcards, pass the collection and
          revision as separate keyword arguments.
        - To ensure you are loading the correct revision, use the revisions kwarg instead of
          relative paths.

        Raises
        ------
        ValueError
            When a relative paths is provided (e.g. 'collection/#revision#/object.attribute.ext'),
            the collection and revision keyword arguments must be None.
        ValueError
            If a list of collections or revisions are provided, they must match the number of
            datasets passed in.
        TypeError
            The datasets argument must be a non-string iterable.
        one.alf.exceptions.ALFObjectNotFound
            One or more of the datasets was not found in the cache or on disk.  To suppress this
            error and return None for missing datasets, use assert_present=False.
        one.alf.exceptions.ALFMultipleCollectionsFound
            One or more of the dataset(s) provided exist in multiple collections. Provide the
            specific collections to load, and if using wildcards/regular expressions, make sure
            the expression is specific enough.

        Warnings
        --------
        UserWarning
            When providing a list of relative dataset paths, this warning occurs if one or more
            of the datasets are not marked as default revisions.  Avoid such warnings by explicitly
            passing in the required revisions with the revisions keyword argument.
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
        query_type = query_type or self.mode
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
        if self.wildcards:  # Append extension wildcard if 'object.attribute' string
            datasets = [x + ('.*' if isinstance(x, str) and len(x.split('.')) == 2 else '')
                        for x in datasets]
        # If collections provided in datasets list, e.g. [collection/x.y.z], do not assert unique
        validate = not any(('/' if isinstance(d, str) else 'collection') in d for d in datasets)
        if not validate and not all(x is None for x in collections + revisions):
            raise ValueError(
                'collection and revision kwargs must be None when dataset is a relative path')
        ops = dict(wildcards=self.wildcards, assert_unique=validate)
        slices = [util.filter_datasets(all_datasets, x, y, z, **ops)
                  for x, y, z in zip(datasets, collections, revisions)]
        present = [len(x) == 1 for x in slices]
        present_datasets = pd.concat(slices)

        # Check if user is blindly downloading all data and warn of non-default revisions
        if 'default_revision' in present_datasets and \
                not any(revisions) and not all(present_datasets['default_revision']):
            old = present_datasets.loc[~present_datasets['default_revision'], 'rel_path'].to_list()
            warnings.warn(
                'The following datasets may have been revised and ' +
                'are therefore not recommended for analysis:\n\t' +
                '\n\t'.join(old) + '\n'
                'To avoid this warning, specify the revision as a kwarg or use load_dataset.'
            )

        if not all(present):
            missing_list = ', '.join(x for x, y in zip(datasets, present) if not y)
            # FIXME include collection and revision also
            message = f'The following datasets are not in the cache: {missing_list}'
            if assert_present:
                raise alferr.ALFObjectNotFound(message)
            else:
                _logger.warning(message)

        # Check files exist / download remote files
        files = self._check_filesystem(present_datasets, **kwargs)

        if any(x is None for x in files):
            missing_list = ', '.join(x for x, y in zip(present_datasets.rel_path, files) if not y)
            message = f'The following datasets were not downloaded: {missing_list}'
            if assert_present:
                raise alferr.ALFObjectNotFound(message)
            else:
                _logger.warning(message)

        # Make list of metadata Bunches out of the table
        records = (present_datasets
                   .reset_index(names='id')
                   .to_dict('records', Bunch))

        # Ensure result same length as input datasets list
        files = [None if not here else files.pop(0) for here in present]
        # Replace missing file records with None
        records = [None if not here else records.pop(0) for here in present]
        if download_only:
            return files, records
        return [alfio.load_file_content(x) for x in files], records

    @util.refresh
    def load_dataset_from_id(self,
                             dset_id: Union[str, UUID],
                             download_only: bool = False,
                             details: bool = False) -> Any:
        """
        Load a dataset given a dataset UUID.

        Parameters
        ----------
        dset_id : uuid.UUID, str
            A dataset UUID to load.
        download_only : bool
            If true the dataset is downloaded (if necessary) and the filepath returned.
        details : bool
            If true a pandas Series is returned in addition to the data.

        Returns
        -------
        np.ndarray, pathlib.Path
            Dataset data (or filepath if download_only) and dataset record if details is True.
        """
        if isinstance(dset_id, UUID):
            dset_id = str(dset_id)
        elif not isinstance(dset_id, str):
            dset_id, = parquet.np2str(dset_id)
        try:
            dataset = self._cache['datasets'].loc[(slice(None), dset_id), :].squeeze()
            assert isinstance(dataset, pd.Series) or len(dataset) == 1
        except AssertionError:
            raise alferr.ALFMultipleObjectsFound('Duplicate dataset IDs')
        except KeyError:
            raise alferr.ALFObjectNotFound('Dataset not found')

        filepath, = self._check_filesystem(dataset)
        if not filepath:
            raise alferr.ALFObjectNotFound('Dataset not found')
        output = filepath if download_only else alfio.load_file_content(filepath)
        if details:
            return output, dataset
        else:
            return output

    @util.refresh
    @util.parse_id
    def load_collection(self,
                        eid: Union[str, Path, UUID],
                        collection: str,
                        object: Optional[str] = None,
                        revision: Optional[str] = None,
                        query_type: Optional[str] = None,
                        download_only: bool = False,
                        **kwargs) -> Union[Bunch, List[Path]]:
        """
        Load all objects in an ALF collection from a Session ID.  Any datasets with matching object
        name(s) will be loaded.  Returns a bunch of objects.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        collection : str
            The collection to which the object belongs, e.g. 'alf/probe01'.
            This is the relative path of the file from the session root.
            Supports asterisks as wildcards.
        object : str
            The ALF object to load.  Supports asterisks as wildcards.
        revision : str
            The dataset revision (typically an ISO date).  If no exact match, the previous
            revision (ordered lexicographically) is returned.  If None, the default revision is
            returned (usually the most recent revision).  Regular expressions/wildcards not
            permitted.
        query_type : str
            Query cache ('local') or Alyx database ('remote')
        download_only : bool
            When true the data are downloaded and the file path is returned.
        **kwargs
            Additional filters for datasets, including namespace and timescale. For full list
            see the one.alf.spec.describe function.

        Returns
        -------
        Bunch of one.alf.io.AlfBunch, list of pathlib.Path
            A Bunch of objects or if download_only is True, a list of Paths objects

        Examples
        --------
        >>> alf_collection = load_collection(eid, 'alf')
        >>> load_collection(eid, '*probe01', object=['spikes', 'clusters'])  # wildcards is True
        >>> files = load_collection(eid, '', download_only=True)  # Base session dir

        Raises
        ------
        alferr.ALFError
            No datasets exist for the provided session collection.
        alferr.ALFObjectNotFound
            No datasets match the object, attribute or revision filters for this collection.
        """
        query_type = query_type or self.mode
        datasets = self.list_datasets(eid, details=True, collection=collection,
                                      query_type=query_type)

        if len(datasets) == 0:
            raise alferr.ALFError(f'{collection} not found for session {eid}')

        dataset = {'object': object, **kwargs}
        datasets = util.filter_datasets(datasets, dataset, revision,
                                        assert_unique=False, wildcards=self.wildcards)

        # Validate result before loading
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(object or '')
        parts = [alfiles.rel_path_parts(x) for x in datasets.rel_path]
        unique_objects = set(x[3] or '' for x in parts)

        # For those that don't exist, download them
        offline = None if query_type == 'auto' else self.mode == 'local'
        files = self._check_filesystem(datasets, offline=offline)
        files = [x for x in files if x]
        if not files:
            raise alferr.ALFObjectNotFound(f'ALF collection "{collection}" not found on disk')

        if download_only:
            return files

        kwargs.update(wildcards=self.wildcards)
        collection = {
            obj: alfio.load_object([x for x, y in zip(files, parts) if y[3] == obj], **kwargs)
            for obj in unique_objects
        }
        return Bunch(collection)

    @staticmethod
    def setup(cache_dir=None, silent=False, **kwargs):
        """Set up One cache tables for a given data directory.

        Parameters
        ----------
        cache_dir : pathlib.Path, str
            A path to the ALF data directory.
        silent : (False) bool
            When True will prompt for cache_dir, if cache_dir is None, and overwrite cache if any.
            When False will use cwd for cache_dir, if cache_dir is None, and use existing cache.
        **kwargs
            Optional arguments to pass to one.alf.cache.make_parquet_db.

        Returns
        -------
        One
            An instance of One for the provided cache directory.
        """
        if not cache_dir:
            if not silent:
                cache_dir = input(f'Select a directory from which to build cache ({Path.cwd()})')
            cache_dir = cache_dir or Path.cwd()
        cache_dir = Path(cache_dir)
        assert cache_dir.exists(), f'{cache_dir} does not exist'

        # Check if cache already exists
        if next(cache_dir.glob('sessions.pqt'), False):
            generate_cache = False
            if not silent:
                answer = input(f'Cache tables exist for {cache_dir}, overwrite? [y/N]')
                generate_cache = True if answer == 'y' else False
            if not generate_cache:
                return One(cache_dir, mode='local')

        # Build cache tables
        make_parquet_db(cache_dir, **kwargs)
        return One(cache_dir, mode='local')


@lru_cache(maxsize=1)
def ONE(*, mode='auto', wildcards=True, **kwargs):
    """ONE API factory
    Determine which class to instantiate depending on parameters passed.

    Parameters
    ----------
    mode : str
        Query mode, options include 'auto', 'local' (offline) and 'remote' (online only).  Most
        methods have a `query_type` parameter that can override the class mode.
    wildcards : bool
        If true all methods use unix shell style pattern matching, otherwise regular expressions
        are used.
    cache_dir : str, pathlib.Path
        Path to the data files.  If Alyx parameters have been set up for this location,
        an OneAlyx instance is returned.  If data_dir and base_url are None, the default
        location is used.
    tables_dir : str, pathlib.Path
        An optional location of the cache tables.  If None, the tables are assumed to be in the
        cache_dir.
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
    One, OneAlyx
        An One instance if mode is 'local', otherwise an OneAlyx instance.
    """
    if kwargs.pop('offline', False):
        _logger.warning('the offline kwarg will probably be removed. '
                        'ONE is now offline by default anyway')
        warnings.warn('"offline" param will be removed; use mode="local"', DeprecationWarning)
        mode = 'local'

    if (any(x in kwargs for x in ('base_url', 'username', 'password')) or
            not kwargs.get('cache_dir', False)):
        return OneAlyx(mode=mode, wildcards=wildcards, **kwargs)

    # If cache dir was provided and corresponds to one configured with an Alyx client, use OneAlyx
    try:
        one.params.check_cache_conflict(kwargs.get('cache_dir'))
        return One(mode='local', wildcards=wildcards, **kwargs)
    except AssertionError:
        # Cache dir corresponds to a Alyx repo, call OneAlyx
        return OneAlyx(mode=mode, wildcards=wildcards, **kwargs)


class OneAlyx(One):
    """An API for searching and loading data through the Alyx database"""
    def __init__(self, username=None, password=None, base_url=None, cache_dir=None,
                 mode='auto', wildcards=True, tables_dir=None, **kwargs):
        """An API for searching and loading data through the Alyx database

        Parameters
        ----------
        mode : str
            Query mode, options include 'auto', 'local' (offline) and 'remote' (online only).  Most
            methods have a `query_type` parameter that can override the class mode.
        wildcards : bool
            If true, methods allow unix shell style pattern matching, otherwise regular
            expressions are supported
        cache_dir : str, pathlib.Path
            Path to the data files.  If Alyx parameters have been set up for this location,
            an OneAlyx instance is returned.  If data_dir and base_url are None, the default
            location is used.
        tables_dir : str, pathlib.Path
            An optional location of the cache tables.  If None, the tables are assumed to be in the
            cache_dir.
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
        super(OneAlyx, self).__init__(
            mode=mode, wildcards=wildcards, tables_dir=tables_dir, cache_dir=cache_dir)

    def __repr__(self):
        return f'One ({"off" if self.offline else "on"}line, {self.alyx.base_url})'

    def load_cache(self, tables_dir=None, clobber=False, tag=None):
        """
        Load parquet cache files.  If the local cache is sufficiently old, this method will query
        the database for the location and creation date of the remote cache.  If newer, it will be
        download and loaded.

        Note: Unlike refresh_cache, this will always reload the local files at least once.

        Parameters
        ----------
        tables_dir : str, pathlib.Path
            An optional directory location of the parquet files, defaults to One._tables_dir.
        clobber : bool
            If True, query Alyx for a newer cache even if current (local) cache is recent.
        tag : str
            An optional Alyx dataset tag for loading cache tables containing a subset of datasets.

        Examples
        --------
        To load the cache tables for a given release tag
        >>> one.load_cache(tag='2022_Q2_IBL_et_al_RepeatedSite')

        To reset the cache tables after loading a tag
        >>> ONE.cache_clear()
        ... one = ONE()
        """
        cache_meta = self._cache.get('_meta', {})
        raw_meta = cache_meta.get('raw', {}).values() or [{}]
        # If user provides tag that doesn't match current cache's tag, always download.
        # NB: In the future 'database_tags' may become a list.
        current_tags = flatten(x.get('database_tags') for x in raw_meta)
        if len(set(filter(None, current_tags))) > 1:
            raise NotImplementedError(
                'Loading cache tables with multiple tags is not currently supported'
            )
        tag = tag or current_tags[0]  # For refreshes take the current tag as default
        different_tag = any(x != tag for x in current_tags)
        if not (clobber or different_tag):
            super(OneAlyx, self).load_cache(tables_dir)  # Load any present cache
            expired = self._cache and (cache_meta := self._cache.get('_meta', {}))['expired']
            if not expired or self.mode in {'local', 'remote'}:
                return

        # Warn user if expired
        if (
            cache_meta['expired'] and
            cache_meta.get('created_time', False) and
            not self.alyx.silent
        ):
            age = datetime.now() - cache_meta['created_time']
            t_str = (f'{age.days} day(s)'
                     if age.days >= 1
                     else f'{np.floor(age.seconds / (60 * 2))} hour(s)')
            _logger.info(f'cache over {t_str} old')

        try:
            # Determine whether a newer cache is available
            cache_info = self.alyx.get(f'cache/info/{tag or ""}'.strip('/'), expires=True)
            assert tag is None or tag in cache_info.get('database_tags', [])

            # Check version compatibility
            min_version = packaging.version.parse(cache_info.get('min_api_version', '0.0.0'))
            if packaging.version.parse(one.__version__) < min_version:
                warnings.warn(f'Newer cache tables require ONE version {min_version} or greater')
                return

            # Check whether remote cache more recent
            remote_created = datetime.fromisoformat(cache_info['date_created'])
            local_created = cache_meta.get('created_time', None)
            fresh = local_created and (remote_created - local_created) < timedelta(minutes=1)
            if fresh and not different_tag:
                _logger.info('No newer cache available')
                return

            # Set the cache table directory location
            if tables_dir:  # If tables directory specified, use that
                self._tables_dir = Path(tables_dir)
            elif different_tag:  # Otherwise use a subdirectory for a given tag
                self._tables_dir = self.cache_dir / tag
                self._tables_dir.mkdir(exist_ok=True)
            else:  # Otherwise use the previous location (default is the data cache directory)
                self._tables_dir = self._tables_dir or self.cache_dir

            # Check if the origin has changed. This is to warn users if downloading from a
            # different database to the one currently loaded.
            prev_origin = list(set(filter(None, (x.get('origin') for x in raw_meta))))
            origin = cache_info.get('origin', 'unknown')
            if prev_origin and origin not in prev_origin:
                warnings.warn(
                    'Downloading cache tables from another origin '
                    f'("{origin}" instead of "{", ".join(prev_origin)}")')

            # Download the remote cache files
            _logger.info('Downloading remote caches...')
            files = self.alyx.download_cache_tables(cache_info.get('location'), self._tables_dir)
            assert any(files)
            super(OneAlyx, self).load_cache(self._tables_dir)  # Reload cache after download
        except (requests.exceptions.HTTPError, wc.HTTPError, requests.exceptions.SSLError) as ex:
            _logger.debug(ex)
            _logger.error(f'{type(ex).__name__}: Failed to load the remote cache file')
            self.mode = 'remote'
        except (ConnectionError, requests.exceptions.ConnectionError, URLError) as ex:
            # NB: URLError may be raised when client SSL configuration is bad
            _logger.debug(ex)
            _logger.error(f'{type(ex).__name__}: Failed to connect to Alyx')
            self.mode = 'local'
        except FileNotFoundError as ex:
            # NB: this error is only raised in online mode
            raise ex from FileNotFoundError(
                f'Cache directory not accessible: {tables_dir or self.cache_dir}\n'
                'Please provide valid tables_dir / cache_dir kwargs '
                'or run ONE.setup to update the default directory.'
            )

    @property
    def alyx(self):
        """one.webclient.AlyxClient: The Alyx Web client"""
        return self._web_client

    @property
    def cache_dir(self):
        """pathlib.Path: The location of the downloaded file cache"""
        return self._web_client.cache_dir

    @util.refresh
    def search_terms(self, query_type=None, endpoint=None):
        """
        Returns a list of search terms to be passed as kwargs to the search method.

        Parameters
        ----------
        query_type : str
            If 'remote', the search terms are largely determined by the REST endpoint used.
        endpoint: str
            If 'remote', specify the endpoint to search terms for.

        Returns
        -------
        tuple
            Tuple of search strings.
        """
        if (query_type or self.mode) != 'remote':
            if endpoint is None or endpoint == self._search_endpoint:
                return self._search_terms
            else:
                return

        endpoint = endpoint or self._search_endpoint
        # Return search terms from REST schema
        fields = self.alyx.rest_schemes[endpoint]['list']['fields']
        excl = ('lab',)  # 'laboratory' already in search terms
        if endpoint != 'sessions':
            return tuple(x['name'] for x in fields)
        return tuple({*self._search_terms, *(x['name'] for x in fields if x['name'] not in excl)})

    def describe_dataset(self, dataset_type=None):
        """Print a dataset type description.

        NB: This requires an Alyx database connection.

        Parameters
        ----------
        dataset_type : str
            A dataset type or dataset name.

        Returns
        -------
        dict
            The Alyx dataset type record.
        """
        assert self.mode != 'local' and not self.offline, 'Unable to connect to Alyx in local mode'
        if not dataset_type:
            return self.alyx.rest('dataset-types', 'list')
        try:
            assert isinstance(dataset_type, str) and not is_uuid_string(dataset_type)
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
    def list_datasets(self, eid=None, filename=None, collection=None, revision=None,
                      details=False, query_type=None) -> Union[np.ndarray, pd.DataFrame]:
        filters = dict(collection=collection, filename=filename, revision=revision)
        if (query_type or self.mode) != 'remote':
            return super().list_datasets(eid, details=details, query_type=query_type, **filters)
        elif not eid:
            warnings.warn('Unable to list all remote datasets')
            return super().list_datasets(eid, details=details, query_type=query_type, **filters)
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if not eid:
            return self._cache['datasets'].iloc[0:0] if details else []  # Return empty
        session, datasets = util.ses2records(self.alyx.rest('sessions', 'read', id=eid))
        # Add to cache tables
        self._update_cache_from_records(sessions=session, datasets=datasets.copy())
        if datasets is None or datasets.empty:
            return self._cache['datasets'].iloc[0:0] if details else []  # Return empty
        datasets = util.filter_datasets(
            datasets, assert_unique=False, wildcards=self.wildcards, **filters)
        # Return only the relative path
        return datasets if details else datasets['rel_path'].sort_values().values.tolist()

    def list_aggregates(self, relation: str, identifier: str = None,
                        dataset=None, revision=None, assert_unique=False):
        """
        List datasets aggregated over a given relation.

        Parameters
        ----------
        relation : str
            The thing over which the data were aggregated, e.g. 'subjects' or 'tags'.
        identifier : str
            The ID of the datasets, e.g. for data over subjects this would be lab/subject.
        dataset : str, dict, list
            Filters datasets and returns only the ones matching the filename.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        revision : str
            Filters datasets and returns only the ones matching the revision.
            Supports asterisks as wildcards.
        assert_unique : bool
            When true an error is raised if multiple collections or datasets are found.

        Returns
        -------
        pandas.DataFrame
            The matching aggregate dataset records.

        Examples
        --------
        List datasets aggregated over a specific subject's sessions

        >>> trials = one.list_aggregates('subjects', 'SP026')
        """
        query = 'session__isnull,True'  # ',data_repository_name__endswith,aggregates'
        all_aggregates = self.alyx.rest('datasets', 'list', django=query)
        records = (util.datasets2records(all_aggregates)
                   .reset_index(level=0)
                   .drop('eid', axis=1)
                   .rename_axis(index={'id': 'did'}))
        records['relation'] = records['rel_path'].map(lambda x: x.split('/')[1].lower())
        records = records[records['relation'] == relation.lower()]

        def path2id(p) -> str:
            """Extract identifier from relative path."""
            parts = alfiles.rel_path_parts(p)[0].split('/')
            idx = list(map(str.lower, parts)).index(relation.lower()) + 1
            return '/'.join(parts[idx:])

        records['identifier'] = records['rel_path'].map(path2id)
        if identifier is not None:
            # NB: We avoid exact matches as most users will only include subject, not lab/subject
            records = records[records['identifier'].str.contains(identifier)]

        # Add exists_aws field for download method
        for i, rec in records.iterrows():
            fr = next(x['file_records'] for x in all_aggregates if x['url'].endswith(i))
            records.loc[i, 'exists_aws'] = any(
                x['data_repository'].startswith('aws') and x['exists'] for x in fr)
        return util.filter_datasets(records, filename=dataset, revision=revision,
                                    wildcards=True, assert_unique=assert_unique)

    def load_aggregate(self, relation: str, identifier: str,
                       dataset=None, revision=None, download_only=False):
        """
        Load a single aggregated dataset for a given string identifier.

        Loads data aggregated over a relation such as subject, project or tag.

        Parameters
        ----------
        relation : str
            The thing over which the data were aggregated, e.g. 'subjects' or 'tags'.
        identifier : str
            The ID of the datasets, e.g. for data over subjects this would be lab/subject.
        dataset : str, dict, list
            Filters datasets and returns only the ones matching the filename.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        revision : str
            Filters datasets and returns only the ones matching the revision.
            Supports asterisks as wildcards.
        download_only : bool
            When true the data are downloaded and the file path is returned.

        Returns
        -------
        pandas.DataFrame, pathlib.Path
            Dataset or a Path object if download_only is true.

        Raises
        ------
        alferr.ALFObjectNotFound
            No datasets match the object, attribute or revision filters for this relation and
             identifier.
            Matching dataset was not found on disk (neither on the remote repository or locally).

        Examples
        --------
        Load a dataset aggregated over a specific subject's sessions

        >>> trials = one.load_aggregate('subjects', 'SP026', '_ibl_subjectTraining.table')

        """
        # If only two parts and wildcards are on, append ext wildcard
        if self.wildcards and isinstance(dataset, str) and len(dataset.split('.')) == 2:
            dataset += '.*'
            _logger.debug('Appending extension wildcard: ' + dataset)

        records = self.list_aggregates(relation, identifier,
                                       dataset=dataset, revision=revision, assert_unique=True)
        if records.empty:
            raise alferr.ALFObjectNotFound(
                f'{dataset or "dataset"} not found for {relation}/{identifier}')
        # update_exists=False because these datasets are not in the cache table
        file, = self._check_filesystem(records, update_exists=False)
        if not file:
            raise alferr.ALFObjectNotFound('Dataset file not found on disk')
        return file if download_only else alfio.load_file_content(file)

    @util.refresh
    def pid2eid(self, pid: str, query_type=None) -> (str, str):
        """
        Given an Alyx probe UUID string, returns the session id string and the probe label
        (i.e. the ALF collection).

        NB: Requires a connection to the Alyx database.

        Parameters
        ----------
        pid : str, uuid.UUID
            A probe UUID.
        query_type : str
            Query mode - options include 'remote', and 'refresh'.

        Returns
        -------
        str
            Experiment ID (eid).
        str
            Probe label.
        """
        query_type = query_type or self.mode
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplementedError('Converting probe IDs required remote connection')
        rec = self.alyx.rest('insertions', 'read', id=str(pid))
        return rec['session'], rec['name']

    @util.refresh
    def eid2pid(self, eid, query_type=None, details=False):
        """
        Given an experiment UUID (eID), returns the probe IDs and the probe labels
        (i.e. the ALF collection).

        NB: Requires a connection to the Alyx database.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        query_type : str
            Query mode - options include 'remote', and 'refresh'.
        details : bool
            Additionally return the complete Alyx records from insertions endpoint.

        Returns
        -------
        list of str
            Probe UUIDs (pID).
        list of str
            Probe labels, e.g. 'probe00'.
        list of dict (optional)
            If details is true, returns the Alyx records from insertions endpoint.
        """
        query_type = query_type or self.mode
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplementedError('Converting probe IDs required remote connection')
        eid = self.to_eid(eid)  # Ensure we have a UUID str
        if not eid:
            return (None,) * (3 if details else 2)
        recs = self.alyx.rest('insertions', 'list', session=eid)
        pids = [x['id'] for x in recs]
        labels = [x['name'] for x in recs]
        if details:
            return pids, labels, recs
        else:
            return pids, labels

    def search_insertions(self, details=False, query_type=None, **kwargs):
        """
        Searches insertions matching the given criteria and returns a list of matching probe IDs.

        For a list of search terms, use the method

            one.search_terms(query_type='remote', endpoint='insertions')

        All of the search parameters, apart from dataset and dataset type require a single value.
        For dataset and dataset type, a single value or a list can be provided. Insertions
        returned will contain all listed datasets.

        Parameters
        ----------
        session : str
            A session eid, returns insertions associated with the session.
        name: str
            An insertion label, returns insertions with specified name.
        lab : str
            A lab name, returns insertions associated with the lab.
        subject : str
            A subject nickname, returns insertions associated with the subject.
        task_protocol : str
            A task protocol name (can be partial, i.e. any task protocol containing that str
            will be found).
        project(s) : str
            The project name (can be partial, i.e. any task protocol containing that str
            will be found).
        dataset : str
            A (partial) dataset name. Returns sessions containing matching datasets.
            A dataset matches if it contains the search string e.g. 'wheel.position' matches
            '_ibl_wheel.position.npy'. C.f. `datasets` argument.
        datasets : str, list
            One or more exact dataset names. Returns insertions containing all these datasets.
        dataset_types : str, list
            One or more dataset_types (exact matching).
        details : bool
            If true also returns a dict of dataset details.
        query_type : str, None
            Query cache ('local') or Alyx database ('remote').
        limit : int
            The number of results to fetch in one go (if pagination enabled on server).

        Returns
        -------
        list
            List of probe IDs (pids).
        (list of dicts)
            If details is True, also returns a list of dictionaries, each entry corresponding to a
            matching insertion.

        Notes
        -----
        - This method does not use the local cache and therefore can not work in 'local' mode.

        Examples
        --------
        List the insertions associated with a given data release

        >>> tag = '2022_Q2_IBL_et_al_RepeatedSite'
        ... ins = one.search_insertions(django='datasets__tags__name,' + tag)
        """
        query_type = query_type or self.mode
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplementedError('Searching on insertions required remote connection')
        elif query_type == 'auto':
            _logger.debug('OneAlyx.search_insertions only supports remote queries')
            query_type = 'remote'
        # Get remote query params from REST endpoint
        search_terms = self.search_terms(query_type=query_type, endpoint='insertions')
        # Add some extra fields to keep compatibility with the search method
        search_terms += ('dataset', 'laboratory', 'number')
        params = {'django': kwargs.pop('django', '')}
        for key, value in sorted(kwargs.items()):
            field = util.autocomplete(key, search_terms)  # Validate and get full name
            # check that the input matches one of the defined filters
            if field == 'dataset':
                if not isinstance(value, str):
                    raise TypeError(
                        '"dataset" parameter must be a string. For lists use "datasets"')
                query = f'datasets__name__icontains,{value}'
                params['django'] += (',' if params['django'] else '') + query
            elif field == 'laboratory':
                params['lab'] = value
            elif field == 'number':
                params['experiment_number'] = value
            else:
                params[field] = value
        if not params['django']:
            params.pop('django')

        ins = self.alyx.rest('insertions', 'list', **params)
        pids = util.LazyId(ins)
        if not details:
            return pids

        return pids, ins

    def search(self, details=False, query_type=None, **kwargs):
        """
        Searches sessions matching the given criteria and returns a list of matching eids.

        For a list of search terms, use the method

            one.search_terms(query_type='remote')

        For all of the search parameters, a single value or list may be provided.  For `dataset`,
        the sessions returned will contain all listed datasets.  For the other parameters,
        the session must contain at least one of the entries.

        For all but `date_range` and `number`, any field that contains the search string is
        returned.  Wildcards are not permitted, however if wildcards property is True, regular
        expressions may be used (see notes and examples).

        Parameters
        ----------
        dataset : str
            A (partial) dataset name. Returns sessions containing matching datasets.
            A dataset matches if it contains the search string e.g. 'wheel.position' matches
            '_ibl_wheel.position.npy'. C.f. `datasets` argument.
        date_range : str, list, datetime.datetime, datetime.date, pandas.timestamp
            A single date to search or a list of 2 dates that define the range (inclusive).  To
            define only the upper or lower date bound, set the other element to None.
        lab : str, list
            A str or list of lab names, returns sessions from any of these labs (can be partial,
            i.e. any task protocol containing that str will be found).
        number : str, int
            Number of session to be returned, i.e. number in sequence for a given date.
        subject : str, list
            A list of subject nicknames, returns sessions for any of these subjects (can be
            partial, i.e. any task protocol containing that str will be found).
        task_protocol : str, list
            The task protocol name (can be partial, i.e. any task protocol containing that str
            will be found).
        project(s) : str, list
            The project name (can be partial, i.e. any task protocol containing that str
            will be found).
        performance_lte / performance_gte : float
            Search only for sessions whose performance is less equal or greater equal than a
            pre-defined threshold as a percentage (0-100).
        users : str, list
            A list of users.
        location : str, list
            A str or list of lab location (as per Alyx definition) name.
            Note: this corresponds to the specific rig, not the lab geographical location per se.
        dataset_types : str, list
            One or more of dataset_types.
        datasets : str, list
            One or more (exact) dataset names. Returns insertions containing all of these datasets.
        details : bool
            If true also returns a dict of dataset details.
        query_type : str, None
            Query cache ('local') or Alyx database ('remote').
        limit : int
            The number of results to fetch in one go (if pagination enabled on server).

        Returns
        -------
        list
            List of eids.
        (list of dicts)
            If details is True, also returns a list of dictionaries, each entry corresponding to a
            matching session.

        Examples
        --------
        Search for sessions with 'training' in the task protocol.

        >>> eids = one.search(task='training')

        Search for sessions by subject 'MFD_04'.

        >>> eids = one.search(subject='MFD_04')

        Do an exact search for sessions by subject 'FD_04'.

        >>> assert one.wildcards is True, 'the wildcards flag must be True for regex expressions'
        >>> eids = one.search(subject='^FD_04$', query_type='local')

        Search for sessions on a given date, in a given lab, containing trials and spike data.

        >>> eids = one.search(date='2023-01-01', lab='churchlandlab', dataset=['trials', 'spikes'])

        Notes
        -----
        - In default and local mode, most queries are case-sensitive partial matches. When lists
          are provided, the search is a logical OR, except for `datasets`, which is a logical AND.
        - All search terms are true for a session to be returned, i.e. subject matches AND project
          matches, etc.
        - In remote mode most queries are case-insensitive partial matches.
        - In default and local mode, when the one.wildcards flag is True (default), queries are
          interpreted as regular expressions. To turn this off set one.wildcards to False.
        - In remote mode regular expressions are only supported using the `django` argument.
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
                if not isinstance(value, str):
                    raise TypeError(
                        '"dataset" parameter must be a string. For lists use "datasets"')
                query = f'data_dataset_session_related__name__icontains,{value}'
                params['django'] += (',' if params['django'] else '') + query
            elif field == 'laboratory':
                params['lab'] = value
            else:
                params[field] = value
        if not params['django']:
            params.pop('django')
        # Make GET request
        ses = self.alyx.rest(self._search_endpoint, 'list', **params)
        # LazyId only transforms records when indexed
        eids = util.LazyId(ses)
        if not details:
            return eids

        def _add_date(records):
            """Add date field for compatibility with One.search output."""
            for s in util.ensure_list(records):
                s['date'] = datetime.fromisoformat(s['start_time']).date()
            return records

        return eids, util.LazyId(ses, func=_add_date)

    def _download_datasets(self, dsets, **kwargs) -> List[Path]:
        """
         Download a single or multitude of datasets if stored on AWS, otherwise calls
         OneAlyx._download_dataset.

         NB: This will not skip files that are already present.  Use check_filesystem instead.

         Parameters
         ----------
         dset : dict, str, pd.Series
             A single or multitude of dataset dictionaries.

         Returns
         -------
         pathlib.Path
             A local file path or list of paths.
         """
        # If all datasets exist on AWS, download from there.
        try:
            if 'exists_aws' in dsets and np.all(np.equal(dsets['exists_aws'].values, True)):
                _logger.info('Downloading from AWS')
                return self._download_aws(map(lambda x: x[1], dsets.iterrows()), **kwargs)
        except Exception as ex:
            _logger.debug(ex)
        return self._download_dataset(dsets, **kwargs)

    def _download_aws(self, dsets, update_exists=True, **_) -> List[Path]:
        # Download datasets from AWS
        import one.remote.aws as aws
        s3, bucket_name = aws.get_s3_from_alyx(self.alyx)
        assert self.mode != 'local'
        # Get all dataset URLs
        dsets = list(dsets)  # Ensure not generator
        uuids = [util.ensure_list(x.name)[-1] for x in dsets]
        # If number of UUIDs is too high, fetch in loop to avoid 414 HTTP status code
        remote_records = []
        N = 100  # Number of UUIDs per query
        for i in range(0, len(uuids), N):
            remote_records.extend(
                self.alyx.rest('datasets', 'list', exists=True, django=f'id__in,{uuids[i:i + N]}')
            )
        remote_records = sorted(remote_records, key=lambda x: uuids.index(x['url'].split('/')[-1]))
        out_files = []
        for dset, uuid, record in zip(dsets, uuids, remote_records):
            # Fetch file record path
            record = next((x for x in record['file_records']
                           if x['data_repository'].startswith('aws') and x['exists']), None)
            if not record and update_exists and 'exists_aws' in self._cache['datasets']:
                _logger.debug('Updating exists field')
                self._cache['datasets'].loc[(slice(None), uuid), 'exists_aws'] = False
                self._cache['_meta']['modified_time'] = datetime.now()
                out_files.append(None)
                continue
            source_path = PurePosixPath(record['data_repository_path'], record['relative_path'])
            source_path = alfiles.add_uuid_string(source_path, uuid)
            local_path = alfiles.remove_uuid_string(
                self.cache_dir.joinpath(dset['session_path'], dset['rel_path']))
            local_path.parent.mkdir(exist_ok=True, parents=True)
            out_files.append(aws.s3_download_file(
                source_path, local_path, s3=s3, bucket_name=bucket_name, overwrite=update_exists))
        return out_files

    def _dset2url(self, dset, update_cache=True):
        """
        Converts a dataset into a remote HTTP server URL.  The dataset may be one or more of the
        following: a dict from returned by the sessions endpoint or dataset endpoint, a record
        from the datasets cache table, or a file path.  Unlike record2url, this method can convert
        dicts and paths to URLs.

        Parameters
        ----------
        dset : dict, str, pd.Series, pd.DataFrame, list
            A single or multitude of dataset dictionary from an Alyx REST query OR URL string.
        update_cache : bool
            If True (default) and the dataset is from Alyx and cannot be converted to a URL,
            'exists' will be set to False in the corresponding entry in the cache table.

        Returns
        -------
        str
            The remote URL of the dataset.
        """
        did = None
        if isinstance(dset, str) and dset.startswith('http'):
            url = dset
        elif isinstance(dset, (str, Path)):
            url = self.path2url(dset)
            if not url:
                _logger.warning(f'Dataset {dset} not found in cache')
                return
        elif isinstance(dset, (list, tuple)):
            dset2url = partial(self._dset2url, update_cache=update_cache)
            return list(flatten(map(dset2url, dset)))
        else:
            # check if dset is dataframe, iterate over rows
            if hasattr(dset, 'iterrows'):
                dset2url = partial(self._dset2url, update_cache=update_cache)
                url = list(map(lambda x: dset2url(x[1]), dset.iterrows()))
            elif 'data_url' in dset:  # data_dataset_session_related dict
                url = dset['data_url']
                did = dset['id']
            elif 'file_records' not in dset:  # Convert dataset Series to alyx dataset dict
                url = self.record2url(dset)  # NB: URL will always be returned but may not exist
                did = util.ensure_list(dset.name)[-1]
            else:  # from datasets endpoint
                repo = getattr(getattr(self._web_client, '_par', None), 'HTTP_DATA_SERVER', None)
                url = next(
                    (fr['data_url'] for fr in dset['file_records']
                     if fr['data_url'] and fr['exists'] and
                     fr['data_url'].startswith(repo or fr['data_url'])), None)
                did = dset['url'][-36:]

        # Update cache if url not found
        if did is not None and not url and update_cache:
            _logger.debug('Updating cache')
            # NB: This will be considerably easier when IndexSlice supports Ellipsis
            idx = [slice(None)] * int(self._cache['datasets'].index.nlevels / 2)
            self._cache['datasets'].loc[(*idx, *util.ensure_list(did)), 'exists'] = False
            self._cache['_meta']['modified_time'] = datetime.now()

        return url

    def _download_dataset(self, dset, cache_dir=None, update_cache=True, **kwargs) -> List[Path]:
        """
        Download a single or multitude of dataset from an Alyx REST dictionary.

        NB: This will not skip files that are already present.  Use check_filesystem instead.

        Parameters
        ----------
        dset : dict, str, pd.Series, pd.DataFrame, list
            A single or multitude of dataset dictionary from an Alyx REST query OR URL string.
        cache_dir : str, pathlib.Path
            The root directory to save the data to (default taken from ONE parameters).
        update_cache : bool
            If true, the cache is updated when filesystem discrepancies are encountered.

        Returns
        -------
        pathlib.Path, list
            A local file path or list of paths.
        """
        cache_dir = cache_dir or self.cache_dir
        url = self._dset2url(dset, update_cache=update_cache)
        if not url:
            return
        if isinstance(url, str):
            target_dir = str(Path(cache_dir, alfiles.get_alf_path(url)).parent)
            return self._download_file(url, target_dir, **kwargs)
        # must be list of URLs
        valid_urls = list(filter(None, url))
        if not valid_urls:
            return [None] * len(url)

        target_dir = []
        for x in valid_urls:
            _path = urllib.parse.urlsplit(x, allow_fragments=False).path.strip('/')
            target_dir.append(str(Path(cache_dir, alfiles.get_alf_path(_path)).parent))
        files = self._download_file(valid_urls, target_dir, **kwargs)
        # Return list of file paths or None if we failed to extract URL from dataset
        return [None if not x else files.pop(0) for x in url]

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
            try:
                self.alyx.rest('files', 'partial_update',
                               id=fr[0]['url'][-36:], data={'json': json_field})
            except requests.exceptions.HTTPError as ex:
                warnings.warn(
                    f'Failed to tag remote file record mismatch: {ex}\n'
                    'Please contact the database administrator.')

    def _download_file(self, url, target_dir, keep_uuid=False, file_size=None, hash=None):
        """
        Downloads a single file or multitude of files from an HTTP webserver.
        The webserver in question is set by the AlyxClient object.

        Parameters
        ----------
        url : str, list
            An absolute or relative URL for a remote dataset.
        target_dir : str, list
            Absolute path of directory to download file to (including alf path).
        keep_uuid : bool
            If true, the UUID is not removed from the file name (default is False).
        file_size : int, list
            The expected file size or list of file sizes to compare with downloaded file.
        hash : str, list
            The expected file hash or list of file hashes to compare with downloaded file.

        Returns
        -------
        pathlib.Path
            The file path of the downloaded file or files.

        Example
        -------
        >>> file_path = OneAlyx._download_file(
        ...    'https://example.com/data.file.npy', '/home/Downloads/subj/1900-01-01/001/alf')
        """
        assert not self.offline
        # Ensure all target directories exist
        [Path(x).mkdir(parents=True, exist_ok=True) for x in set(util.ensure_list(target_dir))]

        # download file(s) from url(s), returns file path(s) with UUID
        local_path, md5 = self.alyx.download_file(url, target_dir=target_dir, return_md5=True)

        # check if url, hash, and file_size are lists
        if isinstance(url, (tuple, list)):
            assert (file_size is None) or len(file_size) == len(url)
            assert (hash is None) or len(hash) == len(url)
        for args in zip(*map(util.ensure_list, (file_size, md5, hash, local_path, url))):
            self._check_hash_and_file_size_mismatch(*args)

        # check if we are keeping the uuid on the list of file names
        if keep_uuid:
            return local_path

        # remove uuids from list of file names
        if isinstance(local_path, (list, tuple)):
            return [x.replace(alfiles.remove_uuid_string(x)) for x in local_path]
        else:
            return local_path.replace(alfiles.remove_uuid_string(local_path))

    def _check_hash_and_file_size_mismatch(self, file_size, hash, expected_hash, local_path, url):
        """
        Check to ensure the hash and file size of a downloaded file matches what is on disk

        Parameters
        ----------
        file_size : int
            The expected file size to compare with downloaded file
        hash : str
            The expected file hash to compare with downloaded file
        local_path: str
            The path of the downloaded file
        url : str
            An absolute or relative URL for a remote dataset
        """
        # verify hash size
        hash = hash or hashfile.md5(local_path)
        hash_mismatch = hash and hash != expected_hash
        # verify file size
        file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
        # check if there is a mismatch in hash or file_size
        if hash_mismatch or file_size_mismatch:
            # post download, if there is a mismatch between Alyx and the newly downloaded file size
            # or hash, flag the offending file record in Alyx for database for maintenance
            hash_mismatch = expected_hash and expected_hash != hash
            file_size_mismatch = file_size and Path(local_path).stat().st_size != file_size
            if hash_mismatch or file_size_mismatch:
                url = url or self.path2url(local_path)
                _logger.debug(f'Tagging mismatch for {url}')
                # tag the mismatched file records
                self._tag_mismatched_file_record(url)

    @staticmethod
    def setup(base_url=None, **kwargs):
        """
        Set up OneAlyx for a given database

        Parameters
        ----------
        base_url : str
            An Alyx database URL.  If None, the current default database is used.
        **kwargs
            Optional arguments to pass to one.params.setup.

        Returns
        -------
        OneAlyx
            An instance of OneAlyx for the newly set up database URL

        See Also
        --------
        one.params.setup
        """
        base_url = base_url or one.params.get_default_client()
        cache_map = one.params.setup(client=base_url, **kwargs)
        return OneAlyx(base_url=base_url or one.params.get(cache_map.DEFAULT).ALYX_URL)

    @util.refresh
    @util.parse_id
    def eid2path(self, eid, query_type=None) -> util.Listable(Path):
        """
        From an experiment ID gets the local session path

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict, list
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        query_type : str
            If set to 'remote', will force database connection

        Returns
        -------
        pathlib.Path, list
            A session path or list of session paths
        """
        # first try avoid hitting the database
        mode = query_type or self.mode
        if mode != 'remote':
            cache_path = super().eid2path(eid)
            if cache_path or mode == 'local':
                return cache_path

        # If eid is a list recurse through it and return a list
        if isinstance(eid, list):
            unwrapped = unwrap(self.path2eid)
            return [unwrapped(self, e, query_type='remote') for e in eid]

        # if it wasn't successful, query Alyx
        ses = self.alyx.rest('sessions', 'list', django=f'pk,{eid}')
        if len(ses) == 0:
            return None
        else:
            return Path(self.cache_dir).joinpath(
                ses[0]['lab'], 'Subjects', ses[0]['subject'], ses[0]['start_time'][:10],
                str(ses[0]['number']).zfill(3))

    @util.refresh
    def path2eid(self, path_obj: Union[str, Path], query_type=None) -> util.Listable(Path):
        """
        From a local path, gets the experiment ID

        Parameters
        ----------
        path_obj : str, pathlib.Path, list
            Local path or list of local paths
        query_type : str
            If set to 'remote', will force database connection

        Returns
        -------
        str, list
            An eid or list of eids
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

        session_path = alfiles.get_session_path(path_obj)
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
    def path2url(self, filepath, query_type=None) -> str:
        """
        Given a local file path, returns the URL of the remote file.

        Parameters
        ----------
        filepath : str, pathlib.Path
            A local file path
        query_type : str
            If set to 'remote', will force database connection

        Returns
        -------
        str
            A URL string
        """
        query_type = query_type or self.mode
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
    def type2datasets(self, eid, dataset_type, details=False):
        """
        Get list of datasets belonging to a given dataset type for a given session

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        dataset_type : str, list
            An Alyx dataset type, e.g. camera.times or a list of dtypes
        details : bool
            If True, a datasets DataFrame is returned

        Returns
        -------
        np.ndarray, dict
            A numpy array of data, or DataFrame if details is true
        """
        assert self.mode != 'local' and not self.offline, 'Unable to connect to Alyx in local mode'
        if isinstance(dataset_type, str):
            restriction = f'session__id,{eid},dataset_type__name,{dataset_type}'
        elif isinstance(dataset_type, collections.abc.Sequence):
            restriction = f'session__id,{eid},dataset_type__name__in,{dataset_type}'
        else:
            raise TypeError('dataset_type must be a str or str list')
        datasets = util.datasets2records(self.alyx.rest('datasets', 'list', django=restriction))
        return datasets if details else datasets['rel_path'].sort_values().values

    def dataset2type(self, dset) -> str:
        """Return dataset type from dataset.

        NB: Requires an Alyx database connection

        Parameters
        ----------
        dset : str, np.ndarray, tuple
            A dataset name, dataset uuid or dataset integer id

        Returns
        -------
        str
            The dataset type
        """
        assert self.mode != 'local' and not self.offline, 'Unable to connect to Alyx in local mode'
        # Ensure dset is a str uuid
        if isinstance(dset, str) and not is_uuid_string(dset):
            dset = self._dataset_name2id(dset)
        if isinstance(dset, np.ndarray):
            dset = parquet.np2str(dset)[0]
        if isinstance(dset, tuple) and all(isinstance(x, int) for x in dset):
            dset = parquet.np2str(np.array(dset))
        if not is_uuid_string(dset):
            raise ValueError('Unrecognized name or UUID')
        return self.alyx.rest('datasets', 'read', id=dset)['dataset_type']

    def describe_revision(self, revision, full=False):
        """Print description of a revision

        Parameters
        ----------
        revision : str
            The name of the revision (without '#')
        full : bool
            If true, returns the matching record

        Returns
        -------
        None, dict
            None if full is false or no record found, otherwise returns record as dict
        """
        assert self.mode != 'local' and not self.offline, 'Unable to connect to Alyx in local mode'
        try:
            rec = self.alyx.rest('revisions', 'read', id=revision)
            print(rec['description'])
            if full:
                return rec
        except requests.exceptions.HTTPError as ex:
            if ex.response.status_code != 404:
                raise ex
            print(f'revision "{revision}" not found')

    def _dataset_name2id(self, dset_name, eid=None):
        # TODO finish function
        datasets = self.list_datasets(eid) if eid else self._cache['datasets']
        # Get ID of fist matching dset
        for idx, rel_path in datasets['rel_path'].items():
            if rel_path.endswith(dset_name):
                return idx[-1]  # (eid, did)
        raise ValueError(f'Dataset {dset_name} not found in cache')

    @util.refresh
    @util.parse_id
    def get_details(self, eid: str, full: bool = False, query_type=None):
        """Return session details for a given session.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict, list
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        full : bool
            If True, returns a DataFrame of session and dataset info.
        query_type : {'local', 'refresh', 'auto', 'remote'}
            The query mode - if 'local' the details are taken from the cache tables; if 'remote'
            the details are returned from the sessions REST endpoint; if 'auto' uses whichever
            mode ONE is in; if 'refresh' reloads the cache before querying.

        Returns
        -------
        pd.Series, pd.DataFrame, dict
            in local mode - a session record or full DataFrame with dataset information if full is
            True; in remote mode - a full or partial session dict.

        Raises
        ------
        ValueError
            Invalid experiment ID (failed to parse into eid string).
        requests.exceptions.HTTPError
            [Errno 404] Remote session not found on Alyx.
        """
        if (query_type or self.mode) == 'local':
            return super().get_details(eid, full=full)
        # If eid is a list of eIDs recurse through list and return the results
        if isinstance(eid, list):
            details_list = []
            for p in eid:
                details_list.append(self.get_details(p, full=full))
            return details_list
        # load all details
        dets = self.alyx.rest('sessions', 'read', eid)
        if full:
            return dets
        # If it's not full return the normal output like from a one.search
        det_fields = ['subject', 'start_time', 'number', 'lab', 'projects',
                      'url', 'task_protocol', 'local_path']
        out = {k: v for k, v in dets.items() if k in det_fields}
        out['projects'] = ','.join(out['projects'])
        out.update({'local_path': self.eid2path(eid),
                    'date': datetime.fromisoformat(out['start_time']).date()})
        return out


def _setup(**kwargs):
    """A setup method for the main ONE function.

    Clears the ONE LRU cache before running setup, which ensures ONE is re-instantiated after
    modifying parameter defaults.  NB: This docstring is overwritten by the one.params.setup
    docstring upon module init.

    Parameters
    ----------
    **kwargs
        See one.params.setup.

    Returns
    -------
    IBLParams
        An updated cache map.
    """
    ONE.cache_clear()
    kwargs['client'] = kwargs.pop('base_url', None)
    return one.params.setup(**kwargs)


ONE.setup = _setup
ONE.setup.__doc__ = one.params.setup.__doc__
ONE.version = __version__
