"""Classes for searching, listing and (down)loading ALyx Files."""
import collections.abc
import urllib.parse
import warnings
import logging
from weakref import WeakMethod
from datetime import datetime, timedelta
from functools import lru_cache, partial
from inspect import unwrap
from pathlib import Path, PurePosixPath
from typing import Any, Union, Optional, List
from uuid import UUID
from urllib.error import URLError
import os
import re

import pandas as pd
import numpy as np
import requests.exceptions
import packaging.version

from iblutil.io import parquet, hashfile
from iblutil.io.params import FileLock
from iblutil.util import Bunch, flatten, ensure_list, Listable

import one.params
import one.webclient as wc
import one.alf.io as alfio
import one.alf.path as alfiles
import one.alf.exceptions as alferr
from one.alf.path import ALFPath
from .alf.cache import (
    make_parquet_db, load_tables, remove_table_files, merge_tables,
    default_cache, cast_index_object)
from .alf.spec import is_uuid, is_uuid_string, QC, to_alf
from . import __version__
from one.converters import ConversionMixin, session_record2path, ses2records, datasets2records
from one import util

_logger = logging.getLogger(__name__)
__all__ = ['ONE', 'One', 'OneAlyx']
SAVE_ON_DELETE = (os.environ.get('ONE_SAVE_ON_DELETE') or '1').casefold() in ('true', '1')
"""bool: Whether to save modified cache tables on delete."""

_logger.debug('ONE_SAVE_ON_DELETE: %s', SAVE_ON_DELETE)


class One(ConversionMixin):
    """An API for searching and loading data on a local filesystem."""

    _search_terms = (
        'datasets', 'date_range', 'laboratory', 'number',
        'projects', 'subject', 'task_protocol', 'dataset_qc_lte'
    )

    uuid_filenames = None
    """bool: whether datasets on disk have a UUID in their filename."""

    def __init__(self, cache_dir=None, mode='local', wildcards=True, tables_dir=None):
        """An API for searching and loading data on a local filesystem.

        Parameters
        ----------
        cache_dir : str, Path
            Path to the data files.  If Alyx parameters have been set up for this location,
            an OneAlyx instance is returned.  If data_dir and base_url are None, the default
            location is used.
        mode : str
            Query mode, options include 'local' (offline) and 'remote' (online).  Most methods
            have a `query_type` parameter that can override the class mode.
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
        self.mode = mode
        self.wildcards = wildcards  # Flag indicating whether to use regex or wildcards
        self.record_loaded = False
        # assign property here as different instances may work on separate filesystems
        self.uuid_filenames = False
        # init the cache file
        self._reset_cache()
        if self.mode == 'local':
            # Ensure that we don't call any subclass method here as we only load local cache
            # tables on init. Direct calls to load_cache can be made by the user or subclass.
            One.load_cache(self)
        elif self.mode != 'remote':
            raise ValueError(f'Mode "{self.mode}" not recognized')

    def __repr__(self):
        return f'One ({"off" if self.offline else "on"}line, {self.cache_dir})'

    def __del__(self):
        """Save cache tables to disk before deleting the object."""
        if SAVE_ON_DELETE:
            self.save_cache()

    @property
    def offline(self):
        """bool: True if mode is local or no Web client set."""
        return self.mode == 'local' or not getattr(self, '_web_client', False)

    def search_terms(self, query_type=None) -> tuple:
        """List the search term keyword args for use in the search method."""
        return self._search_terms

    def _reset_cache(self):
        """Replace the cache object with a Bunch that contains the right fields."""
        self._cache = default_cache()

    def _remove_table_files(self, tables=None):
        """Delete cache tables on disk.

        Parameters
        ----------
        tables : list of str
            A list of table names to removes, e.g. ['sessions', 'datasets'].
            If None, the currently loaded table names are removed. NB: This
            will also delete the cache_info.json metadata file.

        Returns
        -------
        list of pathlib.Path
            A list of the removed files.

        """
        tables = tables or filter(lambda x: x[0] != '_', self._cache)
        return remove_table_files(self._tables_dir, tables)

    def load_cache(self, tables_dir=None, clobber=True, **kwargs):
        """Load parquet cache files from a local directory.

        Parameters
        ----------
        tables_dir : str, pathlib.Path
            An optional directory location of the parquet files, defaults to One._tables_dir.
        clobber : bool
            If true, the cache is loaded without merging with existing table files.

        Returns
        -------
        datetime.datetime
            A timestamp of when the cache was loaded.
        """
        if clobber:
            self._reset_cache()
        else:
            raise NotImplementedError('clobber=False not implemented yet')
        self._tables_dir = Path(tables_dir or self._tables_dir or self.cache_dir)
        self._cache = load_tables(self._tables_dir)

        if self._cache['_meta']['loaded_time'] is None:
            # No tables present
            if self.offline:  # In online mode, the cache tables should be downloaded later
                warnings.warn(f'No cache tables found in {self._tables_dir}')

        # If in remote mode and loading old tables generated on Alyx,
        # prompt the user to delete them to improve load times
        raw_meta = self._cache['_meta'].get('raw', {}).values() or [{}]
        tagged = any(filter(None, flatten(x.get('database_tags') for x in raw_meta)))
        origin = set(filter(None, flatten(ensure_list(x.get('origin', [])) for x in raw_meta)))
        older = (self._cache['_meta']['created_time'] or datetime.now()) < datetime(2025, 2, 13)
        remote = not self.offline and self.mode == 'remote'
        if remote and origin == {'alyx'} and older and not self._web_client.silent and not tagged:
            message = ('Old Alyx cache tables detected on disk. '
                      'It\'s recomended to remove these tables as they '
                      'negatively affect performance.\nDelete these tables? [Y/n]: ')
            if (input(message).casefold().strip() or 'y')[0] == 'y':
                self._remove_table_files()
                self._reset_cache()
        elif len(self._cache.datasets) > 1e6:
            warnings.warn(
                'Large cache tables affect performance. '
                'Consider removing them by calling the `_remove_table_files` method.')

        return self._cache['_meta']['loaded_time']

    def save_cache(self, save_dir=None, clobber=False):
        """Save One._cache attribute into parquet tables if recently modified.

        Checks if another process is writing to file, if so waits before saving.

        Parameters
        ----------
        save_dir : str, pathlib.Path
            The directory path into which the tables are saved.  Defaults to cache directory.
        clobber : bool
            If true, the cache is saved without merging with existing table files, regardless of
            modification time.

        """
        TIMEOUT = 5  # Delete lock file this many seconds after creation/modification or waiting
        save_dir = Path(save_dir or self.cache_dir)
        caches = self._cache
        meta = caches['_meta']
        modified = meta.get('modified_time') or datetime.min
        update_time = max(meta.get(x) or datetime.min for x in ('loaded_time', 'saved_time'))
        all_empty = all(x.empty for x in self._cache.values() if isinstance(x, pd.DataFrame))

        if not clobber:
            if modified < update_time or all_empty:
                return  # Not recently modified; return
            # Merge existing tables with new data
            _logger.debug('Merging cache tables...')
            caches = load_tables(save_dir)
            merge_tables(
                caches, **{k: v for k, v in self._cache.items() if not k.startswith('_')})
            # Ensure we use the minimum created date for each table
            for table in caches['_meta']['raw']:
                raw_meta = [x['_meta']['raw'].get(table, {}) for x in (caches, self._cache)]
                created = filter(None, (x.get('date_created') for x in raw_meta))
                if any(created := list(created)):
                    created = min(map(datetime.fromisoformat, created))
                    created = created.isoformat(sep=' ', timespec='minutes')
                    meta['raw'][table]['date_created'] = created

        with FileLock(save_dir, log=_logger, timeout=TIMEOUT, timeout_action='delete'):
            _logger.info('Saving cache tables...')
            for table in filter(lambda x: not x[0] == '_', caches.keys()):
                metadata = meta['raw'].get(table, {})
                if isinstance(metadata.get('origin'), set):
                    metadata['origin'] = list(metadata['origin'])
                metadata['date_modified'] = modified.isoformat(sep=' ', timespec='minutes')
                filename = save_dir.joinpath(f'{table}.pqt')
                # Cast indices to str before saving
                df = cast_index_object(caches[table].copy(), str)
                parquet.save(filename, df, metadata)
                _logger.debug(f'Saved {filename}')
            meta['saved_time'] = datetime.now()

    def save_loaded_ids(self, sessions_only=False, clear_list=True):
        """Save list of UUIDs corresponding to datasets or sessions where datasets were loaded.

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

    def _download_datasets(self, dsets, **kwargs) -> List[ALFPath]:
        """Download several datasets given a set of datasets.

        NB: This will not skip files that are already present.  Use check_filesystem instead.

        Parameters
        ----------
        dsets : list
            List of dataset dictionaries from an Alyx REST query OR URL strings.

        Returns
        -------
        list of one.alf.path.ALFPath
            A local file path list.

        """
        # Looking to entirely remove method
        pass  # pragma: no cover

    def _download_dataset(self, dset, cache_dir=None, **kwargs) -> ALFPath:
        """Download a dataset from an Alyx REST dictionary.

        Parameters
        ----------
        dset : pandas.Series, dict, str
            A single dataset dictionary from an Alyx REST query OR URL string.
        cache_dir : str, pathlib.Path
            The root directory to save the data in (home/downloads by default).

        Returns
        -------
        one.alf.path.ALFPath
            The local file path.

        """
        pass  # pragma: no cover

    def search(self, details=False, **kwargs):
        """Searches sessions matching the given criteria and returns a list of matching eids.

        For a list of search terms, use the method

            one.search_terms()

        For all search parameters, a single value or list may be provided.  For `dataset`, the
        sessions returned will contain all listed datasets.  For the other parameters, the session
        must contain at least one of the entries.

        For all but `date_range` and `number`, any field that contains the search string is
        returned.  Wildcards are not permitted, however if wildcards property is True, regular
        expressions may be used (see notes and examples).

        Parameters
        ----------
        datasets : str, list
            One or more (exact) dataset names. Returns sessions containing all of these datasets.
        dataset_qc_lte : str, int, one.alf.spec.QC
            A dataset QC value, returns sessions with datasets at or below this QC value, including
            those with no QC set.  If `dataset` not passed, sessions with any passing QC datasets
            are returned, otherwise all matching datasets must have the QC value or below.
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

        Returns
        -------
        list of UUID
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

        >>> eids = one.search(
        ...    date='2023-01-01', lab='churchlandlab',
        ...    datasets=['trials.table.pqt', 'spikes.times.npy'])

        Search for sessions containing trials and spike data where QC for both are WARNING or less.

        >>> eids = one.search(dataset_qc_lte='WARNING', dataset=['trials', 'spikes'])

        Search for sessions with any datasets that have a QC of PASS or NOT_SET.

        >>> eids = one.search(dataset_qc_lte='PASS')

        Notes
        -----
        - In default and local mode, most queries are case-sensitive partial matches. When lists
          are provided, the search is a logical OR, except for `datasets`, which is a logical AND.
        - If `dataset_qc` and `datasets` are defined, the QC criterion only applies to the provided
          datasets and all must pass for a session to be returned.
        - All search terms are true for a session to be returned, i.e. subject matches AND project
          matches, etc.
        - In remote mode most queries are case-insensitive partial matches.
        - In default and local mode, when the one.wildcards flag is True (default), queries are
          interpreted as regular expressions. To turn this off set one.wildcards to False.
        - In remote mode regular expressions are only supported using the `django` argument.

        """

        def all_present(x, dsets, exists=True):
            """Returns true if all datasets present in Series."""
            name = x.str.rsplit('/', n=1, expand=True).iloc[:, -1]
            return all(any(name.str.fullmatch(y) & exists) for y in dsets)

        # Iterate over search filters, reducing the sessions table
        sessions = self._cache['sessions']

        # Ensure sessions filtered in a particular order, with datasets last
        search_order = ('date_range', 'number', 'datasets')

        def sort_fcn(itm):
            return -1 if itm[0] not in search_order else search_order.index(itm[0])

        # Validate and get full name for queries
        search_terms = self.search_terms(query_type='local')
        kwargs.pop('query_type', None)  # used by subclasses
        queries = {util.autocomplete(k, search_terms): v for k, v in kwargs.items()}
        for key, value in sorted(queries.items(), key=sort_fcn):
            # No matches; short circuit
            if sessions.size == 0:
                return ([], None) if details else []
            # String fields
            elif key in ('subject', 'task_protocol', 'laboratory', 'projects'):
                query = '|'.join(ensure_list(value))
                key = 'lab' if key == 'laboratory' else key
                mask = sessions[key].str.contains(query, regex=self.wildcards)
                sessions = sessions[mask.astype(bool, copy=False)]
            elif key == 'date_range':
                start, end = util.validate_date_range(value)
                session_date = pd.to_datetime(sessions['date'])
                sessions = sessions[(session_date >= start) & (session_date <= end)]
            elif key == 'number':
                query = ensure_list(value)
                sessions = sessions[sessions[key].isin(map(int, query))]
            # Dataset/QC check is biggest so this should be done last
            elif key == 'datasets' or (key == 'dataset_qc_lte' and 'datasets' not in queries):
                datasets = self._cache['datasets']
                qc = QC.validate(queries.get('dataset_qc_lte', 'FAIL')).name  # validate value
                has_dset = sessions.index.isin(datasets.index.get_level_values('eid'))
                if not has_dset.any():
                    sessions = sessions.iloc[0:0]  # No datasets for any sessions
                    continue
                datasets = datasets.loc[(sessions.index.values[has_dset], ), :]
                query = ensure_list(value if key == 'datasets' else '')
                # For each session check any dataset both contains query and exists
                mask = (
                    (datasets
                        .groupby('eid', sort=False)
                        .apply(lambda x: all_present(
                            x['rel_path'], query, x['exists'] & x['qc'].le(qc))
                        ))
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
            return eids, sessions.reset_index(drop=True).to_dict('records', into=Bunch)
        else:
            return eids

    def _search_insertions(self, details=False, **kwargs):
        """Search insertions matching the given criteria and return a list of matching probe IDs.

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
        dataset : str, list
            One or more dataset names. Returns sessions containing all these datasets.
            A dataset matches if it contains the search string e.g. 'wheel.position' matches
            '_ibl_wheel.position.npy'.
        dataset_qc_lte : int, str, one.alf.spec.QC
            The maximum QC value for associated datasets.
        details : bool
            If true also returns a dict of dataset details.

        Returns
        -------
        list of UUID
            List of probe IDs (pids).
        (list of dicts)
            If details is True, also returns a list of dictionaries, each entry corresponding to a
            matching insertion.

        Notes
        -----
        - This method does not use the local cache and therefore can not work in 'local' mode.

        Examples
        --------
        List the insertions associated with a given subject

        >>> ins = one.search_insertions(subject='SWC_043')

        """
        # Warn if no insertions table present
        if (insertions := self._cache.get('insertions')) is None:
            warnings.warn('No insertions data loaded.')
            return ([], None) if details else []

        # Validate and get full names
        search_terms = ('model', 'name', 'json', 'serial', 'chronic_insertion')
        search_terms += self._search_terms
        kwargs.pop('query_type', None)  # used by subclasses
        arguments = {util.autocomplete(key, search_terms): value for key, value in kwargs.items()}
        # Apply session filters first
        session_kwargs = {k: v for k, v in arguments.items() if k in self._search_terms}
        if session_kwargs:
            eids = self.search(**session_kwargs, details=False, query_type='local')
            insertions = insertions.loc[eids]
        # Apply insertion filters
        # Iterate over search filters, reducing the insertions table
        for key, value in sorted(filter(lambda x: x[0] not in session_kwargs, kwargs.items())):
            if insertions.size == 0:
                return ([], None) if details else []
            # String fields
            elif key in ('model', 'serial', 'name'):
                query = '|'.join(ensure_list(value))
                mask = insertions[key].str.contains(query, regex=self.wildcards)
                insertions = insertions[mask.astype(bool, copy=False)]
            else:
                raise NotImplementedError(key)

        # Return results
        if insertions.size == 0:
            return ([], None) if details else []
        # Sort insertions
        eids = insertions.index.get_level_values('eid').unique()
        # NB: This will raise if no session in cache; may need to improve error handling here
        sessions = self._cache['sessions'].loc[eids, ['date', 'subject', 'number']]
        insertions = (insertions
                      .join(sessions, how='inner')
                      .sort_values(['date', 'subject', 'number', 'name'], ascending=False))
        pids = insertions.index.get_level_values('id').to_list()

        if details:  # TODO replicate Alyx records here
            return pids, insertions.reset_index(drop=True).to_dict('records', into=Bunch)
        else:
            return pids

    def _check_filesystem(self, datasets, offline=None, update_exists=True, check_hash=True):
        """Update the local filesystem for the given datasets.

        Given a set of datasets, check whether records correctly reflect the filesystem.
        Called by load methods, this returns a list of file paths to load and return.
        This changes datasets frame, calls _update_cache(sessions=None, datasets=None) to
        update and save tables.  Download_datasets may also call this function.

        Parameters
        ----------
        datasets : pandas.Series, pandas.DataFrame, list of dicts
            A list or DataFrame of dataset records.
        offline : bool, None
            If false and Web client present, downloads the missing datasets from a remote
            repository.
        update_exists : bool
            If true, the cache is updated to reflect the filesystem.
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.

        Returns
        -------
        A list of one.alf.path.ALFPath for the datasets (None elements for non-existent datasets).

        """
        if isinstance(datasets, pd.Series):
            datasets = pd.DataFrame([datasets])
            assert datasets.index.nlevels <= 2
            idx_names = ['eid', 'id'] if datasets.index.nlevels == 2 else ['id']
            datasets.index.set_names(idx_names, inplace=True)
        elif not isinstance(datasets, pd.DataFrame):
            # Cast set of dicts (i.e. from REST datasets endpoint)
            datasets = datasets2records(list(datasets))
        elif datasets.empty:
            return []
        else:
            datasets = datasets.copy()
        indices_to_download = []  # indices of datasets that need (re)downloading
        files = []  # file path list to return
        # If the session_path field is missing from the datasets table, fetch from sessions table
        # Typically only aggregate frames contain this column
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
            file = ALFPath(self.cache_dir, *rec[['session_path', 'rel_path']])
            if self.uuid_filenames:
                file = file.with_uuid(i[1] if isinstance(i, tuple) else i)
            if file.exists():
                # Check if there's a hash mismatch
                # If so, add this index to list of datasets that need downloading
                if rec['file_size'] and file.stat().st_size != rec['file_size']:
                    _logger.warning('local file size mismatch on dataset: %s',
                                    PurePosixPath(rec.session_path, rec.rel_path))
                    indices_to_download.append(i)
                elif check_hash and rec['hash'] is not None:
                    if hashfile.md5(file) != rec['hash']:
                        _logger.warning('local md5 mismatch on dataset: %s',
                                        PurePosixPath(rec.session_path, rec.rel_path))
                        indices_to_download.append(i)
                files.append(file)  # File exists so add to file list
            else:
                # File doesn't exist so add None to output file list
                files.append(None)
                # Add this index to list of datasets that need downloading
                indices_to_download.append(i)

        # If online and we have datasets to download, call download_datasets with these datasets
        if not (offline or self.offline) and indices_to_download:
            dsets_to_download = datasets.loc[indices_to_download]
            # Returns list of local file paths and set to variable
            new_files = self._download_datasets(dsets_to_download, update_cache=update_exists)
            # Add each downloaded file to the output list of files
            for i, file in zip(indices_to_download, new_files):
                files[datasets.index.get_loc(i)] = file

        # NB: Currently if not offline and a remote file is missing, an exception will be raised
        # before we reach this point. This could change in the future.
        exists = list(map(bool, files))
        if not all(datasets['exists'] == exists):
            with warnings.catch_warnings():
                # Suppress future warning: exist column should always be present
                msg = '.*indexing on a MultiIndex with a nested sequence of labels.*'
                warnings.filterwarnings('ignore', message=msg)
                datasets['exists'] = exists
                if update_exists:
                    _logger.debug('Updating exists field')
                    i = datasets.index
                    if i.nlevels == 1:
                        # eid index level missing in datasets input
                        i = pd.IndexSlice[:, i]
                    self._cache['datasets'].loc[i, 'exists'] = exists
                    self._cache['_meta']['modified_time'] = datetime.now()

        if self.record_loaded:
            loaded = np.fromiter(map(bool, files), bool)
            loaded_ids = datasets.index.get_level_values('id')[loaded].to_numpy()
            if '_loaded_datasets' not in self._cache:
                self._cache['_loaded_datasets'] = np.unique(loaded_ids)
            else:
                loaded_set = np.hstack([self._cache['_loaded_datasets'], loaded_ids])
                self._cache['_loaded_datasets'] = np.unique(loaded_set)

        # Return full list of file paths
        return files

    @util.parse_id
    def get_details(self, eid: Union[str, Path, UUID], full: bool = False):
        """Return session details for a given session ID.

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

    def list_subjects(self) -> List[str]:
        """List all subjects in database.

        Returns
        -------
        list
            Sorted list of subject names

        """
        return self._cache['sessions']['subject'].sort_values().unique().tolist()

    def list_datasets(
            self, eid=None, filename=None, collection=None, revision=None, qc=QC.FAIL,
            ignore_qc_not_set=False, details=False, query_type=None, default_revisions_only=False,
            keep_eid_index=False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Given an eid, return the datasets for those sessions.

        If no eid is provided, a list of all datasets is returned.  When details is false, a sorted
        array of unique datasets is returned (their relative paths).

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
        qc : str, int, one.alf.spec.QC
            Returns datasets at or below this QC level.  Integer values should correspond to the QC
            enumeration NOT the qc category column codes in the pandas table.
        ignore_qc_not_set : bool
            When true, do not return datasets for which QC is NOT_SET.
        details : bool
            When true, a pandas DataFrame is returned, otherwise a numpy array of
            relative paths (collection/revision/filename) - see one.alf.spec.describe for details.
        query_type : str
            Query cache ('local') or Alyx database ('remote').
        default_revisions_only : bool
            When true, only matching datasets that are considered default revisions are returned.
            If no 'default_revision' column is present, and ALFError is raised.
        keep_eid_index : bool
            If details is true, this determines whether the returned data frame contains the eid
            in the index. When false (default) the returned data frame index is the dataset id
            only, otherwise the index is a MultIndex with levels (eid, id).

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
        if default_revisions_only:
            if 'default_revision' not in datasets.columns:
                raise alferr.ALFError('No default revisions specified')
            datasets = datasets[datasets['default_revision']]

        filter_args = dict(
            collection=collection, filename=filename, wildcards=self.wildcards, revision=revision,
            revision_last_before=False, assert_unique=False, qc=qc,
            ignore_qc_not_set=ignore_qc_not_set)
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
        if details:
            if keep_eid_index and datasets.index.nlevels == 1:
                # Reinstate eid index
                datasets = pd.concat({eid: datasets}, names=['eid'])
            # Return the full data frame
            return datasets
        else:
            # Return only the relative path
            return datasets['rel_path'].sort_values().values.tolist()

    def list_collections(self, eid=None, filename=None, collection=None, revision=None,
                         details=False, query_type=None) -> Union[np.ndarray, dict]:
        """List the collections for a given experiment.

        If no experiment ID is given, all collections are returned.

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

    def list_revisions(self, eid=None, filename=None, collection=None, revision=None,
                       details=False, query_type=None):
        """List the revisions for a given experiment.

        If no experiment id is given, all collections are returned.

        Parameters
        ----------
        eid : str, UUID, Path, dict
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        filename : str, dict, list
            Filters datasets and returns only the revisions containing matching datasets.
            Supports lists asterisks as wildcards.  May be a dict of ALF parts.
        collection : str, list
            Filter by a given collection. Supports asterisks as wildcards.
        revision : str, list
            Filter by a given pattern. Supports asterisks as wildcards.
        details : bool
            If true a dict of pandas datasets tables is returned with collections as keys,
            otherwise a numpy array of unique collections.
        query_type : str
            Query cache ('local') or Alyx database ('remote').

        Returns
        -------
        list, dict
            A list of unique collections or dict of datasets tables.

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

    @util.parse_id
    def load_object(self,
                    eid: Union[str, Path, UUID],
                    obj: str,
                    collection: Optional[str] = None,
                    revision: Optional[str] = None,
                    query_type: Optional[str] = None,
                    download_only: bool = False,
                    check_hash: bool = True,
                    **kwargs) -> Union[alfio.AlfBunch, List[ALFPath]]:
        """Load all attributes of an ALF object from a Session ID and an object name.

        Any datasets with matching object name will be loaded.

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
            Query cache ('local') or Alyx database ('remote').
        download_only : bool
            When true the data are downloaded and the file path is returned. NB: The order of the
            file path list is undefined.
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.
        kwargs
            Additional filters for datasets, including namespace and timescale. For full list
            see the :func:`one.alf.spec.describe` function.

        Returns
        -------
        one.alf.io.AlfBunch, list
            An ALF bunch or if download_only is True, a list of one.alf.path.ALFPath objects.

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
        datasets = self.list_datasets(
            eid, details=True, query_type=query_type, keep_eid_index=True)

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
        offline = self.mode == 'local'
        files = self._check_filesystem(datasets, offline=offline, check_hash=check_hash)
        files = [x for x in files if x]
        if not files:
            raise alferr.ALFObjectNotFound(f'ALF object "{obj}" not found on disk')

        if download_only:
            return files

        return alfio.load_object(files, wildcards=self.wildcards, **kwargs)

    @util.parse_id
    def load_dataset(self,
                     eid: Union[str, Path, UUID],
                     dataset: str,
                     collection: Optional[str] = None,
                     revision: Optional[str] = None,
                     query_type: Optional[str] = None,
                     download_only: bool = False,
                     check_hash: bool = True) -> Any:
        """Load a single dataset for a given session id and dataset name.

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
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.

        Returns
        -------
        np.ndarray, one.alf.path.ALFPath
            Dataset or a ALFPath object if download_only is true.

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
        datasets = self.list_datasets(
            eid, details=True, query_type=query_type or self.mode, keep_eid_index=True)
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
        offline = self.mode == 'local'
        file, = self._check_filesystem(datasets, offline=offline, check_hash=check_hash)

        if not file:
            raise alferr.ALFObjectNotFound('Dataset not found')
        elif download_only:
            return file
        return alfio.load_file_content(file)

    @util.parse_id
    def load_datasets(self,
                      eid: Union[str, Path, UUID],
                      datasets: List[str],
                      collections: Optional[str] = None,
                      revisions: Optional[str] = None,
                      query_type: Optional[str] = None,
                      assert_present=True,
                      download_only: bool = False,
                      check_hash: bool = True) -> Any:
        """Load datasets for a given session id.

        Returns two lists the length of datasets.  The first is the data (or file paths if
        download_data is false), the second is a list of meta data Bunches.  If assert_present is
        false, missing data will be returned as None.

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
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.

        Returns
        -------
        list
            A list of data (or file paths) the length of datasets.
        list
            A list of meta data Bunches. If assert_present is False, missing data will be None.

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
        - To load an exact revision (i.e. not the last revision before a given date), pass in
          a list of relative paths or a data frame.

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
            """Ensure specifiers lists matching datasets length."""
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

        # Check if rel paths have been used (e.g. the output of list_datasets)
        is_frame = isinstance(datasets, pd.DataFrame)
        if is_rel_paths := (is_frame or any('/' in x for x in datasets)):
            if not (collections, revisions) == (None, None):
                raise ValueError(
                    'collection and revision kwargs must be None when dataset is a relative path')
            if is_frame:
                if 'eid' in datasets.index.names:
                    assert set(datasets.index.get_level_values('eid')) == {eid}
                datasets = datasets['rel_path'].tolist()
            datasets = list(map(partial(alfiles.rel_path_parts, as_dict=True), datasets))
            if len(datasets) > 0:
                # Extract collection and revision from each of the parsed datasets
                # None -> '' ensures exact collections and revisions are used in filter
                # NB: f user passes in dicts, any collection/revision keys will be ignored.
                collections, revisions = zip(
                    *((x.pop('collection') or '', x.pop('revision') or '') for x in datasets)
                )

        # Short circuit
        query_type = query_type or self.mode
        all_datasets = self.list_datasets(
            eid, details=True, query_type=query_type, keep_eid_index=True)
        if len(all_datasets) == 0:
            if assert_present:
                raise alferr.ALFObjectNotFound(f'No datasets found for session {eid}')
            else:
                _logger.warning(f'No datasets found for session {eid}')
                return None, all_datasets
        if len(datasets) == 0:
            return None, all_datasets.iloc[0:0]  # Return empty

        # More input validation
        input_types = [(isinstance(x, str), isinstance(x, dict)) for x in datasets]
        if not all(map(any, input_types)) or not any(map(all, zip(*input_types))):
            raise ValueError('`datasets` must be iterable of only str or only dicts')
        if self.wildcards and input_types[0][0]:  # if wildcards and input is iter of str
            # Append extension wildcard if 'object.attribute' string
            datasets = [
                x + ('.*' if isinstance(x, str) and len(x.split('.')) == 2 else '')
                for x in datasets
            ]

        # Check input args
        collections, revisions = _verify_specifiers([collections, revisions])

        # If collections provided in datasets list, e.g. [collection/x.y.z], do not assert unique
        # If not a dataframe, use revision last before (we've asserted no revision in rel_path)
        ops = dict(
            wildcards=self.wildcards, assert_unique=True, revision_last_before=not is_rel_paths)
        slices = [util.filter_datasets(all_datasets, x, y, z, **ops)
                  for x, y, z in zip(datasets, collections, revisions)]
        present = [len(x) == 1 for x in slices]
        present_datasets = pd.concat(slices)

        # Check if user is blindly downloading all data and warn of non-default revisions
        if 'default_revision' in present_datasets and \
                is_rel_paths and not all(present_datasets['default_revision']):
            old = present_datasets.loc[~present_datasets['default_revision'], 'rel_path'].to_list()
            warnings.warn(
                'The following datasets may have been revised and ' +
                'are therefore not recommended for analysis:\n\t' +
                '\n\t'.join(old) + '\n'
                'To avoid this warning, specify the revision as a kwarg or use load_dataset.',
                alferr.ALFWarning
            )

        if not all(present):
            missing_list = (x if isinstance(x, str) else to_alf(**x) for x in datasets)
            missing_list = ('/'.join(filter(None, [c, f'#{r}#' if r else None, d]))
                            for c, r, d in zip(collections, revisions, missing_list))
            missing_list = ', '.join(x for x, y in zip(missing_list, present) if not y)
            message = f'The following datasets are not in the cache: {missing_list}'
            if assert_present:
                raise alferr.ALFObjectNotFound(message)
            else:
                _logger.warning(message)

        # Check files exist / download remote files
        offline = self.mode == 'local'
        files = self._check_filesystem(present_datasets, offline=offline, check_hash=check_hash)

        if any(x is None for x in files):
            missing_list = ', '.join(x for x, y in zip(present_datasets.rel_path, files) if not y)
            message = f'The following datasets were not downloaded: {missing_list}'
            if assert_present:
                raise alferr.ALFObjectNotFound(message)
            else:
                _logger.warning(message)

        # Make list of metadata Bunches out of the table
        records = (present_datasets
                   .reset_index(names=['eid', 'id'])
                   .to_dict('records', into=Bunch))

        # Ensure result same length as input datasets list
        files = [None if not here else files.pop(0) for here in present]
        # Replace missing file records with None
        records = [None if not here else records.pop(0) for here in present]
        if download_only:
            return files, records
        return [alfio.load_file_content(x) for x in files], records

    def load_dataset_from_id(self,
                             dset_id: Union[str, UUID],
                             download_only: bool = False,
                             details: bool = False,
                             check_hash: bool = True) -> Any:
        """Load a dataset given a dataset UUID.

        Parameters
        ----------
        dset_id : uuid.UUID, str
            A dataset UUID to load.
        download_only : bool
            If true the dataset is downloaded (if necessary) and the filepath returned.
        details : bool
            If true a pandas Series is returned in addition to the data.
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.

        Returns
        -------
        np.ndarray, one.alf.path.ALFPath
            Dataset data (or filepath if download_only) and dataset record if details is True.

        """
        if isinstance(dset_id, str):
            dset_id = UUID(dset_id)
        elif not isinstance(dset_id, UUID):
            dset_id, = parquet.np2uuid(dset_id)
        try:
            dataset = self._cache['datasets'].loc[(slice(None), dset_id), :].squeeze()
            assert isinstance(dataset, pd.Series) or len(dataset) == 1
        except AssertionError:
            raise alferr.ALFMultipleObjectsFound('Duplicate dataset IDs')
        except KeyError:
            raise alferr.ALFObjectNotFound('Dataset not found')

        filepath, = self._check_filesystem(dataset, check_hash=check_hash)
        if not filepath:
            raise alferr.ALFObjectNotFound('Dataset not found')
        output = filepath if download_only else alfio.load_file_content(filepath)
        if details:
            return output, dataset
        else:
            return output

    @util.parse_id
    def load_collection(self,
                        eid: Union[str, Path, UUID],
                        collection: str,
                        object: Optional[str] = None,
                        revision: Optional[str] = None,
                        query_type: Optional[str] = None,
                        download_only: bool = False,
                        check_hash: bool = True,
                        **kwargs) -> Union[Bunch, List[ALFPath]]:
        """Load all objects in an ALF collection from a Session ID.

        Any datasets with matching object name(s) will be loaded.  Returns a bunch of objects.

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
        check_hash : bool
            Consider dataset missing if local file hash does not match. In online mode, the dataset
            will be re-downloaded.
        kwargs
            Additional filters for datasets, including namespace and timescale. For full list
            see the one.alf.spec.describe function.

        Returns
        -------
        Bunch of one.alf.io.AlfBunch, list of one.alf.path.ALFPath
            A Bunch of objects or if download_only is True, a list of ALFPath objects.

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
        datasets = self.list_datasets(
            eid, details=True, collection=collection, query_type=query_type, keep_eid_index=True)

        if len(datasets) == 0:
            raise alferr.ALFError(f'{collection} not found for session {eid}')

        dataset = {'object': object, **kwargs}
        datasets = util.filter_datasets(datasets, dataset, revision,
                                        assert_unique=False, wildcards=self.wildcards)

        # Validate result before loading
        if len(datasets) == 0:
            raise alferr.ALFObjectNotFound(object or '')
        parts = [alfiles.rel_path_parts(x) for x in datasets.rel_path]

        # For those that don't exist, download them
        offline = self.mode == 'local'
        files = self._check_filesystem(datasets, offline=offline, check_hash=check_hash)
        if not any(files):
            raise alferr.ALFObjectNotFound(f'ALF collection "{collection}" not found on disk')
        # Remove missing items
        files, parts = zip(*[(x, y) for x, y in zip(files, parts) if x])

        if download_only:
            return files

        unique_objects = set(x[3] or '' for x in parts)
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
        kwargs
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
def ONE(*, mode='remote', wildcards=True, **kwargs):
    """ONE API factory.

    Determine which class to instantiate depending on parameters passed.

    Parameters
    ----------
    mode : str
        Query mode, options include 'local' (offline) and 'remote' (online only).  Most
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
    """An API for searching and loading data through the Alyx database."""

    def __init__(self, username=None, password=None, base_url=None, cache_dir=None,
                 mode='remote', wildcards=True, tables_dir=None, **kwargs):
        """An API for searching and loading data through the Alyx database.

        Parameters
        ----------
        mode : str
            Query mode, options include 'local' (offline) and 'remote' (online only).  Most
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
        """Load parquet cache files.

        Queries the database for the location and creation date of the remote cache.  If newer, it
        will be download and loaded.

        Parameters
        ----------
        tables_dir : str, pathlib.Path
            An optional directory location of the parquet files, defaults to One._tables_dir.
        clobber : bool
            If True, query Alyx for a newer cache even if current (local) cache is recent.
        tag : str
            An optional Alyx dataset tag for loading cache tables containing a subset of datasets.

        Returns
        -------
        datetime.datetime
            A timestamp of when the cache was loaded.

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
            # Load any present cache
            super(OneAlyx, self).load_cache(tables_dir, clobber=True)
            cache_meta = self._cache.get('_meta', {})
            raw_meta = cache_meta.get('raw', {}).values() or [{}]

        try:
            # Determine whether a newer cache is available
            cache_info = self.alyx.get(f'cache/info/{tag or ""}'.strip('/'), expires=True)
            assert tag is None or tag in cache_info.get('database_tags', [])

            # Check version compatibility
            min_version = packaging.version.parse(cache_info.get('min_api_version', '0.0.0'))
            if packaging.version.parse(one.__version__) < min_version:
                warnings.warn(f'Newer cache tables require ONE version {min_version} or greater')
                return cache_meta['loaded_time']

            # Check whether remote cache more recent
            remote_created = datetime.fromisoformat(cache_info['date_created'])
            local_created = cache_meta.get('created_time', None)
            fresh = local_created and (remote_created - local_created) < timedelta(minutes=1)
            # The local cache may have been created locally more recently, but if it doesn't
            # contain the same tag or origin, we need to download the remote one.
            origin = cache_info.get('origin', 'unknown')
            local_origin = (x.get('origin', []) for x in raw_meta)
            local_origin = set(flatten(map(ensure_list, local_origin)))
            different_origin = origin not in local_origin
            if fresh and not (different_tag or different_origin):
                _logger.info('No newer cache available')
                return cache_meta['loaded_time']

            # Set the cache table directory location
            if tables_dir:  # If tables directory specified, use that
                self._tables_dir = Path(tables_dir)
            elif different_tag:  # Otherwise use a subdirectory for a given tag
                self._tables_dir = self.cache_dir / tag
                self._tables_dir.mkdir(exist_ok=True)
            else:  # Otherwise use the previous location (default is the data cache directory)
                self._tables_dir = self._tables_dir or self.cache_dir

            # Check if the origin has changed. This is to warn users if downloading from a
            # different database to the one currently loaded. When building the cache from
            # remote queries the origin is set to the Alyx database URL. If the cache info
            # origin name and URL are different, warn the user.
            if different_origin and local_origin and self.alyx.base_url not in local_origin:
                warnings.warn(
                    'Downloading cache tables from another origin '
                    f'("{origin}" instead of "{", ".join(local_origin)}")')

            # Download the remote cache files
            _logger.info('Downloading remote caches...')
            files = self.alyx.download_cache_tables(cache_info.get('location'), self._tables_dir)
            assert any(files)
            # Reload cache after download
            loaded_time = super(OneAlyx, self).load_cache(self._tables_dir)
            # Add db URL to origin set so we know where the cache came from
            for raw_meta in self._cache['_meta']['raw'].values():
                table_origin = set(filter(None, ensure_list(raw_meta.get('origin', []))))
                if origin in table_origin:
                    table_origin.add(self.alyx.base_url)
                raw_meta['origin'] = table_origin
            return loaded_time
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
        return cache_meta['loaded_time']

    @property
    def alyx(self):
        """one.webclient.AlyxClient: The Alyx Web client."""
        return self._web_client

    @property
    def cache_dir(self):
        """pathlib.Path: The location of the downloaded file cache."""
        return self._web_client.cache_dir

    def search_terms(self, query_type=None, endpoint=None):
        """Returns a list of search terms to be passed as kwargs to the search method.

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
            out = self.alyx.rest('dataset-types', 'read', id=dataset_type)
        except (AssertionError, requests.exceptions.HTTPError):
            # Try to get dataset type from dataset name
            out = self.alyx.rest('dataset-types', 'read', id=self.dataset2type(dataset_type))
        finally:
            _logger.disabled = False
        print(out['description'])
        return out

    def list_datasets(
            self, eid=None, filename=None, collection=None, revision=None, qc=QC.FAIL,
            ignore_qc_not_set=False, details=False, query_type=None, default_revisions_only=False,
            keep_eid_index=False
    ) -> Union[np.ndarray, pd.DataFrame]:
        filters = dict(
            collection=collection, filename=filename, revision=revision, qc=qc,
            ignore_qc_not_set=ignore_qc_not_set, default_revisions_only=default_revisions_only)
        if (query_type or self.mode) != 'remote':
            return super().list_datasets(eid, details=details, keep_eid_index=keep_eid_index,
                                         query_type=query_type, **filters)
        elif not eid:
            warnings.warn('Unable to list all remote datasets')
            return super().list_datasets(eid, details=details, keep_eid_index=keep_eid_index,
                                         query_type=query_type, **filters)
        eid = self.to_eid(eid)  # Ensure we have a UUID str list
        if not eid:
            return self._cache['datasets'].iloc[0:0] if details else []  # Return empty
        session, datasets = ses2records(self.alyx.rest('sessions', 'read', id=eid))
        # Add to cache tables
        merge_tables(
            self._cache, sessions=session, datasets=datasets.copy(), origin=self.alyx.base_url)
        if datasets is None or datasets.empty:
            return self._cache['datasets'].iloc[0:0] if details else []  # Return empty
        assert set(datasets.index.unique('eid')) == {eid}
        del filters['default_revisions_only']
        if not keep_eid_index and 'eid' in datasets.index.names:
            datasets = datasets.droplevel('eid')
        kwargs = dict(assert_unique=False, wildcards=self.wildcards, revision_last_before=False)
        datasets = util.filter_datasets(datasets, **kwargs, **filters)
        # Return only the relative path
        return datasets if details else datasets['rel_path'].sort_values().values.tolist()

    def list_aggregates(self, relation: str, identifier: str = None,
                        dataset=None, revision=None, assert_unique=False):
        """List datasets aggregated over a given relation.

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
        records = datasets2records(all_aggregates).droplevel('eid')
        # Since rel_path for public FI file records starts with 'public/aggregates' instead of just
        # 'aggregates', we should discard the file path parts before 'aggregates' (if present)
        records['rel_path'] = records['rel_path'].str.replace(
            r'^[\w\/]+(?=aggregates\/)', '', n=1, regex=True)
        # The relation is the first part after 'aggregates', i.e. the second part
        records['relation'] = records['rel_path'].map(
            lambda x: x.split('aggregates')[-1].split('/')[1].casefold())
        records = records[records['relation'] == relation.casefold()]

        def path2id(p) -> str:
            """Extract identifier from relative path."""
            parts = alfiles.rel_path_parts(p)[0].split('/')
            idx = list(map(str.casefold, parts)).index(relation.casefold()) + 1
            return '/'.join(parts[idx:])

        records['identifier'] = records['rel_path'].map(path2id)
        if identifier is not None:
            # NB: We avoid exact matches as most users will only include subject, not lab/subject
            records = records[records['identifier'].str.contains(identifier)]

        return util.filter_datasets(records, filename=dataset, revision=revision,
                                    wildcards=True, assert_unique=assert_unique)

    def load_aggregate(self, relation: str, identifier: str,
                       dataset=None, revision=None, download_only=False):
        """Load a single aggregated dataset for a given string identifier.

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
        pandas.DataFrame, one.alf.path.ALFPath
            Dataset or a ALFPath object if download_only is true.

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
        records['session_path'] = ''  # explicitly add session path column
        file, = self._check_filesystem(records, update_exists=False)
        if not file:
            raise alferr.ALFObjectNotFound('Dataset file not found on disk')
        return file if download_only else alfio.load_file_content(file)

    def pid2eid(self, pid: str, query_type=None) -> (UUID, str):
        """Given an Alyx probe UUID string, return the session ID and probe label.

        NB: Requires a connection to the Alyx database.

        Parameters
        ----------
        pid : str, UUID
            A probe UUID.
        query_type : str
            Query mode - options include 'remote', and 'refresh'.

        Returns
        -------
        uuid.UUID
            Experiment ID (eid).
        str
            Probe label.

        """
        query_type = query_type or self.mode
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplementedError('Converting probe IDs required remote connection')
        rec = self.alyx.rest('insertions', 'read', id=str(pid))
        return UUID(rec['session']), rec['name']

    def eid2pid(self, eid, query_type=None, details=False, **kwargs) -> (UUID, str, list):
        """Given an experiment UUID (eID), return the probe IDs and labels (i.e. ALF collection).

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
        kwargs
            Additional parameters to filter insertions Alyx endpoint.

        Returns
        -------
        list of UUID
            Probe UUIDs (pID).
        list of str
            Probe labels, e.g. 'probe00'.
        list of dict (optional)
            If details is true, returns the Alyx records from insertions endpoint.

        Examples
        --------
        Get the probe IDs and details for a given session ID

        >>> pids, labels, recs = one.eid2pid(eid, details=True)

        Get the probe ID for a given session ID and label

        >>> (pid,), _ = one.eid2pid(eid, details=False, name='probe00')
        """
        query_type = query_type or self.mode
        if query_type == 'local' and 'insertions' not in self._cache.keys():
            raise NotImplementedError('Converting probe IDs required remote connection')
        eid = self.to_eid(eid)  # Ensure we have a UUID str
        if not eid:
            return (None,) * (3 if details else 2)
        recs = self.alyx.rest('insertions', 'list', session=eid, **kwargs)
        pids = [UUID(x['id']) for x in recs]
        labels = [x['name'] for x in recs]
        if details:
            return pids, labels, recs
        else:
            return pids, labels

    def search_insertions(self, details=False, query_type=None, **kwargs):
        """Search insertions matching the given criteria and return a list of matching probe IDs.

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
        dataset_qc_lte : int, str, one.alf.spec.QC
            The maximum QC value for associated datasets.
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
        list of UUID
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
        if query_type == 'local':
            return super()._search_insertions(details=details, query_type=query_type, **kwargs)
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
        # Update cache table with results
        if isinstance(ins, list):  # not a paginated response
            if len(ins) > 0:
                self._update_insertions_table(ins)
            pids = util.LazyId.ses2eid(ins)  # immediately extract UUIDs
        else:
            # populate first page
            self._update_insertions_table(ins._cache[:ins.limit])
            # Add callback for updating cache on future fetches
            ins.add_callback(WeakMethod(self._update_insertions_table))
            pids = util.LazyId(ins)

        if not details:
            return pids

        return pids, ins

    def _update_insertions_table(self, insertions_records):
        """Update the insertions tables with a list of insertions records.

        Parameters
        ----------
        insertions_records : list of dict
            A list of insertions records from the /insertions list endpoint.

        Returns
        -------
        datetime.datetime
            A timestamp of when the cache was updated.

        """
        df = (pd.DataFrame(insertions_records)
              .drop(['session_info'], axis=1)
              .rename({'session': 'eid'}, axis=1)
              .set_index(['eid', 'id'])
              .sort_index())
        # Cast indices to UUID
        df = cast_index_object(df, UUID)

        if 'insertions' not in self._cache:
            self._cache['insertions'] = df.iloc[0:0]
        # Build sessions table
        session_records = (x['session_info'] for x in insertions_records)
        sessions_df = pd.DataFrame(next(zip(*map(ses2records, session_records))))
        return merge_tables(
            self._cache, insertions=df, sessions=sessions_df, origin=self.alyx.base_url)

    def search(self, details=False, query_type=None, **kwargs):
        """Searches sessions matching the given criteria and returns a list of matching eids.

        For a list of search terms, use the method

            one.search_terms(query_type='remote')

        For all search parameters, a single value or list may be provided.  For `dataset`,
        the sessions returned will contain all listed datasets.  For the other parameters,
        the session must contain at least one of the entries.

        For all but `date_range` and `number`, any field that contains the search string is
        returned.  Wildcards are not permitted, however if wildcards property is True, regular
        expressions may be used (see notes and examples).

        Parameters
        ----------
        datasets : str, list
            One or more (exact) dataset names. Returns sessions containing all of these datasets.
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
            One or more of dataset_types. Unlike with `datasets`, the dataset types for the
            sessions returned may not be reachable (i.e. for recent sessions the datasets may not
            yet be available).
        dataset_qc_lte : int, str, one.alf.spec.QC
            The maximum QC value for associated datasets. NB: Without `datasets`, not all
            associated datasets with the matching QC values are guarenteed to be reachable.
        details : bool
            If true also returns a dict of dataset details.
        query_type : str, None
            Query cache ('local') or Alyx database ('remote').
        limit : int
            The number of results to fetch in one go (if pagination enabled on server).

        Returns
        -------
        list of UUID
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
        - In remote mode, only the `datasets` argument returns sessions where datasets are
          registered *and* exist. Using `dataset_types` or `dataset_qc_lte` without `datasets`
          will not check that the datasets are reachable.

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

        # Update cache table with results
        if isinstance(ses, list):  # not a paginated response
            if len(ses) > 0:
                self._update_sessions_table(ses)
            eids = util.LazyId.ses2eid(ses)
        else:
            # populate first page
            self._update_sessions_table(ses._cache[:ses.limit])
            # Add callback for updating cache on future fetches
            ses.add_callback(WeakMethod(self._update_sessions_table))
            # LazyId only transforms records when indexed
            eids = util.LazyId(ses)

        if not details:
            return eids

        def _add_date(records):
            """Add date field for compatibility with One.search output."""
            for s in ensure_list(records):
                s['date'] = datetime.fromisoformat(s['start_time']).date()
            return records

        # Return LazyId object only if paginated response
        return eids, _add_date(ses) if isinstance(ses, list) else util.LazyId(ses, func=_add_date)

    def _update_sessions_table(self, session_records):
        """Update the sessions tables with a list of session records.

        Parameters
        ----------
        session_records : list of dict
            A list of session records from the /sessions list endpoint.

        Returns
        -------
        datetime.datetime
            A timestamp of when the cache was updated.

        """
        df = pd.DataFrame(next(zip(*map(ses2records, session_records))))
        return merge_tables(self._cache, sessions=df, origin=self.alyx.base_url)

    def _download_datasets(self, dsets, **kwargs) -> List[ALFPath]:
        """Download a single or multitude of datasets if stored on AWS.

        Falls back to :meth:`OneAlyx._download_dataset` if call to :meth:`OneAlyx._download_aws`
        fails.

        NB: This will not skip files that are already present.  Use check_filesystem instead.

        Parameters
        ----------
        dset : dict, str, pandas.Series, pandas.DataFrame
            A single or multitude of dataset dictionaries. For AWS downloads the input must be a
            data frame.

        Returns
        -------
        list of one.alf.path.ALFPath
            A list of local file paths.

        """
        # determine whether to remove the UUID after download, this may be overridden by user
        kwargs['keep_uuid'] = kwargs.get('keep_uuid', self.uuid_filenames)

        # If all datasets exist on AWS, download from there.
        try:
            if not isinstance(dsets, pd.DataFrame):
                raise TypeError('Input datasets must be a pandas data frame for AWS download.')
            assert 'exists_aws' not in dsets or np.all(np.equal(dsets['exists_aws'].values, True))
            _logger.debug('Downloading from AWS')
            files = self._download_aws(map(lambda x: x[1], dsets.iterrows()), **kwargs)
            # Trigger fallback download of any files missing on AWS
            assert all(files), f'{sum(map(bool, files))} datasets not found on AWS'
            return files
        except Exception as ex:
            _logger.debug(ex)
        return self._download_dataset(dsets, **kwargs)

    def _download_aws(self, dsets, update_exists=True, keep_uuid=None, **_) -> List[ALFPath]:
        """Download datasets from an AWS S3 instance using boto3.

        Parameters
        ----------
        dsets : list of pandas.Series
            An iterable for datasets as a pandas Series.
        update_exists : bool
            If true, the 'exists_aws' field of the cache table is set to False for any missing
            datasets.
        keep_uuid : bool
            If false, the dataset UUID is removed from the downloaded filename. If None, the
            `uuid_filenames` attribute determined whether the UUID is kept (default is false).

        Returns
        -------
        list of one.alf.path.ALFPath
            A list the length of `dsets` of downloaded dataset file paths. Missing datasets are
            returned as None.

        See Also
        --------
        one.remote.aws.s3_download_file - The AWS download function.

        """
        # Download datasets from AWS
        import one.remote.aws as aws
        s3, bucket_name = aws.get_s3_from_alyx(self.alyx)
        assert self.mode != 'local'
        # Get all dataset URLs
        dsets = list(dsets)  # Ensure not generator
        uuids = [str(ensure_list(x.name)[-1]) for x in dsets]
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
            if not record:
                if update_exists and 'exists_aws' in self._cache['datasets']:
                    _logger.debug('Updating exists field')
                    self._cache['datasets'].loc[(slice(None), UUID(uuid)), 'exists_aws'] = False
                    self._cache['_meta']['modified_time'] = datetime.now()
                out_files.append(None)
                continue
            if 'relation' in dset:
                # For non-session datasets the pandas record rel path is the full path
                matches = dset['rel_path'].endswith(record['relative_path'])
            else:
                # For session datasets the pandas record rel path is relative to the session
                matches = record['relative_path'].endswith(dset['rel_path'])
            assert matches, f'Relative path for dataset {uuid} does not match Alyx record'
            source_path = PurePosixPath(record['data_repository_path'], record['relative_path'])
            local_path = self.cache_dir.joinpath(alfiles.get_alf_path(source_path))
            # Add UUIDs to filenames, if required
            source_path = alfiles.add_uuid_string(source_path, uuid)
            if keep_uuid is True or (keep_uuid is None and self.uuid_filenames is True):
                local_path = alfiles.add_uuid_string(local_path, uuid)
            local_path.parent.mkdir(exist_ok=True, parents=True)
            out_files.append(aws.s3_download_file(
                source_path, local_path, s3=s3, bucket_name=bucket_name, overwrite=update_exists))
        return [ALFPath(x) if x else x for x in out_files]

    def _dset2url(self, dset, update_cache=True):
        """Converts a dataset into a remote HTTP server URL.

        The dataset may be one or more of the following: a dict from returned by the sessions
        endpoint or dataset endpoint, a record from the datasets cache table, or a file path.
        Unlike :meth:`ConversionMixin.record2url`, this method can convert dicts and paths to
        URLs.

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
            try:
                url = self.path2url(dset)
            except alferr.ALFObjectNotFound:
                _logger.warning(f'Dataset {dset} not found')
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
                did = UUID(dset['id'])
            elif 'file_records' not in dset:  # Convert dataset Series to alyx dataset dict
                url = self.record2url(dset)  # NB: URL will always be returned but may not exist
                did = ensure_list(dset.name)[-1]
            else:  # from datasets endpoint
                repo = getattr(getattr(self._web_client, '_par', None), 'HTTP_DATA_SERVER', None)
                url = next(
                    (fr['data_url'] for fr in dset['file_records']
                     if fr['data_url'] and fr['exists'] and
                     fr['data_url'].startswith(repo or fr['data_url'])), None)
                did = UUID(dset['url'][-36:])

        # Update cache if url not found
        if did is not None and not url and update_cache:
            _logger.debug('Updating cache')
            # NB: This will be considerably easier when IndexSlice supports Ellipsis
            idx = [slice(None)] * int(self._cache['datasets'].index.nlevels / 2)
            self._cache['datasets'].loc[(*idx, *ensure_list(did)), 'exists'] = False
            self._cache['_meta']['modified_time'] = datetime.now()

        return url

    def _download_dataset(
            self, dset, cache_dir=None, update_cache=True, **kwargs) -> List[ALFPath]:
        """Download a single or multitude of dataset from an Alyx REST dictionary.

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
        list of one.alf.path.ALFPath
            A local file path or list of paths.

        """
        cache_dir = cache_dir or self.cache_dir
        url = self._dset2url(dset, update_cache=update_cache)
        if not url:
            return
        if isinstance(url, str):
            target_dir = str(Path(cache_dir, alfiles.get_alf_path(url)).parent)
            file = self._download_file(url, target_dir, **kwargs)
            return ALFPath(file) if file else None
        # must be list of URLs
        valid_urls = list(filter(None, url))
        if not valid_urls:
            return [None] * len(url)

        target_dir = []
        for x in valid_urls:
            _path = urllib.parse.urlsplit(x, allow_fragments=False).path.strip('/')
            # Since rel_path for public FI file records starts with 'public/aggregates' instead of
            # 'aggregates', we should discard the file path parts before 'aggregates' (if present)
            _path = re.sub(r'^[\w\/]+(?=aggregates\/)', '', _path, count=1)
            target_dir.append(str(Path(cache_dir, alfiles.get_alf_path(_path)).parent))
        files = self._download_file(valid_urls, target_dir, **kwargs)
        # Return list of file paths or None if we failed to extract URL from dataset
        return [None if not x else ALFPath(files.pop(0)) for x in url]

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

    def _download_file(self, url, target_dir, keep_uuid=None, file_size=None, hash=None):
        """Downloads a single file or multitude of files from an HTTP webserver.

        The webserver in question is set by the AlyxClient object.

        Parameters
        ----------
        url : str, list
            An absolute or relative URL for a remote dataset.
        target_dir : str, list
            Absolute path of directory to download file to (including alf path).
        keep_uuid : bool
            If true, the UUID is not removed from the file name.  See `uuid_filenames' property.
        file_size : int, list
            The expected file size or list of file sizes to compare with downloaded file.
        hash : str, list
            The expected file hash or list of file hashes to compare with downloaded file.

        Returns
        -------
        pathlib.Path or list of pathlib.Path
            The file path of the downloaded file or files.

        Example
        -------
        >>> file_path = OneAlyx._download_file(
        ...    'https://example.com/data.file.npy', '/home/Downloads/subj/1900-01-01/001/alf')

        """
        assert not self.offline
        # Ensure all target directories exist
        [Path(x).mkdir(parents=True, exist_ok=True) for x in set(ensure_list(target_dir))]

        # download file(s) from url(s), returns file path(s) with UUID
        local_path, md5 = self.alyx.download_file(url, target_dir=target_dir, return_md5=True)

        # check if url, hash, and file_size are lists
        if isinstance(url, (tuple, list)):
            assert (file_size is None) or len(file_size) == len(url)
            assert (hash is None) or len(hash) == len(url)
        for args in zip(*map(ensure_list, (file_size, md5, hash, local_path, url))):
            self._check_hash_and_file_size_mismatch(*args)

        # check if we are keeping the uuid on the list of file names
        if keep_uuid is True or (keep_uuid is None and self.uuid_filenames):
            return list(local_path) if isinstance(local_path, tuple) else local_path

        # remove uuids from list of file names
        if isinstance(local_path, (list, tuple)):
            return [x.replace(alfiles.remove_uuid_string(x)) for x in local_path]
        else:
            return local_path.replace(alfiles.remove_uuid_string(local_path))

    def _check_hash_and_file_size_mismatch(self, file_size, hash, expected_hash, local_path, url):
        """Check to ensure the hash and file size of a downloaded file matches what is on disk.

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
        """Set up OneAlyx for a given database.

        Parameters
        ----------
        base_url : str
            An Alyx database URL.  If None, the current default database is used.
        kwargs
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

    @util.parse_id
    def eid2path(self, eid, query_type=None) -> Listable(ALFPath):
        """From an experiment ID gets the local session path.

        Parameters
        ----------
        eid : str, UUID, pathlib.Path, dict, list
            Experiment session identifier; may be a UUID, URL, experiment reference string
            details dict or Path.
        query_type : str
            If set to 'remote', will force database connection.

        Returns
        -------
        one.alf.path.ALFPath, list
            A session path or list of session paths.

        """
        # first try avoid hitting the database
        mode = query_type or self.mode
        if mode != 'remote':
            cache_path = super().eid2path(eid)
            if cache_path or mode == 'local':
                return cache_path

        # If eid is a list recurse through it and return a list
        if isinstance(eid, list):
            unwrapped = unwrap(self.eid2path)
            return [unwrapped(self, e, query_type='remote') for e in eid]

        # if it wasn't successful, query Alyx
        ses = self.alyx.rest('sessions', 'list', django=f'pk,{str(eid)}')
        if len(ses) == 0:
            return None
        else:
            return ALFPath(self.cache_dir).joinpath(
                ses[0]['lab'], 'Subjects', ses[0]['subject'], ses[0]['start_time'][:10],
                str(ses[0]['number']).zfill(3))

    def path2eid(self, path_obj: Union[str, Path], query_type=None) -> Listable(str):
        """From a local path, gets the experiment ID.

        Parameters
        ----------
        path_obj : str, pathlib.Path, list
            Local path or list of local paths.
        query_type : str
            If set to 'remote', will force database connection.

        Returns
        -------
        UUID, list
            An eid or list of eids.

        """
        # If path_obj is a list recurse through it and return a list
        if isinstance(path_obj, list):
            eid_list = []
            for p in path_obj:
                eid_list.append(self.path2eid(p))
            return eid_list
        # else ensure the path ends with mouse, date, number
        path_obj = ALFPath(path_obj)

        # try the cached info to possibly avoid hitting database
        mode = query_type or self.mode
        if mode != 'remote':
            cache_eid = super().path2eid(path_obj)
            if cache_eid or mode == 'local':
                return cache_eid

        session_path = path_obj.session_path()
        # if path does not have a date and a number return None
        if session_path is None:
            return None

        # if not search for subj, date, number XXX: hits the DB
        uuid = self.search(subject=session_path.parts[-3],
                           date_range=session_path.parts[-2],
                           number=session_path.parts[-1],
                           query_type='remote')

        # Return the uuid if any
        return uuid[0] if uuid else None

    def path2url(self, filepath, query_type=None) -> str:
        """Given a local file path, returns the URL of the remote file.

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
            params = {'name': Path(filepath).name}
            if eid is None:
                params['django'] = 'session__isnull,True'
            else:
                params['session'] = str(eid)
            dataset, = self.alyx.rest('datasets', 'list', **params)
            return next(
                r['data_url'] for r in dataset['file_records'] if r['data_url'] and r['exists'])
        except (ValueError, StopIteration):
            raise alferr.ALFObjectNotFound(f'File record for {filepath} not found on Alyx')

    @util.parse_id
    def type2datasets(self, eid, dataset_type, details=False):
        """Get list of datasets belonging to a given dataset type for a given session.

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
        datasets = datasets2records(self.alyx.rest('datasets', 'list', django=restriction))
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
        if isinstance(dset, str):
            if is_uuid_string(dset):
                dset = UUID(dset)
            else:
                dset = self._dataset_name2id(dset)
        if isinstance(dset, np.ndarray):
            dset = parquet.np2uuid(dset)[0]
        if isinstance(dset, tuple) and all(isinstance(x, int) for x in dset):
            dset = parquet.np2uuid(np.array(dset))
        if not is_uuid(dset):
            raise ValueError('Unrecognized name or UUID')
        return self.alyx.rest('datasets', 'read', id=dset)['dataset_type']

    def describe_revision(self, revision, full=False):
        """Print description of a revision.

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
        query_type : {'local', 'remote'}
            The query mode - if 'local' the details are taken from the cache tables; if 'remote'
            the details are returned from the sessions REST endpoint.

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
        if isinstance(eid, (list, util.LazyId)):
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
    kwargs
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
