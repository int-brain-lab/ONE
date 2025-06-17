"""Unit tests for the one.api module.

Wherever possible the ONE tests should not rely on an internet connection.

Fixture locations:

- The cache tables for the public test instance are in tests/fixtures/
- The test db parameters can be found in tests/fixtures/params/
- Some REST GET requests can be found in tests/fixtures/rest_responses/
- These can be copied over to a temporary directory using the functions in tests/util.py,
  then construct ONE with the directory as cache_dir, mode='local' and silent=True

Imported constants:

- For tests that do require a remote connection use the tests.OFFLINE_ONLY flag in the skipIf
  decorator.
- For testing REST POST requests use TEST_DB_1 (test.alyx.internationalbrainlab.org).
- For testing download functions, use TEST_DB_2 (openalyx.internationalbrainlab.org).

Note ONE and AlyxClient use caching:

- When verifying remote changes via the rest method, use the no_cache flag to ensure the remote
  databaseis queried.  You can clear the cache using AlyxClient.clear_rest_cache(),
  or mock iblutil.io.params.getfile to return a temporary cache directory.
- An One object created through the one.api.ONE function, make sure you restore the
  properties to their original state on teardown, or call one.api.ONE.cache_clear().

"""
import datetime
import logging
import time
import shutil
from pathlib import Path, PurePosixPath
from itertools import permutations, combinations_with_replacement
from functools import partial
import unittest
from unittest import mock
import tempfile
from uuid import UUID, uuid4
import io

import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from iblutil.io import parquet
from iblutil.util import Bunch

from one import __version__
import one.api  # required for patching SAVE_ON_DELETE
from one.api import ONE, One, OneAlyx
from one.util import (
    validate_date_range, index_last_before, filter_datasets, _collection_spec,
    filter_revision_last_before, parse_id, autocomplete, LazyId
)
from one.webclient import AlyxClient
import one.params
import one.alf.exceptions as alferr
from one.converters import datasets2records
from one.alf import spec
from one.alf.path import get_alf_path
from one.alf.cache import EMPTY_DATASETS_FRAME, EMPTY_SESSIONS_FRAME, cast_index_object
from one.tests import util, OFFLINE_ONLY, TEST_DB_1, TEST_DB_2  # 1 = TestAlyx; 2 = OpenAlyx


class TestONECache(unittest.TestCase):
    """Test methods that use sessions and datasets tables.

    This class loads the parquet tables from the fixtures and builds a file tree in a temp folder
    """

    tempdir = None

    def setUp(self) -> None:
        self.tempdir = util.set_up_env()
        # Create ONE object with temp cache dir
        self.one = ONE(mode='local', cache_dir=self.tempdir.name)
        # Create dset files from cache
        util.create_file_tree(self.one)
        # here we create some variations to get coverage over more case
        # the 10 first records will have the right file size (0) but the wrong hash
        # the next 10 records will have the right file size (0) but the correct empty file md5
        # all remaining records have NaN in file_size and None in hash (default cache table)
        cols = self.one._cache['datasets'].columns
        self.one._cache['datasets'].iloc[:20, cols.get_loc('file_size')] = 0
        self.one._cache['datasets'].iloc[:20, cols.get_loc('hash')]\
            = 'd41d8cd98f00b204e9800998ecf8427e'  # empty hash correct
        self.one._cache['datasets'].iloc[:10, cols.get_loc('hash')]\
            = 'd41d8cda454aaaa4e9800998ecf8497e'  # wrong hash

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        # Before deleting ONE, set the saved timestamp to something in the future so that the
        # cache is not saved
        self.one._cache['_meta']['saved_time'] = datetime.datetime.max
        while Path(self.one.cache_dir).with_suffix('.lock').exists():
            time.sleep(.1)
        self.tempdir.cleanup()

    def test_delete(self):
        """Test One.__del__ and One.save_cache methods."""
        assert self.one._cache.datasets.loc[:, 'exists'].all()
        # Delete an entire session disk
        eid = self.one._cache.sessions.index[0]
        shutil.rmtree(self.one.eid2path(eid))
        datasets = self.one._cache.datasets.loc[eid]
        self.one._check_filesystem(datasets, update_exists=True, check_hash=False)
        self.assertFalse(self.one._cache.datasets.loc[eid, 'exists'].any())
        # Save cache should not be called when SAVE_ON_DELETE is False
        with mock.patch.object(one.api, 'SAVE_ON_DELETE', False), \
                mock.patch.object(self.one, 'save_cache') as save_cache:
            self.one.__del__()
            save_cache.assert_not_called()
        # Check that the cache tables are updated and saved upon delete with SAVE_ON_DELETE = True
        self.one = One(mode='local', cache_dir=self.tempdir.name)
        datasets = self.one._cache.datasets.loc[eid]
        self.one._check_filesystem(datasets, update_exists=True, check_hash=False)
        self.assertFalse(self.one._cache.datasets.loc[eid, 'exists'].any())
        with mock.patch.object(one.api, 'SAVE_ON_DELETE', True):
            self.one.__del__()
        while Path(self.tempdir.name, '.lock').exists():
            time.sleep(.1)
        self.one = One(mode='local', cache_dir=self.tempdir.name)
        self.assertFalse(self.one._cache.datasets.loc[eid, 'exists'].any())

    def test_list_subjects(self):
        """Test One.list_subjects."""
        subjects = self.one.list_subjects()
        expected = ['KS005', 'ZFM-01935', 'ZM_1094', 'ZM_1150',
                    'ZM_1743', 'ZM_335', 'clns0730', 'flowers']
        self.assertCountEqual(expected, subjects)

    def test_mode_validation(self):
        """Test validation of mode kwarg.

        Since auto and refresh modes were removed, a ValueError should be raised.
        """
        with self.assertRaises(ValueError):
            ONE(mode='auto')
        with self.assertRaises(ValueError):
            ONE(mode='refresh')

    def test_offline_repr(self):
        """Test for One.offline property."""
        self.assertTrue('offline' in str(self.one))
        self.assertTrue(str(self.tempdir.name) in str(self.one))

    def test_one_search(self):
        """Test for One.search."""
        one = self.one
        # Search subject
        eids = one.search(subject='ZM_335')
        expected = [UUID('3473f9d2-aa5d-41a6-9048-c65d0b7ab97c'),
                    UUID('dfe99506-b873-45db-bc93-731f9362e304')]
        self.assertEqual(expected, eids)

        # Search lab
        labs = ['mainen', 'cortexlab']
        eids = one.search(laboratory=labs)
        expected = [UUID('d3372b15-f696-4279-9be5-98f15783b5bb'),
                    UUID('3473f9d2-aa5d-41a6-9048-c65d0b7ab97c')]
        self.assertEqual(len(eids), 25)
        self.assertEqual(expected, eids[:2])

        # Search exact date
        eids = one.search(date='2019-06-07')
        self.assertEqual(eids, [UUID('db524c42-6356-4c61-b236-4967c54d2665')])

        # Search date range
        dates = ['2019-04-01', '2019-04-10']
        eids = one.search(date=dates)
        expected = [UUID('13c99443-01ee-462e-b668-717daa526fc0'),
                    UUID('abf5109c-d780-44c8-9561-83e857c7bc01')]
        self.assertEqual(len(eids), 9)
        self.assertEqual(expected, eids[:2])

        # Search from a given date
        dates = ['2021-01-01', None]
        eids = one.search(date_range=dates)
        self.assertEqual(eids, [UUID('d3372b15-f696-4279-9be5-98f15783b5bb')])

        # Search datasets
        query = 'spikes.depths.npy'
        eids = one.search(datasets=query)
        self.assertTrue(eids)
        expected = [
            UUID('d3372b15-f696-4279-9be5-98f15783b5bb'),
            UUID('b1c968ad-4874-468d-b2e4-5ffa9b9964e9'),
            UUID('cf264653-2deb-44cb-aa84-89b82507028a')
        ]
        self.assertEqual(eids, expected)

        # Test with wildcards is False
        try:
            # Regular expressions should be supported here
            query = r'^spikes\..+$'
            eids = one.search(datasets=query)
            self.assertCountEqual(eids, expected)
        finally:
            self.one.wildcards = True

        # Search QC + dataset
        query = ['spikes.depths.npy', 'spikes.times.npy']
        eid = eids[0]
        idx = (eid, UUID('a563480a-6a57-4221-b630-c7be49732ae5'))
        one._cache['datasets'].loc[idx, 'qc'] = 'FAIL'  # Set QC for 1 spikes.times dataset to FAIL
        eids = one.search(datasets=query, dataset_qc='WARNING')
        self.assertEqual(eids, expected[1:], 'failed to filter FAIL QC')

        # Search QC only - the one session with no WARNING or lower datasets should be excluded
        one._cache['datasets'].loc[eid, 'qc'] = 'FAIL'
        self.assertNotIn(eid, one.search(dataset_qc='WARNING'))

        # Filter non-existent
        # Set exist for one of the eids to false
        query = 'spikes.depths.npy'
        mask = (one._cache['datasets']['rel_path'].str.contains(query))
        i = one._cache['datasets'][mask].index[0]
        one._cache['datasets'].loc[i, 'exists'] = False

        self.assertTrue(len(expected) == len(one.search(datasets=query)) + 1)

        # Search task_protocol
        eids = one.search(task='habituation')
        self.assertEqual(eids, [UUID('ac80cd12-49e5-4aff-b5f2-1a718679ceeb')])

        # Search project
        eids = one.search(proj='neuropix')
        self.assertEqual(len(eids), 20)

        # Search number
        number = 1
        eids = one.search(num=number)

        sess_num = self.one._cache.sessions.loc[eids, 'number']
        self.assertTrue(all(sess_num == number))

        number = '002'
        eids = one.search(number=number)

        sess_num = self.one._cache.sessions.loc[eids, 'number']
        self.assertTrue(all(sess_num == int(number)))

        # Empty results
        eids, det = one.search(num=100, subject='KS000', details=True)
        self.assertTrue(len(eids) == 0)
        self.assertIsNone(det)
        # Check works with just one search term
        eids, det = one.search(num=500, details=True)
        self.assertTrue(len(eids) == 0)
        self.assertIsNone(det)

        # Test multiple fields, with short params
        eids = one.search(subj='KS005', date='2019-04-10', num='003', lab='cortexlab')
        self.assertTrue(len(eids) == 1)

        # Test param error validation
        with self.assertRaises(ValueError):
            one.search(dat='2021-03-05')  # ambiguous
        with self.assertRaises(ValueError):
            one.search(user='mister')  # invalid search term

        # Test details parameter
        eids, details = one.search(date='2019-04-10', lab='cortexlab', details=True)
        self.assertEqual(len(eids), len(details))
        self.assertCountEqual(details[0].keys(), self.one._cache.sessions.columns)

        # Test search with empty tables (previously caused error)
        self.one._cache['datasets'] = EMPTY_DATASETS_FRAME.copy()
        self.assertEqual([], self.one.search(datasets='_ibl_trials.table.pqt'))
        self.one._cache['sessions'] = EMPTY_SESSIONS_FRAME.copy()
        self.assertEqual([], self.one.search(subject='KS005'))

    def test_search_insertions(self):
        """Test for One._search_insertions."""
        one = self.one
        # Create some records (eids taken from sessions cache fixture)
        insertions = [
            {'model': '3A', 'name': 'probe00', 'json': {}, 'serial': '19051010091',
             'chronic_insertion': None, 'datasets': [uuid4(), uuid4()], 'id': str(uuid4()),
             'eid': UUID('01390fcc-4f86-4707-8a3b-4d9309feb0a1')},
            {'model': 'Fiber', 'name': 'fiber00', 'json': {}, 'serial': '18000010000',
             'chronic_insertion': str(uuid4()), 'datasets': [], 'id': str(uuid4()),
             'eid': UUID('aaf101c3-2581-450a-8abd-ddb8f557a5ad')}
        ]
        for i in range(2):
            insertions.append({
                'model': '3B2', 'name': f'probe{i:02}', 'json': {}, 'serial': f'19051010{i}90',
                'chronic_insertion': None, 'datasets': [uuid4(), uuid4()], 'id': str(uuid4()),
                'eid': UUID('4e0b3320-47b7-416e-b842-c34dc9004cf8')
            })
        one._cache['insertions'] = pd.DataFrame(insertions).set_index(['eid', 'id']).sort_index()

        # Search model
        pids = one._search_insertions(model='3B2')
        self.assertEqual(2, len(pids))
        pids = one._search_insertions(model=['3B2', '3A'])
        self.assertEqual(3, len(pids))

        # Search name
        pids = one._search_insertions(name='probe00')
        self.assertEqual(2, len(pids))
        pids = one._search_insertions(name='probe00', model='3B2')
        self.assertEqual(1, len(pids))

        # Search session details
        pids = one._search_insertions(subject='flowers')
        self.assertEqual(2, len(pids))
        pids = one._search_insertions(subject='flowers', name='probe00')
        self.assertEqual(1, len(pids))

        # Unimplemented keys
        self.assertRaises(NotImplementedError, one._search_insertions, json='foo')

        # Details
        pids, details = one._search_insertions(name='probe00', details=True)
        self.assertEqual({'probe00'}, set(x['name'] for x in details))

        # Check returned sorted by date, subject, number, and name
        pids, details = one._search_insertions(details=True)
        expected = sorted([d['date'] for d in details], reverse=True)
        self.assertEqual(expected, [d['date'] for d in details])

        # Empty returns
        self.assertEqual([], one._search_insertions(model='3A', name='fiber00', serial='123'))
        self.assertEqual([], one._search_insertions(model='foo'))
        del one._cache['insertions']
        with self.assertWarns(UserWarning):
            self.assertEqual([], one._search_insertions())

    def test_filter(self):
        """Test one.util.filter_datasets."""
        datasets = self.one._cache.datasets.iloc[:5].copy()
        # Test identity
        verifiable = filter_datasets(datasets, None, None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(len(datasets), len(verifiable))

        # Test collection filter
        verifiable = filter_datasets(datasets, None, 'alf', None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(2, len(verifiable))
        with self.assertRaises(alferr.ALFMultipleCollectionsFound):
            filter_datasets(datasets, None, 'raw.*', None, revision_last_before=False)
        # Test filter empty collection
        datasets.rel_path[-1] = '_ibl_trials.rewardVolume.npy'
        verifiable = filter_datasets(datasets, None, '', None, revision_last_before=False)
        self.assertTrue(len(verifiable), 1)

        # Test dataset filter
        verifiable = filter_datasets(datasets, '_iblrig_.*', None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(3, len(verifiable))
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            filter_datasets(datasets, '_iblrig_.*', 'raw_video_data', None,
                            revision_last_before=False)
        # Test list as logical OR
        verifiable = filter_datasets(datasets, ['alf/_ibl_wheel.*', '_ibl_trials.*'], None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertTrue(all(x.startswith('_ibl_trials') or x.startswith('alf/_ibl_wheel')
                            for x in verifiable.rel_path))

        # Test as dict
        dataset = dict(namespace='ibl', object='trials')
        verifiable = filter_datasets(datasets, dataset, None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(1, len(verifiable))
        dataset = dict(timescale='bpod', object='trials')
        verifiable = filter_datasets(self.one._cache.datasets, dataset, None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(verifiable['rel_path'].values[0], 'alf/_ibl_trials.intervals_bpod.npy')

        # As dict with list (should act as logical OR)
        dataset = dict(attribute=['time.+', 'raw'])
        verifiable = filter_datasets(datasets, dataset, None, None,
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(4, len(verifiable))

        # QC
        datasets.loc[:, 'qc'] = ['NOT_SET', 'PASS', 'WARNING', 'FAIL', 'CRITICAL']
        verifiable = filter_datasets(datasets, assert_unique=False)
        self.assertEqual(4, len(verifiable), 'failed to filter QC value')
        self.assertTrue(all(verifiable.qc < 'CRITICAL'), 'did not exclude CRITICAL QC by default')

        # 'ignore_qc_not_set' kwarg should ignore records without QC
        verifiable = filter_datasets(datasets, assert_unique=False, ignore_qc_not_set=True)
        self.assertEqual(3, len(verifiable), 'failed to filter QC value')
        self.assertTrue(all(verifiable.qc > 'NOT_SET'), 'did not exclude NOT_SET QC datasets')

        # Check QC input types
        verifiable = filter_datasets(datasets, assert_unique=False, qc='PASS')
        self.assertEqual(2, len(verifiable), 'failed to filter QC value')
        self.assertTrue(all(verifiable.qc < 'WARNING'))

        verifiable = filter_datasets(datasets, assert_unique=False, qc=30)
        self.assertEqual(3, len(verifiable), 'failed to filter QC value')
        self.assertTrue(all(verifiable.qc < 'FAIL'))

        verifiable = filter_datasets(datasets, qc=spec.QC.PASS, ignore_qc_not_set=True)
        self.assertEqual(1, len(verifiable))
        self.assertTrue(all(verifiable['qc'] == 'PASS'))

        datasets.iat[-1, -1] = 'PASS'  # set CRITICAL dataset to PASS so not excluded by default

        # Revisions
        revisions = [
            'alf/probe00/#2020-01-01#/spikes.times.npy',
            'alf/probe00/#2020-08-31#/spikes.times.npy',
            'alf/probe00/spikes.times.npy',
            'alf/probe00/#2021-xx-xx#/spikes.times.npy',
            'alf/probe01/#2020-01-01#/spikes.times.npy'
        ]
        datasets['rel_path'] = revisions

        # Should return last revision before date for each collection/dataset
        # These comprise mixed revisions which should trigger ALF warning
        revision = '2020-09-06'
        expected_warn = 'Multiple revisions: "2020-08-31", "2020-01-01"'
        with self.assertWarnsRegex(alferr.ALFWarning, expected_warn):
            verifiable = filter_datasets(datasets, None, None, revision, assert_unique=False)
        self.assertEqual(2, len(verifiable))
        self.assertTrue(all(x.split('#')[1] < revision for x in verifiable['rel_path']))

        # with no default_revisions column there should be a warning about return latest revision
        # when no revision is provided.
        with self.assertWarnsRegex(alferr.ALFWarning, 'No default revision for dataset'):
            verifiable = filter_datasets(
                datasets, '*spikes.times*', 'alf/probe00', None,
                assert_unique=True, wildcards=True, revision_last_before=True)
            self.assertEqual(1, len(verifiable))
            self.assertTrue(verifiable['rel_path'].str.contains('#2021-xx-xx#').all())

        # Should return matching revision
        verifiable = filter_datasets(datasets, None, None, r'2020-08-\d{2}',
                                     assert_unique=False, revision_last_before=False)
        self.assertEqual(1, len(verifiable))
        self.assertTrue(verifiable['rel_path'].str.contains('#2020-08-31#').all())

        # Matches more than one revision; should raise error
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_datasets(datasets, None, '.*probe00', r'2020-0[18]-\d{2}',
                            revision_last_before=False)

        # Should return revision that's lexicographically first for each dataset
        verifiable = filter_datasets(datasets, None, None, None, assert_unique=False)
        self.assertEqual(2, len(verifiable))
        actual = tuple(x.split('#')[1] for x in verifiable['rel_path'])
        self.assertEqual(('2021-xx-xx', '2020-01-01'), actual)

        # Should return those without revision
        verifiable = filter_datasets(datasets, None, None, '', assert_unique=False)
        self.assertFalse(verifiable['rel_path'].str.contains('#').any())

        # Should return empty
        verifiable = filter_datasets(datasets, None, '.*01', '', assert_unique=False)
        self.assertEqual(0, len(verifiable))

        verifiable = filter_datasets(datasets, None, '.*01', None, assert_unique=False)
        self.assertEqual(1, len(verifiable))
        self.assertTrue(verifiable['rel_path'].str.contains('#2020-01-01#').all())

        # Should return dataset marked as default
        datasets['default_revision'] = [True] + [False] * 4
        verifiable = filter_datasets(datasets, None, None, None, assert_unique=False)
        self.assertEqual(revisions[0], verifiable.rel_path.values[0])

        # Load dataset with default revision
        verifiable = filter_datasets(datasets, '*spikes.times*', 'alf/probe00', None,
                                     assert_unique=True, wildcards=True, revision_last_before=True)
        self.assertEqual(verifiable.rel_path.to_list(),
                         ['alf/probe00/#2020-01-01#/spikes.times.npy'])
        # When revision_last_before is false, expect multiple revisions error
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_datasets(datasets, '*spikes.times*', 'alf/probe00', None,
                            assert_unique=True, wildcards=True, revision_last_before=False)

    def test_filter_wildcards(self):
        """Test one.util.filter_datasets with wildcards flag set to True."""
        datasets = self.one._cache.datasets.iloc[:5].copy()
        # Test identity
        verifiable = filter_datasets(datasets, '_ibl_*', '*lf', None,
                                     assert_unique=False, wildcards=True)
        self.assertTrue(len(verifiable) == 2)
        # As dict with list (should act as logical OR)
        kwargs = dict(assert_unique=False, revision_last_before=False, wildcards=True)
        dataset = dict(attribute=['timestamp?', 'raw'])
        verifiable = filter_datasets(datasets, dataset, None, None, **kwargs)
        self.assertEqual(4, len(verifiable))

        # Test handling greedy captures of collection parts when there are wildcards at the start
        # of the filename patten.

        # Add some identical files that exist in collections and sub-collections
        # (i.e. raw_ephys_data, raw_ephys_data/probe00, raw_ephys_data/probe01)
        all_datasets = self.one._cache.datasets
        meta_datasets = all_datasets[all_datasets.rel_path.str.contains('meta')].copy()
        datasets = pd.concat([datasets, meta_datasets])

        # Matching *meta should not capture raw_ephys_data/probe00, etc.
        verifiable = filter_datasets(datasets, '*.meta', 'raw_ephys_data', None, **kwargs)
        expected = ['raw_ephys_data/_spikeglx_ephysData_g0_t0.nidq.meta']
        self.assertCountEqual(expected, verifiable.rel_path)
        verifiable = filter_datasets(datasets, '*.meta', 'raw_ephys_data/probe??', None, **kwargs)
        self.assertEqual(2, len(verifiable))
        verifiable = filter_datasets(datasets, '*.meta', 'raw_ephys_data*', None, **kwargs)
        self.assertEqual(3, len(verifiable))

    def test_list_datasets(self):
        """Test One.list_datasets."""
        # test filename
        dsets = self.one.list_datasets(filename='_ibl_trials*')
        self.assertEqual(len(dsets), 18)
        dsets = self.one.list_datasets(filename='gnagnag')
        self.assertEqual(len(dsets), 0)

        # Test no eid
        dsets = self.one.list_datasets(details=True)
        self.assertEqual(len(dsets), len(self.one._cache.datasets))
        self.assertFalse(dsets is self.one._cache.datasets)
        self.assertEqual(2, dsets.index.nlevels, 'details data frame should be with eid index')

        # Test list for eid
        dsets = self.one.list_datasets('KS005/2019-04-02/001', details=True)
        self.assertEqual(27, len(dsets))
        self.assertEqual(1, dsets.index.nlevels, 'details data frame should be without eid index')

        # Test keep_eid_index parameter
        dsets = self.one.list_datasets('KS005/2019-04-02/001', details=True, keep_eid_index=True)
        self.assertEqual(27, len(dsets))
        self.assertEqual(2, dsets.index.nlevels, 'details data frame should be with eid index')

        # Test filters
        filename = {'attribute': ['times', 'intervals'], 'extension': 'npy'}
        dsets = self.one.list_datasets('ZFM-01935/2021-02-05/001', filename)
        self.assertEqual(10, len(dsets))
        self.assertTrue(all(any(y in x for y in ('.times.', '.intervals')) for x in dsets))

        filename['attribute'][0] += '*'  # Include wildcard to match both times and timestamps
        dsets = self.one.list_datasets('ZFM-01935/2021-02-05/001', filename)
        self.assertEqual(13, len(dsets))
        self.assertEqual(3, sum('.timestamps.' in x for x in dsets))

        # Test empty
        dsets = self.one.list_datasets('FMR019/2021-03-18/002', details=True)
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertEqual(len(dsets), 0)

        # Test details=False, with and without eid
        for eid in [None, 'KS005/2019-04-02/001']:
            dsets = self.one.list_datasets(eid, details=False)
            self.assertIsInstance(dsets, list)
            self.assertTrue(len(dsets) == np.unique(dsets).size)

        # Test default_revisions_only=True
        with self.assertRaises(alferr.ALFError):  # should raise as no 'default_revision' column
            self.one.list_datasets('KS005/2019-04-02/001', default_revisions_only=True)
        # Add the column and add some alternates
        datasets = util.revisions_datasets_table(collections=['alf'], revisions=['', '2023-01-01'])
        datasets['default_revision'] = [False, True] * 2
        self.one._cache.datasets['default_revision'] = True
        self.one._cache.datasets = pd.concat([self.one._cache.datasets, datasets]).sort_index()
        eid, *_ = datasets.index.get_level_values(0)
        dsets = self.one.list_datasets(eid, 'spikes.*', default_revisions_only=False)
        self.assertEqual(4, len(dsets))
        dsets = self.one.list_datasets(eid, 'spikes.*', default_revisions_only=True)
        self.assertEqual(2, len(dsets))
        self.assertTrue(all('#2023-01-01#' in x for x in dsets))
        # Should be the same with details=True
        dsets = self.one.list_datasets(eid, 'spikes.*', default_revisions_only=True, details=True)
        self.assertEqual(2, len(dsets))
        self.assertTrue(all('#2023-01-01#' in x for x in dsets.rel_path))

    def test_list_collections(self):
        """Test One.list_collections."""
        # Test no eid
        dsets = self.one.list_collections()
        expected = [
            '', 'alf', 'alf/ks2', 'alf/probe00', 'raw_behavior_data', 'raw_ephys_data',
            'raw_ephys_data/probe00', 'raw_passive_data', 'raw_video_data'
        ]
        self.assertCountEqual(expected, dsets)

        # Test details for eid
        dsets = self.one.list_collections('KS005/2019-04-02/001', details=True)
        self.assertIsInstance(dsets, dict)
        self.assertTrue(set(dsets.keys()) <= set(expected))
        self.assertIsInstance(dsets['alf'], pd.DataFrame)
        self.assertTrue(dsets['alf'].rel_path.str.startswith('alf').all())

        # Test empty
        self.assertFalse(len(self.one.list_collections('FMR019/2021-03-18/002', details=True)))
        self.assertFalse(len(self.one.list_collections('FMR019/2021-03-18/002', details=False)))

    def test_list_revisions(self):
        """Test One.list_revisions."""
        # No revisions in cache fixture so generate our own
        revisions_datasets = util.revisions_datasets_table()
        self.one._cache.datasets = pd.concat([self.one._cache.datasets, revisions_datasets])
        eid, _ = revisions_datasets.index[0]

        # Test no eid
        dsets = self.one.list_revisions()
        expected = ['', '2020-01-08', '2021-07-06']
        self.assertCountEqual(expected, dsets)

        # Test details for eid
        dsets = self.one.list_revisions(eid, details=True)
        self.assertIsInstance(dsets, dict)
        self.assertTrue(set(dsets.keys()) <= set(expected))
        self.assertIsInstance(dsets['2020-01-08'], pd.DataFrame)
        self.assertTrue(dsets['2020-01-08'].rel_path.str.contains('#2020-01-08#').all())

        # Test dataset filter
        dsets = self.one.list_revisions(eid, filename='spikes.times.npy', details=True)
        self.assertTrue(dsets['2020-01-08'].rel_path.str.endswith('spikes.times.npy').all())

        # Test collections filter
        dsets = self.one.list_revisions(eid, collection='alf/probe01', details=True)
        self.assertTrue(dsets['2020-01-08'].rel_path.str.startswith('alf/probe01').all())

        # Test revision filter
        revisions = self.one.list_revisions(eid, revision=['202[01]*'])
        self.assertCountEqual(['2020-01-08', '2021-07-06'], revisions)

        # Test empty
        self.assertFalse(len(self.one.list_revisions('FMR019/2021-03-18/002', details=True)))
        self.assertFalse(len(self.one.list_revisions('FMR019/2021-03-18/002', details=False)))

    def test_get_details(self):
        """Test One.get_details."""
        eid = 'aaf101c3-2581-450a-8abd-ddb8f557a5ad'
        det = self.one.get_details(eid)
        self.assertIsInstance(det, pd.Series)
        self.assertEqual('KS005', det.subject)
        self.assertEqual('2019-04-04', str(det.date))
        self.assertEqual(4, det.number)

        # Test details flag
        det2 = self.one.get_details(eid, full=True)
        self.assertIsInstance(det2, pd.DataFrame)
        self.assertTrue('rel_path' in det2.columns)

        # Test errors
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.get_details(eid.replace('a', 'b'))
        sessions = self.one._cache.sessions
        self.one._cache.sessions = pd.concat([sessions, det.to_frame().T]).sort_index()
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.get_details(eid)

    def test_check_filesystem(self):
        """Test for One._check_filesystem.

        Most is already covered by other tests, this checks that it can deal with dataset frame
        without eid index and without a session_path column.
        """
        # Get two eids to test
        eids = self.one._cache['datasets'].index.get_level_values(0)[[0, -1]]
        datasets = self.one._cache['datasets'].loc[eids]
        files = self.one._check_filesystem(datasets)
        self.assertEqual(53, len(files))

        # Expect same number of unique session paths as eids
        session_paths = set(map(lambda p: p.parents[1], files))
        self.assertEqual(len(eids), len(session_paths))
        expected = map(lambda x: '/'.join(x.parts[-3:]), session_paths)
        session_parts = self.one._cache['sessions'].loc[eids, ['subject', 'date', 'number']].values
        self.assertCountEqual(expected, map(lambda x: f'{x[0]}/{x[1]}/{x[2]:03}', session_parts))

        # Test a very rare occurence of a missing dataset with eid index missing
        # but session_path column present
        idx = self.one._cache.datasets.index[(i := 5)]  # pick a random dataset to make vanish
        _eid2path = {
            e: self.one.eid2path(e).relative_to(self.one.cache_dir).as_posix() for e in eids
        }
        session_paths = list(map(_eid2path.get, datasets.index.get_level_values(0)))
        datasets['session_path'] = session_paths
        datasets = datasets.droplevel(0)
        files[(i := 5)].unlink()
        # For this check the current state should be exists==True in the main cache
        assert self.one._cache.datasets.loc[idx, 'exists'].all()
        _files = self.one._check_filesystem(datasets, update_exists=True)
        self.assertIsNone(_files[i])
        self.assertFalse(
            self.one._cache.datasets.loc[idx, 'exists'].all(), 'failed to update cache exists')
        files[i].touch()  # restore file for next check

        # Attempt to load datasets with both eid index
        # and session_path column missing (most common)
        datasets = datasets.drop('session_path', axis=1)
        self.assertEqual(files, self.one._check_filesystem(datasets))

        # Test with uuid_filenames as True
        self.one.uuid_filenames = True
        try:
            for file, (uuid, _) in zip(files, datasets.iterrows()):
                file.rename(file.with_suffix(f'.{uuid}{file.suffix}'))
            files = self.one._check_filesystem(datasets)
            self.assertTrue(all(files))
            self.assertIn(str(datasets.index[0]), files[0].name)
        finally:
            self.one.uuid_filenames = False

        # Test empty input
        self.assertEqual([], self.one._check_filesystem(datasets.iloc[:0]))

    def test_load_dataset(self):
        """Test One.load_dataset."""
        eid = 'KS005/2019-04-02/001'
        # Check download only
        file = self.one.load_dataset(eid, '_ibl_wheel.position.npy', download_only=True)
        self.assertIsInstance(file, Path)

        # Check loading data
        np.save(str(file), np.arange(3))  # Make sure we have something to load
        dset = self.one.load_dataset(eid, '_ibl_wheel.position.npy')
        self.assertTrue(np.all(dset == np.arange(3)))

        # Check collection filter
        file = self.one.load_dataset(eid, '_iblrig_leftCamera.timestamps.ssv',
                                     download_only=True, collection='raw_video_data')
        self.assertIsNotNone(file)

        # Test errors
        # ID not in cache
        fake_id = str(self.one.to_eid(eid)).replace('b', 'a')
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset(fake_id, '_iblrig_leftCamera.timestamps.ssv')
        file.unlink()
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset(eid, '_iblrig_leftCamera.timestamps.ssv')
        with self.assertRaises(ValueError):
            self.one.load_dataset(eid, 'alf/_ibl_trials.choice', collection='')

        # Check loading without extension
        file = self.one.load_dataset(eid, '_ibl_wheel.position', download_only=True)
        self.assertTrue(str(file).endswith('wheel.position.npy'))

        # Check loading with relative path
        file = self.one.load_dataset(eid, 'alf/_ibl_trials.choice.npy', download_only=True)
        self.assertIsNotNone(file)
        with self.assertWarns(UserWarning):
            self.one.load_dataset(eid, 'alf*/_ibl_trials.choice.npy', download_only=True)

    def test_load_datasets(self):
        """Test One.load_datasets."""
        eid = 'KS005/2019-04-02/001'
        # Check download only
        dsets = ['_ibl_wheel.position.npy', '_ibl_wheel.timestamps.npy']
        files, meta = self.one.load_datasets(eid, dsets, download_only=True, assert_present=False)
        self.assertIsInstance(files, list)
        self.assertTrue(all(isinstance(x, Path) for x in files))

        # Check loading data and missing dataset
        dsets = ['_ibl_wheel.position.npy', '_ibl_wheel.timestamps_bpod.npy']
        np.save(str(files[0]), np.arange(3))  # Make sure we have something to load
        data, meta = self.one.load_datasets(eid, dsets, download_only=False, assert_present=False)
        self.assertEqual(2, len(data))
        self.assertEqual(2, len(meta))
        self.assertTrue(np.all(data[0] == np.arange(3)))

        # Check assert_present raises error
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_datasets(eid, dsets, assert_present=True)

        # Check collection and revision filters
        dsets = ['_ibl_wheel.position.npy', '_ibl_wheel.timestamps.npy']
        files, meta = self.one.load_datasets(eid, dsets, collections='alf', revisions=[None, None],
                                             download_only=True, assert_present=False)
        self.assertTrue(all(files))

        files, meta = self.one.load_datasets(eid, dsets, collections=['alf', ''],
                                             download_only=True, assert_present=False)
        self.assertIsNone(files[-1])

        # Check validations
        with self.assertRaises(ValueError):
            self.one.load_datasets(eid, dsets, collections=['alf', '', 'foo'])
        with self.assertRaises(ValueError):
            self.one.load_datasets(eid, [f'alf/{x}' for x in dsets], collections='alf')
        with self.assertRaises(TypeError):
            self.one.load_datasets(eid, 'spikes.times')
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_datasets('ff812ca5-ce60-44ac-b07e-66c2c37e98eb', dsets)
        with self.assertLogs(logging.getLogger('one.api'), 'WARNING'):
            data, meta = self.one.load_datasets('ff812ca5-ce60-44ac-b07e-66c2c37e98eb', dsets,
                                                assert_present=False)
        self.assertIsNone(data)
        self.assertEqual(0, len(meta))
        self.assertIsNone(self.one.load_datasets(eid, [])[0])
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_datasets(eid, dsets, collections='none', assert_present=True)

        # Check behaviour when data are not downloaded for any reason
        with mock.patch.object(self.one, '_check_filesystem',
                               side_effect=lambda x, **kwargs: [None] * len(x)):
            with self.assertLogs(logging.getLogger('one.api'), 'WARNING'):
                self.one.load_datasets(eid, dsets, collections='alf', assert_present=False)
            with self.assertRaises(alferr.ALFObjectNotFound):
                self.one.load_datasets(eid, dsets, collections='alf', assert_present=True)

        # Check loading without extensions
        # Check download only
        dsets = ['_ibl_wheel.position.npy', '_ibl_wheel.timestamps']
        files, meta = self.one.load_datasets(eid, dsets, download_only=True)
        self.assertTrue(all(isinstance(x, Path) for x in files))

        # Check behaviour when loading with a data frame (undocumented)
        eid = UUID('01390fcc-4f86-4707-8a3b-4d9309feb0a1')
        datasets = self.one._cache.datasets.loc[([eid],), :].iloc[:3, :]
        files, meta = self.one.load_datasets(eid, datasets, download_only=True)
        self.assertTrue(all(isinstance(x, Path) for x in files))
        # Should raise when data frame contains a different eid
        self.assertRaises(AssertionError, self.one.load_datasets, uuid4(), datasets)

        # Mix of str and dict
        # Check download only
        dsets = [
            spec.regex(spec.FILE_SPEC).match('_ibl_wheel.position.npy').groupdict(),
            '_ibl_wheel.timestamps.npy'
        ]
        with self.assertRaises(ValueError):
            self.one.load_datasets('KS005/2019-04-02/001', dsets, assert_present=False)

        # Loading of non default revisions without using the revision kwarg causes user warning.
        # With relative paths provided as input, dataset uniqueness validation is suppressed.
        eid = self.one._cache.sessions.iloc[0].name
        datasets = util.revisions_datasets_table(
            revisions=('', '2020-01-08'), attributes=('times',), touch_path=self.one.eid2path(eid))
        datasets['default_revision'] = [False, True] * 3
        datasets.index = datasets.index.set_levels([eid], level=0)
        self.one._cache.datasets = datasets
        with self.assertWarns(alferr.ALFWarning):
            self.one.load_datasets(eid, datasets['rel_path'].to_list(), download_only=True)

        # When loading without collections in the dataset list (i.e. just the dataset names)
        # an exception should be raised when datasets belong to multiple collections.
        self.assertRaises(
            alferr.ALFMultipleCollectionsFound, self.one.load_datasets, eid, ['spikes.times'])

        # Ensure that when rel paths are passed, a null collection/revision is not interpreted as
        # an ANY. NB this means the output of 'spikes.times.npy' will be different depending on
        # weather other datasets in list include a collection or revision.
        self.one._cache.datasets = datasets.iloc[:2, :].copy()  # only two datasets, one default
        (file,), _ = self.one.load_datasets(eid, ['spikes.times.npy', ], download_only=True)
        self.assertTrue(file.as_posix().endswith('001/#2020-01-08#/spikes.times.npy'))
        (file, _), _ = self.one.load_datasets(
            eid, ['spikes.times.npy', 'xx/obj.attr.ext'], download_only=True, assert_present=False)
        self.assertTrue(file.as_posix().endswith('001/spikes.times.npy'))

    def test_load_dataset_from_id(self):
        """Test One.load_dataset_from_id."""
        uid = np.array([[-9204203870374650458, -6411285612086772563]])
        file = self.one.load_dataset_from_id(uid, download_only=True)
        self.assertIsInstance(file, Path)
        expected = 'ZFM-01935/2021-02-05/001/alf/probe00/_phy_spikes_subset.waveforms.npy'
        self.assertTrue(file.as_posix().endswith(expected))

        # Details
        _, details = self.one.load_dataset_from_id(uid, download_only=True, details=True)
        self.assertIsInstance(details, pd.Series)

        # Load file content with str id
        s_id, = parquet.np2str(uid)
        data = np.arange(3)
        np.save(str(file), data)  # Ensure data to load
        dset = self.one.load_dataset_from_id(s_id)
        self.assertTrue(np.array_equal(dset, data))

        # Load file content with UUID
        dset = self.one.load_dataset_from_id(UUID(s_id))
        self.assertTrue(np.array_equal(dset, data))

        # Test errors
        # ID not in cache
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset_from_id(s_id.replace('a', 'b'))
        # File missing
        file.unlink()
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset_from_id(s_id)
        # Duplicate ids in cache
        details.name = (UUID('d3372b15-f696-4279-9be5-98f15783b5bb'), UUID(s_id))
        datasets = self.one._cache.datasets
        self.one._cache.datasets = pd.concat([datasets, details.to_frame().T]).sort_index()
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.load_dataset_from_id(s_id)

    def test_load_object(self):
        """Test One.load_object."""
        eid = 'aaf101c3-2581-450a-8abd-ddb8f557a5ad'
        files = self.one.load_object(eid, 'wheel', download_only=True)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(isinstance(x, Path) for x in files))

        # Save some data into the files
        N = 10  # length of data
        for f in files:
            np.save(str(f), np.random.rand(N))
        wheel = self.one.load_object(eid, 'wheel')
        self.assertIsInstance(wheel, dict)
        self.assertCountEqual(wheel.keys(), ('position', 'velocity', 'timestamps'))
        self.assertTrue(
            all(x.size == N for x in wheel.values())
        )

        # Test errors
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_object(eid, 'spikes')
        # Test behaviour with missing session
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_object(eid.replace('a', 'b'), 'wheel')
        # Test missing files on disk
        [f.unlink() for f in files]
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_object(eid, 'wheel')
        # Check the three wheel datasets set to exist = False in cache
        self.assertFalse(self.one._cache.datasets.loc[(UUID(eid),), 'exists'].all())

        eid = 'ZFM-01935/2021-02-05/001'
        with self.assertRaises(alferr.ALFMultipleCollectionsFound):
            self.one.load_object(eid, 'ephysData_g0_t0')
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.load_object(eid, '*Camera')

    def test_load_collection(self):
        """Test One.load_collection."""
        # Check download_only output
        eid = 'aaf101c3-2581-450a-8abd-ddb8f557a5ad'
        files = self.one.load_collection(eid, 'alf', download_only=True)
        self.assertEqual(len(files), 18)
        self.assertTrue(all(isinstance(x, Path) for x in files))
        # Check load
        alf = self.one.load_collection(eid, 'alf', attribute='*time*')
        self.assertCountEqual(alf.keys(), ('trials', 'wheel'))
        self.assertIsInstance(alf, Bunch)
        self.assertIn('feedback_times', alf.trials)
        # Check object filter
        alf = self.one.load_collection(eid, 'alf', object='trials')
        self.assertNotIn('wheel', alf)
        # Check errors
        with self.assertRaises(alferr.ALFError):
            self.one.load_collection(eid, '')
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_collection(eid, 'alf', object='foo')
        # Should raise error when no objects found on disk
        with mock.patch.object(self.one, '_check_filesystem',
                               side_effect=lambda x, **kwargs: [None] * len(x)), \
                self.assertRaises(alferr.ALFObjectNotFound) as ex:
            self.one.load_collection(eid, 'alf')
            self.assertIn('not found on disk', str(ex))

    def test_load_cache(self):
        """Test One._load_cache."""
        # Test loading unsorted table with no id index set
        df = cast_index_object(self.one._cache['datasets'], str).reset_index()
        info = self.one._cache['_meta']['raw']['datasets'].copy()
        info['origin'] = list(info['origin'])  # ensures serializable
        with tempfile.TemporaryDirectory() as tdir:
            # Loading from empty dir
            # In offline mode, load_cache will not raise, but should warn when no tables found
            self.assertWarns(UserWarning, self.one.load_cache, tdir)
            # Save unindexed
            parquet.save(Path(tdir) / 'datasets.pqt', df, info)
            del self.one._cache['datasets']
            self.one.load_cache(tdir)
            self.assertIsInstance(self.one._cache['datasets'].index, pd.MultiIndex)
            # Save shuffled
            id_keys = ['eid', 'id']
            df[id_keys] = np.random.permutation(df[id_keys])
            assert not df.set_index(id_keys).index.is_monotonic_increasing
            parquet.save(Path(tdir) / 'datasets.pqt', df, info)
            del self.one._cache['datasets']
            self.one.load_cache(tdir)
            self.assertTrue(self.one._cache['datasets'].index.is_monotonic_increasing)
            # Save a parasitic table that will not be loaded
            pd.DataFrame().to_parquet(Path(tdir).joinpath('gnagna.pqt'))
            with self.assertLogs(logging.getLogger('one.alf.cache'), logging.WARNING) as log:
                self.one.load_cache(tdir)
                self.assertIn('gnagna.pqt', log.output[0])
            # Save table with missing id columns
            df.drop(id_keys, axis=1, inplace=True)
            parquet.save(Path(tdir) / 'datasets.pqt', df, info)
            with self.assertRaises(KeyError):
                self.one.load_cache(tdir)
            # Loading cache in local mode with clobber=False should raise NotImplementedError
            with self.assertRaises(NotImplementedError):
                self.one.load_cache(tdir, clobber=False)

        # Test loading large Alyx tables
        raw = {'origin': {'alyx'}}
        cache = Bunch({
            'datasets': EMPTY_DATASETS_FRAME.copy(),
            'sessions': EMPTY_SESSIONS_FRAME.copy(),
            '_meta': {
                'created_time': datetime.datetime(2025, 2, 1, 12, 0),
                'loaded_time': datetime.datetime.now(),
                'raw': {'datasets': raw, 'sessions': raw}}
        })
        with mock.patch('one.api.load_tables', return_value=cache), \
                mock.patch('builtins.input', return_value='yes'), \
                mock.patch.object(self.one, '_remove_table_files') as m:
            self.one.load_cache()
            m.assert_not_called()
        # Remote mode
        self.one.mode = 'remote'
        self.one._web_client = mock.MagicMock()
        self.one._web_client.silent = False
        with mock.patch('one.api.load_tables', return_value=cache), \
                mock.patch('builtins.input', return_value='yes'), \
                mock.patch.object(self.one, '_remove_table_files') as m:
            self.one.load_cache()
            m.assert_called_once()
        # Test large table warning
        cache.datasets = mock.MagicMock()
        cache.datasets.__len__.return_value = int(1.5e6)
        self.one._web_client = None
        self.one.mode = 'local'
        with mock.patch('one.api.load_tables', return_value=cache), \
                mock.patch('builtins.input', return_value='n'), \
                mock.patch.object(self.one, '_remove_table_files') as m:
            self.assertWarns(UserWarning, self.one.load_cache)

    def test_save_cache(self):
        """Test One.save_cache method."""
        self.one._cache['_meta'].pop('modified_time', None)
        # Should be no cache save as it's not been modified
        with tempfile.TemporaryDirectory() as tdir:
            self.one.save_cache(save_dir=tdir)
            self.assertFalse(any(Path(tdir).glob('*')))
            self.assertIsNone(self.one._cache['_meta']['saved_time'])
            # Should save two tables
            self.one._cache['_meta']['modified_time'] = datetime.datetime.now()
            self.one.save_cache(save_dir=tdir)
            self.assertEqual(2, len(list(Path(tdir).glob('*.pqt'))))
            self.assertIsNotNone(self.one._cache['_meta']['saved_time'])
            # Without clobber, the tables should be merged with existing tables
            n_datasets = np.unique(self.one._cache.datasets.index.values).size
            dataset_1 = self.one._cache.datasets.iloc[-1:].copy()
            dataset_1.loc[dataset_1.index[0], 'exists'] = False
            dataset_2 = self.one._cache.datasets.iloc[0]
            dataset_2.name = (dataset_2.name[0], uuid4())
            self.one._cache.datasets = pd.concat([dataset_1, dataset_2.to_frame().T])
            self.one._cache.datasets.index.set_names(('eid', 'id'), inplace=True)
            self.one._cache['_meta']['modified_time'] = datetime.datetime.now()
            # The cache on disk has a created date of 2021 for both tables
            # Check how merge handles conflicting dates
            raw_meta = self.one._cache['_meta']['raw']
            assert raw_meta['datasets']['date_created'] == '2021-05-13 20:38'
            self.one._cache['_meta']['raw']['datasets']['date_created'] = '2024-01-01 00:00'
            self.one._cache['_meta']['raw']['sessions']['date_created'] = '2020-01-01 00:00'
            self.one.save_cache(save_dir=tdir)
            # Load with One and check modified time is preserved
            merged = One(cache_dir=tdir)._cache
            raw_modified = merged['_meta']['raw']['datasets']['date_modified']
            expected = self.one._cache['_meta']['modified_time'].strftime('%Y-%m-%d %H:%M')
            self.assertEqual(raw_modified, expected)
            # Date created for each table should be the minimum of the two dates
            self.assertEqual(
                '2021-05-13 20:38', merged['_meta']['raw']['datasets']['date_created'])
            self.assertEqual(
                '2020-01-01 00:00', merged['_meta']['raw']['sessions']['date_created'])
            # Should drop duplicates and add new dataset
            self.assertEqual(n_datasets + 1, len(merged.datasets))
            self.assertIn(dataset_2.name, merged.datasets.index)
            # Should have modified existing dataset
            self.assertFalse(merged.datasets.loc[dataset_1.index, 'exists'].all())
            # Check clobber command
            self.one._reset_cache()
            save_time = self.one._cache['_meta']['saved_time']
            self.one.save_cache(save_dir=tdir, clobber=False)
            self.assertEqual(save_time, self.one._cache['_meta']['saved_time'])
            self.one.save_cache(save_dir=tdir, clobber=True)
            self.assertNotEqual(save_time, self.one._cache['_meta']['saved_time'])
            One(cache_dir=tdir)._cache.sessions.empty

    def test_reset_cache(self):
        """Test One._reset_cache method, namely that cache types are correct."""
        # Assert cache dtypes are indeed what are expected
        self.one._reset_cache()
        self.assertCountEqual(['datasets', 'sessions', '_meta'], self.one._cache.keys())
        self.assertTrue(self.one._cache.datasets.empty)
        self.assertCountEqual(EMPTY_SESSIONS_FRAME.columns, self.one._cache.sessions.columns)
        self.assertTrue(self.one._cache.sessions.empty)
        self.assertCountEqual(EMPTY_DATASETS_FRAME.columns, self.one._cache.datasets.columns)
        # Check sessions data frame types
        sessions_types = EMPTY_SESSIONS_FRAME.reset_index().dtypes.to_dict()
        s_types = self.one._cache.sessions.reset_index().dtypes.to_dict()
        self.assertDictEqual(sessions_types, s_types)
        # Check datasets data frame types
        datasets_types = EMPTY_DATASETS_FRAME.reset_index().dtypes.to_dict()
        d_types = self.one._cache.datasets.reset_index().dtypes.to_dict()
        self.assertDictEqual(datasets_types, d_types)

    def test_save_loaded_ids(self):
        """Test One.save_loaded_ids and logic within One._check_filesystem."""
        self.one.record_loaded = True  # Turn on saving UUIDs
        self.one._cache.pop('_loaded_datasets', None)  # Ensure we start with a clean slate
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        files = self.one.load_object(eid, 'trials', download_only=True)
        # Check datasets added to list
        self.assertIn('_loaded_datasets', self.one._cache)
        loaded = self.one._cache['_loaded_datasets']
        self.assertEqual(len(files), len(loaded))
        # Ensure all are from the same session
        eids = self.one._cache.datasets.loc[(slice(None), loaded), ].index.get_level_values(0)
        self.assertTrue(np.all(eids == UUID(eid)))

        # Test loading a dataset that doesn't exist
        dset = self.one.list_datasets(eid, filename='*trials*', details=True).iloc[-1]
        dset['rel_path'] = dset['rel_path'].replace('.npy', '.pqt')
        dset.name = (UUID(eid), uuid4())
        old_cache = self.one._cache['datasets']
        try:
            datasets = [self.one._cache.datasets, dset.to_frame().T]
            datasets = pd.concat(datasets).astype(old_cache.dtypes).sort_index()
            datasets.index.set_names(('eid', 'id'), inplace=True)
            self.one._cache['datasets'] = datasets
            dsets = [dset['rel_path'], '_ibl_trials.feedback_times.npy']
            new_files, rec = self.one.load_datasets(eid, dsets, assert_present=False)
            loaded = self.one._cache['_loaded_datasets']
            # One dataset is already in the list (test for duplicates) and other doesn't exist
            self.assertEqual(len(files), len(loaded), 'No new UUIDs should have been added')
            self.assertEqual(len(loaded), len(np.unique(loaded)))
            self.assertNotIn(dset.name[1], loaded)  # Wasn't loaded as doesn't exist on disk
        finally:
            self.one._cache['datasets'] = old_cache

        # Test saving the loaded datasets list
        ids, filename = self.one.save_loaded_ids(clear_list=False)
        self.assertTrue(loaded is ids)
        self.assertTrue(filename.exists())
        # Load from file
        ids = pd.read_csv(filename)
        self.assertCountEqual(ids['dataset_uuid'].map(UUID), loaded)
        self.assertTrue(self.one._cache['_loaded_datasets'].size, 'List unexpectedly cleared')

        # Test as session UUIDs
        ids, filename = self.one.save_loaded_ids(sessions_only=True, clear_list=False)
        self.assertCountEqual([UUID(eid)], ids)
        self.assertEqual(pd.read_csv(filename)['session_uuid'][0], eid)

        # Check dataset int IDs and clear list
        ids, _ = self.one.save_loaded_ids(clear_list=True)
        self.assertCountEqual(loaded, ids)
        self.assertFalse(self.one._cache['_loaded_datasets'].size, 'List not cleared')

        # Test clear list and warn on empty
        with self.assertWarns(Warning):
            ids, filename = self.one.save_loaded_ids()
        self.assertEqual(ids, [])
        self.assertIsNone(filename)

    def test_remove_cache_table_files(self):
        """Test One._remove_cache_table_files method."""
        root = self.one._tables_dir
        for name in ('cache_info.json', 'foo.pqt'):
            root.joinpath(name).touch()
        removed = self.one._remove_table_files()
        expected = ['sessions.pqt', 'datasets.pqt', 'cache_info.json']
        self.assertCountEqual(expected, [str(x.relative_to(root)) for x in removed])
        with self.assertLogs('one.alf.cache', 30) as cm:
            removed = self.one._remove_table_files(tables=('foo',))
        self.assertEqual([root / 'foo.pqt'], removed)
        self.assertIn('cache_info.json not found', cm.records[0].message)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneAlyx(unittest.TestCase):
    """Test OneAlyx methods that use the Alyx REST API.

    This could be an offline test: would need to add /docs REST cache fixture.
    """

    tempdir = None
    one = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = util.set_up_env()
        with mock.patch('one.params.iopar.getfile', new=partial(util.get_file, cls.tempdir.name)):
            # util.setup_test_params(token=True)
            cls.one = OneAlyx(
                **TEST_DB_1,
                cache_dir=cls.tempdir.name,
                mode='local'
            )

    def tearDown(self) -> None:
        """Ensure the ONE mode is local and that the cache isn't saved when ONE is deleted."""
        self.one.mode = 'local'
        self.one._cache['_meta']['saved_time'] = datetime.datetime.max

    def test_type2datasets(self):
        """Test OneAlyx.type2datasets."""
        self.one.mode = 'remote'
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        # when the dataset is at the root, there shouldn't be the separator
        dsets = self.one.type2datasets(eid, 'eye.blink')
        self.assertCountEqual(dsets, ['eye.blink.npy'])
        # test multiples datasets with collections
        eid = '8dd0fcb0-1151-4c97-ae35-2e2421695ad7'
        dtypes = ['trials.feedback_times', '_iblrig_codeFiles.raw']
        dsets = self.one.type2datasets(eid, dtypes)
        expected = ['alf/_ibl_trials.feedback_times.npy',
                    'raw_behavior_data/_iblrig_codeFiles.raw.zip']
        self.assertCountEqual(dsets, expected)
        # this returns a DataFrame
        dsets = self.one.type2datasets(eid, dtypes, details=True)
        self.assertIsInstance(dsets, pd.DataFrame)
        # check validation
        with self.assertRaises(TypeError):
            self.one.type2datasets(eid, 14)

    def test_dataset2type(self):
        """Test for OneAlyx.dataset2type."""
        # Test dataset ID
        did = np.array([[-1058666951852871669, -6012002505064768322]])

        # Check assertion in local mode
        with self.assertRaises(AssertionError):
            self.one.dataset2type(did)

        self.one.mode = 'remote'
        dset_type = self.one.dataset2type(did)
        self.assertEqual('wheelMoves.peakAmplitude', dset_type)
        # Check with tuple int
        dset_type = self.one.dataset2type(tuple(did.squeeze().tolist()))
        self.assertEqual('wheelMoves.peakAmplitude', dset_type)
        # Check with str id
        did, = parquet.np2str(did)
        dset_type = self.one.dataset2type(did)
        self.assertEqual('wheelMoves.peakAmplitude', dset_type)

        dset_type = self.one.dataset2type('_ibl_wheelMoves.peakAmplitude.npy')
        self.assertEqual('wheelMoves.peakAmplitude', dset_type)

        bad_id = np.array([[-1058666951852871669, -6012002505064768312]])
        with self.assertRaises(ValueError):
            self.one.dataset2type(bad_id)

    def test_pid2eid(self):
        """Test OneAlyx.pid2eid.

        For a more complete test see `test_converters.TestOnlineConverters.test_pid2eid`.
        This test uses the REST fixtures and therefore can be run offline.
        """
        pid = 'b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        eid, collection = self.one.pid2eid(pid, query_type='remote')
        self.assertEqual(UUID('fc737f3c-2a57-4165-9763-905413e7e341'), eid)
        self.assertEqual('probe00', collection)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_describe_revision(self, mock_stdout):
        """Test OneAlyx.describe_revision."""
        self.one.mode = 'remote'
        # Choose a date in the past so as to not conflict with registration tests
        record = {
            'name': str(datetime.date.today() - datetime.timedelta(days=5)) + 'a',
            'description': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
        }
        try:
            self.one.alyx.rest('revisions', 'partial_update', id=record['name'], data=record)
        except HTTPError:
            self.one.alyx.rest('revisions', 'create', data=record)
        self.one.describe_revision(record['name'])
        self.assertEqual(record['description'], mock_stdout.getvalue().strip())
        self.one.describe_revision('foobar')
        self.assertTrue('not found' in mock_stdout.getvalue())

        # Check full kwarg
        full = self.one.describe_revision(record['name'], full=True)
        self.assertIsInstance(full, dict)

        # Check raises non-404 error
        err = HTTPError()
        err.response = Bunch({'status_code': 500})
        with mock.patch.object(self.one.alyx, 'get', side_effect=err), \
                self.assertRaises(HTTPError):
            self.one.describe_revision(record['name'])

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_describe_dataset(self, mock_stdout):
        """Test OneAlyx.describe_dataset.

        NB This could be offline: REST responses in fixtures.
        """
        self.one.mode = 'remote'
        # Test all datasets
        dset_types = self.one.describe_dataset()
        self.assertEqual(7, len(dset_types))
        self.assertEqual('unknown', dset_types[0]['name'])

        # Test dataset type
        out = self.one.describe_dataset('wheel.velocity')
        expected = 'Signed velocity of wheel'
        self.assertTrue(expected in mock_stdout.getvalue())
        self.assertEqual(expected, out['description'])

        # Test dataset name
        expected = 'amplitude of the wheel move'
        out = self.one.describe_dataset('_ibl_wheelMoves.peakAmplitude.npy')
        self.assertTrue(expected in mock_stdout.getvalue())
        self.assertEqual(expected, out['description'])

        # Test unknown dataset name
        with self.assertRaises(ValueError):
            self.one.describe_dataset('_ibl_foo.bar.baz')

    def test_url_from_path(self):
        """Test OneAlyx.path2url."""
        file = Path(self.tempdir.name).joinpath('cortexlab', 'Subjects', 'KS005', '2019-04-04',
                                                '004', 'alf', '_ibl_wheel.position.npy')
        url = self.one.path2url(file)
        self.assertTrue(url.startswith(self.one.alyx._par.HTTP_DATA_SERVER))
        self.assertTrue('91546fc6-b67c-4a69-badc-5e66088519c4' in url)
        # Check remote mode
        url = self.one.path2url(file, query_type='remote')
        # Local data server path ends with '/public'; remote path does not
        self.assertTrue(url.startswith(self.one.alyx._par.HTTP_DATA_SERVER.rsplit('/', 1)[0]))

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.path2url(file))
        # Check remote mode  FIXME Different behaviour between remote and local modes
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.path2url(file, query_type='remote')

    def test_url_from_record(self):
        """Test ConversionMixin.record2url."""
        idx = (slice(None), UUID('91546fc6-b67c-4a69-badc-5e66088519c4'))
        dataset = self.one._cache['datasets'].loc[idx, :]
        url = self.one.record2url(dataset.squeeze())
        expected = ('https://ibl.flatironinstitute.org/public/'
                    'cortexlab/Subjects/KS005/2019-04-04/004/alf/'
                    '_ibl_wheel.position.91546fc6-b67c-4a69-badc-5e66088519c4.npy')
        self.assertEqual(expected, url)

    def test_load_cache(self):
        """Test loading the remote cache."""
        self.one.alyx.silent = False  # For checking log
        try:
            with self.assertLogs(logging.getLogger('one.api'), logging.INFO) as lg:
                with mock.patch.object(self.one.alyx, 'get', side_effect=HTTPError):
                    self.one.load_cache(clobber=True)
                self.assertEqual('remote', self.one.mode)
                self.assertTrue('Failed to load' in lg.output[0])

                with mock.patch.object(self.one.alyx, 'get', side_effect=ConnectionError):
                    self.one.load_cache(clobber=True)
                    self.assertEqual('local', self.one.mode)
                self.assertTrue('Failed to connect' in lg.output[-1])

                # In online mode, the error cause should suggest re-running setup
                with mock.patch.object(self.one.alyx, 'get', side_effect=FileNotFoundError), \
                        self.assertRaises(FileNotFoundError) as cm:
                    self.one.load_cache(clobber=True)
                self.assertIn('run ONE.setup', str(cm.exception.__cause__))

            cache_info = {'min_api_version': '200.0.0'}
            # Check version verification
            with mock.patch.object(self.one.alyx, 'get', return_value=cache_info), \
                    self.assertWarns(UserWarning):
                self.one.load_cache(clobber=True)

            # Check cache tags
            raw_meta = self.one._cache._meta['raw']
            raw_meta['sessions']['database_tags'] = 'Q3-2020-TAG'
            raw_meta['datasets']['database_tags'] = 'Q3-2020-TAG'
            # Make the remote cache older than current one: for a different tag, the created
            # date should be ignored while for the same tag, the remote cache should not be
            # downloaded if older
            created = datetime.datetime.fromisoformat(raw_meta['datasets']['date_created'])
            created -= datetime.timedelta(days=5)
            cache_info = {
                'min_api_version': '2.11.0', 'database_tags': ['ANOTHER_TAG', 'Q3-2020-TAG'],
                'date_created': created.isoformat(sep=' ', timespec='minutes'),
                'origin': 'ibl_test'}
            files = list(self.one.cache_dir.glob('*.pqt'))
            now = lambda *args, **kwargs: datetime.datetime.now()  # noqa
            with mock.patch.object(
                self.one.alyx, 'download_cache_tables', return_value=files) as cm, \
                    mock.patch.object(self.one.alyx, 'get', return_value=cache_info), \
                    mock.patch('one.api.One.load_cache', side_effect=now):
                self.one.load_cache(tag='ANOTHER_TAG')
            cm.assert_called_once_with(None, Path(self.tempdir.name, 'ANOTHER_TAG'))

            # With the same tag should return with no newer tables log message
            with mock.patch.object(
                self.one.alyx, 'download_cache_tables', return_value=files) as cm, \
                    mock.patch.object(self.one.alyx, 'get', return_value=cache_info), \
                    mock.patch('one.api.One.load_cache', side_effect=now), \
                    self.assertLogs('one.api', 'INFO') as lg:
                expected = self.one._cache['_meta']['loaded_time']
                self.assertEqual(expected, self.one.load_cache(tag='Q3-2020-TAG'))
            cm.assert_not_called()
            self.assertRegex(lg.output[-1], 'No newer cache available')

            # Loaded tables with heterogeneous tags should cause a NotImplemented error when
            # attempting to load another tag
            raw_meta = self.one._cache._meta['raw']
            raw_meta['sessions']['database_tags'] = 'Q3-2020-TAG'
            raw_meta['datasets']['database_tags'] = 'ANOTHER_TAG'
            with mock.patch.object(self.one.alyx, 'get', return_value=cache_info):
                self.assertRaises(NotImplementedError, self.one.load_cache)

            # Check for warning when origins are mixed
            raw_meta['sessions']['origin'] = raw_meta['datasets']['origin'] = 'alyx'
            cache_info['origin'] = 'public'
            cache_info['min_api_version'] = __version__
            cache_info['date_created'] = datetime.datetime.now().isoformat()
            raw_meta['datasets']['database_tags'] = 'Q3-2020-TAG'

            with mock.patch.object(self.one.alyx, 'download_cache_tables', return_value=files), \
                    mock.patch.object(self.one.alyx, 'get', return_value=cache_info), \
                    self.assertWarns(UserWarning, msg='another origin'):
                self.one.load_cache(tag='ANOTHER_TAG')
            self.assertTrue(str(self.one._tables_dir).endswith('ANOTHER_TAG'))
            self.assertTrue(self.one._tables_dir.exists(), 'failed to create tag dir')

            # Check table_dir behaviour
            prev_loc = self.one._tables_dir  # should be same location as previous
            del cache_info['database_tags']
            with mock.patch.object(self.one.alyx, 'download_cache_tables', return_value=files), \
                    mock.patch.object(self.one.alyx, 'get', return_value=cache_info):
                self.one.load_cache(clobber=True)
            self.assertEqual(prev_loc, self.one._tables_dir)
            new_loc = prev_loc.parent  # user input should override default
            with mock.patch.object(self.one.alyx, 'download_cache_tables', return_value=files), \
                    mock.patch.object(self.one.alyx, 'get', return_value=cache_info):
                self.one.load_cache(tables_dir=new_loc, clobber=True)
            self.assertEqual(new_loc, self.one._tables_dir)

        finally:  # Restore properties
            self.one.alyx.silent = True

    def test_check_filesystem(self):
        """Test for One._check_filesystem.

        Most is already covered by other tests, this just checks that it can deal with dataset
        dicts as input.
        """
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        dataset_type = 'probes.description'
        dsets = self.one.alyx.rest('datasets', 'list', session=eid, dataset_type=dataset_type)
        # Create file on disk
        file_ = self.one.eid2path(eid).joinpath('alf', 'probes.description.json')
        file_.parent.mkdir(parents=True)
        file_.touch()
        # Test method
        file, = self.one._check_filesystem(dsets)
        self.assertIsNotNone(file)

    @mock.patch('boto3.Session')
    def test_download_aws(self, boto3_mock):
        """Tests for the OneAlyx._download_aws method."""
        N = 5
        dsets = self.one._cache['datasets'].iloc[:N].copy()
        dsets['exists_aws'] = True

        self.one.mode = 'remote'  # Can't download in local mode

        # Return a file size so progress bar callback hook functions
        file_object = mock.MagicMock()
        file_object.content_length = 1024
        boto3_mock().resource().Object.return_value = file_object

        # Mock _download_dataset for safety: method should not be called
        with mock.patch.object(self.one, '_download_dataset') as fallback_method:
            out_paths = self.one._download_datasets(dsets)
            fallback_method.assert_not_called()
        self.assertEqual(len(out_paths), N, 'Returned list should equal length of input datasets')
        self.assertTrue(all(isinstance(x, Path) for x in out_paths))
        # These values come from REST cache fixture
        boto3_mock.assert_called_with(aws_access_key_id='ABCDEF', aws_secret_access_key='shhh',
                                      region_name=None)
        ((bucket, path), _), *_ = boto3_mock().resource().Object.call_args_list
        self.assertEqual(bucket, 's3_bucket')
        self.assertTrue(dsets['rel_path'][0].split('.')[0] in path)
        self.assertTrue(str(dsets.index[0][1]) in path, 'Dataset UUID not in filepath')

        # Should fall back to usual method if any datasets do not exist on AWS
        dsets['exists_aws'] = False
        with mock.patch.object(self.one, '_download_dataset') as fallback_method:
            self.one._download_datasets(dsets)
            fallback_method.assert_called()

    def test_list_aggregates(self):
        """Test OneAlyx.list_aggregates."""
        # Test listing by relation
        datasets = self.one.list_aggregates('subjects')
        self.assertTrue(all(datasets['rel_path'].str.startswith('aggregates/Subjects')))
        self.assertTrue(self.one.list_aggregates('foobar').empty)
        # Test filtering with an identifier
        datasets = self.one.list_aggregates('subjects', 'ZM_1085')
        expected = 'aggregates/Subjects/mainenlab/ZM_1085'
        self.assertTrue(all(datasets['rel_path'].str.startswith(expected)))
        self.assertTrue(self.one.list_aggregates('subjects', 'foobar').empty)
        # Test that additional parts of data path are correctly removed
        # specifically /public in FI openalyx file rec
        mock_ds = [
            {'url': '3ef042c6-82a4-426c-9aa9-be3b48d86652',
             'session': None, 'file_size': None, 'hash': '',
             'file_records': [{'data_url': 'https://ibl-brain-wide-map-public.s3.amazonaws.com/'
                                           'aggregates/Subjects/cortexlab/KS051/'
                                           'trials.table.3ef042c6-82a4-426c-9aa9-be3b48d86652.pqt',
                               'data_repository': 'aws_aggregates',
                               'exists': True}],
             'default_dataset': True,
             'qc': 'NOT_SET'},
            {'url': '7bdb08d6-b166-43d8-8981-816cf616d291',
             'session': None, 'file_size': None, 'hash': '',
             'file_records': [{'data_url': 'https://ibl.flatironinstitute.org/'
                                           'public/aggregates/Subjects/mrsicflogellab/IBL_005/'
                                           'trials.table.7bdb08d6-b166-43d8-8981-816cf616d291.pqt',
                               'data_repository': 'flatiron_aggregates',
                               'exists': True}],
             'default_dataset': True,
             'qc': 'NOT_SET'},
        ]
        with mock.patch.object(self.one.alyx, 'rest', return_value=mock_ds):
            dsets = self.one.list_aggregates('subjects')
            self.assertEqual(len(dsets), 2)
            # Should handle null file_size values correctly
            self.assertIsInstance(dsets.file_size.dtype, pd.UInt64Dtype)
            self.assertTrue(dsets.file_size.isna().all())

    def test_load_aggregate(self):
        """Test OneAlyx.load_aggregate."""
        # Test object not found on disk
        assert self.one.offline
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_aggregate('subjects', 'ZM_1085', '_ibl_subjectTraining.table.pqt')

        # Touch a file to ensure that we do not try downloading
        expected = self.one.cache_dir.joinpath(
            'aggregates/Subjects/mainenlab/ZM_1085/_ibl_subjectTraining.table.pqt')
        expected.parent.mkdir(parents=True), expected.touch()

        # Test loading with different input dataset formats
        datasets = ['_ibl_subjectTraining.table.pqt',
                    '_ibl_subjectTraining.table',
                    {'object': 'subjectTraining', 'attribute': 'table'}]
        for dset in datasets:
            with self.subTest(dataset=dset):
                file = self.one.load_aggregate('subjects', 'ZM_1085', dset, download_only=True)
                self.assertEqual(expected, file)

        # Test object not found
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_aggregate('subjects', 'ZM_1085', 'foo.bar')

        # Test download file from HTTP dataserver
        expected.unlink()
        self.one.mode = 'remote'  # Can't download in local mode
        with mock.patch.object(self.one, '_download_file', return_value=[expected]) as m, \
                mock.patch.object(self.one, '_download_aws', side_effect=AssertionError):
            file = self.one.load_aggregate('subjects', 'ZM_1085', dset, download_only=True)
            # Check correct url passed to download_file
            self.assertEqual(expected, file)
            expected_src = (self.one.alyx._par.HTTP_DATA_SERVER +
                            '/aggregates/Subjects/mainenlab/ZM_1085/' +
                            '_ibl_subjectTraining.table.74dfb745-a7dc-4672-ace6-b556876c80cb.pqt')
            expected_dst = str(file.parent)  # should be without 'public' part
            m.assert_called_once_with([expected_src], [expected_dst], keep_uuid=False)
        # Test download file from AWS
        with mock.patch('one.remote.aws.s3_download_file', return_value=expected) as m, \
                mock.patch.object(self.one, '_download_file', side_effect=AssertionError):
            file = self.one.load_aggregate('subjects', 'ZM_1085', dset, download_only=True)
            # Check correct url passed to download_file
            self.assertEqual(expected, file)
            expected_src = PurePosixPath(
                expected_src[len(self.one.alyx._par.HTTP_DATA_SERVER) + 1:])
            m.assert_called_once_with(
                expected_src, expected, s3=mock.ANY, bucket_name='s3_bucket', overwrite=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneRemote(unittest.TestCase):
    """Test remote queries using OpenAlyx."""

    def setUp(self) -> None:
        self.one = OneAlyx(**TEST_DB_2, mode='remote')
        self.eid = UUID('4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a')
        self.pid = UUID('da8dfec1-d265-44e8-84ce-6ae9c109b8bd')
        # Set cache directory to a temp dir to ensure that we re-download files
        self.tempdir = tempfile.TemporaryDirectory()
        self.one.alyx._par = self.one.alyx._par.set('CACHE_DIR', Path(self.tempdir.name))

    def test_online_repr(self):
        """Tests OneAlyx.__repr__."""
        self.assertTrue('online' in str(self.one))
        self.assertTrue(TEST_DB_2['base_url'] in str(self.one))

    def test_list_datasets(self):
        """Test OneAlyx.list_datasets."""
        # Test list for eid
        # Ensure remote by making local datasets table empty
        self.addCleanup(self.one.load_cache)
        self.one._cache['datasets'] = self.one._cache['datasets'].iloc[0:0].copy()

        dsets = self.one.list_datasets(self.eid, details=True, query_type='remote')
        expected_n_datasets = 253  # this may change after a BWM release or patch
        self.assertEqual(expected_n_datasets, len(dsets))
        self.assertEqual(1, dsets.index.nlevels, 'details data frame should be without eid index')

        # Test keep_eid_index
        dsets = self.one.list_datasets(
            self.eid, details=True, query_type='remote', keep_eid_index=True)
        self.assertEqual(2, dsets.index.nlevels, 'details data frame should be with eid index')

        # Test missing eid
        dsets = self.one.list_datasets('FMR019/2021-03-18/008', details=True, query_type='remote')
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertEqual(len(dsets), 0)

        # Test empty datasets
        with mock.patch('one.api.ses2records', return_value=(pd.DataFrame(), pd.DataFrame())):
            dsets = self.one.list_datasets(self.eid, details=True, query_type='remote')
            self.assertIsInstance(dsets, pd.DataFrame)
            self.assertEqual(len(dsets), 0)

        # Test details=False, with eid
        dsets = self.one.list_datasets(self.eid, details=False, query_type='remote')
        self.assertIsInstance(dsets, list)
        self.assertEqual(expected_n_datasets, len(dsets))

        # Test with other filters
        dsets = self.one.list_datasets(self.eid, collection='*probe*', filename='*channels*',
                                       details=False, query_type='remote')
        self.assertEqual(36, len(dsets))
        self.assertTrue(all(x in y for x in ('probe', 'channels') for y in dsets))

        with self.assertWarns(Warning):
            self.one.list_datasets(query_type='remote')

    def test_search(self):
        """Test OneAlyx.search method in remote mode."""
        # Load the cache tables
        self.one.load_cache()

        # Modify sessions dataframe so we can check that the records get updated
        records = self.one._cache.sessions[self.one._cache.sessions.subject == 'SWC_043']
        self.one._cache.sessions.loc[records.index, 'lab'] = 'foolab'  # change a field
        self.one._cache.sessions.drop(self.eid, inplace=True)  # remove a row

        # Check remote seach of subject
        eids = self.one.search(subject='SWC_043', query_type='remote')
        self.assertIn(self.eid, list(eids))
        updated = self.one._cache.sessions[self.one._cache.sessions.subject == 'SWC_043']
        self.assertCountEqual(eids, updated.index)
        self.assertFalse('foolab' in updated['lab'])

        eids, d = self.one.search(subject='SWC_043', query_type='remote', details=True)
        correct = len(d) == len(eids) and 'url' in d[0] and d[0]['url'].endswith(str(eids[0]))
        self.assertTrue(correct)

        # Check minimum set of keys present (these are present in One.search output)
        expected = {'lab', 'subject', 'date', 'number', 'projects'}
        self.assertTrue(d[0].keys() >= expected)
        # Test dataset search with Django
        query = ['data_dataset_session_related__collection__iexact,alf',
                 'data_dataset_session_related__name__startswith,probes.description']
        eids = self.one.search(subject='SWC_043', number=1, django=query, query_type='remote')
        self.assertIn(self.eid, list(eids))

        # Test date range
        eids = self.one.search(subject='SWC_043', date='2020-09-21', query_type='remote')
        self.assertCountEqual(eids, [self.eid])

        date_range = [datetime.date(2020, 9, 21), datetime.date(2020, 9, 22)]
        eids = self.one.search(date=date_range, lab='hoferlab', query_type='remote')
        self.assertIn(self.eid, list(eids))
        dates = set(map(lambda x: self.one.get_details(x)['date'], eids))
        self.assertTrue(dates <= set(date_range))

        # Test limit arg, LazyId, and update with paginated response callback
        self.one._reset_cache()  # Remove sessions table
        assert self.one._cache.sessions.empty
        eids = self.one.search(date='2020-03-23', limit=2, query_type='remote')
        self.assertEqual(2, len(self.one._cache.sessions),
                         'failed to update cache with first page of search results')
        self.assertIsInstance(eids, LazyId)
        assert len(eids) > 5, 'in order to check paginated response callback we need several pages'
        e = eids[-3]  # access an uncached value
        self.assertEqual(
            4, len(self.one._cache.sessions), 'failed to update cache after page access')
        self.assertTrue(e in self.one._cache.sessions.index)
        self.assertTrue(all(isinstance(x, UUID) for x in eids))
        self.assertEqual(len(eids), len(self.one._cache.sessions))

        # Test laboratory kwarg
        eids = self.one.search(laboratory='hoferlab', query_type='remote')
        self.assertIn(self.eid, list(eids))

        eids = self.one.search(lab='hoferlab', query_type='remote')
        self.assertIn(self.eid, list(eids))

        # Test dataset and dataset_types kwargs
        eids = self.one.search(datasets='_ibl_trials.table.pqt', query_type='remote')
        self.assertIn(self.eid, list(eids))
        eids = self.one.search(datasets=['_ibl_trials.intervals.npy'], query_type='remote')
        self.assertNotIn(self.eid, list(eids))
        # The dataset arg with partial matching has been retired and should raise a value error
        self.assertRaises(ValueError, self.one.search, dataset='wheel.times', query_type='remote')

        eids = self.one.search(dataset_type='_ibl_trials.table.pqt', query_type='remote')
        self.assertEqual(0, len(eids))
        eids = self.one.search(dataset_type='trials.table', date='2020-09-21', query_type='remote')
        self.assertIn(self.eid, list(eids))

        # Ensure that when calling with anything other than remote mode, One.search is used
        with mock.patch('one.api.One.search') as offline_search, \
                mock.patch.object(self.one.alyx, 'rest', return_value=[]) as alyx:
            # In remote mode
            self.one.search(subject='SWC_043', query_type='remote')
            offline_search.assert_not_called(), alyx.assert_called()
            alyx.reset_mock()
            # In another mode
            self.one.search(subject='SWC_043', query_type='local')
            offline_search.assert_called_with(details=False, query_type='local', subject='SWC_043')
            alyx.assert_not_called()

    def test_search_insertions(self):
        """Test OneAlyx.search_insertion method in remote mode."""
        # Test search on subject
        pids = self.one.search_insertions(subject='SWC_043', query_type='remote')
        self.assertIn(self.pid, list(pids))

        # Test search on session with details
        pids, det = self.one.search_insertions(session=self.eid, query_type='remote', details=True)
        self.assertIn(self.pid, list(pids))
        correct = len(det) == len(pids) and 'id' in det[0] and det[0]['id'] == str(self.pid)
        self.assertTrue(correct)

        # Test search on session and insertion name
        pids = self.one.search_insertions(session=self.eid, name='probe00', query_type='remote')
        self.assertEqual(pids[0], self.pid)

        # Should work in local mode but with debug message
        pids = self.one.search_insertions(name='probe00', query_type='local')
        self.assertIn(self.pid, list(pids))

        # Test search with acronym (remote only) raises value error
        with self.assertRaises(ValueError):
            self.one.search_insertions(atlas_acronym='STR', query_type='local')

        # Expect this list of acronyms to return nothing
        pids = self.one.search_insertions(atlas_acronym=['STR', 'CA3'], query_type='remote')
        self.assertEqual(0, len(pids))

        # Test 'special' params (these have different names on the REST side)
        # - full 'laboratory' word
        # - 'dataset' as singular with fuzzy match
        # - 'number' -> 'experiment_number'
        lab = 'cortexlab'
        _, det = self.one.search_insertions(
            laboratory=lab, number=1, dataset='_ibl_log.info',
            atlas_acronym='STR', query_type='remote', details=True)
        self.assertEqual(14, len(det))
        self.assertEqual({lab}, {x['session_info']['lab'] for x in det})

        # Test mode and field validation
        self.assertRaises(TypeError, self.one.search_insertions,
                          dataset=['wheel.times'], query_type='remote')
        # Ensure that when calling with anything other than remote mode, One is used
        with mock.patch('one.api.One._search_insertions') as offline_search, \
                mock.patch.object(self.one.alyx, 'rest', return_value=[]) as alyx:
            # In remote mode
            self.one.search_insertions(subject='SWC_043', query_type='remote')
            offline_search.assert_not_called(), alyx.assert_called()
            alyx.reset_mock()
            # In local mode
            self.one.search_insertions(subject='SWC_043', query_type='local')
            offline_search.assert_called_with(details=False, query_type='local', subject='SWC_043')
            alyx.assert_not_called()

        # Test limit arg, LazyId, and update with paginated response callback
        self.one._reset_cache()  # Remove insertions table
        assert 'insertions' not in self.one._cache
        pids = self.one.search_insertions(limit=2, query_type='remote')
        self.assertEqual(2, len(self.one._cache.insertions),
                         'failed to update insertions cache with first page of search results')
        self.assertEqual(2, len(self.one._cache.sessions),
                         'failed to update sessions cache with first page of search results')
        self.assertIsInstance(pids, LazyId)
        assert len(pids) > 5, 'in order to check paginated response callback we need several pages'
        p = pids[-3]  # access an uncached value
        self.assertEqual(4, len(self.one._cache.insertions),
                         'failed to update insertions cache after page access')
        self.assertEqual(4, len(self.one._cache.sessions),
                         'failed to update insertions cache after page access')
        self.assertTrue(p in self.one._cache.insertions.index.get_level_values('id'))

    def test_search_terms(self):
        """Test OneAlyx.search_terms."""
        self.one.mode = 'local'
        search1 = self.one.search_terms()
        self.assertIn('datasets', search1)

        search2 = self.one.search_terms(endpoint='sessions')
        self.assertEqual(search1, search2)

        search3 = self.one.search_terms(query_type='remote', endpoint='sessions')
        self.assertIn('django', search3)

        search4 = self.one.search_terms(endpoint='insertions')
        self.assertIsNone(search4)

        search5 = self.one.search_terms(query_type='remote', endpoint='insertions')
        self.assertIn('model', search5)

    def test_load_dataset(self):
        """Test OneAlyx.load_dataset."""
        file = self.one.load_dataset(self.eid, '_spikeglx_sync.channels.npy',
                                     collection='raw_ephys_data', query_type='remote',
                                     download_only=True)
        self.assertIsInstance(file, Path)
        self.assertTrue(file.as_posix().endswith('raw_ephys_data/_spikeglx_sync.channels.npy'))
        # Test validations
        with self.assertRaises(alferr.ALFMultipleCollectionsFound):
            self.one.load_dataset(self.eid, 'spikes.clusters', query_type='remote')
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.load_dataset(self.eid, '_iblrig_*Camera.raw', query_type='remote')
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset(self.eid, '_iblrig_encoderEvents.raw.ssv',
                                  collection='alf', query_type='remote')

    def test_load_object(self):
        """Test OneAlyx.load_object."""
        files = self.one.load_object(self.eid, 'wheel',
                                     collection='alf', query_type='remote',
                                     download_only=True)
        self.assertIsInstance(files[0], Path)
        self.assertTrue(
            files[0].as_posix().endswith('SWC_043/2020-09-21/001/alf/_ibl_wheel.timestamps.npy')
        )

    def test_get_details(self):
        """Test OneAlyx.get_details."""
        det = self.one.get_details(self.eid, query_type='remote')
        self.assertIsInstance(det, dict)
        self.assertEqual('SWC_043', det['subject'])
        self.assertEqual('ibl_neuropixel_brainwide_01', det['projects'])
        self.assertEqual('2020-09-21', str(det['date']))
        self.assertEqual(1, det['number'])
        self.assertNotIn('data_dataset_session_related', det)

        # Test with a list
        # For duplicate eids, we should avoid multiple queries
        with mock.patch.object(AlyxClient, 'rest', wraps=self.one.alyx.rest) as m:
            det = self.one.get_details([self.eid, self.eid], full=True)
        m.assert_called_once_with('sessions', 'read', id=str(self.eid))
        self.assertIsInstance(det, list)
        self.assertEqual(2, len(det))
        self.assertIn('data_dataset_session_related', det[0])
        # Check that the details dicts are copies (modifying one should not affect the other)
        self.assertEqual(det[0], det[1])  # details should be the same
        self.assertIsNot(det[0], det[1])  # should be different objects

    def test_cache_buildup(self):
        """Test build up of cache table via remote queries.

        Tests a regression where a cache table built up from remote queries could not be loaded
        as the cache table meta data was empty.
        """
        assert not any(self.one.cache_dir.glob('*.pqt'))  # tempdir must be empty
        # Check that default cache is empty with raw meta data fields defined
        meta_raw = self.one._cache['_meta']['raw']
        self.assertCountEqual(('sessions', 'datasets'), meta_raw)
        # Should be empty origin and a defined created date
        self.assertTrue(all(
            x['origin'] == set() and isinstance(x['date_created'], str)
            and len(x['date_created']) > 0) for x in meta_raw.values())
        # Update the session table from a remote query
        self.one.search(subject='KS005')
        # Database URL should be added as origin to the sessions table meta only
        self.assertEqual(set(), meta_raw['datasets']['origin'])
        self.assertEqual({self.one.alyx.base_url}, meta_raw['sessions']['origin'])
        self.one.save_cache()  # write the tables to disk
        self.assertEqual(2, len(list(self.one.cache_dir.glob('*.pqt'))))
        # Load the cache from disk into a new offline instance
        # Previously this skipped the tables for being invalid
        one = ONE(cache_dir=self.one.cache_dir, mode='local')
        self.assertTrue(one._cache.datasets.empty)
        self.assertFalse(one._cache.sessions.empty)
        meta_raw = one._cache['_meta']['raw']
        # save method should have added the date_modified field
        self.assertIn('date_modified', meta_raw['sessions'])
        self.assertEqual(set(), meta_raw['datasets']['origin'])
        self.assertEqual({self.one.alyx.base_url}, meta_raw['sessions']['origin'])

    def tearDown(self):
        """Ensure the cache is not saved when the object is deleted."""
        self.one._cache['_meta']['saved_time'] = datetime.datetime.max
        self.tempdir.cleanup()


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneDownload(unittest.TestCase):
    """Test downloading datasets using OpenAlyx."""

    tempdir = None
    one = None

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = mock.patch('one.params.iopar.getfile',
                                new=partial(util.get_file, self.tempdir.name))
        self.patch.start()
        self.one = OneAlyx(**TEST_DB_2, cache_dir=self.tempdir.name)
        self.one.load_cache()
        self.fid = UUID('6f175a7a-e20b-4622-81fc-08947a4fd1d3')  # File record of wiring.json
        self.eid = UUID('aad23144-0e52-4eac-80c5-c4ee2decb198')
        self.did = UUID('d693fbf9-2f90-4123-839e-41474c44742d')

    def test_download_datasets(self):
        """Test OneAlyx._download_dataset, _download_file and _dset2url."""
        det = self.one.get_details(self.eid, True)
        rec = next(x for x in det['data_dataset_session_related'] if x['id'] == str(self.did))
        # FIXME hack because data_url may be AWS
        rec['data_url'] = self.one.alyx.rel_path2url(get_alf_path(rec['data_url']))
        file = self.one._download_dataset(rec)
        self.assertIsInstance(file, Path)
        self.assertTrue(file.exists())

        url = rec['data_url']
        file = self.one._download_dataset(url)
        self.assertIsNotNone(file)

        rec = self.one.alyx.get(rec['url'])
        file = self.one._download_dataset(rec)
        self.assertIsNotNone(file)

        # Check behaviour when hash mismatch
        self.one.alyx.silent = False  # So we can check for warning
        file_hash = rec['hash'].replace('a', 'd')
        # Check three things:
        # 1. The mismatch should be logged at the debug level
        # 2. As we don't have permission to update this db we should see a failure warning
        with self.assertLogs(logging.getLogger('one.api'), logging.DEBUG), \
                self.assertWarns(Warning, msg=f'files/{self.fid}'):
            self.one._download_dataset(rec, hash=file_hash)
        self.one.alyx.silent = True  # Remove console clutter

        # Check JSON field added
        # 3. The files endpoint should be called with a 'mismatch_hash' json key
        fr = [{'url': f'files/{str(self.fid)}', 'json': None}]
        with mock.patch.object(self.one.alyx, '_generic_request', return_value=fr) as patched:
            self.one._download_dataset(rec, hash=file_hash)
            args, kwargs = patched.call_args
            self.assertEqual(kwargs.get('data', {}), {'json': {'mismatch_hash': True}})

        # Check keep_uuid kwarg
        # FIXME Another hack: for this to work the file records order must be correct.
        fi = next(i for i, x in enumerate(rec['file_records'])
                  if x['data_url'].startswith(self.one.alyx._par.HTTP_DATA_SERVER))
        if fi != 0:
            rec['file_records'] = [rec['file_records'].pop(fi), *rec['file_records']]
        file = self.one._download_dataset(rec, keep_uuid=True)
        self.assertEqual(file.stem.split('.')[-1], rec['url'].split('/')[-1])

        # Check list input
        recs = [rec, sorted(det['data_dataset_session_related'], key=lambda x: x['file_size'])[1]]
        files = self.one._download_dataset(recs)
        self.assertIsInstance(files, list)
        self.assertTrue(all(isinstance(x, Path) for x in files))

        # Check Series input
        r_ = datasets2records(rec).squeeze()
        file = self.one._download_dataset(r_)
        self.assertIn('imec0.wiring.json', file.name)

        # Check behaviour when URL invalid
        did = UUID(rec['url'].split('/')[-1])
        self.assertTrue(self.one._cache.datasets.loc[(slice(None), did), 'exists'].all())
        for fr in rec['file_records']:
            fr['data_url'] = None
        file = self.one._download_dataset(rec)
        self.assertIsNone(file)
        self.assertFalse(self.one._cache.datasets.loc[(slice(None), did), 'exists'].all())
        # With multiple dsets
        files = self.one._download_dataset([rec, rec])
        self.assertTrue(all(x is None for x in files))
        self.one._cache.datasets.loc[(slice(None), did), 'exists'] = True  # Reset values

        # Check with invalid path
        path = self.one.cache_dir.joinpath(
            'lab', 'Subjects', 'subj', '2020-01-01', '001', 'spikes.times.npy')
        with self.assertLogs(logging.getLogger('one.api'), logging.WARNING):
            file = self.one._download_dataset(path)
            self.assertIsNone(file)

        # Check data frame record
        rec = self.one.list_datasets(self.eid, details=True)
        rec = rec[rec.rel_path.str.contains('00/_spikeglx_ephysData_g0_t0.imec0.wiring')]
        rec.loc[self.did, 'exist_aws'] = False  # Ensure we use FlatIron for this
        rec = pd.concat({self.eid: rec}, names=['eid'])  # Add back eid index

        files = self.one._download_datasets(rec)
        self.assertFalse(None in files)

        # Check behaviour when dataset missing
        with mock.patch.object(self.one, 'record2url', return_value=None):
            self.assertIsNone(self.one._download_dataset(rec.squeeze()))

    def test_download_aws(self):
        """Test for OneAlyx._download_aws method."""
        # Test download datasets via AWS
        dsets = self.one.list_datasets(
            self.eid, filename='*wiring.json', collection='raw_ephys_data/probe??', details=True)
        dsets = pd.concat({self.eid: dsets}, names=['eid'])
        assert len(dsets) == 2

        file = self.one.eid2path(self.eid) / dsets['rel_path'].values[0]
        with mock.patch('one.remote.aws.get_s3_from_alyx', return_value=(None, None)), \
                mock.patch('one.remote.aws.s3_download_file', return_value=file) as method:
            self.one._download_datasets(dsets)
            self.assertEqual(len(dsets), method.call_count)
            # Check output filename
            _, local = method.call_args.args
            self.assertTrue(local.as_posix().startswith(self.one.cache_dir.as_posix()))
            self.assertTrue(local.as_posix().endswith(dsets.iloc[-1]['rel_path']))
            # Check keep_uuid = True
            self.one._download_datasets(dsets, keep_uuid=True)
            _, local = method.call_args.args
            self.assertIn(str(dsets.iloc[-1].name[1]), local.name)

        # Test behaviour when dataset not remotely accessible on S3
        dsets = dsets[:1].copy()
        rec = self.one.alyx.rest('datasets', 'read', id=dsets.index[0][1])
        # need to find the index of matching aws repo, this is not constant across releases
        iaws = list(map(lambda x: x['data_repository'].startswith('aws'),
                        rec['file_records'])).index(True)
        rec['file_records'][iaws]['exists'] = False  # Set AWS file record to non-existent
        self.one._cache.datasets['exists_aws'] = True  # Only changes column if exists
        with mock.patch('one.remote.aws.get_s3_from_alyx', return_value=(None, None)), \
                mock.patch.object(self.one.alyx, 'rest', return_value=[rec]), \
                mock.patch.object(self.one, '_download_dataset') as mock_method, \
                self.assertLogs('one.api', logging.DEBUG) as log:
            # should still download file via fallback method
            self.one._download_datasets(dsets)
            mock_method.assert_called_once()
            self.assertTrue(all(dsets == mock_method.call_args.args[0]), 'failed to pass on dsets')
            self.assertRegex(log.output[1], 'Updating exists field')
            # Expect it to have updated datasets table
            datasets = self.one._cache['datasets']
            self.assertFalse(
                datasets.loc[pd.IndexSlice[:, dsets.index[0][1]], 'exists_aws'].any()
            )

        # Check falls back to HTTP when error raised
        with mock.patch('one.remote.aws.get_s3_from_alyx', side_effect=RuntimeError), \
                mock.patch.object(self.one, '_download_dataset') as mock_method:
            self.one._download_datasets(dsets)
            mock_method.assert_called_with(dsets, keep_uuid=False)
        # Test type check (download_aws only works with data frames)
        with mock.patch.object(self.one, '_download_dataset') as mock_method:
            self.one._download_datasets(dsets.to_dict('records'))
            mock_method.assert_called()

    def test_tag_mismatched_file_record(self):
        """Test for OneAlyx._tag_mismatched_file_record.

        This method is also tested in test_download_datasets.
        """
        did = '4a1500c2-60f3-418f-afa2-c752bb1890f0'
        url = f'https://example.com/channels.brainLocationIds_ccf_2017.{did}.npy'
        data = [{'json': {'mismatch_hash': False}, 'url': f'https://example.com/files/{did}'}]
        with mock.patch.object(self.one.alyx, 'rest', return_value=data) as mk:
            self.one._tag_mismatched_file_record(url)
        data[0]['json']['mismatch_hash'] = True
        mk.assert_called_with('files', 'partial_update', id=did, data={'json': data[0]['json']})

    def tearDown(self) -> None:
        """Remove any created file records and ensure cache not saved when object deleted."""
        self.one._cache['_meta']['saved_time'] = datetime.datetime.max
        try:
            # In case we did indeed have remote REST permissions, try resetting the json field
            self.one.alyx.rest('files', 'partial_update', id=self.fid, data={'json': None})
        except HTTPError as ex:
            if ex.errno != 403:
                raise ex
        self.patch.stop()
        self.tempdir.cleanup()


class TestOneSetup(unittest.TestCase):
    """Test parameter setup upon ONE instantiation and calling setup methods."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.get_file = partial(util.get_file, self.tempdir.name)
        # Change default cache dir to temporary directory
        patch = mock.patch('one.params.CACHE_DIR_DEFAULT', Path(self.tempdir.name))
        patch.start()
        self.addCleanup(patch.stop)

    def test_local_cache_setup_prompt(self):
        """Test One.setup."""
        path = Path(self.tempdir.name).joinpath('subject', '2020-01-01', '1', 'spikes.times.npy')
        path.parent.mkdir(parents=True)
        path.touch()
        with mock.patch('builtins.input', return_value=self.tempdir.name):
            one_obj = One.setup()
        self.assertCountEqual(one_obj.list_datasets(), ['spikes.times.npy'])

        # Check prompts warns about cache existing
        path.parent.joinpath('spikes.clusters.npy').touch()
        one_obj = One.setup(cache_dir=self.tempdir.name, silent=True)
        self.assertEqual(1, len(one_obj.list_datasets()))
        with mock.patch('builtins.input', side_effect=['n', 'y']):
            one_obj = One.setup(cache_dir=self.tempdir.name)  # Reply no to cache overwrite
            self.assertEqual(1, len(one_obj.list_datasets()))
            one_obj = One.setup(cache_dir=self.tempdir.name)  # Reply yes to cache overwrite
            self.assertEqual(2, len(one_obj.list_datasets()))

        # Test with no prompt
        one_obj = One.setup(cache_dir=self.tempdir.name, silent=True)
        assert len(one_obj.list_datasets()) == 2

    def test_local_tables_only(self):
        """Test first time instantiation of ONE with only tables_dir arg.

        Previously this would have raised and AttributeError as cache_map.DEFAULT
        doesn't exist on a new install.
        """
        # Expect warning as table_dir empty
        with mock.patch('iblutil.io.params.getfile', new=self.get_file), \
                self.assertWarns(UserWarning):
            # Should instantiate without error
            one_obj = One(tables_dir=Path(self.tempdir.name, 'tables'))
        self.assertEqual(self.tempdir.name, str(one_obj.cache_dir))
        self.assertEqual('tables', one_obj._tables_dir.name)

    def test_setup_silent(self):
        """Test setting up parameters with silent flag.

        - Mock getfile to return temp dir as param file location
        - Mock input function as fail safe in case function erroneously prompts user for input
        """
        with mock.patch('iblutil.io.params.getfile', new=self.get_file), \
                mock.patch('one.params.input', new=self.assertFalse):
            one_obj = ONE(silent=True, mode='local', password=TEST_DB_2['password'])
            self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

        # Check param files were saved
        self.assertEqual(len(list(Path(self.tempdir.name).rglob('.caches'))), 1)
        client_pars = Path(self.tempdir.name).rglob(f'.{one_obj.alyx.base_url.split("/")[-1]}')
        self.assertEqual(len(list(client_pars)), 1)

        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            # Check uses defaults on second instantiation
            one_obj = ONE(mode='local')
            url = one.params.default().ALYX_URL
            self.assertEqual(url, one_obj.alyx.base_url)
            # With the default database in silent mode the defaults should be used
            one.params.save(one_obj.alyx._par.set('ALYX_LOGIN', 'foobar'), url)
            one.params.setup(url, silent=True)
            one_obj = ONE(mode='local')
            self.assertEqual(one.params.default().ALYX_LOGIN, one_obj.alyx._par.ALYX_LOGIN)

        # Check saves base_url arg
        with self.subTest('Test setup with base URL'):
            if OFFLINE_ONLY:
                self.skipTest('Requires remote db connection')
            with mock.patch('iblutil.io.params.getfile', new=self.get_file):
                one_obj = ONE(**TEST_DB_1)
                self.assertEqual(one_obj.alyx.base_url, TEST_DB_1['base_url'])
                params_url = one.params.get(client=TEST_DB_1['base_url']).ALYX_URL
                self.assertEqual(params_url, one_obj.alyx.base_url)
                # With non-default database in silent mode the previous pars should be used
                one.params.setup(params_url, silent=True)
                user = one.params.get(client=params_url).ALYX_LOGIN
                self.assertEqual(TEST_DB_1['username'], user)

    def test_setup_username(self):
        """Test setting up parameters with a provided username.

        - Mock getfile to return temp dir as param file location
        - Mock input function as fail safe in case function erroneously prompts user for input
        - Mock requests.post returns a fake user authentication response
        """
        credentials = {'username': 'foobar', 'password': '123'}
        with mock.patch('iblutil.io.params.getfile', new=self.get_file), \
                mock.patch('one.params.input', new=self.assertFalse), \
                mock.patch('one.webclient.requests.post') as req_mock:
            req_mock().json.return_value = {'token': 'shhh'}
            # In remote mode the cache endpoint will not be queried
            one_obj = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                          silent=True, mode='remote', **credentials)
            params_username = one.params.get(client=TEST_DB_1['base_url']).ALYX_LOGIN
            self.assertEqual(params_username, one_obj.alyx.user)
            self.assertEqual(credentials['username'], one_obj.alyx.user)
            self.assertEqual(req_mock.call_args.kwargs.get('data', {}), credentials)

            # Reinstantiate as a different user
            one_obj = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                          username='baz', password='123', mode='remote')
            self.assertEqual(one_obj.alyx.user, 'baz')
            self.assertEqual(one_obj.alyx.user, one_obj.alyx._par.ALYX_LOGIN)
            # After initial set up the username in the pars file should remain unchanged
            params_username = one.params.get(client=TEST_DB_1['base_url']).ALYX_LOGIN
            self.assertNotEqual(one_obj.alyx.user, params_username)

    @unittest.skipIf(OFFLINE_ONLY, 'online only test')
    def test_static_setup(self):
        """Test OneAlyx.setup."""
        with mock.patch('iblutil.io.params.getfile', new=self.get_file), \
                mock.patch('one.webclient.getpass', return_value='international'):
            one_obj = OneAlyx.setup(silent=True)
        self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

    def test_setup(self):
        """Test one.params.setup."""
        url = TEST_DB_1['base_url']

        def mock_input(prompt):
            if prompt.casefold().startswith('warning'):
                if not getattr(mock_input, 'conflict_warn', False):    # Checks both responses
                    mock_input.conflict_warn = True
                    return 'y'
                return 'n'
            elif 'download cache' in prompt.casefold():
                return Path(self.tempdir.name).joinpath('downloads').as_posix()
            elif 'url' in prompt.casefold():
                return url
            else:
                return 'mock_input'
        one.params.input = mock_input
        one.params.getpass = lambda prompt: 'mock_pwd'
        one.params.print = lambda text: 'mock_print'
        # Mock getfile function to return a path to non-existent file instead of usual one pars
        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            one_obj = OneAlyx(mode='local',
                              username=TEST_DB_1['username'],
                              password=TEST_DB_1['password'])
            pars = one.params.get(url)
            self.assertFalse('ALYX_PWD' in pars.as_dict())
        self.assertEqual(one_obj.alyx._par.ALYX_URL, url)
        client = f'.{one_obj.alyx.base_url.split("/")[-1]}'.replace(':', '_')
        client_pars = Path(self.tempdir.name).rglob(client)
        self.assertEqual(len(list(client_pars)), 1)
        # Save ALYX_PWD into params and see if setup modifies it
        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            one.params.save(pars.set('ALYX_PWD', 'foobar'), url)
            one.params.setup(url)
            self.assertEqual(one.params.get(url).ALYX_PWD, 'mock_pwd')

        # Check conflict warning
        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            OneAlyx(mode='local',
                    base_url=TEST_DB_2['base_url'],
                    username=TEST_DB_2['username'],
                    password=TEST_DB_2['password'])
        self.assertTrue(getattr(mock_input, 'conflict_warn', False))

    def test_patch_params(self):
        """Test patching legacy params to the new location."""
        # Save some old-style params
        old_pars = one.params.default().set('HTTP_DATA_SERVER', 'openalyx.org')
        # Save a REST query in the old location
        old_rest = Path(self.tempdir.name, '.one', '.rest', old_pars.ALYX_URL[8:], 'https')
        old_rest.mkdir(parents=True, exist_ok=True)
        old_rest.joinpath('1baff95c2d0e31059720a3716ad5b5a34b61a207').touch()

        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            one.params.setup(silent=True)
            one.params.save(old_pars, old_pars.ALYX_URL)
            one_obj = ONE(base_url=old_pars.ALYX_URL, mode='local')
        self.assertEqual(one_obj.alyx._par.HTTP_DATA_SERVER, one.params.default().HTTP_DATA_SERVER)
        self.assertFalse(Path(self.tempdir.name, '.rest').exists())
        self.assertTrue(any(one_obj.alyx.cache_dir.joinpath('.rest').glob('*')))

    def test_one_factory(self):
        """Tests the ONE class factory."""
        with mock.patch('iblutil.io.params.getfile', new=self.get_file), \
                mock.patch('one.params.input', new=self.assertFalse):
            # Cache dir not in client cache map; use One (light)
            one_obj = ONE(cache_dir=self.tempdir.name)
            self.assertIsInstance(one_obj, One)

            # Test setup with virtual ONE method
            assert ONE.cache_info().currsize > 0
            ONE.setup(silent=True, make_default=True)
            self.assertFalse(ONE.cache_info().currsize, 'failed to reset LRU cache')

            with self.subTest('ONE setup with database URL'):
                if OFFLINE_ONLY:
                    self.skipTest('Requires remote db connection')
                # No cache dir provided; use OneAlyx (silent setup mode)
                one_obj = ONE(silent=True, mode='local', password=TEST_DB_2['password'])
                self.assertIsInstance(one_obj, OneAlyx)

                # The cache dir is in client cache map; use OneAlyx
                one_obj = ONE(cache_dir=one_obj.alyx.cache_dir, mode='local')
                self.assertIsInstance(one_obj, OneAlyx)

                # A db URL was provided; use OneAlyx
                # mode = 'local' ensures we don't download cache (could also set cache_dir)
                one_obj = ONE(**TEST_DB_1, mode='local')
                self.assertIsInstance(one_obj, OneAlyx)


class TestOneMisc(unittest.TestCase):
    """Test functions in one.util."""

    def test_validate_date_range(self):
        """Test one.util.validate_date_range."""
        # Single string date
        actual = validate_date_range('2020-01-01')  # On this day
        expected = (pd.Timestamp('2020-01-01 00:00:00'),
                    pd.Timestamp('2020-01-01 23:59:59.999000'))
        self.assertEqual(actual, expected)

        # Single datetime.date object
        actual = validate_date_range(pd.Timestamp('2020-01-01 00:00:00').date())
        self.assertEqual(actual, expected)

        # Single pandas Timestamp
        actual = validate_date_range(pd.Timestamp(2020, 1, 1))
        self.assertEqual(actual, expected)

        # Array of two datetime64
        actual = validate_date_range(np.array(['2022-01-30', '2022-01-30'],
                                              dtype='datetime64[D]'))
        expected = (pd.Timestamp('2022-01-30 00:00:00'), pd.Timestamp('2022-01-30 00:00:00'))
        self.assertEqual(actual, expected)

        # From date (lower bound)
        actual = validate_date_range(['2020-01-01'])  # from date
        self.assertEqual(actual[0], pd.Timestamp('2020-01-01 00:00:00'))
        dt = actual[1] - pd.Timestamp.now()
        self.assertTrue(dt.days > 10 * 365)

        actual = validate_date_range(['2020-01-01', None])  # from date
        self.assertEqual(actual[0], pd.Timestamp('2020-01-01 00:00:00'))
        dt = actual[1] - pd.Timestamp.now()
        self.assertTrue(dt.days > 10 * 365)  # Upper bound at least 60 years in the future

        # To date (upper bound)
        actual = validate_date_range([None, '2020-01-01'])  # up to date
        self.assertEqual(actual[1], pd.Timestamp('2020-01-01 00:00:00'))
        dt = pd.Timestamp.now().date().year - actual[0].date().year
        self.assertTrue(dt > 60)  # Lower bound at least 60 years in the past

        self.assertIsNone(validate_date_range(None))
        with self.assertRaises(ValueError):
            validate_date_range(['2020-01-01', '2019-09-06', '2021-10-04'])

    def test_index_last_before(self):
        """Test one.util.index_last_before."""
        revisions = ['2021-01-01', '2020-08-01', '', '2020-09-30']
        verifiable = index_last_before(revisions, '2021-01-01')
        self.assertEqual(0, verifiable)

        verifiable = index_last_before(revisions, '2020-09-15')
        self.assertEqual(1, verifiable)

        verifiable = index_last_before(revisions, '')
        self.assertEqual(2, verifiable)

        self.assertIsNone(index_last_before([], '2009-01-01'))

        verifiable = index_last_before(revisions, None)
        self.assertEqual(0, verifiable, 'should return most recent')

    def test_collection_spec(self):
        """Test one.util._collection_spec."""
        # Test every combination of input
        inputs = []
        _collection = {None: '({collection}/)?', '': '', '-': '{collection}/'}
        _revision = {None: '(#{revision}#/)?', '': '', '-': '#{revision}#/'}
        combs = combinations_with_replacement((None, '', '-'), 2)
        [inputs.extend(set(permutations(x))) for x in combs]
        for collection, revision in inputs:
            with self.subTest(collection=collection, revision=revision):
                verifiable = _collection_spec(collection, revision)
                expected = _collection[collection] + _revision[revision]
                self.assertEqual(expected, verifiable)

    def test_revision_last_before(self):
        """Test one.util.filter_revision_last_before."""
        datasets = util.revisions_datasets_table()
        df = datasets[datasets.rel_path.str.startswith('alf/probe00')].copy()
        verifiable = filter_revision_last_before(df, revision='2020-09-01', assert_unique=False)
        self.assertTrue(len(verifiable) == 2)

        # Remove one of the datasets' revisions to test assert unique on mixed revisions
        df_mixed = df.drop((df['revision'] == '2020-01-08').idxmax())
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_revision_last_before(df_mixed, revision='2020-09-01', assert_consistent=True)

        # Test with default revisions
        df['default_revision'] = False
        with self.assertWarnsRegex(alferr.ALFWarning, 'No default revision for dataset'):
            verifiable = filter_revision_last_before(df.copy(), assert_unique=False)
        self.assertTrue(len(verifiable) == 2)

        # Should have fallen back on lexicographical ordering
        self.assertTrue(verifiable.rel_path.str.contains('#2021-07-06#').all())
        with self.assertRaises(alferr.ALFError):
            filter_revision_last_before(df.copy(), assert_unique=True)

        # Add unique default revisions
        df.iloc[[0, 4], -1] = True
        # Should log mixed revisions
        with self.assertWarnsRegex(alferr.ALFWarning, 'Multiple revisions'):
            verifiable = filter_revision_last_before(df.copy(), assert_unique=True)
        self.assertEqual(2, len(verifiable))
        self.assertCountEqual(verifiable['rel_path'], df['rel_path'].iloc[[0, 4]])

        # Add multiple default revisions
        df['default_revision'] = True
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_revision_last_before(df.copy(), assert_unique=True)

    def test_parse_id(self):
        """Test one.util.parse_id."""
        obj = unittest.mock.MagicMock()  # Mock object to decorate
        obj.to_eid.return_value = 'parsed_id'  # Method to be called
        input = 'subj/date/num'  # Input id to pass to `to_eid`
        parse_id(obj.method)(obj, input)
        obj.to_eid.assert_called_with(input)
        obj.method.assert_called_with(obj, 'parsed_id')

        # Test raises value error when None returned
        obj.to_eid.return_value = None  # Simulate failure to parse id
        with self.assertRaises(ValueError):
            parse_id(obj.method)(obj, input)

    def test_autocomplete(self):
        """Test one.util.autocomplete."""
        search_terms = ('subject', 'date_range', 'dataset', 'dataset_type')
        self.assertEqual('subject', autocomplete('Subj', search_terms))
        self.assertEqual('dataset', autocomplete('dataset', search_terms))
        with self.assertRaises(ValueError):
            autocomplete('dtypes', search_terms)
        with self.assertRaises(ValueError):
            autocomplete('dat', search_terms)

    def test_LazyID(self):
        """Test one.util.LazyID."""
        uuids = [
            'c1a2758d-3ce5-4fa7-8d96-6b960f029fa9',
            '0780da08-a12b-452a-b936-ebc576aa7670',
            'ff812ca5-ce60-44ac-b07e-66c2c37e98eb'
        ]
        ses = [{'url': f'https://website.org/foo/{x}'} for x in uuids]
        ez = LazyId(ses)
        self.assertEqual(len(uuids), len(ez))
        self.assertCountEqual(map(str, ez), uuids)
        self.assertEqual(ez[0], UUID(uuids[0]))
        self.assertEqual(ez[0:2], [UUID(x) for x in uuids[0:2]])
        ez = LazyId([{'id': x} for x in uuids])
        self.assertCountEqual(map(str, ez), uuids)


if __name__ == '__main__':
    unittest.main(exit=True, verbosity=2)
