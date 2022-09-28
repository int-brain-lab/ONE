"""Unit tests for the one.api module

Wherever possible the ONE tests should not rely on an internet connection

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
  or mock iblutil.io.params.getfile to return a temporary cache directory
- An One object created through the one.api.ONE function, make sure you restore the
  properties to their original state on teardown, or call one.api.ONE.cache_clear()

"""
import datetime
import logging
import time
from pathlib import Path
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

from one.api import ONE, One, OneAlyx
from one.util import (
    ses2records, validate_date_range, index_last_before, filter_datasets, _collection_spec,
    filter_revision_last_before, parse_id, autocomplete, LazyId, datasets2records
)
import one.params
import one.alf.exceptions as alferr
from . import util
from . import OFFLINE_ONLY, TEST_DB_1, TEST_DB_2  # 1 = TestAlyx; 2 = OpenAlyx


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

    def tearDown(self) -> None:
        while Path(self.one.cache_dir).joinpath('.cache.lock').exists():
            time.sleep(.1)
        self.tempdir.cleanup()

    def test_list_subjects(self):
        """Test One.list_subejcts"""
        subjects = self.one.list_subjects()
        expected = ['KS005', 'ZFM-01935', 'ZM_1094', 'ZM_1150',
                    'ZM_1743', 'ZM_335', 'clns0730', 'flowers']
        self.assertCountEqual(expected, subjects)

    def test_offline_repr(self):
        """Test for One.offline property"""
        self.assertTrue('offline' in str(self.one))
        self.assertTrue(str(self.tempdir.name) in str(self.one))

    def test_one_search(self):
        """Test for One.search"""
        one = self.one
        # Search subject
        eids = one.search(subject='ZM_335')
        expected = ['3473f9d2-aa5d-41a6-9048-c65d0b7ab97c',
                    'dfe99506-b873-45db-bc93-731f9362e304']
        self.assertEqual(expected, eids)

        # Search lab
        labs = ['mainen', 'cortexlab']
        eids = one.search(laboratory=labs)
        expected = ['d3372b15-f696-4279-9be5-98f15783b5bb',
                    '3473f9d2-aa5d-41a6-9048-c65d0b7ab97c']
        self.assertEqual(len(eids), 25)
        self.assertEqual(expected, eids[:2])

        # Search exact date
        eids = one.search(date='2019-06-07')
        self.assertEqual(eids, ['db524c42-6356-4c61-b236-4967c54d2665'])

        # Search date range
        dates = ['2019-04-01', '2019-04-10']
        eids = one.search(date=dates)
        expected = ['13c99443-01ee-462e-b668-717daa526fc0',
                    'abf5109c-d780-44c8-9561-83e857c7bc01']
        self.assertEqual(len(eids), 9)
        self.assertEqual(expected, eids[:2])

        # Search from a given date
        dates = ['2021-01-01', None]
        eids = one.search(date_range=dates)
        self.assertEqual(eids, ['d3372b15-f696-4279-9be5-98f15783b5bb'])

        # Search datasets
        query = 'spikes.depths'
        eids = one.search(data=query)
        self.assertTrue(eids)
        expected = [
            'd3372b15-f696-4279-9be5-98f15783b5bb',
            'b1c968ad-4874-468d-b2e4-5ffa9b9964e9',
            'cf264653-2deb-44cb-aa84-89b82507028a'
        ]
        self.assertEqual(eids, expected)

        # Filter non-existent
        # Set exist for one of the eids to false
        mask = (one._cache['datasets']['rel_path'].str.contains(query))
        i = one._cache['datasets'][mask].index[0]
        one._cache['datasets'].loc[i, 'exists'] = False

        self.assertTrue(len(eids) == len(one.search(data=query)) + 1)

        # Search task_protocol
        eids = one.search(task='habituation')
        self.assertEqual(eids, ['ac80cd12-49e5-4aff-b5f2-1a718679ceeb'])

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

        # Test search with integer ids
        util.caches_str2int(one._cache)
        query = 'clusters'
        eids = one.search(data=query)
        self.assertTrue(all(isinstance(x, str) for x in eids))
        self.assertEqual(3, len(eids))

    def test_filter(self):
        """Test one.util.filter_datasets"""
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
        revision = '2020-09-06'
        verifiable = filter_datasets(datasets, None, None, revision, assert_unique=False)
        self.assertEqual(2, len(verifiable))
        self.assertTrue(all(x.split('#')[1] < revision for x in verifiable['rel_path']))

        # Should return single dataset with last revision when default specified
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_datasets(datasets, '*spikes.times*', 'alf/probe00', None,
                            assert_unique=True, wildcards=True, revision_last_before=True)

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
        # When revision_last_before is false, expect multiple objects error
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            filter_datasets(datasets, '*spikes.times*', 'alf/probe00', None,
                            assert_unique=True, wildcards=True, revision_last_before=False)

    def test_filter_wildcards(self):
        """Test one.util.filter_datasets with wildcards flag set to True"""
        datasets = self.one._cache.datasets.iloc[:5].copy()
        # Test identity
        verifiable = filter_datasets(datasets, '_ibl_*', '*lf', None,
                                     assert_unique=False, wildcards=True)
        self.assertTrue(len(verifiable) == 2)
        # As dict with list (should act as logical OR)
        dataset = dict(attribute=['timestamp?', 'raw'])
        verifiable = filter_datasets(datasets, dataset, None, None,
                                     assert_unique=False, revision_last_before=False,
                                     wildcards=True)
        self.assertEqual(4, len(verifiable))

    def test_list_datasets(self):
        """Test One.list_datasets"""
        # test filename
        dsets = self.one.list_datasets(filename='_ibl_trials*')
        self.assertEqual(len(dsets), 18)
        dsets = self.one.list_datasets(filename='gnagnag')
        self.assertEqual(len(dsets), 0)

        # Test no eid
        dsets = self.one.list_datasets(details=True)
        self.assertEqual(len(dsets), len(self.one._cache.datasets))
        self.assertFalse(dsets is self.one._cache.datasets)

        # Test list for eid
        dsets = self.one.list_datasets('KS005/2019-04-02/001', details=True)
        self.assertEqual(27, len(dsets))

        # Test filters
        filename = {'attribute': ['times', 'intervals'], 'extension': 'npy'}
        dsets = self.one.list_datasets('ZFM-01935/2021-02-05/001', filename)
        self.assertEqual(10, len(dsets))
        self.assertTrue(all(any(y in x for y in ('.times.', '.intervals')) for x in dsets))

        filename['attribute'][0] += '*'  # Include wildcard to match both times and timestamps
        dsets = self.one.list_datasets('ZFM-01935/2021-02-05/001', filename)
        self.assertEqual(13, len(dsets))
        self.assertEqual(3, sum('.timestamps.' in x for x in dsets))

        # Test using str ids as index
        util.caches_str2int(self.one._cache)
        dsets = self.one.list_datasets('KS005/2019-04-02/001')
        self.assertEqual(27, len(dsets))

        # Test empty
        dsets = self.one.list_datasets('FMR019/2021-03-18/002', details=True)
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertEqual(len(dsets), 0)

        # Test details=False, with and without eid
        for eid in [None, 'KS005/2019-04-02/001']:
            dsets = self.one.list_datasets(eid, details=False)
            self.assertIsInstance(dsets, list)
            self.assertTrue(len(dsets) == np.unique(dsets).size)

    def test_list_collections(self):
        """Test One.list_collections"""
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
        """Test One.list_revisions"""
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
        """Test One.get_details"""
        eid = 'aaf101c3-2581-450a-8abd-ddb8f557a5ad'
        det = self.one.get_details(eid)
        self.assertIsInstance(det, pd.Series)
        self.assertEqual('KS005', det.subject)
        self.assertEqual('2019-04-04', str(det.date))
        self.assertEqual(4, det.number)

        # Test details flag
        det = self.one.get_details(eid, full=True)
        self.assertIsInstance(det, pd.DataFrame)
        self.assertTrue('rel_path' in det.columns)

        # Test with int index ids
        util.caches_str2int(self.one._cache)
        det = self.one.get_details(eid)
        self.assertIsInstance(det, pd.Series)

        # Test errors
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.get_details(eid.replace('a', 'b'))
        sessions = self.one._cache.sessions
        self.one._cache.sessions = pd.concat([sessions, det.to_frame().T]).sort_index()
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.get_details(eid)

    def test_index_type(self):
        """Test One._index_type"""
        self.assertIs(str, self.one._index_type())
        util.caches_str2int(self.one._cache)
        self.assertIs(int, self.one._index_type())
        self.one._cache.datasets.reset_index(inplace=True)
        with self.assertRaises(IndexError):
            self.one._index_type('datasets')

    def test_load_dataset(self):
        """Test One.load_dataset"""
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
        fake_id = self.one.to_eid(eid).replace('b', 'a')
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset(fake_id, '_iblrig_leftCamera.timestamps.ssv')
        file.unlink()
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_dataset(eid, '_iblrig_leftCamera.timestamps.ssv')

        # Check loading without extension
        file = self.one.load_dataset(eid, '_ibl_wheel.position', download_only=True)
        self.assertTrue(str(file).endswith('wheel.position.npy'))

    def test_load_datasets(self):
        """Test One.load_datasets"""
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

    def test_load_dataset_from_id(self):
        """Test One.load_dataset_from_id"""
        id = np.array([[-9204203870374650458, -6411285612086772563]])
        file = self.one.load_dataset_from_id(id, download_only=True)
        self.assertIsInstance(file, Path)
        expected = 'ZFM-01935/2021-02-05/001/alf/probe00/_phy_spikes_subset.waveforms.npy'
        self.assertTrue(file.as_posix().endswith(expected))

        # Details
        _, details = self.one.load_dataset_from_id(id, download_only=True, details=True)
        self.assertIsInstance(details, pd.Series)

        # Load file content with str id
        s_id, = parquet.np2str(id)
        data = np.arange(3)
        np.save(str(file), data)  # Ensure data to load
        dset = self.one.load_dataset_from_id(s_id)
        self.assertTrue(np.array_equal(dset, data))

        # Load file content with UUID
        dset = self.one.load_dataset_from_id(UUID(s_id))
        self.assertTrue(np.array_equal(dset, data))

        # Load with int ids as index
        util.caches_str2int(self.one._cache)
        dset = self.one.load_dataset_from_id(s_id)
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
        details.name = (8737712210713458643, -4920882604093676133, *id.flatten().tolist())
        datasets = self.one._cache.datasets
        self.one._cache.datasets = pd.concat([datasets, details.to_frame().T]).sort_index()
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.load_dataset_from_id(s_id)

    def test_load_object(self):
        """Test One.load_object"""
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
        self.assertFalse(self.one._cache.datasets.loc[(eid,), 'exists'].all())

        eid = 'ZFM-01935/2021-02-05/001'
        with self.assertRaises(alferr.ALFMultipleCollectionsFound):
            self.one.load_object(eid, 'ephysData_g0_t0')
        with self.assertRaises(alferr.ALFMultipleObjectsFound):
            self.one.load_object(eid, '*Camera')

    def test_load_collection(self):
        """Test One.load_collection"""
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
                               side_effect=lambda x, **kwargs: [None] * len(x)),\
                self.assertRaises(alferr.ALFObjectNotFound) as ex:
            self.one.load_collection(eid, 'alf')
            self.assertIn('not found on disk', str(ex))

    def test_load_cache(self):
        """Test One._load_cache"""
        # Test loading unsorted table with no id index set
        df = self.one._cache['datasets'].reset_index()
        info = self.one._cache['_meta']['raw']['datasets']
        with tempfile.TemporaryDirectory() as tdir:
            # Loading from empty dir
            self.one.load_cache(tdir)
            self.assertTrue(self.one._cache['_meta']['expired'])
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
            with self.assertLogs(logging.getLogger('one.api'), logging.WARNING) as log:
                self.one.load_cache(tdir)
                self.assertTrue('gnagna.pqt' in log.output[0])
            # Save table with missing id columns
            df.drop(id_keys, axis=1, inplace=True)
            parquet.save(Path(tdir) / 'datasets.pqt', df, info)
            with self.assertRaises(KeyError):
                self.one.load_cache(tdir)

    def test_refresh_cache(self):
        """Test One.refresh_cache"""
        self.one._cache.datasets = self.one._cache.datasets.iloc[0:0].copy()
        prev_loaded = self.one._cache['_meta']['loaded_time']
        for mode in ('auto', 'local', 'remote'):
            with self.subTest("Refresh modes", mode=mode):
                loaded = self.one.refresh_cache(mode)
                self.assertFalse(len(self.one._cache.datasets))
                self.assertEqual(prev_loaded, loaded)
        loaded = self.one.refresh_cache('refresh')
        self.assertTrue(len(self.one._cache.datasets))
        self.assertTrue(loaded > prev_loaded)
        self.one.cache_expiry = datetime.timedelta()  # Immediately expire
        self.one._cache.datasets = self.one._cache.datasets.iloc[0:0].copy()
        self.one.refresh_cache('auto')
        self.assertTrue(len(self.one._cache.datasets))
        with self.assertRaises(ValueError):
            self.one.refresh_cache('double')

    def test_save_cache(self):
        """Test one.util.save_cache"""
        self.one._cache['_meta'].pop('modified_time', None)
        # Should be no cache save as it's not been modified
        with tempfile.TemporaryDirectory() as tdir:
            self.one._save_cache(save_dir=tdir)
            self.assertFalse(any(Path(tdir).glob('*')))
            # Should save two tables
            self.one._cache['_meta']['modified_time'] = datetime.datetime.now()
            self.one._save_cache(save_dir=tdir)
            self.assertEqual(2, len(list(Path(tdir).glob('*.pqt'))))
            # Load with One and check modified time is preserved
            raw_modified = One(cache_dir=tdir)._cache['_meta']['raw']['datasets']['date_modified']
            expected = self.one._cache['_meta']['modified_time'].strftime('%Y-%m-%d %H:%M')
            self.assertEqual(raw_modified, expected)

    def test_update_cache_from_records(self):
        """Test One._update_cache_from_records"""
        # Update with single record (pandas.Series), one exists, one doesn't
        session = self.one._cache.sessions.iloc[0].squeeze()
        session.name = str(uuid4())  # New record
        dataset = self.one._cache.datasets.iloc[0].squeeze()
        dataset['exists'] = not dataset['exists']
        self.one._update_cache_from_records(sessions=session, datasets=dataset)
        self.assertTrue(session.name in self.one._cache.sessions.index)
        updated, = dataset['exists'] == self.one._cache.datasets.loc[dataset.name, 'exists']
        self.assertTrue(updated)

        # Update a number of records
        datasets = self.one._cache.datasets.iloc[:3].copy()
        datasets.loc[:, 'exists'] = ~datasets.loc[:, 'exists']
        # Make one of the datasets a new record
        idx = datasets.index.values
        idx[-1] = (idx[-1][0], str(uuid4()))
        datasets.index = pd.MultiIndex.from_tuples(idx)
        self.one._update_cache_from_records(datasets=datasets)
        self.assertTrue(idx[-1] in self.one._cache.datasets.index)
        verifiable = self.one._cache.datasets.loc[datasets.index.values, 'exists']
        self.assertTrue(np.all(verifiable == datasets.loc[:, 'exists']))

        # Check behaviour when columns don't match
        datasets.loc[:, 'exists'] = ~datasets.loc[:, 'exists']
        datasets['extra_column'] = True
        self.one._cache.datasets['new_column'] = False
        self.addCleanup(self.one._cache.datasets.drop, 'new_column', axis=1, inplace=True)
        with self.assertRaises(AssertionError):
            self.one._update_cache_from_records(datasets=datasets, strict=True)
        self.one._update_cache_from_records(datasets=datasets)
        verifiable = self.one._cache.datasets.loc[datasets.index.values, 'exists']
        self.assertTrue(np.all(verifiable == datasets.loc[:, 'exists']))

        # Check fringe cases
        with self.assertRaises(KeyError):
            self.one._update_cache_from_records(unknown=datasets)
        self.assertIsNone(self.one._update_cache_from_records(datasets=None))

    def test_save_loaded_ids(self):
        """Test One.save_loaded_ids and logic within One._check_filesystem"""
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
        self.assertTrue(np.all(eids == eid))

        # Test loading a dataset that doesn't exist
        dset = self.one.list_datasets(eid, filename='*trials*', details=True).iloc[-1]
        dset['rel_path'] = dset['rel_path'].replace('.npy', '.pqt')
        dset.name = (eid, str(uuid4()))
        old_cache = self.one._cache['datasets']
        try:
            self.one._cache['datasets'] = self.one._cache.datasets.append(dset)
            dsets = [dset['rel_path'], '_ibl_trials.feedback_times.npy']
            new_files, rec = self.one.load_datasets(eid, dsets, assert_present=False)
            loaded = self.one._cache['_loaded_datasets']
            # One dataset is already in the list (test for duplicates) and other doesn't exist
            self.assertEqual(len(files), len(loaded), 'No new UUIDs should have been added')
            self.assertIn(rec[1]['id'], loaded)  # Already in list
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
        self.assertCountEqual(ids['dataset_uuid'], loaded)
        self.assertTrue(self.one._cache['_loaded_datasets'].size, 'List unexpectedly cleared')

        # Test as session UUIDs
        ids, filename = self.one.save_loaded_ids(sessions_only=True, clear_list=False)
        self.assertCountEqual([eid], ids)
        self.assertEqual(pd.read_csv(filename)['session_uuid'][0], eid)

        # Test int IDs.
        self.one._cache['_loaded_datasets'] = parquet.str2np(self.one._cache['_loaded_datasets'])
        # IDs should be cast to string
        with self.assertRaises(NotImplementedError):
            self.one.save_loaded_ids(clear_list=False, sessions_only=True)

        # Check dataset int IDs and clear list
        ids, _ = self.one.save_loaded_ids(clear_list=True)
        self.assertCountEqual(loaded, ids)
        self.assertFalse(self.one._cache['_loaded_datasets'].size, 'List not cleared')

        # Test clear list and warn on empty
        with self.assertWarns(Warning):
            ids, filename = self.one.save_loaded_ids()
        self.assertEqual(ids, [])
        self.assertIsNone(filename)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneAlyx(unittest.TestCase):
    """
    This could be an offline test.  Would need to add /docs REST cache fixture.
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
        self.one.mode = 'local'

    def test_type2datasets(self):
        """Test OneAlyx.type2datasets"""
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
        """Test for OneAlyx.dataset2type"""
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

    def test_ses2records(self):
        """Test one.util.ses2records"""
        eid = '8dd0fcb0-1151-4c97-ae35-2e2421695ad7'
        ses = self.one.alyx.rest('sessions', 'read', id=eid)
        session, datasets = ses2records(ses)

        # Verify returned tables are compatible with cache tables
        self.assertIsInstance(session, pd.Series)
        self.assertIsInstance(datasets, pd.DataFrame)
        self.assertEqual(session.name, eid)
        self.assertCountEqual(session.keys(), self.one._cache['sessions'].columns)
        self.assertEqual(len(datasets), len(ses['data_dataset_session_related']))
        expected = [x for x in self.one._cache['datasets'].columns] + ['default_revision']
        self.assertCountEqual(expected, datasets.columns)
        self.assertEqual(tuple(datasets.index.names), ('eid', 'id'))
        self.assertTrue(datasets.default_revision.all())

        # Check int_id as True
        session, datasets = ses2records(ses, int_id=True)
        self.assertEqual(session.name, (-7544566139326771059, -2928913016589240914))
        self.assertEqual(tuple(datasets.index.names), ('eid_0', 'eid_1', 'id_0', 'id_1'))

        # Check behaviour when no datasets present
        ses['data_dataset_session_related'] = []
        _, datasets = ses2records(ses)
        self.assertIsNone(datasets)

    def test_datasets2records(self):
        """Test one.util.datasets2records"""
        eid = '8dd0fcb0-1151-4c97-ae35-2e2421695ad7'
        dsets = self.one.alyx.rest('datasets', 'list', session=eid)
        datasets = datasets2records(dsets)

        # Verify returned tables are compatible with cache tables
        self.assertIsInstance(datasets, pd.DataFrame)
        self.assertTrue(len(datasets) >= len(dsets))
        expected = self.one._cache['datasets'].columns
        self.assertCountEqual(expected, (x for x in datasets.columns if x != 'default_revision'))
        self.assertEqual(tuple(datasets.index.names), ('eid', 'id'))

        # Check behaviour when ind_id is True
        datasets = datasets2records(dsets, int_id=True)
        self.assertEqual(tuple(datasets.index.names), ('eid_0', 'eid_1', 'id_0', 'id_1'))

        # Test single input
        dataset = datasets2records(dsets[0])
        self.assertTrue(len(dataset) == 1)
        # Test records when data missing
        dsets[0]['file_records'][0]['exists'] = False
        empty = datasets2records(dsets[0])
        self.assertTrue(isinstance(empty, pd.DataFrame) and len(empty) == 0)

    def test_pid2eid(self):
        """Test OneAlyx.pid2eid"""
        pid = 'b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        eid, collection = self.one.pid2eid(pid, query_type='remote')
        self.assertEqual('fc737f3c-2a57-4165-9763-905413e7e341', eid)
        self.assertEqual('probe00', collection)
        with self.assertRaises(NotImplementedError):
            self.one.pid2eid(pid, query_type='local')

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_describe_revision(self, mock_stdout):
        """Test OneAlyx.describe_revision"""
        self.one.mode = 'remote'
        record = {
            'name': str(datetime.date.today()) + 'a',
            'description': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
        }
        try:
            self.one.alyx.rest('revisions', 'read', id=record['name'], no_cache=True)
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
        with mock.patch.object(self.one.alyx, 'get', side_effect=err),\
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
        """Test OneAlyx.path2url"""
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
        """Test ConversionMixin.record2url"""
        idx = (slice(None), '91546fc6-b67c-4a69-badc-5e66088519c4')
        dataset = self.one._cache['datasets'].loc[idx, :]
        url = self.one.record2url(dataset.squeeze())
        expected = ('https://ibl.flatironinstitute.org/public/'
                    'cortexlab/Subjects/KS005/2019-04-04/004/alf/'
                    '_ibl_wheel.position.91546fc6-b67c-4a69-badc-5e66088519c4.npy')
        self.assertEqual(expected, url)

    def test_load_cache(self):
        """Test loading the remote cache"""
        self.one.alyx.silent = False  # For checking log
        self.one._cache._meta['expired'] = True
        try:
            with self.assertLogs(logging.getLogger('one.api'), logging.INFO) as lg:
                with mock.patch.object(self.one.alyx, 'get', side_effect=HTTPError()):
                    self.one.load_cache(clobber=True)
                self.assertEqual('remote', self.one.mode)
                self.assertRegex(lg.output[0], 'cache over .+ old')
                self.assertTrue('Failed to load' in lg.output[1])

                with mock.patch.object(self.one.alyx, 'get', side_effect=ConnectionError()):
                    self.one.load_cache(clobber=True)
                    self.assertEqual('local', self.one.mode)
                self.assertTrue('Failed to connect' in lg.output[-1])

            cache_info = {'min_api_version': '200.0.0'}
            # Check version verification
            with mock.patch.object(self.one.alyx, 'get', return_value=cache_info),\
                    self.assertWarns(UserWarning):
                self.one.load_cache(clobber=True)

        finally:  # Restore properties
            self.one.mode = 'auto'
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
        """ Tests for the OneAlyx._download_aws method"""
        N = 5
        dsets = self.one._cache['datasets'].iloc[:N].copy()
        dsets['exists_aws'] = True

        self.one.mode = 'auto'  # Can't download in local mode

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
        self.assertTrue(dsets.index[0][1] in path, 'Dataset UUID not in filepath')

        # Should fall back to usual method if any datasets do not exist on AWS
        dsets['exists_aws'] = False
        with mock.patch.object(self.one, '_download_dataset') as fallback_method:
            self.one._download_datasets(dsets)
            fallback_method.assert_called()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneRemote(unittest.TestCase):
    """Test remote queries"""
    def setUp(self) -> None:
        self.one = OneAlyx(**TEST_DB_2)
        self.eid = '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a'

    def test_online_repr(self):
        """Tests OneAlyx.__repr__"""
        self.assertTrue('online' in str(self.one))
        self.assertTrue(TEST_DB_2['base_url'] in str(self.one))

    def test_list_datasets(self):
        """Test OneAlyx.list_datasets"""
        # Test list for eid
        # Ensure remote by making local datasets table empty
        self.addCleanup(self.one.load_cache)
        self.one._cache['datasets'] = self.one._cache['datasets'].iloc[0:0].copy()

        dsets = self.one.list_datasets(self.eid, details=True, query_type='remote')
        self.assertEqual(108, len(dsets))

        # Test missing eid
        dsets = self.one.list_datasets('FMR019/2021-03-18/002', details=True, query_type='remote')
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertEqual(len(dsets), 0)

        # Test empty datasets
        with mock.patch('one.util.ses2records', return_value=(None, None)):
            dsets = self.one.list_datasets(self.eid, details=True, query_type='remote')
            self.assertIsInstance(dsets, pd.DataFrame)
            self.assertEqual(len(dsets), 0)

        # Test details=False, with eid
        dsets = self.one.list_datasets(self.eid, details=False, query_type='remote')
        self.assertIsInstance(dsets, list)
        self.assertEqual(108, len(dsets))

        # Test with other filters
        dsets = self.one.list_datasets(self.eid, collection='*probe*', filename='*channels*',
                                       details=False, query_type='remote')
        self.assertEqual(13, len(dsets))
        self.assertTrue(all(x in y for x in ('probe', 'channels') for y in dsets))

        with self.assertWarns(Warning):
            self.one.list_datasets(query_type='remote')

    def test_search(self):
        """Test OneAlyx.search"""
        eids = self.one.search(subject='SWC_043', query_type='remote')
        self.assertIn(self.eid, list(eids))

        eids, det = self.one.search(subject='SWC_043', query_type='remote', details=True)
        correct = len(det) == len(eids) and 'url' in det[0] and det[0]['url'].endswith(eids[0])
        self.assertTrue(correct)

        # Check minimum set of keys present (these are present in One.search output)
        expected = {'lab', 'subject', 'date', 'number', 'projects'}
        self.assertTrue(det[0].keys() >= expected)
        # Test dataset search with Django
        eids = self.one.search(subject='SWC_043', dataset=['probes.description'], number=1,
                               django='data_dataset_session_related__collection__iexact,alf',
                               query_type='remote')
        self.assertCountEqual(eids, [self.eid])

        # Test date range
        eids = self.one.search(subject='SWC_043', date='2020-09-21', query_type='remote')
        self.assertCountEqual(eids, [self.eid])

        eids = self.one.search(date=[datetime.date(2020, 9, 21), datetime.date(2020, 9, 22)],
                               lab='hoferlab', query_type='remote')
        self.assertCountEqual(eids, [self.eid])

        # Test limit arg and LazyId
        eids = self.one.search(date='2020-03-23', limit=2, query_type='remote')
        self.assertIsInstance(eids, LazyId)
        self.assertTrue(all(len(x) == 36 for x in eids))

        # Test laboratory kwarg
        eids = self.one.search(laboratory='hoferlab', query_type='remote')
        self.assertIn(self.eid, list(eids))

        eids = self.one.search(lab='hoferlab', query_type='remote')
        self.assertIn(self.eid, list(eids))

    def test_load_dataset(self):
        """Test OneAlyx.load_dataset"""
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
        """Test OneAlyx.load_object"""
        files = self.one.load_object(self.eid, 'wheel',
                                     collection='alf', query_type='remote',
                                     download_only=True)
        self.assertIsInstance(files[0], Path)
        self.assertTrue(
            files[0].as_posix().endswith('SWC_043/2020-09-21/001/alf/_ibl_wheel.position.npy')
        )

    def test_get_details(self):
        """Test OneAlyx.get_details"""
        det = self.one.get_details(self.eid, query_type='remote')
        self.assertIsInstance(det, dict)
        self.assertEqual('SWC_043', det['subject'])
        self.assertEqual('ibl_neuropixel_brainwide_01', det['projects'])
        self.assertEqual('2020-09-21', str(det['date']))
        self.assertEqual(1, det['number'])
        self.assertNotIn('data_dataset_session_related', det)

        # Test list
        det = self.one.get_details([self.eid, self.eid], full=True)
        self.assertIsInstance(det, list)
        self.assertIn('data_dataset_session_related', det[0])


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestOneDownload(unittest.TestCase):
    """Test downloading datasets using OpenAlyx"""
    tempdir = None
    one = None

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = mock.patch('one.params.iopar.getfile',
                                new=partial(util.get_file, self.tempdir.name))
        self.patch.start()
        self.one = OneAlyx(**TEST_DB_2, cache_dir=self.tempdir.name)
        self.fid = '17ab5b57-aaf6-4016-9251-66daadc200c7'  # File record of channels.brainLocation
        self.eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'

    def test_download_datasets(self):
        """Test OneAlyx._download_dataset, _download_file and _dset2url"""
        det = self.one.get_details(self.eid, True)
        rec = next(x for x in det['data_dataset_session_related']
                   if 'channels.brainLocation' in x['dataset_type'])
        # FIXME hack because data_url may be AWS
        from one.alf.files import get_alf_path
        rec['data_url'] = self.one.alyx.rel_path2url(get_alf_path(rec['data_url']))
        # FIXME order may not be stable, this only works
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
        fr = [{'url': f'files/{self.fid}', 'json': None}]
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
        self.assertEqual(str(file).split('.')[2], rec['url'].split('/')[-1])

        # Check list input
        files = self.one._download_dataset([rec] * 2)
        self.assertIsInstance(files, list)
        self.assertTrue(all(isinstance(x, Path) for x in files))

        # Check Series input
        r_ = datasets2records(rec, int_id=True).squeeze()
        file = self.one._download_dataset(r_)
        self.assertIn('channels.brainLocation', file.as_posix())

        # Check behaviour when URL invalid
        did = rec['url'].split('/')[-1]
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
        path = self.one.cache_dir.joinpath('lab', 'Subjects', 'subj', '2020-01-01', '001',
                                           'spikes.times.npy')
        with self.assertLogs(logging.getLogger('one.api'), logging.WARNING):
            file = self.one._download_dataset(path)
            self.assertIsNone(file)

        rec = self.one.list_datasets(self.eid, details=True)
        rec = rec[rec.rel_path.str.contains('pykilosort/channels.brainLocation')]
        rec['exists_aws'] = False  # Ensure we use FlatIron for this
        files = self.one._download_datasets(rec)
        self.assertFalse(None in files)

        # Check update cache when id is int and cache table ids are str
        int_id, = parquet.str2np(np.array(rec.index.tolist())).tolist()
        self.one._download_dataset({'data_url': None, 'id': np.array(int_id)})
        exists, = self.one._cache.datasets.loc[(slice(None), rec.index[0]), 'exists']
        self.assertFalse(exists, 'failed to update dataset cache with str index')
        self.one._cache.datasets.loc[(slice(None), rec.index[0]), 'exists'] = True  # Reset values

        # Check works with int index
        util.caches_str2int(self.one._cache)
        self.assertIsNotNone(self.one._download_dataset(files[0]))
        self.assertIsNotNone(self.one._download_datasets(rec))
        # and when dataset missing
        with mock.patch.object(self.one, 'record2url', return_value=None):
            self.assertIsNone(self.one._download_dataset(rec.squeeze()))

        exists, = self.one._cache.datasets.loc[(slice(None), slice(None), *int_id), 'exists']
        self.assertFalse(exists, 'failed to update dataset cache with str index')

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
        try:
            # In case we did indeed have remote REST permissions, try resetting the json field
            self.one.alyx.rest('files', 'partial_update', id=self.fid, data={'json': None})
        except HTTPError as ex:
            if ex.errno != 403:
                raise ex
        self.patch.stop()
        self.tempdir.cleanup()


class TestOneSetup(unittest.TestCase):
    """Test parameter setup upon ONE instantiation and calling setup methods"""
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.get_file = partial(util.get_file, self.tempdir.name)
        # Change default cache dir to temporary directory
        patch = mock.patch('one.params.CACHE_DIR_DEFAULT', Path(self.tempdir.name))
        patch.start()
        self.addCleanup(patch.stop)

    def test_local_cache_setup_prompt(self):
        """Test One.setup"""
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

    def test_setup_silent(self):
        """Test setting up parameters with silent flag.
        - Mock getfile to return temp dir as param file location
        - Mock input function as fail safe in case function erroneously prompts user for input
        """
        with mock.patch('iblutil.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse):
            one_obj = ONE(silent=True, mode='local', password=TEST_DB_2['password'])
            self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

        # Check param files were saved
        self.assertEqual(len(list(Path(self.tempdir.name).rglob('.caches'))), 1)
        client_pars = Path(self.tempdir.name).rglob(f'.{one_obj.alyx.base_url.split("/")[-1]}')
        self.assertEqual(len(list(client_pars)), 1)

        # Check uses defaults on second instantiation
        with mock.patch('iblutil.io.params.getfile', new=self.get_file):
            one_obj = ONE(mode='local')
            self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

        # Check saves base_url arg
        with self.subTest('Test setup with base URL'):
            if OFFLINE_ONLY:
                self.skipTest('Requires remote db connection')
            with mock.patch('iblutil.io.params.getfile', new=self.get_file):
                one_obj = ONE(**TEST_DB_1)
                self.assertEqual(one_obj.alyx.base_url, TEST_DB_1['base_url'])
                params_url = one.params.get(client=TEST_DB_1['base_url']).ALYX_URL
                self.assertEqual(params_url, one_obj.alyx.base_url)

    def test_setup_username(self):
        """Test setting up parameters with a provided username.
        - Mock getfile to return temp dir as param file location
        - Mock input function as fail safe in case function erroneously prompts user for input
        - Mock requests.post returns a fake user authentication response
        """
        credentials = {'username': 'foobar', 'password': '123'}
        with mock.patch('iblutil.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse),\
                mock.patch('one.webclient.requests.post') as req_mock:
            req_mock().json.return_value = {'token': 'shhh'}
            # In remote mode the cache endpoint will not be queried
            one_obj = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                          silent=True, mode='remote', **credentials)
            params_username = one.params.get(client=TEST_DB_1['base_url']).ALYX_LOGIN
            self.assertEqual(params_username, one_obj.alyx.user)
            self.assertEqual(credentials['username'], one_obj.alyx.user)
            _, kwargs = req_mock.call_args
            self.assertEqual(kwargs.get('data', {}), credentials)

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
        """Test OneAlyx.setup"""
        with mock.patch('iblutil.io.params.getfile', new=self.get_file),\
                mock.patch('one.webclient.getpass', return_value='international'):
            one_obj = OneAlyx.setup(silent=True)
        self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

    def test_setup(self):
        """Test one.params.setup"""
        url = TEST_DB_1['base_url']

        def mock_input(prompt):
            if prompt.lower().startswith('warning'):
                if not getattr(mock_input, 'conflict_warn', False):    # Checks both responses
                    mock_input.conflict_warn = True
                    return 'y'
                return 'n'
            elif 'download cache' in prompt.lower():
                return Path(self.tempdir.name).joinpath('downloads').as_posix()
            elif 'url' in prompt.lower():
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
        client_pars = Path(self.tempdir.name).rglob(f'.{one_obj.alyx.base_url.split("/")[-1]}')
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
        """Test patching legacy params to the new location"""
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
        """Tests the ONE class factory"""
        with mock.patch('iblutil.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse):
            # Cache dir not in client cache map; use One (light)
            one_obj = ONE(cache_dir=self.tempdir.name)
            self.assertIsInstance(one_obj, One)

            # The offline param was given, raise deprecation warning (via log)
            # with self.assertLogs(logging.getLogger('ibllib'), logging.WARNING):
            #     ONE(offline=True, cache_dir=self.tempdir.name)
            with self.assertWarns(DeprecationWarning):
                ONE(offline=True, cache_dir=self.tempdir.name)

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
    """Test functions in one.util"""
    def test_validate_date_range(self):
        """Test one.util.validate_date_range"""
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
        """Test one.util.index_last_before"""
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
        """Test one.util._collection_spec"""
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
        """Test one.util.filter_revision_last_before"""
        datasets = util.revisions_datasets_table()
        df = datasets[datasets.rel_path.str.startswith('alf/probe00')].copy()
        verifiable = filter_revision_last_before(df,
                                                 revision='2020-09-01', assert_unique=False)
        self.assertTrue(len(verifiable) == 2)

        # Test assert unique
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_revision_last_before(df, revision='2020-09-01', assert_unique=True)

        # Test with default revisions
        df['default_revision'] = False
        with self.assertLogs(logging.getLogger('one.util')):
            verifiable = filter_revision_last_before(df.copy(), assert_unique=False)
        self.assertTrue(len(verifiable) == 2)

        # Should have fallen back on lexicographical ordering
        self.assertTrue(verifiable.rel_path.str.contains('#2021-07-06#').all())
        with self.assertRaises(alferr.ALFError):
            filter_revision_last_before(df.copy(), assert_unique=True)

        # Add unique default revisions
        df.iloc[[0, 4], -1] = True
        verifiable = filter_revision_last_before(df.copy(), assert_unique=True)
        self.assertTrue(len(verifiable) == 2)
        self.assertCountEqual(verifiable['rel_path'], df['rel_path'].iloc[[0, 4]])

        # Add multiple default revisions
        df['default_revision'] = True
        with self.assertRaises(alferr.ALFMultipleRevisionsFound):
            filter_revision_last_before(df.copy(), assert_unique=True)

    def test_parse_id(self):
        """Test one.util.parse_id"""
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
        """Test one.util.autocomplete"""
        search_terms = ('subject', 'date_range', 'dataset', 'dataset_type')
        self.assertEqual('subject', autocomplete('Subj', search_terms))
        self.assertEqual('dataset', autocomplete('dataset', search_terms))
        with self.assertRaises(ValueError):
            autocomplete('dtypes', search_terms)
        with self.assertRaises(ValueError):
            autocomplete('dat', search_terms)

    def test_LazyID(self):
        """Test one.util.LazyID"""
        uuids = [
            'c1a2758d-3ce5-4fa7-8d96-6b960f029fa9',
            '0780da08-a12b-452a-b936-ebc576aa7670',
            'ff812ca5-ce60-44ac-b07e-66c2c37e98eb'
        ]
        ses = [{'url': f'https://website.org/foo/{x}'} for x in uuids]
        ez = LazyId(ses)
        self.assertEqual(len(uuids), len(ez))
        self.assertCountEqual(map(str, ez), uuids)
        self.assertEqual(ez[0], uuids[0])
        self.assertEqual(ez[0:2], uuids[0:2])
        ez = LazyId([{'id': x} for x in uuids])
        self.assertCountEqual(map(str, ez), uuids)
