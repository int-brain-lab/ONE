"""Unit tests for the one.converters module."""
import unittest
from unittest import mock
from pathlib import Path, PurePosixPath, PureWindowsPath
from requests.exceptions import HTTPError
from uuid import UUID, uuid4
import datetime

import pandas as pd

from one.api import ONE
from one import converters
from one.alf.path import add_uuid_string
from one.alf.cache import EMPTY_DATASETS_FRAME
from one.alf.path import ALFPath, PurePosixALFPath, PureWindowsALFPath
from one.tests import util, OFFLINE_ONLY, TEST_DB_2


class TestConverters(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = util.set_up_env()
        # Create ONE object with temp cache dir
        cls.one = ONE(mode='local', cache_dir=cls.tempdir.name)

    def test_to_eid(self):
        """Test for ConversionMixin.to_eid method."""
        expected = UUID('d3372b15-f696-4279-9be5-98f15783b5bb')
        # Path str
        eid = self.one.to_eid('ZFM-01935/2021-02-05/001')
        self.assertEqual(eid, expected)
        # eid str
        eid = self.one.to_eid(str(eid))
        self.assertEqual(eid, expected)
        # Path
        session_path = Path(self.one.cache_dir).joinpath(
            'mainenlab', 'Subjects', 'ZFM-01935', '2021-02-05', '001', 'alf'
        )
        eid = self.one.to_eid(session_path)
        self.assertEqual(eid, expected)
        # exp ref str
        eid = self.one.to_eid('2021-02-05_001_ZFM-01935')
        self.assertEqual(eid, expected)
        # UUID
        eid = self.one.to_eid(eid)
        self.assertEqual(eid, expected)
        # session URL
        base_url = 'https://alyx.internationalbrainlab.org/'
        eid = self.one.to_eid(base_url + 'sessions/' + str(eid))
        self.assertEqual(eid, expected)
        # None
        self.assertIsNone(self.one.to_eid(None))

        # Test value errors
        with self.assertRaises(ValueError):
            self.one.to_eid('fakeid')
        with self.assertRaises(ValueError):
            self.one.to_eid(util)

    def test_path2eid(self):
        """Test for ConversionMixin.path2eid (offline mode)."""
        verifiable = self.one.path2eid('CSK-im-007/2021-03-21/002')
        self.assertIsNone(verifiable)

        verifiable = self.one.path2eid('ZFM-01935/2021-02-05/001')
        expected = UUID('d3372b15-f696-4279-9be5-98f15783b5bb')
        self.assertEqual(verifiable, expected)

        session_path = Path.home().joinpath('mainenlab', 'Subjects', 'ZFM-01935',
                                            '2021-02-05', '001', 'alf')
        verifiable = self.one.path2eid(session_path)
        self.assertEqual(verifiable, expected)

        # Test short circuit
        with mock.patch.object(self.one._cache, 'sessions', new=pd.DataFrame([])):
            self.assertIsNone(self.one.path2eid(session_path))

    def test_eid2path(self):
        """Test for ConversionMixin.eid2path."""
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        verifiable = self.one.eid2path(eid)
        expected = ALFPath(self.tempdir.name).joinpath(
            'mainenlab', 'Subjects', 'ZFM-01935', '2021-02-05', '001')
        self.assertIsInstance(verifiable, ALFPath)
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            self.one.eid2path('fakeid')
        self.assertIsNone(self.one.eid2path(eid.replace('d', 'b')))

        # Test list
        verifiable = self.one.eid2path([eid, eid])
        self.assertIsInstance(verifiable, list)
        self.assertTrue(len(verifiable) == 2)

        # Test short circuit
        with mock.patch.object(self.one._cache, 'sessions', new=pd.DataFrame([])):
            self.assertIsNone(self.one.eid2path(eid))

    def test_eid2ref(self):
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        verifiable = self.one.eid2ref(eid, parse=False, as_dict=True)
        expected = {'subject': 'ZFM-01935', 'date': '2021-02-05', 'sequence': '001'}
        self.assertCountEqual(expected, verifiable)
        verifiable = self.one.eid2ref(eid, parse=True, as_dict=True)
        expected = {'subject': 'ZFM-01935', 'date': datetime.date(2021, 2, 5), 'sequence': 1}
        self.assertCountEqual(expected, verifiable)
        verifiable = self.one.eid2ref(eid, parse=False, as_dict=False)
        self.assertCountEqual('2021-02-05_001_ZFM-01935', verifiable)
        verifiable = self.one.eid2ref(eid, parse=True, as_dict=False)
        self.assertCountEqual('2021-02-05_1_ZFM-01935', verifiable)

    def test_path2record(self):
        """Tests for ConversionMixin.path2record method."""
        file = Path(self.tempdir.name).joinpath('cortexlab', 'Subjects', 'KS005', '2019-04-02',
                                                '001', 'alf', '_ibl_wheel.position.npy')
        rec = self.one.path2record(file)
        self.assertIsInstance(rec, pd.Series)
        self.assertTrue(file.as_posix().endswith(rec['rel_path']))

        # Test URL
        uuid = '6cbb724e-c7ec-4eab-b24b-555001502d10'
        parts = add_uuid_string(file, uuid).parts[-7:]
        url = TEST_DB_2['base_url'] + '/'.join(('', *parts))
        rec = self.one.path2record(url)
        self.assertIsInstance(rec, pd.Series)
        self.assertTrue(file.as_posix().endswith(rec['rel_path']))
        # With a UUID missing from cache, should return None
        uuid = '94285bfd-7500-4583-83b1-906c420cc667'
        parts = add_uuid_string(file, uuid).parts[-7:]
        url = TEST_DB_2['base_url'] + '/'.join(('', *parts))
        self.assertIsNone(self.one.path2record(url))

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.path2record(file))

        # Test short circuit
        empty = self.one._cache.datasets.iloc[0:0].copy()
        with mock.patch.object(self.one._cache, 'datasets', new=empty):
            self.assertIsNone(self.one.path2record(file))

        # Test empty session
        with mock.patch.object(self.one._cache, 'datasets', new=self.one._cache.datasets.iloc[:2]):
            self.assertIsNone(self.one.path2record(file))

        # Test session path input
        session_path = Path(self.tempdir.name).joinpath(
            'cortexlab', 'Subjects', 'KS005', '2019-04-02', '001'
        )
        rec = self.one.path2record(session_path)
        self.assertIsInstance(rec, pd.Series)
        self.assertEqual(rec.name, UUID('bc93a3b2-070d-47a8-a2b8-91b3b6e9f25c'))

        with mock.patch.object(self.one._cache, 'sessions', new=empty):
            self.assertIsNone(self.one.path2record(session_path))

    def test_is_exp_ref(self):
        ref = {'date': datetime.datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
        self.assertTrue(self.one.is_exp_ref(ref))
        self.assertTrue(self.one.is_exp_ref('2018-07-13_001_flowers'))
        self.assertTrue(self.one.is_exp_ref('2018-07-13_1_flowers'))
        self.assertTrue(self.one.is_exp_ref('2023-03-14_11_HB_003'))
        self.assertFalse(self.one.is_exp_ref('2018-invalid_ref-s'))
        # Test recurse
        refs = ('2018-07-13_001_flowers', '2018-07-13_1_flowers')
        self.assertTrue(all(self.one.is_exp_ref(refs)))

    def test_ref2dict(self):
        # Test ref string (none padded)
        d = self.one.ref2dict('2018-07-13_1_flowers')
        expected = {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
        self.assertEqual(d, expected)
        # Test short circuit
        self.assertEqual(self.one.ref2dict(d), expected)
        # Test padded number and parse
        d = self.one.ref2dict('2018-07-13_001_flowers', parse=False)
        expected = {'date': '2018-07-13', 'sequence': '001', 'subject': 'flowers'}
        self.assertEqual(d, expected)
        # Test list input
        d = self.one.ref2dict(['2018-07-13_1_flowers', '2020-01-23_002_ibl_witten_01'])
        expected = [
            {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'},
            {'date': datetime.date(2020, 1, 23), 'sequence': 2, 'subject': 'ibl_witten_01'}
        ]
        self.assertEqual(d, expected)

    def test_dict2ref(self):
        d1 = {'date': '2018-07-13', 'sequence': '001', 'subject': 'flowers'}
        self.assertEqual(self.one.dict2ref(d1), '2018-07-13_001_flowers')
        d2 = {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
        self.assertEqual(self.one.dict2ref(d2), '2018-07-13_1_flowers')
        self.assertIsNone(self.one.dict2ref({}))
        expected = ['2018-07-13_001_flowers', '2018-07-13_1_flowers']
        self.assertCountEqual(expected, self.one.dict2ref([d1, d2]))
        d3 = {'start_time': '2023-04-07T17:34:11.403825', 'number': '001', 'subject': 'flowers'}
        self.assertEqual(self.one.dict2ref(d3), '2023-04-07_001_flowers')
        start_date_time = datetime.datetime.fromisoformat(d3['start_time'])
        d4 = {'start_time': start_date_time, 'number': 1, 'subject': 'flowers'}
        self.assertEqual(self.one.dict2ref(d4), '2023-04-07_1_flowers')

    def test_path2ref(self):
        path_str = Path('E:/FlatIron/Subjects/zadorlab/flowers/2018-07-13/001')
        ref = self.one.path2ref(path_str)
        expected = {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1}
        self.assertEqual(expected, ref)
        ref = self.one.path2ref(path_str, as_dict=False)
        expected = '2018-07-13_001_flowers'
        self.assertEqual(expected, ref)
        expected = {'subject': 'flowers', 'date': '2018-07-13', 'sequence': '001'}
        ref = self.one.path2ref(path_str, parse=False)
        self.assertEqual(expected, ref)
        path_str2 = 'E:/FlatIron/Subjects/churchlandlab/CSHL046/2020-06-20/002'
        refs = self.one.path2ref([path_str, path_str2])
        expected = [
            {'subject': 'flowers', 'date': datetime.date(2018, 7, 13), 'sequence': 1},
            {'subject': 'CSHL046', 'date': datetime.date(2020, 6, 20), 'sequence': 2}
        ]
        self.assertCountEqual(expected, refs)
        # Check support of non-zero-padded sequences
        ref = self.one.path2ref(path_str.with_name('1'), as_dict=False)
        self.assertEqual('2018-07-13_1_flowers', ref)
        # The regex matches sequence length between 1 and 3. If zero-padded, must be 3 digits.
        path_str4 = path_str.with_name('01')
        self.assertIsNone(self.one.path2ref(path_str4, as_dict=False))

    def test_ref2path(self):
        ref = {'subject': 'flowers', 'date': datetime.datetime(2018, 7, 13).date(), 'sequence': 1}
        path = self.one.ref2path(ref)
        self.assertIsInstance(path, Path)
        self.assertTrue(path.as_posix().endswith('zadorlab/Subjects/flowers/2018-07-13/001'))
        paths = self.one.ref2path(['2018-07-13_1_flowers', '2019-04-11_1_KS005'])
        expected = ['zadorlab/Subjects/flowers/2018-07-13/001',
                    'cortexlab/Subjects/KS005/2019-04-11/001']
        self.assertTrue(all(x.as_posix().endswith(y) for x, y in zip(paths, expected)))


@unittest.skipIf(OFFLINE_ONLY, 'online only tests')
class TestOnlineConverters(unittest.TestCase):
    """Currently these methods hit the /docs endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        # Create ONE object with temp cache dir
        cls.one = ONE(**TEST_DB_2)
        cls.one.load_cache()  # load local cache tables
        cls.eid = UUID('4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a')
        cls.pid = UUID('da8dfec1-d265-44e8-84ce-6ae9c109b8bd')
        cls.session_record = cls.one.get_details(cls.eid)

    def test_to_eid(self):
        """Test for ConversionMixin.to_eid."""
        eid = self.one.to_eid(self.session_record)
        self.assertEqual(eid, self.eid)

    def test_record2url(self):
        """Test for ConversionMixin.record2url."""
        rec = self.one.get_details(self.eid, full=True, query_type='local')
        idx = rec.rel_path == 'alf/probe00/_phy_spikes_subset.channels.npy'
        # As pd.Series
        url = self.one.record2url(rec[idx].squeeze())
        expected = ('https://ibl.flatironinstitute.org/public/hoferlab/Subjects/'
                    'SWC_043/2020-09-21/001/alf/probe00/'
                    '_phy_spikes_subset.channels.00c234a3-a4ff-4f97-a522-939d15528a45.npy')
        self.assertEqual(expected, url)
        # As pd.DataFrame
        url = self.one.record2url(rec[idx])
        self.assertEqual([expected], url)
        # Session record
        rec = self.one._cache['sessions'].loc[self.eid]
        url = self.one.record2url(rec)
        expected = 'https://ibl.flatironinstitute.org/public/' \
                   'hoferlab/Subjects/SWC_043/2020-09-21/001'
        self.assertEqual(expected, url)
        # Check type checking
        self.assertRaises(TypeError, self.one.record2url, rec.to_dict())

    def test_record2path(self):
        """Test for ConversionMixin.record2path."""
        rec = self.one.get_details(self.eid, full=True, query_type='local')
        # As pd.Series
        alf_path = ('hoferlab/Subjects/SWC_043/2020-09-21/001/'
                    'alf/probe00/_phy_spikes_subset.channels.npy')
        expected = ALFPath(self.one.alyx.cache_dir).joinpath(*alf_path.split('/'))
        data_id = UUID('00c234a3-a4ff-4f97-a522-939d15528a45')
        path = self.one.record2path(rec.loc[(self.eid, data_id)])
        self.assertIsInstance(path, ALFPath)
        self.assertEqual(expected, path)
        # As pd.DataFrame
        idx = rec.rel_path == 'alf/probe00/_phy_spikes_subset.channels.npy'
        path = self.one.record2path(rec[idx])
        self.assertEqual([expected], path)
        # Test validation
        self.assertRaises(AssertionError, self.one.record2path, rec[idx].droplevel(0))  # no eid
        self.assertRaises(TypeError, self.one.record2path, rec[idx].to_dict())
        unknown = rec[idx].squeeze().rename(index=(str(uuid4()), data_id))
        self.assertRaises(ValueError, self.one.record2path, unknown)  # unknown eid
        # With UUID in file name
        try:
            self.one.uuid_filenames = True
            expected = expected.with_suffix(f'.{data_id}.npy')
            self.assertEqual([expected], self.one.record2path(rec[idx]))  # as pd.DataFrame
            verifiable = self.one.record2path(rec[idx].squeeze())
            self.assertEqual(expected, verifiable)  # as pd.Series
            self.assertIsInstance(verifiable, ALFPath)
        finally:
            self.one.uuid_filenames = False

    def test_eid2path(self):
        """Test for OneAlyx.eid2path."""
        verifiable = self.one.eid2path(self.eid, query_type='remote')
        expected = ALFPath(self.one.cache_dir).joinpath(
            'hoferlab', 'Subjects', 'SWC_043', '2020-09-21', '001',)
        self.assertIsInstance(verifiable, ALFPath)
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            self.one.eid2path('fakeid', query_type='remote')
        self.assertIsNone(self.one.eid2path(str(self.eid).replace('d', 'b')))

        # Test list
        verifiable = self.one.eid2path([self.eid, self.eid], query_type='remote')
        self.assertIsInstance(verifiable, list)
        self.assertTrue(len(verifiable) == 2)

    def test_path2eid(self):
        """Test for OneAlyx.path2eid method."""
        test_path = Path(self.one.cache_dir).joinpath(
            'hoferlab', 'Subjects', 'SWC_043', '2020-09-21', '001')
        verifiable = self.one.path2eid(test_path, query_type='remote')
        self.assertEqual(self.eid, verifiable)
        # Check works with list
        verifiable = self.one.path2eid([test_path, test_path], query_type='remote')
        self.assertEqual([self.eid, self.eid], verifiable)
        # Check returns None when path invalid session
        verifiable = self.one.path2eid(test_path.parent, query_type='remote')
        self.assertIsNone(verifiable)

    def test_pid2eid(self):
        """Test for OneAlyx.pid2eid method."""
        if 'insertions' in self.one._cache:
            del self.one._cache['insertions']
        self.assertRaises(NotImplementedError, self.one.pid2eid, self.pid, query_type='local')
        self.assertEqual((self.eid, 'probe00'), self.one.pid2eid(self.pid))
        # Check cache table updated
        self.assertIn('insertions', self.one._cache)
        self.assertIn(self.eid, self.one._cache['insertions'].index)
        # Makes sure the meta data of the newly created insertion table is populated
        meta_data = self.one._cache['_meta']['raw']['insertions']
        self.assertIn('date_created', meta_data.keys())
        self.assertEqual(meta_data['origin'], {'https://openalyx.internationalbrainlab.org'})
        # Local mode should now work
        self.assertEqual((self.eid, 'probe00'), self.one.pid2eid(self.pid, query_type='local'))
        # Test behaviour when pid not found
        pid = UUID('00000000-0000-0000-0000-000000000000')
        self.assertEqual((None, None), self.one.pid2eid(pid, query_type='local'))
        self.assertEqual((None, None), self.one.pid2eid(pid, query_type='remote'))
        # Non-404 status code should raise
        err = HTTPError()
        err.response = self.one._cache.__class__({'status_code': 500})
        with mock.patch.object(self.one.alyx, 'get', side_effect=err):
            self.assertRaises(HTTPError, self.one.pid2eid, pid, query_type='remote')

    def test_eid2pid(self):
        """Test for OneAlyx.eid2pid method."""
        if 'insertions' in self.one._cache:
            del self.one._cache['insertions']
        self.assertRaises(NotImplementedError, self.one.eid2pid, self.eid, query_type='local')
        # Check invalid eid
        self.assertEqual((None, None), self.one.eid2pid(None))
        self.assertEqual((None, None, None), self.one.eid2pid(None, details=True))
        # Check valid eid
        expected = (
            [self.pid, UUID('6638cfb3-3831-4fc2-9327-194b76cf22e1')], ['probe00', 'probe01']
        )
        self.assertEqual(expected, self.one.eid2pid(self.eid))
        *_, det = self.one.eid2pid(self.eid, details=True)
        self.assertEqual(2, len(det))
        expected_keys = {'id', 'name', 'model', 'serial'}
        for d in det:
            self.assertTrue(set(d.keys()) >= expected_keys)
        # Check cache table updated
        cache = self.one._cache
        self.assertIn('insertions', cache)
        self.assertTrue(cache['insertions'].index.get_level_values(1).isin(expected[0]).all())
        # Check local mode should now work
        self.assertEqual(expected, self.one.eid2pid(self.eid, query_type='local'))
        *_, det = self.one.eid2pid(self.eid, details=True, query_type='local')
        for d in det:
            self.assertTrue(set(d.keys()) >= expected_keys)
        # Check behaviour when eid not found in local mode
        eid = UUID('00000000-0000-0000-0000-000000000000')
        self.assertEqual((None, None), self.one.eid2pid(eid, query_type='local'))
        out = self.one.eid2pid(eid, query_type='local', details=True)
        self.assertEqual((None, None, None), out)

    def test_ses2records(self):
        """Test one.converters.ses2records function."""
        ses = self.one.alyx.rest('sessions', 'read', id=self.eid)
        session, datasets = converters.ses2records(ses)

        # Verify returned tables are compatible with cache tables
        self.assertIsInstance(session, pd.Series)
        self.assertIsInstance(datasets, pd.DataFrame)
        self.assertEqual(session.name, self.eid)
        self.assertCountEqual(session.keys(), self.one._cache['sessions'].columns)
        self.assertEqual(len(datasets), len(ses['data_dataset_session_related']))
        expected = list(EMPTY_DATASETS_FRAME.columns) + ['default_revision']
        self.assertCountEqual(expected, datasets.columns)
        self.assertEqual(tuple(datasets.index.names), ('eid', 'id'))
        self.assertIsInstance(datasets.qc.dtype, pd.CategoricalDtype)

        # Check behaviour when no datasets present
        ses['data_dataset_session_related'] = []
        _, datasets = converters.ses2records(ses)
        self.assertTrue(datasets.empty)

    def test_datasets2records(self):
        """Test one.converters.datasets2records function."""
        dsets = self.one.alyx.rest('datasets', 'list', session=str(self.eid))
        datasets = converters.datasets2records(dsets)

        # Verify returned tables are compatible with cache tables
        self.assertIsInstance(datasets, pd.DataFrame)
        self.assertTrue(len(datasets) >= len(dsets))
        expected = list(EMPTY_DATASETS_FRAME.columns) + ['default_revision']
        self.assertCountEqual(expected, datasets.columns)
        self.assertEqual(tuple(datasets.index.names), ('eid', 'id'))
        self.assertIsInstance(datasets.qc.dtype, pd.CategoricalDtype)

        # Test extracts additional fields
        fields = ('url', 'auto_datetime')
        datasets = converters.datasets2records(dsets, additional=fields)
        self.assertTrue(set(datasets.columns) >= set(fields))
        self.assertTrue(all(datasets['url'].str.startswith('http')))

        # Test single input
        dataset = converters.datasets2records(dsets[0])
        self.assertTrue(len(dataset) == 1)
        # Test records when data missing
        for fr in dsets[0]['file_records']:
            fr['exists'] = False
        empty = converters.datasets2records(dsets[0])
        self.assertTrue(isinstance(empty, pd.DataFrame) and empty.empty)


class TestAlyx2Path(unittest.TestCase):
    dset = {
        'url': 'https://alyx.internationalbrainlab.org/'
               'datasets/00059298-1b33-429c-a802-fa51bb662d72',
        'name': 'channels.localCoordinates.npy', 'created_by': 'nate',
        'created_datetime': '2020-02-07T22:08:08.053982',
        'dataset_type': 'channels.localCoordinates', 'data_format': 'npy',
        'collection': 'alf/probe00',
        'session': ('https://alyx.internationalbrainlab.org/'
                    'sessions/7cffad38-0f22-4546-92b5-fd6d2e8b2be9'),
        'file_size': 6064, 'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6', 'version': '1.5.36',
        'experiment_number': 1,
        'file_records': [
            {'id': 'c9ae1b6e-03a6-41c9-9e1b-4a7f9b5cfdbf', 'data_repository': 'ibl_floferlab_SR',
             'data_repository_path': '/mnt/s0/Data/Subjects/',
             'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
             'data_url': None, 'exists': True},
            {'id': 'f434a638-bc61-4695-884e-70fd1e521d60', 'data_repository': 'flatiron_hoferlab',
             'data_repository_path': '/hoferlab/Subjects/',
             'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
             'data_url': (
                 'https://ibl.flatironinstitute.org/hoferlab/Subjects/SWC_014/2019-12-11/001/'
                 'alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'),
             'exists': True}],
        'auto_datetime': '2021-02-10T20:24:31.484939'}

    def test_dsets_2_path(self):
        one_path = ('/one_root/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                    'channels.localCoordinates.npy')

        # Test one_path_from_dataset
        root = PurePosixPath('/one_root')
        testable = converters.one_path_from_dataset(self.dset, one_cache=root)
        self.assertIsInstance(testable, PurePosixALFPath)
        self.assertEqual(str(testable), one_path)
        # Check list input
        testable = converters.one_path_from_dataset([self.dset], one_cache=root)
        self.assertIsInstance(testable, list)
        # Check handles string inputs
        testable = converters.one_path_from_dataset(self.dset, one_cache='/one_root')
        self.assertTrue(hasattr(testable, 'is_absolute'), 'Failed to return Path object')
        self.assertIsInstance(testable, ALFPath)
        self.assertEqual(str(testable).replace('\\', '/'), one_path)

        # Test one_path_from_dataset using Windows path
        one_path = PureWindowsPath(r'C:/Users/User/')
        testable = converters.one_path_from_dataset(self.dset, one_cache=one_path)
        self.assertIsInstance(testable, PureWindowsALFPath)
        self.assertTrue(str(testable).startswith(str(one_path)))
        self.assertTrue('hoferlab/Subjects' in testable.as_posix())
        # Check repository arg
        testable = converters.path_from_dataset(self.dset,
                                                root_path=root, repository='ibl_floferlab_SR')
        self.assertTrue('mnt/s0/Data/Subjects' in testable.as_posix())

        # Tests path_from_filerecord: when given a string, a system path object should be returned
        fr = self.dset['file_records'][0]
        testable = converters.path_from_filerecord(fr, root_path='C:\\')
        self.assertIsInstance(testable, ALFPath)
        # Check list
        testable = converters.path_from_filerecord([fr], root_path='C:\\')
        self.assertIsInstance(testable, list)
        # Check uuid
        uuid = '00059298-1b33-429c-a802-fa51bb662d72'
        testable = converters.path_from_filerecord(fr, root_path='C:\\', uuid=uuid)
        self.assertTrue(uuid in testable.as_posix())

    def test_session_record2path(self):
        """Test one.converters.session_record2path."""
        rec = {'subject': 'ALK01', 'date': '2020-01-01', 'number': 1}
        path = converters.session_record2path(rec)
        self.assertIsInstance(path, PurePosixALFPath)
        self.assertEqual(path, PurePosixALFPath('ALK01/2020-01-01/001'))

        rec = {'date': datetime.datetime.fromisoformat('2020-01-01').date(),
               'number': '001', 'lab': 'foo', 'subject': 'ALK01'}
        path = converters.session_record2path(rec, str(Path.home()))
        self.assertIsInstance(path, ALFPath)
        self.assertEqual(path, Path.home() / 'foo/Subjects/ALK01/2020-01-01/001')


class TestWrappers(unittest.TestCase):
    """Test for one.converters decorators."""

    def test_recurse(self):
        """Test converters.recurse decorator."""
        # Check accepts different numbers of input args
        wrapped = converters.recurse(lambda x, y, z: y * 2)
        self.assertEqual(wrapped(1, 2, 3), 4)
        wrapped = converters.recurse(lambda x, y: y * 2)
        self.assertEqual(wrapped(1, 2), 4)
        wrapped = converters.recurse(lambda: 8)
        self.assertEqual(wrapped(), 8)
        # Check recurse of lists/tuples
        wrapped = converters.recurse(lambda x, y: y * 2)
        self.assertEqual(wrapped(None, [1, 2, 3, 4]), [2, 4, 6, 8])
        self.assertEqual(wrapped(None, (1, 2, 3, 4)), [2, 4, 6, 8])

    def test_parse_values(self):
        wrapped = converters.parse_values(lambda x: x)
        # Check ignores when str
        ref = 'subject_1_2021-01-01'
        self.assertEqual(ref, wrapped(ref, parse=True))
        # Check parse values false (should be identity)
        ref = {'subject': 'flowers', 'sequence': '001', 'date': '2021-01-01'}
        self.assertEqual(ref, wrapped(ref.copy(), parse=False))
        # Check parse values true (should change sequence and date)
        expected = {'subject': 'flowers', 'sequence': 1, 'date': datetime.date(2021, 1, 1)}
        self.assertEqual(expected, wrapped(ref.copy(), parse=True))
        # Check handles datetime ISO
        ref['date'] += 'T14:53:53.586024'
        self.assertEqual(expected, wrapped(ref.copy(), parse=True))
        # Check handles list
        self.assertIsInstance(wrapped([ref.copy(), expected], parse=True), list)


if __name__ == "__main__":
    unittest.main(exit=False)
