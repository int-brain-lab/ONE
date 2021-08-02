"""Tests for the one.converters module"""
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID
import datetime

import pandas as pd

from one.api import ONE
from one import converters
from . import util, OFFLINE_ONLY, TEST_DB_2


class TestConverters(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = util.set_up_env()
        # Create ONE object with temp cache dir
        cls.one = ONE(mode='local', cache_dir=cls.tempdir.name)

    def test_to_eid(self):
        expected = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        # Path str
        eid = self.one.to_eid('ZFM-01935/2021-02-05/001')
        self.assertEqual(eid, expected)
        # eid
        eid = self.one.to_eid(eid)
        self.assertEqual(eid, expected)
        # Path
        session_path = Path(self.one._cache_dir).joinpath(
            'mainenlab', 'Subjects', 'ZFM-01935', '2021-02-05', '001', 'alf'
        )
        eid = self.one.to_eid(session_path)
        self.assertEqual(eid, expected)
        # exp ref str
        eid = self.one.to_eid('2021-02-05_001_ZFM-01935')
        self.assertEqual(eid, expected)
        # UUID
        eid = self.one.to_eid(UUID(eid))
        self.assertEqual(eid, expected)
        # session URL
        base_url = 'https://alyx.internationalbrainlab.org/'
        eid = self.one.to_eid(base_url + 'sessions/' + eid)
        self.assertEqual(eid, expected)

        # Test value errors
        with self.assertRaises(ValueError):
            self.one.to_eid('fakeid')
        with self.assertRaises(ValueError):
            self.one.to_eid(util)

    def test_path2eid(self):
        verifiable = self.one.path2eid('CSK-im-007/2021-03-21/002')
        self.assertIsNone(verifiable)

        verifiable = self.one.path2eid('ZFM-01935/2021-02-05/001')
        expected = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        self.assertEqual(verifiable, expected)

        session_path = Path.home().joinpath('mainenlab', 'Subjects', 'ZFM-01935',
                                            '2021-02-05', '001', 'alf')
        verifiable = self.one.path2eid(session_path)
        self.assertEqual(verifiable, expected)

    def test_eid2path(self):
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        verifiable = self.one.eid2path(eid)
        expected = Path(self.tempdir.name).joinpath('mainenlab', 'Subjects', 'ZFM-01935',
                                                    '2021-02-05', '001',)
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            self.one.eid2path('fakeid')
        self.assertIsNone(self.one.eid2path(eid.replace('d', 'b')))

        # Test list
        verifiable = self.one.eid2path([eid, eid])
        self.assertIsInstance(verifiable, list)
        self.assertTrue(len(verifiable) == 2)

    def test_path2record(self):
        file = Path(self.tempdir.name).joinpath('cortexlab', 'Subjects', 'KS005', '2019-04-02',
                                                '001', 'alf', '_ibl_wheel.position.npy')
        rec = self.one.path2record(file)
        self.assertIsInstance(rec, pd.DataFrame)
        rel_path, = rec['rel_path'].values
        self.assertTrue(file.as_posix().endswith(rel_path))

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.path2record(file))

    def test_is_exp_ref(self):
        ref = {'date': datetime.datetime(2018, 7, 13).date(), 'sequence': 1, 'subject': 'flowers'}
        self.assertTrue(self.one.is_exp_ref(ref))
        self.assertTrue(self.one.is_exp_ref('2018-07-13_001_flowers'))
        self.assertTrue(self.one.is_exp_ref('2018-07-13_1_flowers'))
        self.assertFalse(self.one.is_exp_ref('2018-invalid_ref-s'))
        # Test recurse
        refs = ('2018-07-13_001_flowers', '2018-07-13_1_flowers')
        self.assertTrue(all(x is True for x in self.one.is_exp_ref(refs)))

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
        self.assertEqual('2018-07-13_001_flowers', self.one.dict2ref(d1))
        d2 = {'date': datetime.date(2018, 7, 13), 'sequence': 1, 'subject': 'flowers'}
        self.assertEqual('2018-07-13_1_flowers', self.one.dict2ref(d2))
        self.assertIsNone(self.one.dict2ref({}))
        expected = ['2018-07-13_001_flowers', '2018-07-13_1_flowers']
        self.assertCountEqual(expected, self.one.dict2ref([d1, d2]))

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
    """Currently these methods hit the /docs endpoint"""
    @classmethod
    def setUpClass(cls) -> None:
        # Create ONE object with temp cache dir
        cls.one = ONE(**TEST_DB_2)
        cls.eid = '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a'
        cls.session_record = cls.one.get_details(cls.eid)

    def test_to_eid(self):
        eid = self.one.to_eid(self.session_record)
        self.assertEqual(eid, self.eid)

    def test_record2url(self):
        rec = self.one.get_details(self.eid, full=True, query_type='local')
        # As pd.Series
        url = self.one.record2url(rec.iloc[0])
        expected = ('https://ibl.flatironinstitute.org/public/hoferlab/Subjects/'
                    'SWC_043/2020-09-21/001/raw_ephys_data/probe00/'
                    '_spikeglx_ephysData_g0_t0.imec0.ap.94285bfd-7500-4583-83b1-906c420cc667.cbin')
        self.assertEqual(expected, url)
        # As pd.DataFrame
        url = self.one.record2url(rec.iloc[[0]])
        expected = ('https://ibl.flatironinstitute.org/public/hoferlab/Subjects/'
                    'SWC_043/2020-09-21/001/raw_ephys_data/probe00/'
                    '_spikeglx_ephysData_g0_t0.imec0.ap.94285bfd-7500-4583-83b1-906c420cc667.cbin')
        self.assertEqual(expected, url)

    def test_record2path(self):
        rec = self.one.get_details(self.eid, full=True, query_type='local')
        # As pd.Series
        alf_path = ('public/hoferlab/Subjects/SWC_043/2020-09-21/001/raw_ephys_data/probe00/'
                    '_spikeglx_ephysData_g0_t0.imec0.ap.cbin')
        expected = Path(self.one.alyx.cache_dir).joinpath(*alf_path.split('/'))
        path = self.one.record2path(rec.iloc[0])
        self.assertIsInstance(path, Path)
        self.assertEqual(expected, path)
        # As pd.DataFrame
        path = self.one.record2path(rec.iloc[[0]])
        self.assertEqual(expected, path)

    def test_ref2dj(self):
        try:
            ref = '2020-09-21_1_SWC_043'
            restriction = self.one.ref2dj(ref)
        except ModuleNotFoundError:
            self.skipTest('requires ibl_pipeline')
        self.assertTrue(hasattr(restriction, 'fetch'))
        expected = {
            'subject_uuid': UUID('70bf8cbd-d312-4654-a4ea-3a21ea2f541b'),
            'session_start_time': datetime.datetime(2020, 9, 21, 19, 2, 17),
            'session_number': 1,
            'session_date': datetime.date(2020, 9, 21),
            'subject_nickname': 'SWC_043'}
        self.assertEqual(restriction.fetch1(), expected)


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
        self.assertEqual(str(testable), one_path)
        # Check handles string inputs
        testable = converters.one_path_from_dataset(self.dset, one_cache='/one_root')
        self.assertTrue(hasattr(testable, 'is_absolute'), 'Failed to return Path object')
        self.assertEqual(str(testable).replace('\\', '/'), one_path)

        # Test one_path_from_dataset using Windows path
        one_path = PureWindowsPath(r'C:/Users/User/')
        testable = converters.one_path_from_dataset(self.dset, one_cache=one_path)
        self.assertIsInstance(testable, PureWindowsPath)
        self.assertTrue(str(testable).startswith(str(one_path)))

        # Tests path_from_filerecord: when given a string, a system path object should be returned
        fr = self.dset['file_records'][0]
        testable = converters.path_from_filerecord(fr, root_path='C:\\')
        self.assertIsInstance(testable, Path)


class TestWrappers(unittest.TestCase):
    def test_recurse(self):
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


if __name__ == '__main__':
    unittest.main()
