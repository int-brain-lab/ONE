from pathlib import Path, PurePosixPath, PureWindowsPath
from functools import partial
import unittest
from unittest import mock
import tempfile
import shutil
from uuid import UUID
import json

import numpy as np
import pandas as pd

from one import webclient as wc
from one.api import ONE, One, OneAlyx
import one.lib.io.params
import one.params
import one.alf.exceptions as alferr
from one.lib.brainbox.io import parquet

dset = {
    'url': 'https://alyx.internationalbrainlab.org/datasets/00059298-1b33-429c-a802-fa51bb662d72',
    'name': 'channels.localCoordinates.npy',
    'created_by': 'nate',
    'created_datetime': '2020-02-07T22:08:08.053982',
    'dataset_type': 'channels.localCoordinates',
    'data_format': 'npy',
    'collection': 'alf/probe00',
    'session': ('https://alyx.internationalbrainlab.org/sessions/'
                '7cffad38-0f22-4546-92b5-fd6d2e8b2be9'),
    'file_size': 6064,
    'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6',
    'version': '1.5.36',
    'experiment_number': 1,
    'file_records': [
        {'id': 'c9ae1b6e-03a6-41c9-9e1b-4a7f9b5cfdbf',
         'data_repository': 'ibl_floferlab_SR',
         'data_repository_path': '/mnt/s0/Data/Subjects/',
         'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
         'data_url': None,
         'exists': True},
        {'id': 'f434a638-bc61-4695-884e-70fd1e521d60',
         'data_repository': 'flatiron_hoferlab',
         'data_repository_path': '/hoferlab/Subjects/',
         'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
         'data_url': ('https://ibl.flatironinstitute.org/hoferlab/Subjects/SWC_014/2019-12-11/001/'
                      'alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802'
                      '-fa51bb662d72.npy'),
         'exists': True}
    ],
    'auto_datetime': '2021-02-10T20:24:31.484939'
}


class TestAlyx2Path(unittest.TestCase):

    def test_dsets_2_path(self):
        self.assertEqual(len(wc.globus_path_from_dataset([dset] * 3)), 3)
        sdsc_path = ('/mnt/ibl/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                     'channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy')
        one_path = ('/one_root/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                    'channels.localCoordinates.npy')
        globus_path_sdsc = ('/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                            'channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy')
        globus_path_sr = ('/mnt/s0/Data/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                          'channels.localCoordinates.npy')

        # Test sdsc_path_from_dataset
        testable = wc.sdsc_path_from_dataset(dset)
        self.assertEqual(str(testable), sdsc_path)
        self.assertIsInstance(testable, PurePosixPath)

        # Test one_path_from_dataset
        testable = wc.one_path_from_dataset(dset, one_cache=PurePosixPath('/one_root'))
        self.assertEqual(str(testable), one_path)
        # Check handles string inputs
        testable = wc.one_path_from_dataset(dset, one_cache='/one_root')
        self.assertTrue(hasattr(testable, 'is_absolute'), 'Failed to return Path object')
        self.assertEqual(str(testable).replace('\\', '/'), one_path)

        # Test one_path_from_dataset using Windows path
        one_path = PureWindowsPath(r'C:/Users/User/')
        testable = wc.one_path_from_dataset(dset, one_cache=one_path)
        self.assertIsInstance(testable, PureWindowsPath)
        self.assertTrue(str(testable).startswith(str(one_path)))

        # Test sdsc_globus_path_from_dataset
        testable = wc.sdsc_globus_path_from_dataset(dset)
        self.assertEqual(str(testable), globus_path_sdsc)
        self.assertIsInstance(testable, PurePosixPath)

        # Test globus_path_from_dataset
        testable = wc.globus_path_from_dataset(dset, repository='ibl_floferlab_SR')
        self.assertEqual(str(testable), globus_path_sr)
        self.assertIsInstance(testable, PurePosixPath)

        # Tests _path_from_filerecord: when given a string, a system path object should be returned
        fr = dset['file_records'][0]
        testable = wc._path_from_filerecord(fr, root_path='C:\\')
        self.assertIsInstance(testable, Path)


class TestONECache(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        # FIXME Use util fixture
        fixture = Path(__file__).parent.joinpath('fixtures')
        cls.tempdir = tempfile.TemporaryDirectory()
        # Copy cache files to temporary directory
        for cache_file in ('sessions', 'datasets'):
            filename = shutil.copy(fixture / f'{cache_file}.pqt', cls.tempdir.name)
            assert Path(filename).exists()
        # Create ONE object with temp cache dir
        cls.one = ONE(mode='local', cache_dir=cls.tempdir.name)
        # Create dset files from cache
        for session_path, rel_path in cls.one._cache.datasets[['session_path', 'rel_path']].values:
            filepath = Path(cls.tempdir.name).joinpath(session_path, rel_path)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            filepath.touch()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_list_subjects(self):
        subjects = self.one.list_subjects()
        expected = ['KS005', 'ZFM-01935', 'ZM_1094', 'ZM_1150',
                    'ZM_1743', 'ZM_335', 'clns0730', 'flowers']
        self.assertCountEqual(expected, subjects)

    def test_one_search(self):
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
        query = 'gpio'.upper()
        eids = one.search(data=query)
        self.assertTrue(eids)
        self.assertEqual(eids, ['d3372b15-f696-4279-9be5-98f15783b5bb'])

        # Filter non-existent
        # Set exist for one of the eids to false
        mask = (one._cache['datasets'][['eid_0', 'eid_1']] == parquet.str2np(eids[0]))
        i = one._cache['datasets'][mask].index[0]
        one._cache['datasets'].loc[i, 'exists'] = False  # TODO not lexsorted

        self.assertTrue(len(eids) == len(one.search(data=query, exists_only=True)) + 1)

        # Search task_protocol
        n = 4
        one._cache['sessions'].iloc[:n, -2] = '_iblrig_tasks_biasedChoiceWorld6.4.2'
        eids = one.search(task='biased')
        self.assertEqual(len(eids), n)

        # Search project
        one._cache['sessions'].iloc[:n, -1] = 'ibl_certif_neuropix_recording'
        eids = one.search(proj='neuropix')
        self.assertEqual(len(eids), n)

        # Search number
        number = 1
        eids = one.search(num=number)
        self.assertTrue(all(x.endswith(str(number)) for x in eids))
        number = '002'
        eids = one.search(number=number)
        self.assertTrue(all(x.endswith(number) for x in eids))

        # Test multiple fields, with short params
        eids = one.search(subj='ZFM-02183', date='2021-03-05', num='002', lab='mainen')
        self.assertTrue(len(eids) == 1)

        # Test param error validation
        with self.assertRaises(ValueError):
            one.search(dat='2021-03-05')  # ambiguous
        with self.assertRaises(ValueError):
            one.search(user='mister')  # invalid search term

        # Test details parameter
        eids, details = one.search(date='2021-03-16', lab='witten', details=True)
        self.assertEqual(len(eids), len(details))
        self.assertTrue(all(eid == det.eid for eid, det in zip(eids, details)))

        # Test search without integer ids
        for table in ('sessions', 'datasets'):
            # Set integer uuids to NaN
            cache = self.one._cache[table].reset_index()
            cache[cache.filter(regex=r'_\d{1}$').columns] = np.nan
            self.one._cache[table] = cache.set_index('eid' if table == 'sessions' else 'dset_id')
        query = 'clusters'
        eids = one.search(data=query)
        self.assertTrue(eids)
        self.assertTrue(all(any(Path(self.tempdir.name, x).rglob(f'*{query}*')) for x in eids))

    def test_eid_from_path(self):
        verifiable = self.one.eid_from_path('CSK-im-007/2021-03-21/002')
        self.assertIsNone(verifiable)

        verifiable = self.one.eid_from_path('ZFM-01935/2021-02-05/001')
        expected = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        self.assertEqual(verifiable, expected)

        session_path = Path.home().joinpath('mainenlab', 'Subjects', 'ZFM-01935',
                                            '2021-02-05', '001', 'alf')
        verifiable = self.one.eid_from_path(session_path)
        self.assertEqual(verifiable, expected)

    def test_path_from_eid(self):
        eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        verifiable = self.one.path_from_eid(eid)
        expected = Path(self.tempdir.name).joinpath('mainenlab', 'Subjects', 'ZFM-01935',
                                                    '2021-02-05', '001',)
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            self.one.path_from_eid('fakeid')
        self.assertIsNone(self.one.path_from_eid(eid.replace('d', 'b')))

    @unittest.skip('TODO Move this test?')
    def test_check_exists(self):
        pass

    def test_list_datasets(self):
        dsets = self.one.list_datasets()
        self.assertIsInstance(dsets, np.ndarray)
        self.assertTrue(len(dsets) == np.unique(dsets).size)

        # Test list for eid
        dsets = self.one.list_datasets('KS005/2019-04-02/001')
        self.assertTrue(len(dsets), 27)

        # Test empty
        dsets = self.one.list_datasets('FMR019/2021-03-18/002')
        self.assertIsInstance(dsets, pd.DataFrame)
        self.assertEqual(len(dsets), 0)

    def test_load_session_dataset(self):
        eid = 'KS005/2019-04-02/001'
        # Check download only
        file = self.one.load_session_dataset(eid, '_ibl_wheel.position.npy', download_only=True)
        self.assertIsInstance(file, Path)

        # Check loading data
        np.save(str(file), np.arange(3))  # Make sure we have something to load
        dset = self.one.load_session_dataset(eid, '_ibl_wheel.position.npy')
        self.assertTrue(np.all(dset == np.arange(3)))

        # Check revision filter
        with self.assertRaises(alferr.ALFObjectNotFound):
            self.one.load_session_dataset(eid, '_ibl_wheel.position.npy', revision='v2.3.4')

        # Check collection filter
        file = self.one.load_session_dataset(eid, '_iblrig_leftCamera.timestamps.ssv',
                                             download_only=True, collection='raw_video_data')
        self.assertIsNotNone(file)

    def test_load_dataset_from_id(self):
        id = np.array([[-9204203870374650458, -6411285612086772563]])
        file = self.one.load_dataset_from_id(id, download_only=True)
        self.assertIsInstance(file, Path)
        expected = 'ZFM-01935/2021-02-05/001/alf/probe00/_phy_spikes_subset.waveforms.npy'
        self.assertTrue(file.as_posix().endswith(expected))

        # Details
        _, details = self.one.load_dataset_from_id(id, download_only=True, details=True)
        self.assertIsInstance(details, pd.Series)

        # Load file content with str id
        np.save(str(file), np.arange(3))  # Ensure data to load
        dset = self.one.load_dataset_from_id('a61d0b8a-5819-4480-adbc-fe49b48906a7')
        self.assertTrue(np.all(dset == np.arange(3)))

        # Load file content with UUID
        dset = self.one.load_dataset_from_id(UUID('a61d0b8a-5819-4480-adbc-fe49b48906a7'))
        self.assertTrue(np.all(dset == np.arange(3)))

    def test_load_object(self):
        wheel = self.one.load_object('aaf101c3-2581-450a-8abd-ddb8f557a5ad', 'wheel')
        self.assertIsInstance(wheel, dict)
        self.assertCountEqual(wheel.keys(), ('position', 'velocity', 'timestamps'))

    def test_record_from_path(self):
        file = Path(self.tempdir.name).joinpath('cortexlab', 'Subjects', 'KS005', '2019-04-02',
                                                '001', 'alf', '_ibl_wheel.position.npy')
        rec = self.one.record_from_path(file)
        self.assertIsInstance(rec, pd.DataFrame)
        rel_path, = rec['rel_path'].values
        self.assertTrue(file.as_posix().endswith(rel_path))

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.record_from_path(file))


class TestOneAlyx(unittest.TestCase):
    tempdir = None
    one = None

    @classmethod
    def setUpClass(cls) -> None:
        # TODO Use util fixture
        fixture = Path(__file__).parent.joinpath('fixtures')
        cls.tempdir = tempfile.TemporaryDirectory()
        # Copy cache files to temporary directory
        for cache_file in ('sessions', 'datasets'):
            filename = shutil.copy(fixture / f'{cache_file}.pqt', cls.tempdir.name)
            assert Path(filename).exists()
        # Copy cached rest responses
        rest_cache_dir = Path(one.params.get_params_dir(),
                              '.rest', 'test.alyx.internationalbrainlab.org', 'https')
        rest_cache_dir.mkdir(parents=True, exist_ok=True)
        for file in fixture.joinpath('rest_responses').glob('*'):
            filename = shutil.copy(file, rest_cache_dir)
            assert Path(filename).exists()

        with mock.patch('one.lib.io.params.getfile', new=partial(get_file, cls.tempdir.name)):
            cls.one = OneAlyx(
                base_url='https://test.alyx.internationalbrainlab.org',
                username='test_user',
                password='TapetesBloc18',
                cache_dir=cls.tempdir.name,
                silent=True,
                mode='local'
            )


    def setUp(self) -> None:
        # Create ONE object with temp cache dir
        # TODO Copy over params fixture instead

    @unittest.skip
    def test_download_datasets(self):
        eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
        files = one.download_datasets(['channels.brainLocation.tsv'])

    def test_pid2eid(self):
        pid = 'b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        eid, collection = self.one.pid2eid(pid, query_type='remote')
        self.assertEqual('fc737f3c-2a57-4165-9763-905413e7e341', eid)
        self.assertEqual('probe00', collection)

    def test_url_from_path(self):
        file = Path(self.tempdir.name).joinpath('cortexlab', 'Subjects', 'KS005', '2019-04-04',
                                                '004', 'alf', '_ibl_wheel.position.npy')
        url = self.one.url_from_path(file)
        self.assertTrue(url.startswith(self.one._par.HTTP_DATA_SERVER))
        self.assertTrue('257ff7ae-9ab4-35dc-bac0-245cdc81f6d1' in url)

        file = file.parent / '_fake_obj.attr.npy'
        self.assertIsNone(self.one.url_from_path(file))

    def test_url_from_record(self):
        dataset = self.one._cache['datasets'].loc[[[-2578956635146322139, -3317321292073090886]]]
        url = self.one.url_from_record(dataset)
        expected = ('https://ibl.flatironinstitute.org/'
                    'angelakilab/Subjects/FMR019/2021-03-18/002/alf/'
                    '_ibl_wheel.position.257ff7ae-9ab4-35dc-bac0-245cdc81f6d1.npy')
        self.assertEqual(expected, url)


class TestOneSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.get_file = partial(get_file, self.tempdir.name)
        wc.UniqueSingletons._instances = []  # Delete any active instances

    def test_setup_silent(self):
        """Test setting up parameters with silent flag
        - Mock getfile to return temp dir as param file location
        - Mock input function as fail safe in case function erroneously prompts user for input
        """
        with mock.patch('one.lib.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse):
            one_obj = ONE(silent=True, mode='local')
            self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

        # Check param files were saved
        self.assertEqual(len(list(Path(self.tempdir.name).rglob('.caches'))), 1)
        client_pars = Path(self.tempdir.name).rglob(f'.{one_obj.alyx.base_url.split("/")[-1]}')
        self.assertEqual(len(list(client_pars)), 1)

        # Check uses defaults on second instantiation
        with mock.patch('one.lib.io.params.getfile', new=self.get_file):
            one_obj = ONE(mode='local')
            self.assertEqual(one_obj.alyx.base_url, one.params.default().ALYX_URL)

    def test_setup(self):
        url = 'https://test.alyx.internationalbrainlab.org'
        one.params.input = lambda prompt: url if 'url' in prompt.lower() else 'mock_input'
        one.params.getpass = lambda prompt: 'mock_pwd'
        one.params.print = lambda text: 'mock_print'
        # Mock getfile function to return a path to non-existent file instead of usual one pars
        with mock.patch('one.lib.io.params.getfile', new=self.get_file):
            one_obj = OneAlyx(mode='local', username='test_user', password='TapetesBloc18')
        self.assertEqual(one_obj.alyx._par.ALYX_URL, url)
        client_pars = Path(self.tempdir.name).rglob(f'.{one_obj.alyx.base_url.split("/")[-1]}')
        self.assertEqual(len(list(client_pars)), 1)

    def test_patch_params(self):
        """Test patching legacy params to the new location"""
        # Save some old-style params
        old_pars = (one.params.default()
                    .set('CACHE_DIR', self.tempdir.name)
                    .set('HTTP_DATA_SERVER_PWD', '123'))
        with open(Path(self.tempdir.name, '.one_params'), 'w') as f:
            json.dump(old_pars.as_dict(), f)

        with mock.patch('one.lib.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse):
            one_obj = ONE(silent=False, mode='local')
        self.assertEqual(one_obj.alyx._par.HTTP_DATA_SERVER_PWD, '123')

    def test_one_factory(self):
        """Tests the ONE class factory"""
        with mock.patch('one.lib.io.params.getfile', new=self.get_file),\
                mock.patch('one.params.input', new=self.assertFalse):
            # Cache dir not in client cache map; use One (light)
            one_obj = ONE(cache_dir=self.tempdir.name)
            self.assertIsInstance(one_obj, One)

            # No cache dir provided; use OneAlyx (silent setup mode)
            one_obj = ONE(silent=True, mode='local')
            self.assertIsInstance(one_obj, OneAlyx)

            # The cache dir is in client cache map; use OneAlyx
            one_obj = ONE(cache_dir=one_obj.alyx.cache_dir, mode='local')
            self.assertIsInstance(one_obj, OneAlyx)

            # A db URL was provided; use OneAlyx
            one_obj = ONE(base_url='https://test.alyx.internationalbrainlab.org',
                          username='test_user',
                          password='TapetesBloc18',
                          mode='local',  # Don't download cache (could also set cache_dir)
                          silent=True)
            self.assertIsInstance(one_obj, OneAlyx)

            # The offline param was given, raise deprecation warning (via log)
            # with self.assertLogs(logging.getLogger('ibllib'), logging.WARNING):
            #     ONE(offline=True, cache_dir=self.tempdir.name)
            with self.assertWarns(DeprecationWarning):
                ONE(offline=True, cache_dir=self.tempdir.name)


def get_file(root: str, str_id: str) -> str:
    """
    A stub function for one.lib.io.params.getfile.  Allows the injection of a different param dir.
    :param root: The root directory of the new parameters
    :param str_id: The parameter string identifier
    :return: The parameter filename
    """
    parts = ['.' + p if not p.startswith('.') else p for p in Path(str_id).parts]
    pfile = Path(root, *parts).as_posix()
    return pfile
