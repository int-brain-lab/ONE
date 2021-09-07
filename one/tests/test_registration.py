import logging
import unittest
import unittest.mock
import string
import random
import datetime
import fnmatch
from io import StringIO
from sys import version_info  # For 3.7
from pkg_resources import parse_version

from one.api import ONE
from one import registration
import one.alf.exceptions as alferr
from one.alf.io import next_num_folder
from . import TEST_DB_1, OFFLINE_ONLY
from one.tests import util


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestRegistrationClient(unittest.TestCase):
    one = None
    subject = None
    temp_dir = None

    @classmethod
    def setUpClass(cls) -> None:
        temp_dir = util.set_up_env(use_temp_cache=False)
        if parse_version('.'.join(map(str, version_info[:2]))) >= parse_version('3.8'):
            cls.addClassCleanup(temp_dir.cleanup)  # py3.8
        cls.one = ONE(**TEST_DB_1, cache_dir=temp_dir.name)
        cls.subject = ''.join(random.choices(string.ascii_letters, k=10))
        cls.one.alyx.rest('subjects', 'create', data={'lab': 'mainenlab', 'nickname': cls.subject})
        # Create some files for this subject
        session_path = cls.one.alyx.cache_dir / cls.subject / str(datetime.date.today()) / '001'
        cls.session_path = session_path
        for rel_path in cls.one.list_datasets():
            filepath = session_path.joinpath(rel_path)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            filepath.touch()

    def setUp(self) -> None:
        self.client = registration.RegistrationClient(one=self.one)

    def test_water_administration(self):
        record = self.client.register_water_administration(self.subject, 35.10000000235)
        self.assertEqual(record['subject'], self.subject)
        self.assertEqual(record['water_administered'], 35.1)
        self.assertEqual(record['water_type'], 'Water')
        self.assertEqual(record['user'], self.one.alyx.user)
        # Create session to associate
        d = {'subject': self.subject,
             'procedures': ['Behavior training/tasks'],
             'type': 'Base',
             'users': [self.one.alyx.user],
             }
        ses = self.one.alyx.rest('sessions', 'create', data=d)
        volume = random.random()
        record = self.client.register_water_administration(self.subject, volume,
                                                           session=ses['url'])
        self.assertEqual(record['subject'], self.subject)
        self.assertEqual(record['session'], ses['url'][-36:])
        # Check validations
        with self.assertRaises(ValueError):
            self.client.register_water_administration(self.subject, volume, session='NaN')
        with self.assertRaises(ValueError):
            self.client.register_water_administration(self.subject, .0)

    def test_register_weight(self):
        record = self.client.register_weight(self.subject, 35.10000000235)
        self.assertEqual(record['subject'], self.subject)
        self.assertEqual(record['weight'], 35.1)
        self.assertEqual(record['user'], self.one.alyx.user)
        # Check validations
        with self.assertRaises(ValueError):
            self.client.register_weight(self.subject, 0.0)

    def test_ensure_ISO8601(self):
        date = datetime.datetime(2021, 7, 14, 15, 53, 15, 525119)
        self.assertEqual(self.client.ensure_ISO8601(date), '2021-07-14T15:53:15.525119')
        self.assertEqual(self.client.ensure_ISO8601(date.date()), '2021-07-14T00:00:00')
        date_str = '2021-07-14T15:53:15.525119'
        self.assertEqual(self.client.ensure_ISO8601(date_str), date_str)
        with self.assertRaises(ValueError):
            self.client.ensure_ISO8601(f'{date:%D}')

    def test_exists(self):
        # Check user endpoint
        with self.assertRaises(alferr.AlyxSubjectNotFound):
            self.client.assert_exists('foobar', 'subjects')
        self.client.assert_exists(self.subject, 'subjects')
        # Check user endpoint with list
        with self.assertRaises(alferr.ALFError) as ex:
            self.client.assert_exists([self.one.alyx.user, 'foobar'], 'users')
        self.assertIn('foobar', str(ex.exception))

    def test_find_files(self):
        # Remove a dataset type from the client to check that the dataset(s) are ignored
        existing = (x['filename_pattern'] and any(self.session_path.rglob(x['filename_pattern']))
                    for x in self.client.dtypes)
        removed = self.client.dtypes.pop(next(i for i, x in enumerate(existing) if x))
        files = list(self.client.find_files(self.session_path))
        self.assertEqual(6, len(files))
        self.assertTrue(all(x.is_file() for x in files))
        # Check removed file pattern not in file list
        self.assertFalse(fnmatch.filter([x.name for x in files], removed['filename_pattern']))

    def test_create_new_session(self):
        # Check register = True
        session_path, eid = self.client.create_new_session(self.subject, date='2020-01-01')
        expected = self.one.alyx.cache_dir.joinpath(self.subject, '2020-01-01', '001').as_posix()
        self.assertEqual(session_path.as_posix(), expected)
        self.assertIsNotNone(eid)
        # Check register = False
        session_path, eid = self.client.create_new_session(
            self.subject, date='2020-01-01', register=False)
        expected = self.one.alyx.cache_dir.joinpath(self.subject, '2020-01-01', '002').as_posix()
        self.assertEqual(session_path.as_posix(), expected)
        self.assertIsNone(eid)

    def test_register_session(self):
        datasets = self.one.list_datasets(self.one.search()[0])  # Some datasets to create
        session_path = self.one.alyx.cache_dir.joinpath(self.subject, '2020-01-01', '001')
        # Ensure session exists
        file_list = [session_path.joinpath(x) for x in datasets]
        # Create the files before registering
        for x in file_list:
            x.parent.mkdir(exist_ok=True, parents=True)
            x.touch()

        ses, recs = self.client.register_session(str(session_path))
        self.assertTrue(len(ses['data_dataset_session_related']))
        self.assertEqual(len(ses['data_dataset_session_related']), len(recs))

    def test_create_sessions(self):
        session_path = self.session_path.parent / next_num_folder(self.session_path.parent)
        session_path.mkdir(parents=True)
        session_path.joinpath('create_me.flag').touch()
        # Should print session path in dry mode
        with unittest.mock.patch('sys.stdout', new_callable=StringIO) as stdout:
            session_paths, ses = self.client.create_sessions(self.one.alyx.cache_dir, dry=True)
            self.assertTrue(str(session_path) in stdout.getvalue())
            self.assertTrue(len(ses) == 1 and ses[0] is None)
            self.assertTrue(session_path.joinpath('create_me.flag').exists())

        # Should find and register session
        session_paths, ses = self.client.create_sessions(self.one.alyx.cache_dir)
        self.assertTrue(len(ses) == 1 and len(session_paths) == 1)
        self.assertFalse(session_path.joinpath('create_me.flag').exists())
        self.assertEqual(ses[0]['number'], int(session_path.parts[-1]))
        self.assertEqual(session_paths[0], session_path)

    def test_register_files(self):
        # Test a few things not checked in register_session
        session_path, eid = self.client.create_new_session(self.subject)
        # Check registering single file, dry run, default False
        file_name = session_path.joinpath('wheel.position.npy')
        file_name.touch()
        rec = self.client.register_files(str(file_name), default=False, dry=True)
        self.assertIsInstance(rec, dict)
        self.assertFalse(rec['default'])
        self.assertNotIn('id', rec)
        # Add ambiguous dataset type to types list
        self.client.dtypes.append(self.client.dtypes[-1].copy())
        self.client.dtypes[-1]['name'] += '1'
        # Try registering ambiguous / invalid datasets
        ambiguous = self.client.dtypes[-1]['filename_pattern'].replace('*', 'npy')
        files = [session_path.joinpath('wheel.position.xxx'),  # Unknown ext
                 session_path.joinpath('foo.bar.npy'),  # Unknown dtype
                 session_path.joinpath(ambiguous)  # Ambiguous dtype
                 ]
        version = ['1.2.9'] * len(files)
        with self.assertLogs('one.registration', logging.DEBUG) as dbg:
            rec = self.client.register_files(files, versions=version)
            self.assertIn('wheel.position.xxx: No matching extension', dbg.records[0].message)
            self.assertIn('foo.bar.npy: No matching dataset type', dbg.records[1].message)
            self.assertIn(f'{ambiguous}: Multiple matching', dbg.records[2].message)
        self.assertFalse(len(rec))

    @classmethod
    def tearDownClass(cls) -> None:
        for admin in cls.one.alyx.rest('water-administrations', 'list',
                                       django='subject__nickname,' + cls.subject, no_cache=True):
            cls.one.alyx.rest('water-administrations', 'delete', id=admin['url'][-36:])
        for ses in cls.one.alyx.rest('sessions', 'list', subject=cls.subject, no_cache=True):
            cls.one.alyx.rest('sessions', 'delete', id=ses['url'][-36:])
        cls.one.alyx.rest('subjects', 'delete', id=cls.subject)


if __name__ == '__main__':
    unittest.main()
