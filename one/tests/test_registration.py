"""Unit tests for the one.registration module."""
import json
import logging
import unittest
import unittest.mock
import string
import random
import datetime
import fnmatch
from io import StringIO
from pathlib import Path

from iblutil.util import Bunch
from requests.exceptions import HTTPError
from requests.models import Response

from one.api import ONE
from one import registration
import one.alf.exceptions as alferr
from one.alf.io import next_num_folder
from . import TEST_DB_1, OFFLINE_ONLY
from one.tests import util


class TestDatasetTypes(unittest.TestCase):
    """Tests for dataset type validation."""
    def test_get_dataset_type(self):
        """Test one.registration.get_dataset_type function."""
        dtypes = [
            {'name': 'obj.attr', 'filename_pattern': ''},
            {'name': 'foo.bar', 'filename_pattern': '*FOO.b?r*'},
            {'name': 'bar.baz', 'filename_pattern': ''},
            {'name': 'some_file', 'filename_pattern': 'some_file.*'}
        ]
        dtypes = list(map(Bunch, dtypes))
        filename_typename = (
            ('foo.bar.npy', 'foo.bar'), ('foo.bir.npy', 'foo.bar'),
            ('_ns_obj.attr_clock.extra.npy', 'obj.attr'),
            ('bar.baz.ext', 'bar.baz'), ('some_file.ext', 'some_file'))
        for filename, dataname in filename_typename:
            with self.subTest(filename=filename):
                dtype = registration.get_dataset_type(filename, dtypes)
                self.assertEqual(dtype.name, dataname)

        self.assertRaises(ValueError, registration.get_dataset_type, 'no_match', dtypes)
        # Add an ambiguous dataset type pattern
        dtypes.append(Bunch({'name': 'foo.bor', 'filename_pattern': 'foo.bor*'}))
        self.assertRaises(ValueError, registration.get_dataset_type, 'foo.bor', dtypes)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestRegistrationClient(unittest.TestCase):
    """Test class for RegistrationClient class."""
    one = None
    subject = None
    temp_dir = None
    tag = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = util.set_up_env()
        cls.one = ONE(**TEST_DB_1, cache_dir=cls.temp_dir.name)
        cls.subject = ''.join(random.choices(string.ascii_letters, k=10))
        cls.one.alyx.rest('subjects', 'create', data={'lab': 'mainenlab', 'nickname': cls.subject})
        # Create a tag if doesn't already exist
        tag = ''.join(random.choices(string.ascii_letters, k=5))
        cls.tag = cls.one.alyx.rest('tags', 'create', data={'name': tag, 'protected': True})
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
        """Test for RegistrationClient.register_water_administration"""
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
        with unittest.mock.patch.object(self.client.one, 'to_eid', return_value=None), \
             self.assertRaises(ValueError):
            self.client.register_water_administration(self.subject, 3.6, session=ses['url'])

    def test_register_weight(self):
        """Test for RegistrationClient.register_weight"""
        record = self.client.register_weight(self.subject, 35.10000000235)
        self.assertEqual(record['subject'], self.subject)
        self.assertEqual(record['weight'], 35.1)
        self.assertEqual(record['user'], self.one.alyx.user)
        # Check validations
        with self.assertRaises(ValueError):
            self.client.register_weight(self.subject, 0.0)

    def test_ensure_ISO8601(self):
        """Test for RegistrationClient.ensure_ISO8601"""
        date = datetime.datetime(2021, 7, 14, 15, 53, 15, 525119)
        self.assertEqual(self.client.ensure_ISO8601(date), '2021-07-14T15:53:15.525119')
        self.assertEqual(self.client.ensure_ISO8601(date.date()), '2021-07-14T00:00:00')
        date_str = '2021-07-14T15:53:15.525119'
        self.assertEqual(self.client.ensure_ISO8601(date_str), date_str)
        with self.assertRaises(ValueError):
            self.client.ensure_ISO8601(f'{date:%D}')

    def test_exists(self):
        """Test for RegistrationClient.assert_exists"""
        # Check user endpoint
        with self.assertRaises(alferr.AlyxSubjectNotFound):
            self.client.assert_exists('foobar', 'subjects')
        self.client.assert_exists(self.subject, 'subjects')
        # Check user endpoint with list
        with self.assertRaises(alferr.ALFError) as ex:
            self.client.assert_exists([self.one.alyx.user, 'foobar'], 'users')
        self.assertIn('foobar', str(ex.exception))
        # Check raises non-404
        err = HTTPError()
        err.response = Bunch({'status_code': 500})
        with unittest.mock.patch.object(self.one.alyx, 'get', side_effect=err), \
                self.assertRaises(HTTPError):
            self.client.assert_exists('foobar', 'subjects')

    def test_find_files(self):
        """Test for RegistrationClient.find_files"""
        # Remove a dataset type from the client to check that the dataset(s) are ignored
        existing = (x['filename_pattern'] and any(self.session_path.rglob(x['filename_pattern']))
                    for x in self.client.dtypes)
        removed = self.client.dtypes.pop(next(i for i, x in enumerate(existing) if x))
        files = list(self.client.find_files(self.session_path))
        self.assertEqual(6, len(files))
        self.assertTrue(all(map(Path.is_file, files)))
        # Check removed file pattern not in file list
        self.assertFalse(fnmatch.filter([x.name for x in files], removed['filename_pattern']))

    def test_create_new_session(self):
        """Test for RegistrationClient.create_new_session"""
        # Check register = True
        session_path, eid = self.client.create_new_session(
            self.subject, date='2020-01-01', projects='ibl_neuropixel_brainwide_01'
        )
        expected = self.one.alyx.cache_dir.joinpath(self.subject, '2020-01-01', '001').as_posix()
        self.assertEqual(session_path.as_posix(), expected)
        self.assertIsNotNone(eid)
        # Check projects added to Alyx session
        projects = self.one.get_details(eid, full=True)['projects']
        self.assertEqual(projects, ['ibl_neuropixel_brainwide_01'])
        # Check register = False
        session_path, eid = self.client.create_new_session(
            self.subject, date='2020-01-01', register=False)
        expected = self.one.alyx.cache_dir.joinpath(self.subject, '2020-01-01', '002').as_posix()
        self.assertEqual(session_path.as_posix(), expected)
        self.assertIsNone(eid)

    def test_register_session(self):
        """Test for RegistrationClient.register_session"""
        # Find some datasets to create
        datasets = self.one.list_datasets(self.one.search(dataset='raw')[0])
        session_path = self.one.alyx.cache_dir.joinpath(
            'mainenlab', 'Subjects', self.subject, '2020-01-01', '001'
        )
        # Ensure session exists
        file_list = [session_path.joinpath(x) for x in datasets]
        # Create the files before registering
        for x in file_list:
            x.parent.mkdir(exist_ok=True, parents=True)
            x.touch()

        ses, recs = self.client.register_session(str(session_path),
                                                 end_time='2020-01-02',
                                                 procedures='Behavior training/tasks')
        self.assertTrue(len(ses['data_dataset_session_related']))
        self.assertEqual(len(ses['data_dataset_session_related']), len(recs))
        self.assertTrue(isinstance(ses['procedures'], list) and len(ses['procedures']) == 1)
        self.assertEqual(ses['end_time'], '2020-01-02T00:00:00')

        # Check value error raised when provided lab name doesn't match
        with self.assertRaises(ValueError):
            self.client.register_session(str(session_path), lab='cortexlab')

        # Check updating existing session
        start_time = '2020-01-01T10:36:10'
        ses, _ = self.client.register_session(str(session_path), start_time=start_time)
        self.assertEqual(start_time, ses['start_time'])

    def test_create_sessions(self):
        """Test for RegistrationClient.create_sessions"""
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
        """Test for RegistrationClient.register_files"""
        # Test a few things not checked in register_session
        session_path, eid = self.client.create_new_session(self.subject)

        # Check registering single file, dry run, default False
        file_name = session_path.joinpath('wheel.position.npy')
        file_name.touch()
        labs = 'mainenlab,cortexlab'
        rec = self.client.register_files(str(file_name), default=False, dry=True, labs=labs)
        self.assertIsInstance(rec, dict)
        self.assertFalse(rec['default'])
        self.assertNotIn('id', rec)
        self.assertEqual(labs, rec['labs'], 'failed to include optional kwargs in request')

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
            self.assertRegex(dbg.records[1].message, 'No dataset type .* "foo.bar.npy"')
            self.assertRegex(dbg.records[2].message, f'Multiple matching .* "{ambiguous}"')
        self.assertFalse(len(rec))

        # Check the handling of revisions
        rec, = self.client.register_files(str(file_name))
        # Add a protected tag to all the datasets
        tag = self.tag['name']
        self.one.alyx.rest('datasets', 'partial_update', id=rec['id'], data={'tags': [tag]})

        # Test registering with a revision already in the file path,
        # should use this rather than create one with today's date
        rev = self.one.alyx.rest('revisions', 'create', data={'name': f'{tag}1'})
        self.addCleanup(self.one.alyx.rest, 'revisions', 'delete', id=rev['name'])
        file = file_name.parent.joinpath(f'#{rev["name"]}#', file_name.name)
        file.parent.mkdir(), file.touch()  # Create file in new revision folder
        r, = self.client.register_files(file_list=[file])
        self.assertEqual(r['revision'], rev['name'])
        self.assertTrue(r['default'])
        self.assertIsNone(r['collection'])

        # Register exact dataset revision again - it should append an 'a'
        # When we re-register the original it should move them into revision with today's date
        self.one.alyx.rest('datasets', 'partial_update', id=r['id'], data={'tags': [tag]})
        files = [file,
                 session_path.joinpath('wheel.position.npy'),
                 session_path.joinpath('wheel.position.ssv')]  # Last dataset unprotected
        files[-1].touch()
        r1, r2, r3 = self.client.register_files(file_list=files)
        self.assertEqual(r1['revision'], rev['name'] + 'a')
        self.assertTrue(file.parents[1].joinpath(f'#{r1["revision"]}#', file.name).exists())
        self.assertFalse(file.exists(), 'failed to move protected dataset to new revision')

        self.assertEqual(r2['revision'], str(datetime.date.today()))
        self.assertFalse(files[1].exists(), 'failed to move protected dataset to new revision')
        file = files[1].parent.joinpath(f'#{r2["revision"]}#', file.name)
        self.assertTrue(file.exists())

        self.assertIsNone(r3['revision'])
        self.assertTrue(files[2].exists())

        # Protect the latest datasets
        self.one.alyx.rest('datasets', 'partial_update', id=r2['id'], data={'tags': [tag]})
        # Same day revision
        today = datetime.date.today()
        r, = self.client.register_files(file_list=[file])
        self.assertEqual(r['revision'], f'{today}a')
        self.assertTrue(file.parents[1].joinpath(f'#{r["revision"]}#', file.name).exists())
        self.assertFalse(file.exists(), 'failed to move protected dataset to new revision')

        # Re-register a protected date; given original and revision a are protected
        # we expect to create revision b
        file = session_path.joinpath(f'#{today}#', file_name.name)
        file.touch()
        self.one.alyx.rest('datasets', 'partial_update', id=r['id'], data={'tags': [tag]})
        r, = self.client.register_files(file_list=[file])
        self.assertEqual(r['revision'], str(datetime.date.today()) + 'b')

        # For this logic it's simpler to spoof the response from Alyx
        with unittest.mock.patch.object(self.one.alyx, 'post') as alyx_mock:
            mock_resp = Response()
            mock_resp.status_code = 403
            response = {'status_code': 403, 'error': 'One or more datasets is protected',
                        'details': [
                            # Dataset protected but latest (blank) revision is unprotected
                            {'wheel.position.ssv': [
                                {'': False},
                                {'puHIt1': True}]},
                            # Dataset protected but latest (date) revision is unprotected
                            {'wheel.position.npy': [
                                {f'{today}': False},
                                {'puHIt1': True}]},
                            # Dataset protected and latest (date) revision is today
                            {'wheel.timestamps.npy': [
                                {str(today): True},
                                {'puHIt1': True}]},
                            # Dataset protected and latest (date) revision was updated twice
                            {'wheel.timestamps.ssv': [
                                {f'{today}a': True},
                                {str(today): True}]},
                            # Dataset protected but latest (misc) revision not protected
                            {'wheel.velocity.ssv': [
                                {'puHIt1': False},
                                {'foo': True}]},
                            # Dataset protected and latest (date) revision end with a different
                            # character
                            {'wheel.velocity.npy': [
                                {f'{today}A': True},
                                {str(today): True}]},
                        ]}
            mock_resp._content = bytes(json.dumps(response), 'utf-8')
            alyx_mock.side_effect = [HTTPError(response=mock_resp), [{}]]
            files = [session_path.joinpath(next(iter(x.keys()))) for x in response['details']]
            [x.touch() for x in files]
            self.client.register_files(file_list=files)
            self.assertEqual(2, alyx_mock.call_count)
            actual = alyx_mock.call_args.kwargs.get('data', {}).get('filenames', [])
            expected = [
                'wheel.position.ssv',
                f'#{today}#/wheel.position.npy',
                f'#{today}a#/wheel.timestamps.npy',
                f'#{today}b#/wheel.timestamps.ssv',
                f'#{today}#/wheel.velocity.ssv',
                f'#{today}B#/wheel.velocity.npy']
            self.assertEqual(expected, actual)

            # Finally, simply check that the exception is re-raised for non-revision related errors
            # NB Unfortunately this passes even if the exception isn't re-raised after being caught
            response['status_code'] = 404
            mock_resp._content = bytes(json.dumps(response), 'utf-8')
            alyx_mock.side_effect = HTTPError(response=mock_resp)
            self.assertRaises(HTTPError, self.client.register_files, file_list=[file])

    def test_next_revision(self):
        """Test RegistrationClient._next_revision method"""
        self.assertEqual('2020-01-01a', self.client._next_revision('2020-01-01'))
        reserved = ['2020-01-01a', '2020-01-01b']
        self.assertEqual('2020-01-01c', self.client._next_revision('2020-01-01', reserved))
        new_revision = self.client._next_revision('2020-01-01', alpha=chr(945))
        self.assertEqual(945, ord(new_revision[-1]))
        self.assertRaises(TypeError, self.client._next_revision, '2020-01-01', alpha='do')

    def test_instantiation(self):
        """Test RegistrationClient.__init__ with no args"""
        with unittest.mock.patch('one.registration.ONE') as mk:
            client = registration.RegistrationClient()
        self.assertIsInstance(client.one, unittest.mock.MagicMock)
        mk.assert_called_with(cache_rest=None)

    @classmethod
    def tearDownClass(cls) -> None:
        for admin in cls.one.alyx.rest('water-administrations', 'list',
                                       django='subject__nickname,' + cls.subject, no_cache=True):
            cls.one.alyx.rest('water-administrations', 'delete', id=admin['url'][-36:])
        # Note: datasets deleted in cascade
        for ses in cls.one.alyx.rest('sessions', 'list', subject=cls.subject, no_cache=True):
            cls.one.alyx.rest('sessions', 'delete', id=ses['url'][-36:])
        cls.one.alyx.rest('subjects', 'delete', id=cls.subject)
        cls.one.alyx.rest('tags', 'delete', id=cls.tag['id'])
        cls.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main(exit=False)
