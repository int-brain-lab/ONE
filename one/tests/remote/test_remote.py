"""Unit tests for the one.remote package."""
import logging
import tempfile
import unittest
from unittest import mock
from pathlib import Path, PurePosixPath
import shutil
from functools import partial
from tempfile import TemporaryDirectory
import io
import json
import sys
import uuid
import random

try:
    import globus_sdk
except ModuleNotFoundError:
    raise unittest.skip('globus_sdk module not installed')
from iblutil.io import params as iopar

from one.tests.util import get_file, setup_rest_cache
from one.tests import TEST_DB_1
from one.webclient import AlyxClient
from one.remote import base, globus


class TestBase(unittest.TestCase):
    """Tests for the one.remote.base module"""

    """unittest.mock._patch: Mock object for setting parameter location as temporary directory"""
    path_mock = None
    """tempfile.TemporaryDirectory: The temporary location of remote parameters file"""
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = TemporaryDirectory()
        cls.path_mock = mock.patch('one.remote.base.iopar.getfile',
                                   new=partial(get_file, cls.tempdir.name))

    def setUp(self) -> None:
        self.path_mock.start()

    def test_load_client_params(self):
        """Tests for one.remote.base.load_client_params function"""
        # Check behaviour when no parameters file exists
        with self.assertRaises(FileNotFoundError):
            base.load_client_params(assert_present=True)
        self.assertIsNone(base.load_client_params(assert_present=False))

        # Check behaviour with parameters file
        iopar.write(base.PAR_ID_STR, {'foo': {'bar': 'baz'}})
        p = base.load_client_params(assert_present=True)
        self.assertIn('foo', p.as_dict())

        # Check behaviour when existing key provided
        p = base.load_client_params('foo', assert_present=True)
        self.assertEqual(p.bar, 'baz')

        with self.assertRaises(AttributeError):
            base.load_client_params('bar', assert_present=True)
        self.assertIsNone(base.load_client_params('bar', assert_present=False))

        # Loading a sub-key
        iopar.write(base.PAR_ID_STR, {'globus': {'default': {'par1': 'par2'}}})
        p = base.load_client_params('globus.default')
        self.assertIn('par1', p.as_dict())

    def test_save_client_params(self):
        """Tests for one.remote.base.save_client_params function"""
        # Check behaviour when saving all params
        expected = {'foo': {'bar': 'baz'}}
        base.save_client_params(expected)
        par_path = next(Path(self.tempdir.name).rglob('*remote'), None)
        self.assertIsNotNone(par_path)
        with open(par_path, 'r') as f:
            p = json.load(f)
        self.assertEqual(p, expected)

        # Check validation
        with self.assertRaises(ValueError):
            base.save_client_params({'foo': 'bar'})

        # Check behaviour when saving into client key
        base.save_client_params({'new': {'par1': 1}})
        with open(par_path, 'r') as f:
            p = json.load(f)
        self.assertIn('new', p)
        self.assertEqual(1, p['new']['par1'])

    def test_repo_from_alyx(self):
        """Test for DownloadClient.repo_from_alyx method"""
        ac = AlyxClient(**TEST_DB_1)
        setup_rest_cache(ac.cache_dir)  # Copy REST cache fixtures to temp dir
        record = base.DownloadClient.repo_from_alyx('mainenlab', ac)
        self.assertEqual('mainenlab', record['name'])

    def tearDown(self) -> None:
        par_path = Path(iopar.getfile('.one'))
        assert str(par_path).startswith(self.tempdir.name)
        shutil.rmtree(par_path)
        self.path_mock.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


class TestGlobus(unittest.TestCase):
    """Tests for the one.remote.globus module"""

    """unittest.mock._patch: Mock object for setting parameter location as temporary directory"""
    path_mock = None
    """tempfile.TemporaryDirectory: The temporary location of remote parameters file"""
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = TemporaryDirectory()
        cls.path_mock = mock.patch('one.remote.base.iopar.getfile',
                                   new=partial(get_file, cls.tempdir.name))

    def setUp(self) -> None:
        self.path_mock.start()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setup(self, _):
        """Tests for one.remote.globus.setup function"""
        # Check behaviour when no parameters file exists, local endpoint ID found
        ans = ('', '123', '', 'new_path/to/thing', 'c')
        with mock.patch('builtins.input', side_effect=ans),\
                mock.patch('one.remote.globus.get_local_endpoint_id', return_value='456'):
            globus.setup()

        p = globus.load_client_params('globus.default').as_dict()
        expected = {
            'GLOBUS_CLIENT_ID': '123',
            'local_endpoint': '456',
            'local_path': 'new_path/to/thing'
        }
        self.assertDictEqual(p, expected)

        # Set up again with globus login
        ans = ('', '', '', '', 'abc')
        d = dict(refresh_token=1, access_token=2, expires_at_seconds=3)
        with mock.patch('builtins.input', side_effect=ans),\
                mock.patch('one.remote.globus.globus_sdk.NativeAppAuthClient') as client,\
                mock.patch('one.remote.globus.get_local_endpoint_id'):
            (
                client().
                oauth2_exchange_code_for_tokens().
                by_resource_server.
                __getitem__
            ).return_value = d
            globus.setup()

        p = globus.load_client_params('globus.default').as_dict()
        expected.update(d)
        self.assertDictEqual(p, expected)

        # Check for input validations
        # 1. New profile and no Globus ID inputted
        with mock.patch('builtins.input', side_effect=['']),\
                self.assertRaises(ValueError) as ex:
            globus.setup(par_id='foo')
            self.assertIn('Globus client ID', str(ex))

        # 2. New profile, no local endpoint ID found and none inputted
        with mock.patch('builtins.input', side_effect=['foo', '123', '']),\
             mock.patch('one.remote.globus.get_local_endpoint_id', side_effect=AssertionError),\
             self.assertRaises(ValueError) as ex, self.assertWarns(Warning):
            globus.setup()
            self.assertIn('local endpoint ID', str(ex))

    def test_as_globus_path(self):
        """Tests for one.remote.globus.as_globus_path"""
        # A Windows path
        # "/E/FlatIron/integration"
        # Only test this on windows
        if sys.platform == 'win32':
            actual = globus.as_globus_path('E:\\FlatIron\\integration')
            self.assertTrue(actual.startswith('/E/'))

        # A globus path
        actual = globus.as_globus_path('/E/FlatIron/integration')
        expected = '/E/FlatIron/integration'
        self.assertEqual(expected, actual)

    def test_get_local_endpoint_id(self):
        """Test for one.remote.globus.get_local_endpoint_id function"""
        def _check_path(x):
            self.assertTrue(str(x).endswith('client-id.txt'))
            return True

        # Function should check for path existence
        with self.assertRaises(AssertionError),\
                mock.patch.object(Path, 'exists', return_value=False):
            globus.get_local_endpoint_id()

        # Function should look for 'client-id.txt' file and return contents
        with mock.patch.object(Path, 'exists', _check_path),\
             mock.patch.object(Path, 'read_text', return_value=' 123 '):
            x = globus.get_local_endpoint_id()
        self.assertEqual('123', x)

    def test_get_local_endpoint_paths(self):
        """Tests for one.remote.globus.get_local_endpoint_paths function"""
        with mock.patch('one.remote.globus.sys.platform', 'win32'):
            self.assertEqual([], globus.get_local_endpoint_paths())

        with mock.patch('one.remote.globus.sys.platform', 'linux'),\
             mock.patch.object(Path, 'exists', return_value=False), self.assertWarns(Warning):
            self.assertEqual([], globus.get_local_endpoint_paths())

        expected = [Path('path', 'one'), Path('path', 'two')]
        with mock.patch('one.remote.globus.sys.platform', 'linux'),\
             mock.patch.object(Path, 'exists', return_value=True),\
             mock.patch.object(Path, 'read_text', return_value='path/one,path/two '):
            self.assertCountEqual(expected, globus.get_local_endpoint_paths())

    def test_get_lab_from_endpoint_id(self):
        """Tests for one.remote.globus.get_lab_from_endpoint_id function"""
        # Set up REST cache fixtures
        ac = AlyxClient(**TEST_DB_1)
        setup_rest_cache(ac.cache_dir)
        endpoint_id = '2dc8ccc6-2f8e-11e9-9351-0e3d676669f4'
        name = globus.get_lab_from_endpoint_id(endpoint_id, ac)[0]
        self.assertEqual(name, 'mainenlab')

        # Check behaviour when unknown UUID
        with mock.patch.object(ac, 'rest', return_value=[]):
            self.assertIsNone(globus.get_lab_from_endpoint_id('123', ac))

        # Check behaviour when multiple labs returned
        with mock.patch.object(ac, 'rest', return_value=[{'name': 'lab_A'}, {'name': 'lab_B'}]):
            self.assertEqual(len(globus.get_lab_from_endpoint_id('123', ac)), 2)

        # Check behaviour when no input ID returned
        with mock.patch('one.remote.globus.get_local_endpoint_id', return_value=endpoint_id):
            name = globus.get_lab_from_endpoint_id(alyx=ac)[0]
            self.assertEqual(name, 'mainenlab')

    @mock.patch('one.remote.globus.globus_sdk')
    def test_create_globus_client(self, globus_mock):
        """Tests for one.remote.globus.create_globus_client function"""
        # Check setup run when no params exist, check raises exception when missing params
        incomplete_pars = iopar.from_dict({'GLOBUS_CLIENT_ID': 123})
        with mock.patch('one.remote.globus.setup') as setup_mock,\
             self.assertRaises(ValueError),\
             mock.patch('one.remote.base.load_client_params',
                        side_effect=[AssertionError, incomplete_pars]):
            globus.create_globus_client()
            setup_mock.assert_called()

        # Check behaviour with complete params
        pars = iopar.from_dict({'GLOBUS_CLIENT_ID': 123, 'refresh_token': 456})
        with mock.patch('one.remote.globus.load_client_params', return_value=pars) as par_mock:
            client = globus.create_globus_client('admin')
            par_mock.assert_called_once_with('globus.admin')
        globus_mock.NativeAppAuthClient.assert_called_once_with(123)
        globus_mock.RefreshTokenAuthorizer.assert_called()
        self.assertEqual(client, globus_mock.TransferClient())

    def tearDown(self) -> None:
        par_path = Path(iopar.getfile('.one'))
        assert str(par_path).startswith(self.tempdir.name)
        if par_path.exists():
            shutil.rmtree(par_path)
        self.path_mock.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


class TestGlobusClient(unittest.TestCase):
    """Tests for the GlobusClient class"""

    """unittest.mock._patch: Mock object for globus_sdk package"""
    globus_sdk_mock = None

    @mock.patch('one.remote.globus.setup')
    def setUp(self, _) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        pars = iopar.from_dict({
            'GLOBUS_CLIENT_ID': '123',
            'refresh_token': '456',
            'local_endpoint': '987',
            'local_path': self.tempdir.name,
            'access_token': 'abc',
            'expires_at_seconds': 1636109267
        })
        self.globus_sdk_mock = mock.patch('one.remote.globus.globus_sdk')
        self.globus_sdk_mock.start()
        self.addCleanup(self.globus_sdk_mock.stop)
        with mock.patch('one.remote.globus.load_client_params', return_value=pars):
            self.client = globus.Globus()

    def test_constructor(self):
        """Test for Globus.__init__ method"""
        # self.assertEqual(self.client.client, self.globus_sdk_mock.TransferClient())
        expected = {'local': {'id': '987', 'root_path': self.tempdir.name}}
        self.assertDictEqual(self.client.endpoints, expected)

    def test_add_endpoint(self):
        """Test for Globus.add_endpoint method"""
        # Test with UUID
        # 1. Should raise exception when label not defined
        endpoint_id = '2dc8ccc6-2f8e-11e9-9351-0e3d676669f4'
        with self.assertRaises(ValueError):
            self.client.add_endpoint(endpoint_id)
        # 2. Should add UUID to endpoints along with root path
        name = 'lab1'
        self.client.add_endpoint(endpoint_id, label=name, root_path='/mnt')
        self.assertIn(name, self.client.endpoints)
        expected = {'id': endpoint_id, 'root_path': '/mnt'}
        self.assertDictEqual(self.client.endpoints[name], expected)

        # Test with Alyx repo name
        # Set up REST cache fixtures
        ac = AlyxClient(**TEST_DB_1)
        setup_rest_cache(ac.cache_dir)
        name = 'mainenlab'
        self.client.add_endpoint(name, alyx=ac)
        self.assertIn(name, self.client.endpoints)
        expected = {
            'id': '0b6f5a7c-a7a9-11e8-96fa-0a6d4e044368',
            'root_path': '/mnt/globus/mainenlab/Subjects'
        }
        self.assertDictEqual(self.client.endpoints[name], expected)

        # Test behaviour when label exists
        with self.assertLogs(logging.getLogger('one.remote.globus'), logging.ERROR):
            self.client.add_endpoint(name, root_path='/', alyx=ac)
            self.assertNotEqual(self.client.endpoints[name]['root_path'], '/')
        self.client.add_endpoint(name, root_path='/', overwrite=True, alyx=ac)
        self.assertEqual(self.client.endpoints[name]['root_path'], '/')

    def test_endpoint_path(self):
        """Test for Globus._endpoint_path method"""
        expected = PurePosixPath('/mnt/foo/bar')
        self.assertEqual(str(expected), self.client._endpoint_path(expected))
        expected = '/foo/bar/baz'
        self.assertEqual(expected, self.client._endpoint_path('bar/baz', root_path='/foo'))
        with self.assertRaises(ValueError):
            self.client._endpoint_path('bar', root_path='foo')

    def test_endpoint_id_root(self):
        """Test for Globus._endpoint_id_root method"""
        id, path = self.client._endpoint_id_root('local')
        self.assertEqual('987', id)
        self.assertEqual(path, self.tempdir.name)

        expected = uuid.uuid4()
        id, path = self.client._endpoint_id_root(expected)
        self.assertEqual(expected, id)
        self.assertIsNone(path)

        with self.assertRaises(ValueError):
            self.client._endpoint_id_root('remote')

    def test_ls(self):
        """Test for Globus.ls method"""
        response = dict(
            name=Path(self.tempdir.name, f'some.{uuid.uuid4()}.file').as_posix(),
            type='file', size=1024)
        err = globus_sdk.GlobusConnectionError('', ConnectionError)
        self.client.client.operation_ls.side_effect = (err, [response])
        path = globus.as_globus_path(self.tempdir.name)
        out = self.client.ls('local', path)
        self.assertEqual(2, self.client.client.operation_ls.call_count)
        self.assertIsInstance(out[0], PurePosixPath)
        self.assertEqual(response['name'], out[0].as_posix())

        # Remove uuid and return size args
        self.client.client.operation_ls.side_effect = ([response],)
        out, = self.client.ls('local', path, remove_uuid=True, return_size=True)
        self.assertEqual(response['size'], out[1])
        self.assertNotIn(response['name'].split('.')[-2], str(out[0]))

        self.client.client.operation_ls.reset_mock()
        self.client.client.operation_ls.side_effect = err
        with self.assertRaises(globus_sdk.GlobusConnectionError):
            self.client.ls('local', path, max_retries=2)
        self.assertEqual(3, self.client.client.operation_ls.call_count)

    def test_mv(self):
        """Test for Globus.mv and Globus.run_task methods"""
        source = ('some.file', 'some2.file')
        destination = ('new.file', 'new2.file')
        # Ensure root path is Globus compliant (required on Windows)
        self.client.endpoints['local']['root_path'] = globus.as_globus_path(self.tempdir.name)
        # Mock transfer output
        task_id = random.randint(0, int(1e6))
        self.client.client.submit_transfer.return_value = {'task_id': task_id}
        self.client.client.get_task.return_value = {'status': 'SUCCEEDED'}
        self.client.client.task_successful_transfers.return_value = \
            [dict(source_path=src, destination_path=dst) for src, dst in zip(source, destination)]
        self.client.client.task_skipped_errors.return_value = []
        task_response = self.client.mv('local', 'local', source, destination)
        self.assertEqual(task_id, task_response)

        # Test errors
        # Check timeout behaviour
        self.client.client.task_wait.reset_mock()
        self.client.client.task_wait.return_value = False
        timeout = 10
        with self.assertRaises(IOError) as ex:
            self.client.mv('local', 'local', source, destination, timeout=timeout)
            self.assertIn(str(task_id), str(ex))
        self.assertEqual(timeout, self.client.client.task_wait.call_count)

        # Check status check error behaviour
        self.client.client.task_wait.return_value = True
        self.client.client.task_successful_transfers.side_effect = \
            globus_sdk.TransferAPIError(mock.MagicMock())
        with self.assertLogs(logging.getLogger('one.remote.globus'), logging.WARNING):
            self.client.mv('local', 'local', source, destination)

        # Check failed transfer
        self.client.client.get_task.return_value = {'status': 'FAILED'}
        self.client.client.task_successful_transfers.reset_mock()
        with self.assertRaises(IOError) as ex:
            self.client.mv('local', 'local', source, destination)
            self.assertIn(self.client.client.get_task.return_value['status'], str(ex))

        # Check submission error behaviour
        self.client.client.submit_transfer.side_effect = \
            globus_sdk.GlobusConnectionError('', ConnectionError)
        default_n_retries = 3
        with self.assertLogs(logging.getLogger('one.remote.globus')) as log, \
                self.assertRaises(globus_sdk.GlobusConnectionError):
            self.client.mv('local', 'local', source, destination)
            warnings = filter(lambda x: x.levelno == 30, log.records)
            self.assertEqual(default_n_retries, len(list(warnings)))
            self.assertIn(log.records[-1].msg)
            self.assertRegex(log.records[-1].msg, 'Max retries')
            self.assertEqual('ERROR', log.records[-1].levelname)


if __name__ == '__main__':
    unittest.main()
