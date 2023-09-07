import logging
import tempfile
import unittest
from unittest import mock
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from functools import partial
from tempfile import TemporaryDirectory
from datetime import datetime
import io
import sys
import uuid

try:
    import globus_sdk
except ModuleNotFoundError:
    raise unittest.skip('globus_sdk module not installed')
from iblutil.io import params as iopar

from one.tests.util import get_file, setup_rest_cache
from one.tests import TEST_DB_1
from one.webclient import AlyxClient
from one.remote import globus

ENDPOINT_ID = uuid.uuid1()


class TestGlobus(unittest.TestCase):
    """Tests for the one.remote.globus module."""

    """unittest.mock._patch: Mock object for setting parameter location as temporary directory."""
    path_mock = None
    """tempfile.TemporaryDirectory: The temporary location of remote parameters file."""
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
        """Tests for one.remote.globus._setup function."""
        local_id = str(ENDPOINT_ID)
        gc_id = str(uuid.uuid4())
        # Check behaviour when no parameters file exists, local endpoint ID found
        ans = ('', gc_id, '', 'new_path/to/thing', 'c')
        with mock.patch('builtins.input', side_effect=ans), \
                mock.patch('one.remote.globus.get_local_endpoint_id', return_value=local_id):
            globus._setup(login=False)

        p = globus.load_client_params('globus.default').as_dict()
        expected = {
            'GLOBUS_CLIENT_ID': gc_id,
            'local_endpoint': local_id,
            'local_path': 'new_path/to/thing'
        }
        self.assertDictEqual(p, expected)

        # Set up again with globus login
        ans = ('', '', '', '', 'abc')
        d = dict(refresh_token=1, access_token=2, expires_at_seconds=3)
        with mock.patch('builtins.input', side_effect=ans), \
                mock.patch('one.remote.globus.globus_sdk.NativeAppAuthClient') as client, \
                mock.patch('one.remote.globus.get_local_endpoint_id'):
            (
                client().
                oauth2_exchange_code_for_tokens().
                by_resource_server.
                __getitem__
            ).return_value = d
            globus._setup()

        p = globus.load_client_params('globus.default').as_dict()
        expected.update(d)
        self.assertDictEqual(p, expected)

        # Check for input validations
        # 1. New profile and no Globus ID inputted
        with mock.patch('builtins.input', side_effect=['']), \
                self.assertRaises(ValueError) as ex:
            globus._setup(par_id='foo')
            self.assertIn('Globus client ID', str(ex))

        # 2. New profile, Globus ID invalid
        with mock.patch('builtins.input', side_effect=['bar', '123']), \
                self.assertRaises(ValueError) as ex:
            globus._setup(par_id='foo')
            self.assertIn('Invalid Globus client ID', str(ex))

        # 3. New profile, no local endpoint ID found and none inputted
        with mock.patch('builtins.input', side_effect=['foo', gc_id, '']), \
             mock.patch('one.remote.globus.get_local_endpoint_id', side_effect=AssertionError), \
             self.assertRaises(ValueError) as ex, self.assertWarns(Warning):
            globus._setup()
            self.assertIn('local endpoint ID', str(ex))

    def test_as_globus_path(self):
        """Tests for one.remote.globus.as_globus_path."""
        # A Windows path
        # "/E/FlatIron/integration"
        # Only test this on windows
        if sys.platform == 'win32':
            actual = globus.as_globus_path('/foo/bar')
            self.assertEqual(actual, f'/{Path.cwd().drive[0]}/foo/bar')

        # On all systems an explicit Windows path should be converted to a POSIX one
        actual = globus.as_globus_path(PureWindowsPath('E:\\FlatIron\\integration'))
        self.assertTrue(actual.startswith('/E/'))

        # On all systems an explicit POSIX path should be left unchanged
        actual = globus.as_globus_path(PurePosixPath('E:\\FlatIron\\integration'))
        self.assertEqual(actual, 'E:\\FlatIron\\integration')

        # A valid globus path should be unchanged
        path = '/mnt/ibl'
        actual = globus.as_globus_path(PurePosixPath(path))
        self.assertEqual(actual, path)
        path = '/E/FlatIron/integration'
        actual = globus.as_globus_path(path)
        self.assertEqual(actual, path)

    def test_get_local_endpoint_id(self):
        """Test for one.remote.globus.get_local_endpoint_id function."""
        def _check_path(x):
            self.assertTrue(str(x).endswith('client-id.txt'))
            return True

        # Function should check for path existence
        with self.assertRaises(AssertionError), \
                mock.patch.object(Path, 'exists', return_value=False):
            globus.get_local_endpoint_id()

        # Function should look for 'client-id.txt' file and return contents
        with mock.patch.object(Path, 'exists', _check_path), \
             mock.patch.object(Path, 'read_text', return_value=f' {ENDPOINT_ID} '):
            x = globus.get_local_endpoint_id()
        self.assertEqual(ENDPOINT_ID, x)

    def test_get_local_endpoint_paths(self):
        """Tests for one.remote.globus.get_local_endpoint_paths function."""
        with mock.patch('one.remote.globus.sys.platform', 'win32'):
            self.assertEqual([], globus.get_local_endpoint_paths())

        with mock.patch('one.remote.globus.sys.platform', 'linux'), \
             mock.patch.object(Path, 'exists', return_value=False), self.assertWarns(Warning):
            self.assertEqual([], globus.get_local_endpoint_paths())

        expected = [Path('path', 'one'), Path('path', 'two')]
        with mock.patch('one.remote.globus.sys.platform', 'linux'), \
             mock.patch.object(Path, 'exists', return_value=True), \
             mock.patch.object(Path, 'read_text', return_value='path/one,path/two '):
            self.assertCountEqual(expected, globus.get_local_endpoint_paths())

    def test_get_lab_from_endpoint_id(self):
        """Tests for one.remote.globus.get_lab_from_endpoint_id function."""
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
        """Tests for one.remote.globus.create_globus_client function."""
        # Check setup run when no params exist, check raises exception when missing params
        gc_id = str(uuid.uuid4())
        incomplete_pars = iopar.from_dict({'GLOBUS_CLIENT_ID': gc_id})
        with mock.patch('one.remote.globus._setup') as setup_mock, \
             self.assertRaises(ValueError), \
             mock.patch('one.remote.base.load_client_params',
                        side_effect=[AssertionError, incomplete_pars]):
            globus.create_globus_client()
            setup_mock.assert_called()

        # Check behaviour with complete params
        pars = iopar.from_dict({'GLOBUS_CLIENT_ID': gc_id, 'refresh_token': 456})
        with mock.patch('one.remote.globus.load_client_params', return_value=pars) as par_mock:
            client = globus.create_globus_client('admin')
            par_mock.assert_called_once_with('globus.admin')
        globus_mock.NativeAppAuthClient.assert_called_once_with(gc_id)
        globus_mock.RefreshTokenAuthorizer.assert_called()
        self.assertEqual(client, globus_mock.TransferClient())

        # Check without refresh tokens
        pars = pars.set('refresh_token', None).set('access_token', 456)
        globus_mock.RefreshTokenAuthorizer.reset_mock()
        with mock.patch('one.remote.globus.load_client_params', return_value=pars) as par_mock:
            client = globus.create_globus_client('admin')
            par_mock.assert_called_once_with('globus.admin')
        globus_mock.AccessTokenAuthorizer.assert_called_once_with(456)
        globus_mock.RefreshTokenAuthorizer.assert_not_called()
        self.assertEqual(client, globus_mock.TransferClient())

    def test_remove_token_fields(self):
        """Test for one.remote.globus._remove_token_fields function."""
        par = iopar.from_dict({
            'local_path': 'foo', 'GLOBUS_CLIENT_ID': ENDPOINT_ID, 'refresh_token': None,
            'access_token': str(uuid.uuid4()), 'expires_at_seconds': 12345678})
        newpar = globus._remove_token_fields(par)
        self.assertTrue(hasattr(newpar, '_fields'))
        self.assertEqual(newpar._fields, ('local_path', 'GLOBUS_CLIENT_ID'))
        # Check works with a dict
        newpar = globus._remove_token_fields(par.as_dict())
        self.assertTrue(hasattr(newpar, '_fields'))
        self.assertEqual(newpar._fields, ('local_path', 'GLOBUS_CLIENT_ID'))
        self.assertIsNone(globus._remove_token_fields(None))

    def test_get_token(self):
        """Test for one.remote.globus.get_token function."""
        auth_code = 'a1b2c3d4e5f6g7h8'
        # Test without refresh tokens
        with mock.patch('builtins.input', return_value=auth_code), \
                mock.patch('one.remote.globus.globus_sdk.NativeAppAuthClient') as client:
            token = globus.get_token(str(ENDPOINT_ID), refresh_tokens=False)
            client().oauth2_start_flow.assert_called_with(refresh_tokens=False)
            client().oauth2_exchange_code_for_tokens.assert_called_with('a1b2c3d4e5f6g7h8')
            self.assertIsInstance(token, dict)
            expected = ('refresh_token', 'access_token', 'expires_at_seconds')
            self.assertCountEqual(expected, token.keys())

        # Test with refresh tokens
        with mock.patch('builtins.input', return_value=auth_code), \
                mock.patch('one.remote.globus.globus_sdk.NativeAppAuthClient') as client:
            token = globus.get_token(str(ENDPOINT_ID), refresh_tokens=True)
            client().oauth2_start_flow.assert_called_with(refresh_tokens=True)

        # Test cancel
        with mock.patch('builtins.input', return_value='c '), \
                mock.patch('one.remote.globus.globus_sdk.NativeAppAuthClient') as client:
            token = globus.get_token(str(ENDPOINT_ID), refresh_tokens=True)
            client().oauth2_exchange_code_for_tokens.assert_not_called()
            self.assertCountEqual(token.keys(), expected)
            self.assertFalse(any(token.values()))

    def tearDown(self) -> None:
        par_path = Path(iopar.getfile('.one'))
        assert str(par_path).startswith(self.tempdir.name)
        if par_path.exists():
            shutil.rmtree(par_path)
        self.path_mock.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


class _GlobusClientTest(unittest.TestCase):
    """Globus Client test setup routines."""

    """unittest.mock._patch: Mock object for globus_sdk package."""
    globus_sdk_mock = None

    @mock.patch('one.remote.globus._setup')
    def setUp(self, _) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        # The github CI root dir contains an alias/symlink so we must resolve it
        self.root_path = Path(self.tempdir.name).resolve()
        self.addCleanup(self.tempdir.cleanup)
        self. pars = iopar.from_dict({
            'GLOBUS_CLIENT_ID': '123',
            'refresh_token': '456',
            'local_endpoint': str(ENDPOINT_ID),
            'local_path': str(self.root_path),
            'access_token': 'abc',
            'expires_at_seconds': datetime.now().timestamp() + 60**2
        })
        self.globus_sdk_mock = mock.patch('one.remote.globus.globus_sdk')
        self.globus_sdk_mock.start()
        self.addCleanup(self.globus_sdk_mock.stop)
        with mock.patch('one.remote.globus.load_client_params', return_value=self.pars):
            self.globus = globus.Globus()


class TestGlobusClient(_GlobusClientTest):
    """Tests for the GlobusClient class."""

    def test_constructor(self):
        """Test for Globus.__init__ method."""
        # self.assertEqual(self.client.client, self.globus_sdk_mock.TransferClient())
        expected = {'local': {'id': ENDPOINT_ID, 'root_path': str(self.root_path)}}
        self.assertDictEqual(self.globus.endpoints, expected)

    def test_setup(self):
        """Test for Globus.setup static method.

        TestGlobus.test_setup tests the setup function. Here we just check it's called.
        """
        with mock.patch('one.remote.globus._setup') as setup_mock, \
                mock.patch('one.remote.globus.create_globus_client'), \
                mock.patch('one.remote.globus.load_client_params', return_value=self.pars):
            self.assertIsInstance(globus.Globus.setup(), globus.Globus)
            setup_mock.assert_called_once()

    def test_to_address(self):
        """Test for Globus.to_address method."""
        # Check with Windows path
        root_path = PureWindowsPath(r'C:\root\path')
        uid = uuid.uuid1()
        # Check works with endpoint label
        self.globus.add_endpoint(uid, label='data-repo', root_path=root_path)
        addr = self.globus.to_address('path/to/file', 'data-repo')
        self.assertEqual(addr, '/C/root/path/path/to/file')
        # Check works with UUID
        addr = self.globus.to_address('foo/bar', str(uid))
        self.assertEqual(addr, '/C/root/path/foo/bar')

    def test_add_endpoint(self):
        """Test for Globus.add_endpoint method."""
        # Test with UUID
        # 1. Should raise exception when label not defined
        endpoint_id = '2dc8ccc6-2f8e-11e9-9351-0e3d676669f4'
        with self.assertRaises(ValueError):
            self.globus.add_endpoint(endpoint_id)
        # 2. Should add UUID to endpoints along with root path
        name = 'lab1'
        self.globus.add_endpoint(endpoint_id, label=name, root_path='/mnt')
        self.assertIn(name, self.globus.endpoints)
        expected = {'id': uuid.UUID(endpoint_id), 'root_path': '/mnt'}
        self.assertDictEqual(self.globus.endpoints[name], expected)

        # Test with Alyx repo name
        # Set up REST cache fixtures
        ac = AlyxClient(**TEST_DB_1)
        setup_rest_cache(ac.cache_dir)
        name = 'mainenlab'
        self.globus.add_endpoint(name, alyx=ac)
        self.assertIn(name, self.globus.endpoints)
        expected = {
            'id': uuid.UUID('0b6f5a7c-a7a9-11e8-96fa-0a6d4e044368'),
            'root_path': '/mnt/globus/mainenlab/Subjects'
        }
        self.assertDictEqual(self.globus.endpoints[name], expected)

        # Test behaviour when label exists
        with self.assertLogs(logging.getLogger('one.remote.globus'), logging.ERROR):
            self.globus.add_endpoint(name, root_path='/', alyx=ac)
            self.assertNotEqual(self.globus.endpoints[name]['root_path'], '/')
        self.globus.add_endpoint(name, root_path='/', overwrite=True, alyx=ac)
        self.assertEqual(self.globus.endpoints[name]['root_path'], '/')

    def test_fetch_endpoints_from_alyx(self):
        """Test for Globus.fetch_endpoints_from_alyx method."""
        alyx = AlyxClient(**TEST_DB_1)
        uid = uuid.uuid1()
        repos = [{'name': 'foo', 'globus_endpoint_id': '', 'globus_path': '/some/path'},
                 {'name': 'bar', 'globus_endpoint_id': str(uid), 'globus_path': '/foo/path'}]
        with mock.patch.object(alyx, 'rest', return_value=repos):
            added = self.globus.fetch_endpoints_from_alyx(alyx)
        # Repos without an endpoint ID should have been ignored
        self.assertEqual(added, {'bar': {'id': uid, 'root_path': '/foo/path'}})
        self.assertIn('bar', self.globus.endpoints)

    def test_endpoint_path(self):
        """Test for Globus._endpoint_path method."""
        expected = PurePosixPath('/mnt/foo/bar')
        self.assertEqual(str(expected), self.globus._endpoint_path(expected))
        expected = '/foo/bar/baz'
        self.assertEqual(expected, self.globus._endpoint_path('bar/baz', root_path='/foo'))
        with self.assertRaises(ValueError):
            self.globus._endpoint_path('bar', root_path='foo')

    def test_endpoint_id_root(self):
        """Test for Globus._endpoint_id_root method."""
        id, path = self.globus._endpoint_id_root('local')
        self.assertEqual(ENDPOINT_ID, id)
        self.assertEqual(path, globus.as_globus_path(self.root_path))

        # Check behaviour when endpoint not in list
        expected = uuid.uuid4()
        id, path = self.globus._endpoint_id_root(expected)
        self.assertEqual(expected, id)
        self.assertIsNone(path)

        # Should warn when ambiguous
        self.globus.add_endpoint(ENDPOINT_ID, label='foo', root_path='/foo/bar')
        with self.assertWarns(UserWarning):
            id, path = self.globus._endpoint_id_root(ENDPOINT_ID)
        self.assertEqual(ENDPOINT_ID, id)
        self.assertEqual(path, globus.as_globus_path(self.root_path))

        with self.assertRaises(ValueError):
            self.globus._endpoint_id_root('remote')

    def test_ls(self):
        """Test for Globus.ls method."""
        response = dict(
            name=Path(self.root_path, f'some.{uuid.uuid4()}.file').as_posix(),
            type='file', size=1024)
        err = globus_sdk.GlobusConnectionError('', ConnectionError)
        self.globus.client.operation_ls.side_effect = (err, [response])
        path = globus.as_globus_path(self.root_path)
        out = self.globus.ls('local', path)
        self.assertEqual(2, self.globus.client.operation_ls.call_count)
        self.assertIsInstance(out[0], PurePosixPath)
        self.assertEqual(response['name'], out[0].as_posix())

        # Remove uuid and return size args
        self.globus.client.operation_ls.side_effect = ([response],)
        out, = self.globus.ls('local', path, remove_uuid=True, return_size=True)
        self.assertEqual(response['size'], out[1])
        self.assertNotIn(response['name'].split('.')[-2], str(out[0]))

        self.globus.client.operation_ls.reset_mock()
        self.globus.client.operation_ls.side_effect = err
        with self.assertRaises(globus_sdk.GlobusConnectionError):
            self.globus.ls('local', path, max_retries=2)
        self.assertEqual(3, self.globus.client.operation_ls.call_count)

    def test_mv(self):
        """Test for Globus.mv and Globus.run_task methods."""
        source = ('some.file', 'some2.file')
        destination = ('new.file', 'new2.file')
        # Mock transfer output
        task_id = uuid.uuid1()
        self.globus.client.submit_transfer.return_value = {'task_id': str(task_id)}
        self.globus.client.get_task.return_value = {'status': 'SUCCEEDED'}
        self.globus.client.task_successful_transfers.return_value = \
            [dict(source_path=src, destination_path=dst) for src, dst in zip(source, destination)]
        self.globus.client.task_skipped_errors.return_value = []
        task_response = self.globus.mv('local', 'local', source, destination)
        self.assertEqual(task_id, task_response)

        # Test errors
        # Check timeout behaviour
        self.globus.client.task_wait.reset_mock()
        self.globus.client.task_wait.return_value = False
        timeout = 10
        with self.assertRaises(IOError) as ex:
            self.globus.mv('local', 'local', source, destination, timeout=timeout)
            self.assertIn(str(task_id), str(ex))
        self.assertEqual(timeout, self.globus.client.task_wait.call_count)

        # Check status check error behaviour
        self.globus.client.task_wait.return_value = True
        self.globus.client.task_successful_transfers.side_effect = \
            globus_sdk.TransferAPIError(mock.MagicMock())
        with self.assertLogs(logging.getLogger('one.remote.globus'), logging.WARNING):
            self.globus.mv('local', 'local', source, destination)

        # Check failed transfer
        self.globus.client.get_task.return_value = {'status': 'FAILED'}
        self.globus.client.task_successful_transfers.reset_mock()
        with self.assertRaises(IOError) as ex:
            self.globus.mv('local', 'local', source, destination)
            self.assertIn(self.globus.client.get_task.return_value['status'], str(ex))

        # Check submission error behaviour
        self.globus.client.submit_transfer.side_effect = \
            globus_sdk.GlobusConnectionError('', ConnectionError)
        default_n_retries = 3
        with self.assertLogs(logging.getLogger('one.remote.globus')) as log, \
                self.assertRaises(globus_sdk.GlobusConnectionError):
            self.globus.mv('local', 'local', source, destination)
            warnings = filter(lambda x: x.levelno == 30, log.records)
            self.assertEqual(default_n_retries, len(list(warnings)))
            self.assertRegex(log.records[-1].msg, 'Max retries')
            self.assertEqual('ERROR', log.records[-1].levelname)

    def test_transfer_data(self):
        """Test for Globus.transfer_data method."""
        src_id, dst_id = uuid.uuid1(), uuid.uuid1()
        self.globus.endpoints['repo_01'] = {'id': dst_id, 'root_path': '/mnt/s0/'}
        self.globus.endpoints['repo_00'] = {'id': src_id, 'root_path': '/mnt/h0/Data'}
        sdk_mock, _ = self.globus_sdk_mock.get_original()
        response_mock = mock.create_autospec(globus_sdk.response.GlobusHTTPResponse)
        response_mock.data = {'task_id': str(uuid.uuid1())}
        self.globus.client.submit_transfer.return_value = response_mock

        out = self.globus.transfer_data('path/to/file', 'repo_00', 'repo_01', foo='bar')

        # SDK should be called with endpoint IDs and optional kwargs
        sdk_mock.TransferData.assert_called_once_with(
            self.globus.client, foo='bar', source_endpoint=src_id, destination_endpoint=dst_id)
        sdk_mock.TransferData().add_item.assert_called_once_with(
            '/mnt/h0/Data/path/to/file', '/mnt/s0/path/to/file', recursive=False)
        self.globus.client.submit_transfer.assert_called_once()
        self.assertEqual(out, uuid.UUID(response_mock.data['task_id']))

        # Test passing list of files
        sdk_mock.reset_mock()
        files = ['path/to/file', 'foo/bar.baz']
        self.globus.transfer_data(files, 'repo_00', 'repo_01')
        self.assertEqual(sdk_mock.TransferData().add_item.call_count, len(files))

    def test_download_file(self):
        """Test for Globus.download_file method."""
        self.globus.endpoints['repo_01'] = {'id': ENDPOINT_ID, 'root_path': '/mnt/s0/'}
        task_id = 'abc123'
        files = ['foo/bar.file', 'foo/foo/bar.file', 'baz.file']
        # create files on disk
        transferred = []
        for f in reversed(files):  # reversed to check files reordered
            full_path = self.root_path / f
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.touch()
            transferred.append({'destination_path': globus.as_globus_path(full_path)})

        self.globus.client.task_successful_transfers.return_value = transferred
        with mock.patch.object(self.globus, 'run_task', return_value=task_id):
            downloaded = self.globus.download_file(files, 'repo_01')
        expected = list(map(self.root_path.joinpath, files))
        self.assertEqual(downloaded, expected)  # asserts order of list identical

        # Behaviour should be similar when a folder is downloaded (i.e. recursive is True)
        with mock.patch.object(self.globus, 'run_task', return_value=task_id):
            downloaded = self.globus.download_file('folder', 'repo_01', recursive=True)
        self.assertCountEqual(downloaded, expected)

        # Should return a single element if one file downloaded
        self.globus.client.task_successful_transfers.return_value = [transferred[0]]
        with mock.patch.object(self.globus, 'run_task', return_value=task_id):
            downloaded = self.globus.download_file(files[-1], 'repo_01', recursive=False)
        self.assertIsInstance(downloaded, Path)

        # Should raise assertion error if file doesn't exist on disk (unlikely!)
        self.root_path.joinpath(files[0]).unlink()
        self.globus.client.task_successful_transfers.return_value = transferred
        with mock.patch.object(self.globus, 'run_task', return_value=task_id):
            self.assertRaises(AssertionError, self.globus.download_file, files, 'repo_01')

    def test_delete_data(self):
        """Test for Globus.delete_data method."""
        globus_id = uuid.uuid1()
        self.globus.endpoints['repo_00'] = {'id': globus_id, 'root_path': '/mnt/h0/Data'}
        sdk_mock, _ = self.globus_sdk_mock.get_original()
        response_mock = mock.create_autospec(globus_sdk.response.GlobusHTTPResponse)
        response_mock.data = {'task_id': str(uuid.uuid1())}
        self.globus.client.submit_delete.return_value = response_mock

        out = self.globus.delete_data('path/to/file', 'repo_00', foo='bar')

        # SDK should be called with endpoint IDs and optional kwargs
        sdk_mock.DeleteData.assert_called_once_with(
            self.globus.client, recursive=False, foo='bar', endpoint=globus_id)
        sdk_mock.DeleteData().add_item.assert_called_once_with('/mnt/h0/Data/path/to/file')
        self.globus.client.submit_delete.assert_called_once()
        self.assertEqual(out, uuid.UUID(response_mock.data['task_id']))

        # Test passing list of files
        sdk_mock.reset_mock()
        files = ['path/to/file', 'foo/bar.baz']
        self.globus.delete_data(files, 'repo_00')
        self.assertEqual(sdk_mock.DeleteData().add_item.call_count, len(files))

    def test_globus_headless(self):
        """Test for Globus object in headless mode."""
        self.assertRaises(RuntimeError, globus.Globus, 'foobar', headless=True)
        pars = self.globus._pars
        with mock.patch('one.remote.globus._setup', return_value=pars) as setup_function:
            globus.Globus('foobar', headless=False, connect=False)
            setup_function.assert_called()

    def test_login_logout(self):
        """Test for Globus.login and Globus.logout methods."""
        assert self.globus.is_logged_in
        with self.assertLogs('one.remote.globus', 10):
            self.globus.login()
            self.globus.client.authorizer.ensure_valid_token.assert_called()

        # Log out
        # Change client name in order to avoid overwriting parameters
        with mock.patch('one.remote.globus.save_client_params') as save_func, \
                mock.patch('one.remote.globus.load_client_params', return_value=self.pars):
            self.globus.logout()
            save_func.assert_called()
            (all_pars, *_), _ = save_func.call_args
            self.assertNotIn('access_token', all_pars[self.globus.client_name])

        self.globus.logout()  # check repeat calls don't raise errors
        self.assertIsNone(self.globus.client.authorizer.get_authorization_header())
        self.assertFalse(hasattr(self.globus.client.authorizer, 'access_token'))
        self.assertFalse(hasattr(self.globus._pars, 'access_token'))
        self.assertFalse(self.globus.is_logged_in)
        self.assertIsNone(self.globus._token_expired)

        # Test what happens when authenticate called with invalid token
        self.assertRaises(RuntimeError, self.globus._authenticate)

        # Check login in headless mode
        self.globus.headless = True
        self.assertRaises(RuntimeError, self.globus.login)

        self.globus.headless = False
        # Test login cancel
        with mock.patch('one.remote.globus.load_client_params', return_value=self.pars), \
                mock.patch('builtins.input', return_value='c'), \
                self.assertLogs('one.remote.globus', 10):
            self.globus.login()
            self.assertFalse(self.globus.is_logged_in)

        token = {'refresh_token': None, 'expires_at_seconds': datetime.now().timestamp() + 60**2,
                 'access_token': 'a1b2c3d4e5f6g7h8'}
        # Stop and start mock in order to reset MagicMock attributes
        self.globus_sdk_mock.stop()
        self.globus_sdk_mock = self.globus_sdk_mock.start()
        self.addCleanup(self.globus_sdk_mock.stop)
        with mock.patch('one.remote.globus.save_client_params') as save_func, \
                mock.patch('one.remote.globus.get_token', return_value=token), \
                mock.patch('one.remote.globus.load_client_params', return_value=self.pars):
            # Expected refresh token warning as stay_logged_in is True
            # In reality this will only happen when loading saved taken where refresh_token = False
            self.assertWarns(UserWarning, self.globus.login, stay_logged_in=True)
            self.assertTrue(self.globus.is_logged_in)


class TestGlobusAsync(unittest.IsolatedAsyncioTestCase, _GlobusClientTest):
    """Asynchronous Globus method tests."""

    async def test_task_wait_async(self):
        """Test for Globus.task_wait_async method."""
        task_id = uuid.uuid4()
        statuses = ({'status': 'ACTIVE'}, {'status': 'SUCCESSFUL'})
        with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as sleep_mock, \
                mock.patch.object(self.globus.client, 'get_task', side_effect=statuses):
            self.assertTrue(await self.globus.task_wait_async(task_id, polling_interval=5))
            sleep_mock.assert_awaited_once_with(5)  # polling_interval value

        # Check timeout behaviour
        status = statuses[0]
        with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as sleep_mock, \
                mock.patch.object(self.globus.client, 'get_task', return_value=status):
            self.assertFalse(await self.globus.task_wait_async(task_id, polling_interval=3))
            sleep_mock.assert_awaited_with(3)  # polling_interval value
            self.assertEqual(round(10 / 3), sleep_mock.await_count)  # timeout = 10

        # Check input validation
        with self.assertRaises(globus_sdk.GlobusSDKUsageError):
            await self.globus.task_wait_async(task_id, polling_interval=.5)
        with self.assertRaises(globus_sdk.GlobusSDKUsageError):
            await self.globus.task_wait_async(task_id, timeout=.5)


if __name__ == '__main__':
    unittest.main()
