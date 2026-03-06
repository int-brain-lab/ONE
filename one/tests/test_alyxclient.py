"""Unit tests for the one.webclient module."""

from pathlib import Path
import unittest
from unittest import mock
import http.client
import urllib.parse
import random
import weakref
import uuid
import os
import io
import one.webclient as wc
import one.params
import tempfile
import shutil
import requests
import json
import logging
from datetime import datetime, timedelta, timezone

from iblutil.io import hashfile
import iblutil.io.params as iopar

from one.tests import OFFLINE_ONLY, TEST_DB_1, TEST_DB_2
from one.tests import util

par = one.params.get(silent=True)


class TestRestDocumentation(unittest.TestCase):
    """Tests for AlyxClient REST API schema parsing and printing."""

    def setUp(self) -> None:
        self.ac = wc.AlyxClient()
        self.path_fixtures = Path(__file__).parent.joinpath('fixtures', 'rest_responses')
        with open(self.path_fixtures.joinpath('coreapi.json'), 'r') as f:
            rest_scheme = json.load(f)
        self.scheme_coreapi = wc.RestSchemeCoreApi(rest_scheme)

        with open(self.path_fixtures.joinpath('openapiv3.json'), 'r') as f:
            rest_scheme = json.load(f)
        self.scheme_openapi = wc.RestSchemeOpenApi(rest_scheme)

    def test_rest_schema_core_api(self):
        """Test parsing of coreapi REST schema."""
        # test the list of endpoints
        self.assertEqual(set(self.scheme_openapi.endpoints), set(self.scheme_coreapi.endpoints))
        # test the list of actions for each endpoint
        self.assertEqual(
            set(self.scheme_coreapi.actions('sessions')),
            {'create', 'delete', 'list', 'partial_update', 'read', 'update'},
        )
        self.assertEqual(
            set(self.scheme_openapi.actions('sessions')),
            {'create', 'delete', 'list', 'partial_update', 'read', 'update'},
        )
        # test getting the URL for each endpoint/action
        self.assertEqual(self.scheme_openapi.url('sessions', 'read'), '/sessions/{id}')
        self.assertEqual(self.scheme_coreapi.url('sessions', 'read'), '/sessions/{id}')
        # test getting the parameters for each endpoint/action
        self.assertTrue(len(self.scheme_openapi.fields('sessions', 'partial_update')) >= 19)
        self.assertTrue(len(self.scheme_coreapi.fields('sessions', 'partial_update')) >= 19)
        self.assertTrue(len(self.scheme_openapi.field_names('sessions', 'partial_update')) >= 19)
        self.assertTrue(len(self.scheme_coreapi.field_names('sessions', 'partial_update')) >= 19)

    def test_print_endpoint_info(self):
        """Test endpoint query params are printed when calling AlyxClient.rest without action."""
        # Check behaviour when endpoint invalid
        endpoint = 'foobar'
        for scheme in [self.scheme_openapi, self.scheme_coreapi]:
            with self.subTest(scheme=scheme):
                with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                    scheme.print_endpoint_info(endpoint)
                    self.assertRegex(stdout.getvalue(), f'"{endpoint}" does not exist')
                # Check returns endpoint info as well as printing
                with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                    scheme.print_endpoint_info('sessions', 'partial_update')
                    self.assertRegex(stdout.getvalue(), 'parent_session')
                    self.assertRegex(stdout.getvalue(), 'extended_qc')

                with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                    scheme.print_endpoint_info('sessions')
                    self.assertRegex(stdout.getvalue(), 'partial_update')

                with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                    scheme.print_endpoint_info('insertions', 'erase')
                    self.assertRegex(stdout.getvalue(),
                                     'Endpoint "insertions" does not have action "erase"')

    @unittest.skipIf(OFFLINE_ONLY, 'online only test')
    def test_alyx_client_methods(self):
        """Test AlyxClient.list_endpoints and AlyxClient.print_endpoint_info."""
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.assertTrue(len(self.ac.list_endpoints()) > 20)
            self.assertRegex(stdout.getvalue(), 'sessions')

        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.ac.print_endpoint_info('sessions')
            self.assertRegex(stdout.getvalue(), 'partial_update')

    def test_schema_support(self):
        """Test that both coreapi and openapiv3 REST schemas are supported."""
        # The new alyx uses openapiv3 schema on the /api/schema endpoint
        output = self.scheme_openapi._rest_scheme
        with mock.patch.object(self.ac, 'get', return_value=output) as mock_get:
            assert self.ac._rest_schemes is None  # Ensure not cached
            scheme = self.ac.rest_schemes
        self.assertIsInstance(scheme, wc.RestSchemeOpenApi)
        mock_get.assert_called_once_with('/api/schema', expires=mock.ANY)
        self.ac._rest_schemes = None  # Reset cached scheme
        # Old alyx uses coreapi schema on the /docs endpoint
        err = requests.HTTPError()
        rep = requests.Response()
        rep.status_code = 404
        err.response = rep
        responses = (err, self.scheme_coreapi._rest_scheme)
        with mock.patch.object(self.ac, 'get', side_effect=responses) as mock_get:
            scheme = self.ac.rest_schemes
        self.assertIsInstance(scheme, wc.RestSchemeCoreApi)
        # Check the correct accept request header when querying the legacy
        rep.status_code = 200
        rep._content = json.dumps(self.scheme_coreapi._rest_scheme).encode()
        get_mock = mock.MagicMock(return_value=rep)
        self.ac._generic_request.__wrapped__(self.ac, get_mock, '/docs')
        get_mock.assert_called_once()
        expected = get_mock.call_args.kwargs['headers'] | {'Accept': 'application/coreapi+json'}
        self.assertEqual(expected, get_mock.call_args.kwargs['headers'])
        # Should try new schema endpoint then fallback to old one
        self.assertEqual(mock_get.call_count, 2)
        endpoints = [x.args[0] for x in mock_get.call_args_list]
        expected = ['/api/schema', '/docs']
        self.assertEqual(endpoints, expected)
        # If the status code is not 404, should raise the error
        rep.status_code = 500
        self.ac._rest_schemes = None  # Reset cached scheme
        with mock.patch.object(self.ac, 'get', side_effect=err), \
                self.assertRaises(requests.HTTPError):
            scheme = self.ac.rest_schemes


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestAuthentication(unittest.TestCase):
    """Tests for AlyxClient authentication, token storage, login/out methods and user prompts."""

    def setUp(self) -> None:
        self.ac = wc.AlyxClient(**TEST_DB_2)

    def test_authentication(self):
        """Test for AlyxClient.authenticate and AlyxClient.is_logged_in property."""
        ac = self.ac
        self.assertTrue(ac.is_logged_in)
        ac.logout()
        self.assertFalse(ac.is_logged_in)
        # Check token removed from cache
        cached_token = getattr(one.params.get(TEST_DB_2['base_url']), 'TOKEN', {})
        self.assertFalse(TEST_DB_2['username'] in cached_token)
        # Test with pars set
        login_keys = {'ALYX_LOGIN', 'ALYX_PWD'}
        if not set(ac._par.as_dict().keys()) >= login_keys:
            for k, v in zip(sorted(login_keys), (TEST_DB_2['username'], TEST_DB_2['password'])):
                ac._par = ac._par.set(k, v)
        with mock.patch('builtins.input') as mock_input:
            ac.authenticate()
            mock_input.assert_not_called()
        self.assertTrue(ac.is_logged_in)

        # When password is None and in silent mode, there should be a warning
        # followed by a failed login attempt
        ac._par = ac._par.set('ALYX_PWD', None)
        ac.logout()
        with self.assertWarns(UserWarning), self.assertRaises(requests.HTTPError):
            self.ac.authenticate(password=None)

        # Test using input args
        ac._par = iopar.from_dict(
            {k: v for k, v in ac._par.as_dict().items() if k not in login_keys}
        )
        with mock.patch('builtins.input') as mock_input:
            ac.authenticate(TEST_DB_2['username'], TEST_DB_2['password'], cache_token=False)
            mock_input.assert_not_called()
        # Check token not saved in cache
        cached_token = getattr(one.params.get(TEST_DB_2['base_url']), 'TOKEN', {})
        self.assertFalse(TEST_DB_2['username'] in cached_token)
        # Test user prompts
        ac.logout()
        ac.silent = False
        with (
            mock.patch('builtins.input', return_value=TEST_DB_2['username']),
            mock.patch('one.webclient.getpass', return_value=TEST_DB_2['password']),
        ):
            ac.authenticate(cache_token=True)
        self.assertTrue(ac.is_logged_in)
        # Check token saved in cache
        ac.authenticate(cache_token=True)
        cached_token = getattr(one.params.get(TEST_DB_2['base_url']), 'TOKEN', {})
        self.assertTrue(TEST_DB_2['username'] in cached_token)
        # Check force flag
        with mock.patch('one.webclient.getpass', return_value=TEST_DB_2['password']) as mock_pwd:
            ac.authenticate(cache_token=True, force=True)
            mock_pwd.assert_called()
        # If a password is passed, should always force re-authentication
        rep = requests.Response()
        rep.status_code = 200
        rep.json = lambda **_: {'token': 'abc'}
        assert self.ac.is_logged_in
        with mock.patch('one.webclient.requests.post', return_value=rep) as m:
            self.ac.authenticate(password='foo', force=False)
            expected = {'username': TEST_DB_2['username'], 'password': 'foo'}
            m.assert_called_once_with(TEST_DB_2['base_url'] + '/auth-token', data=expected)

        # Check non-silent double logout
        ac.logout()
        ac.logout()  # Shouldn't complain

    def test_auth_methods(self):
        """Test behaviour when calling AlyxClient._generic_request when logged out."""
        # Check that authentication happens when making a logged out request
        self.ac.logout()
        assert self.ac.is_logged_in is False
        # Set pars for auto login
        login_keys = {'ALYX_LOGIN', 'ALYX_PWD'}
        if not set(self.ac._par.as_dict().keys()) >= login_keys:
            for k, v in zip(sorted(login_keys), (TEST_DB_2['username'], TEST_DB_2['password'])):
                self.ac._par = self.ac._par.set(k, v)

        # Test generic request
        self.ac._generic_request(requests.get, '/sessions?user=Hamish', clobber=True)
        self.assertTrue(self.ac.is_logged_in)

        # Test behaviour when token invalid
        self.ac._token['token'] = '1NVAL1DT0K3N'
        self.ac._headers['Authorization'] = 'Token ' + self.ac._token['token']
        self.ac._generic_request(requests.get, '/sessions?user=Hamish', clobber=True)
        self.assertTrue(self.ac.is_logged_in)

        # Test download cache tables
        self.ac.logout()
        self.assertFalse(self.ac.is_logged_in)
        url = self.ac.get('cache/info').get('location')
        self.ac.download_cache_tables(url)
        self.assertTrue(self.ac.is_logged_in)

    def test_auth_errors(self):
        """Test behaviour when authentication fails."""
        self.ac.logout()  # Make sure logged out
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac.authenticate(password='wrong_pass')
            self.assertTrue('user = intbrainlab' in str(ex))
            self.assertFalse('wrong_pass' in str(ex))
        # Check behaviour when connection error occurs (should mention firewall settings)
        with (
            mock.patch('one.webclient.requests.post', side_effect=requests.ConnectionError),
            self.assertRaises(ConnectionError) as ex,
        ):
            self.ac.authenticate()
            self.assertTrue('firewall' in str(ex))
        # Check behaviour when server error occurs
        rep = requests.Response()
        rep.status_code = 500
        with (
            mock.patch('one.webclient.requests.post', return_value=rep),
            self.assertRaises(requests.HTTPError),
        ):
            self.ac.authenticate()


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestJsonFieldMethods(unittest.TestCase):
    """Tests for AlyxClient methods that modify the JSON field of a REST endpoint.

    These tests are over-engineered in order to test Alyx Django queries with JSON fields.
    Django queries are also tested in TestRemote.test_search.
    """

    def setUp(self):
        self.ac = wc.AlyxClient(**TEST_DB_1, cache_rest=None)

        # Create new subject and two new sessions
        name = '0A' + str(random.randint(0, 10000))
        self.subj = self.ac.rest('subjects', 'create', data={'nickname': name, 'lab': 'cortexlab'})
        sessions = [
            self.ac.rest(
                'sessions',
                'create',
                data={
                    'subject': name,
                    'start_time': datetime.isoformat(datetime.now()),
                    'number': random.randint(1, 999),
                    'type': 'Experiment',
                    'users': [TEST_DB_1['username']],
                },
            )
            for _ in range(2)
        ]

        self.eids = [uuid.UUID(x['url'].split('/')[-1]) for x in sessions]
        self.endpoint = 'sessions'
        self.field_name = 'extended_qc'
        # We filter by key value so we use randint to avoid race condition in concurrent tests
        i = str(random.randint(0, 10000))
        self.data_dict = {'low_' + i: 0, 'high_' + i: 1}

    def _json_field_write(self):
        written1 = self.ac.json_field_write(
            self.endpoint, self.eids[0], self.field_name, self.data_dict
        )
        written2 = self.ac.json_field_write(
            self.endpoint, self.eids[1], self.field_name, self.data_dict
        )
        self.assertTrue(written1 == written2)
        self.assertTrue(written1 == self.data_dict)
        data_field = next(filter(lambda x: x.startswith('low'), self.data_dict))
        url = f'/{self.endpoint}?&{self.field_name}={data_field}__lt,0.5'
        sess_dict = self.ac.get(url, expires=True)
        self.assertTrue(len(sess_dict) == 2)

    def _json_field_update(self):
        data_field = next(filter(lambda x: x.startswith('low'), self.data_dict))
        modified = self.ac.json_field_update(
            self.endpoint, self.eids[0], self.field_name, {data_field: 0.6}
        )
        self.assertCountEqual(modified.keys(), self.data_dict.keys())
        url = f'/{self.endpoint}?&{self.field_name}={data_field}__lt,0.5'
        self.assertTrue(len(self.ac.get(url, expires=True)) == 1)

    def _json_field_remove_key(self):
        eid = self.eids[1]
        data_field = next(filter(lambda x: x.startswith('hi'), self.data_dict))
        url = f'/{self.endpoint}?&{self.field_name}={data_field}__gte,0.5'
        pre_delete = self.ac.get(url, expires=True)
        self.assertTrue(len(pre_delete) == 2)
        deleted = self.ac.json_field_remove_key(self.endpoint, eid, self.field_name, data_field)
        self.assertTrue(data_field not in deleted)
        post_delete = self.ac.get(url, expires=True)
        self.assertTrue(len(post_delete) == 1)

    def _json_field_delete(self):
        data_field = next(filter(lambda x: x.startswith('hi'), self.data_dict))
        deleted = self.ac.json_field_delete(self.endpoint, self.eids[1], self.field_name)
        self.assertTrue(deleted is None)
        url = f'/{self.endpoint}?&{self.field_name}={data_field}__gte,0.5'
        ses = self.ac.get(url, expires=True)
        self.assertTrue(len(ses) == 1)

    def test_json_methods(self):
        """Test for AlyxClient.json_field* methods (write, update, remove_key and delete)."""
        self._json_field_write()
        self._json_field_update()
        self._json_field_remove_key()
        self._json_field_delete()

    def test_empty(self):
        """Test for AlyxClient.json_field* methods when JSON field is empty."""
        eid = self.eids[0]
        # Check behaviour when fields are empty
        self.ac.rest(self.endpoint, 'partial_update', id=eid, data={self.field_name: None})
        # Should return None as no keys exist
        modified = self.ac.json_field_remove_key(self.endpoint, eid, self.field_name, 'foo')
        self.assertIsNone(modified)
        # Should return data
        data = {'some': 0.6}
        modified = self.ac.json_field_update(self.endpoint, eid, self.field_name, data)
        self.assertTrue(modified == data)
        # Should warn if key not in dict
        with self.assertLogs(logging.getLogger('one.webclient'), logging.WARNING):
            self.ac.json_field_remove_key(self.endpoint, eid, self.field_name, 'foo')
        # Check behaviour when fields not a dict
        data = {self.field_name: json.dumps(data)}
        self.ac.rest(self.endpoint, 'partial_update', id=eid, data=data)
        # Update field
        with self.assertLogs(logging.getLogger('one.webclient'), logging.WARNING):
            modified = self.ac.json_field_update(self.endpoint, eid, self.field_name, data)
        self.assertEqual(data[self.field_name], modified)
        # Remove key
        with self.assertLogs(logging.getLogger('one.webclient'), logging.WARNING):
            modified = self.ac.json_field_remove_key(self.endpoint, eid, self.field_name)
        self.assertIsNone(modified)

    def test_uuid_serialize(self):
        """Check that UUID objects are serialized to JSON."""
        data = {'uid': self.eids[-1], **self.data_dict}
        written = self.ac.json_field_write(self.endpoint, self.eids[0], self.field_name, data)
        self.assertIsInstance(written, dict)
        # Encoder should have cast uuid to str
        self.assertEqual(str(self.eids[-1]), written.get('uid'))

    def tearDown(self):
        self.ac.rest('subjects', 'delete', id=self.subj['nickname'])


class TestRestCache(unittest.TestCase):
    """Tests for REST caching system, the cache decorator and cache flags."""

    def setUp(self):
        util.setup_test_params()  # Ensure test alyx set up
        self.ac = wc.AlyxClient(**TEST_DB_1)
        util.setup_rest_cache(self.ac.cache_dir)  # Copy rest cache fixtures
        self.query = '/insertions/b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        self.tempdir = util.set_up_env()
        self.addCleanup(self.tempdir.cleanup)
        one.webclient.datetime = _FakeDateTime
        _FakeDateTime._now = None
        self.cache_dir = self.ac.cache_dir.joinpath('.rest')

    def test_loads_cached(self):
        """Test for one.webclient._cache_response decorator, checks returns cached result."""
        # Check returns cache
        wrapped = wc._cache_response(lambda *args: self.assertTrue(False))
        res = wrapped(self.ac, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

    def test_expired_cache(self):
        """Test behaviour when cached REST query is expired."""
        # Checks expired
        wrapped = wc._cache_response(lambda *args: 'called')
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')
        res = wrapped(self.ac, requests.get, self.query)
        self.assertTrue(res == 'called')

    def test_caches_response(self):
        """Test caches query response before returning."""
        # Default expiry time
        self.ac.default_expiry = timedelta(minutes=1)
        wrapped = wc._cache_response(lambda *args: 'called')
        _FakeDateTime._now = datetime(2021, 5, 13)  # Freeze time
        res = wrapped(self.ac, requests.get, '/endpoint?id=5')
        self.assertTrue(res == 'called')

        # Check cache file created
        filename = '64b5b3476c015e04ee7c4753606b5e967325d34a'
        cache_file = self.cache_dir / filename
        self.assertTrue(cache_file.exists())
        with open(cache_file, 'r') as f:
            q, when = json.load(f)
        self.assertEqual('called', q)
        self.assertEqual(when, '2021-05-13T00:01:00')

    def test_cache_mode(self):
        """Test for AlyxClient.cache_mode property."""
        # With cache mode off, wrapped method should be called even in presence of valid cache
        self.ac.cache_mode = None  # cache nothing
        wrapped = wc._cache_response(lambda *args: 'called')
        res = wrapped(self.ac, requests.get, self.query)
        self.assertTrue(res == 'called')

    def test_expiry_param(self):
        """Test for expires kwarg in one.webclient._cache_response decorator."""
        # Check expiry param
        wrapped = wc._cache_response(lambda *args: '123')
        res = wrapped(self.ac, requests.get, '/endpoint?id=5', expires=True)
        self.assertTrue(res == '123')

        # A second call should yield a new response as cache immediately expired
        wrapped = wc._cache_response(lambda *args: '456')
        res = wrapped(self.ac, requests.get, '/endpoint?id=5', expires=False)
        self.assertTrue(res == '456')

        # With clobber=True the cache should be overwritten
        wrapped = wc._cache_response(lambda *args: '789')
        res = wrapped(self.ac, requests.get, '/endpoint?id=5', clobber=True)
        self.assertTrue(res == '789')

    def test_cache_returned_on_error(self):
        """Test behaviour when connection error occurs and cached response exists."""
        func = mock.Mock(side_effect=requests.ConnectionError())
        wrapped = wc._cache_response(func)
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')  # Expired
        with self.assertWarns(RuntimeWarning):
            res = wrapped(self.ac, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

        # With clobber=True exception should be raised
        with self.assertRaises(requests.ConnectionError):
            wrapped(self.ac, requests.get, self.query, clobber=True)

    def test_decode_error_cache(self):
        """Test behaviour when cached file is corrupted."""
        func = mock.Mock(return_value='called')
        wrapped = wc._cache_response(func)
        # Should not call wrapped function as cache valid
        res = wrapped(self.ac, requests.get, self.query)
        self.assertNotEqual('called', res)
        # Corrupt the cache file by adding a character
        filename = 'f530d6022f61cdc9e38cc66beb3cb71f3003c9a1'
        with open(self.cache_dir / filename, 'a') as f:
            f.write('"')  # Incomplete JSON
        with self.assertLogs(logging.getLogger('one.webclient'), logging.DEBUG) as log:
            res = wrapped(self.ac, requests.get, self.query)
            self.assertTrue('corrupted cache file' in log.output[1])
            self.assertEqual('called', res)

    def test_clear_cache(self):
        """Test for AlyxClient.clear_rest_cache."""
        assert any(self.cache_dir.glob('*'))
        self.ac.clear_rest_cache()
        self.assertFalse(any(self.cache_dir.glob('*')))


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
@mock.patch('one.webclient.sleep')
class TestThrottleBackoff(unittest.TestCase):
    """Test behaviour when AlyxClient receives a 429 Too Many Requests response."""

    def setUp(self):
        self.ac = wc.AlyxClient(**TEST_DB_1, cache_rest=None)
        self.ac.max_retry_attempts = 1
        self.retry_after = 5

    def _get(self, *args, **kwargs):
        """Mock request.get method that simulates a 429 response with Retry-After header."""
        rep = requests.Response()
        rep.status_code = 429
        if self.retry_after:
            rep._content = ('Request was throttled. Expected available in '
                            f'{self.retry_after} seconds.').encode()
            rep.headers = {'Retry-After': str(self.retry_after)}
        else:
            rep._content = b'Request was throttled.'
            rep.headers = {}
        return rep

    def test_backoff_with_retry_after(self, sleep_mock):
        """Test that AlyxClient respects Retry-After header."""
        self.ac.max_retry_attempts = 1  # expected to wait once then raise
        # Mock a 429 response with Retry-After header
        with self.assertRaises(requests.HTTPError) as ex, \
                self.assertLogs(wc.__name__, 30) as log:
            self.ac._generic_request(self._get, '/sessions')
        # Should raise with response message after max retry attempts
        self.assertIn(ex.exception.response.text, str(ex.exception))
        # Should log warning about throttling and retrying
        pattern = \
            r'Rate limited for query: /sessions. Retrying in \d+.\d+ seconds \(attempt 1/1\)\.'
        self.assertRegex(log.records[0].message, pattern)
        # Should wait once before retrying
        sleep_mock.assert_called_once()
        # Retry wait time should be at least as long as response header value
        # For first attempt the extra jitter should be 1 second more
        wait_time, = sleep_mock.call_args.args
        self.assertTrue(
            self.retry_after + 1 > wait_time >= self.retry_after,
            f'wait time = {wait_time}, retry_after = {self.retry_after}'
        )

    def test_backoff_without_retry_after(self, sleep_mock):
        """Test that AlyxClient uses exponential backoff when Retry-After header is missing."""
        self.ac.max_retry_attempts = 10
        self.retry_after = None  # Mock a 429 response without Retry-After header
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac._generic_request(self._get, '/sessions')
        self.assertIn(ex.exception.response.text, str(ex.exception))
        self.assertEqual(self.ac.max_retry_attempts, sleep_mock.call_count)
        self.assertEqual(
            0, self.ac._attempt_counter, 'failed to reset attempt counter after max retries')
        wait_times = [x.args[0] for x in sleep_mock.call_args_list]
        # Wait times should generally increase but should never be greater than 60s
        self.assertTrue(max(wait_times) <= 60, f'wait times > 60s, t = {wait_times}')

    def test_maximum_thresholds(self, sleep_mock):
        """Test that AlyxClient raises an error after maximum retry attempts."""
        self.ac.max_retry_attempts = 0  # Should not retry, should raise immediately
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac._generic_request(self._get, '/sessions')
        self.assertIn(ex.exception.response.text, str(ex.exception))
        sleep_mock.assert_not_called()
        # When the retry delay is excessively long, should raise immediately without retrying
        self.ac.max_retry_attempts = 10
        self.retry_after = 60 * 60 * 24  # Mock a 429 response with long Retry-After header
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac._generic_request(self._get, '/sessions')
        self.assertIn(ex.exception.response.text, str(ex.exception))
        sleep_mock.assert_not_called()

    def test_retry_after_date(self, sleep_mock):
        """Test that AlyxClient correctly parses Retry-After header when it is a date."""
        self.ac.max_retry_attempts = 1
        rep = requests.Response()
        rep.status_code = 503  # Service Unavailable, can also include Retry-After header
        rep._content = b'Service Unavailable.'
        retry_after_date = datetime.now(tz=timezone.utc) + timedelta(seconds=self.retry_after)
        rep.headers = {'Retry-After': retry_after_date.strftime('%a, %d %b %Y %H:%M:%S GMT')}
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac._generic_request(lambda *args, **kwargs: rep, '/sessions')
        self.assertIn(rep.text, str(ex.exception))
        sleep_mock.assert_called_once()
        # Retry wait time should be at least as long as response header value
        # For first attempt the extra jitter should be 1 second more
        wait_time, = sleep_mock.call_args.args
        self.assertTrue(self.retry_after + 1 > wait_time, f'{wait_time} > {self.retry_after} + 1')
        # Wait time may be slightly smaller than expected due to small timing
        # differences parsing header, so we allow it to be up to 1 second less
        self.assertTrue(wait_time >= self.retry_after - 1, f'{wait_time} < {self.retry_after}')


class _FakeDateTime(datetime):
    _now = None

    @staticmethod
    def now(*args, **kwargs):
        return _FakeDateTime._now or datetime.now(*args, **kwargs)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestDownloadHTTP(unittest.TestCase):
    def setUp(self):
        # Init connection to the database
        self.ac = wc.AlyxClient(**TEST_DB_1)
        # Remove /public from data server url
        if 'public' in self.ac._par.HTTP_DATA_SERVER:
            self.ac._par = self.ac._par.set(
                'HTTP_DATA_SERVER', self.ac._par.HTTP_DATA_SERVER.rsplit('/', 1)[0])
        self.test_data_uuid = '40af4a49-1b9d-45ec-b443-a151c010ea3c'  # OpenAlyx dataset

    def test_download_datasets_with_api(self):
        ac_public = wc.AlyxClient(**TEST_DB_2)
        cache_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(cache_dir))

        # Test 1: empty dir, dict mode
        dset = ac_public.get('/datasets/' + self.test_data_uuid)
        urls = wc.dataset_record_to_url(dset)
        url = [u for u in urls if u.startswith('https://ibl.flatiron')]
        (file_name,) = ac_public.download_file(url, target_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 2: empty dir, list mode
        dset = ac_public.get('/datasets?id=' + self.test_data_uuid)
        urls = wc.dataset_record_to_url(dset)
        url = [u for u in urls if u.startswith('https://ibl.flatiron')]
        (file_name,) = ac_public.download_file(url, target_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 3: Log unauthorized error with url (using test alyx)
        url = next(x['data_url'] for x in self.ac.get('/datasets?exists=True')[0]['file_records'])
        old_par = self.ac._par
        self.ac._par = self.ac._par.set('HTTP_DATA_SERVER_PWD', 'foobar')
        with self.assertLogs(logging.getLogger('one.webclient'), logging.ERROR) as log:
            raised = False
            try:
                self.ac.download_file(url, target_dir=cache_dir)
                self.assertTrue(url in log.output[-1])
            except Exception as ex:
                # Check error message mentions the HTTP_DATA_SERVER params
                self.assertTrue('HTTP_DATA_SERVER_PWD' in str(ex))
                raised = True
            finally:
                self.assertTrue(raised)
                self.ac._par = old_par

    def test_download_datasets(self):
        # test downloading a single file
        full_link_to_file = (
            'https://ibl.flatironinstitute.org/public/mrsicflogellab/Subjects/SWC_038/'
            '2020-07-29/001/alf/probes.description.f67570ac-1e54-4ce1-be5d-de2017a42116.json'
        )
        file_name, md5 = wc.http_download_file(full_link_to_file, return_md5=True, clobber=True)
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        self.assertTrue(len(data) > 0)
        self.assertTrue(hashfile.md5(file_name) == md5)

        self.assertFalse(wc.http_download_file('', clobber=True))

        # test downloading a list of files
        links = [
            full_link_to_file,
            r'https://ibl.flatironinstitute.org/public/hoferlab/Subjects/SWC_043/'
            r'2020-09-21/001/alf/probes.description.c4df1eea-c92c-479f-a907-41fa6e770094.json',
        ]
        file_list = wc.http_download_file_list(
            links, username=par.HTTP_DATA_SERVER_LOGIN, password=par.HTTP_DATA_SERVER_PWD
        )
        for file in file_list:
            with open(file, 'r') as json_file:
                data = json.load(json_file)
            self.assertTrue(len(data) > 0)

    @mock.patch('one.webclient.zipfile.ZipFile')
    @mock.patch('one.webclient.http_download_file')
    def test_download_cache_tables_auth(self, download_file_mock, zipfile_mock):
        """Test for AlyxClient.download_cache_tables with authentication.

        NB: This test simply checks that alyx is authenticated automatically before
        downloading the tables.
        """
        try:
            token = self.ac._token
            self.ac._token = None  # Force re-authentication
            with mock.patch.object(self.ac, 'authenticate') as mock_auth:
                # When the URL is different from the database, no need to authenticate
                self.ac.download_cache_tables('https://example.com/cache.zip')
                mock_auth.assert_not_called()
                download_file_mock.assert_called_once()
                zipfile_mock.assert_called_once()
                # When the URL is the same as the database, should authenticate
                self.ac.download_cache_tables(self.ac.base_url + '/cache.zip')
                mock_auth.assert_called_once()
        finally:
            self.ac._token = token

    @mock.patch('one.webclient.urllib.request')
    @mock.patch('builtins.open')
    def test_http_server_auth(self, open_mock, urllib_mock):
        """Test for http_download_file authentication and headers."""
        url_response_mock = mock.MagicMock(spec_set=http.client.HTTPResponse)
        # Simulate file content then end of file
        url_response_mock.read.side_effect = [b'file content', None]
        urllib_mock.urlopen.return_value = url_response_mock
        # When a username and password are set in the parameters, should attempt to authenticate
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name, md5 = wc.http_download_file(
                'https://example.com/file.txt',
                target_dir=temp_dir,
                username='user',
                password='pass',
                return_md5=True,
                chunks=(4, 12),
                headers={'Custom-Header': 'value'}
            )
        expected = Path(temp_dir).joinpath('file.txt')
        # Check file is written to expected location
        self.assertEqual(expected, Path(file_name))
        open_mock.assert_called_once_with(expected, 'wb')
        fid_mock = open_mock()
        fid_mock.write.assert_called_once_with(b'file content')
        fid_mock.close.assert_called_once()
        # Check urlopen called with correct auth header
        urllib.request.HTTPPasswordMgrWithDefaultRealm.assert_called_once()
        manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        manager.add_password.assert_called_once_with(None, 'https://example.com', 'user', 'pass')
        # Check the request headers
        urllib.request.urlopen.assert_called_once()
        req, = urllib.request.urlopen.call_args[0]
        req.add_header.assert_any_call('Custom-Header', 'value')
        req.add_header.assert_any_call('Range', 'bytes=4-15')  # Chunks


class TestMisc(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        self.ac = wc.AlyxClient(**TEST_DB_1)

    def test_update_url_params(self):
        """Test for one.webclient.update_url_params."""
        url = wc.update_url_params('website.com/?q=', {'pg': 5})
        self.assertEqual('website.com/?pg=5', url)

        # Check handles lists
        url = wc.update_url_params('website.com?q=xxx', {'pg': 5, 'foo': ['bar', 'baz']})
        self.assertEqual('website.com?q=xxx&pg=5&foo=bar&foo=baz', url)

        # Check encodes special chars; handles partial URL
        url = '/path?param1=foo bar'
        new_url = wc.update_url_params(url, {'param2': '#2020-01-03#,#2021-02-01#'})
        expected = '/path?param1=foo+bar&param2=%232020-01-03%23%2C%232021-02-01%23'
        self.assertEqual(expected, new_url)

        # Without pars
        url = url.split('?')[0]
        self.assertEqual(url, wc.update_url_params(url, {}))

    def test_validate_file_url(self):
        """Test for AlyxClient._validate_file_url."""
        # Should assert that domain matches data server parameter
        with self.assertRaises(AssertionError):
            self.ac._validate_file_url('https://webserver.net/path/to/file')
        # Should check that the domain is equal and return same URL
        expected = self.ac._par.HTTP_DATA_SERVER + '/path/to/file.ext'
        self.assertEqual(self.ac._validate_file_url(expected), expected)
        # Should prepend data server URL
        self.assertEqual(self.ac._validate_file_url('/path/to/file.ext'), expected)

    def test_no_cache_context_manager(self):
        """Test for one.webclient.no_cache function."""
        assert self.ac.cache_mode is not None
        with wc.no_cache(self.ac):
            self.assertIsNone(self.ac.cache_mode)
        self.assertIsNotNone(self.ac.cache_mode)

    def test_cache_dir_setter(self):
        """Tests setter for AlyxClient.cache_dir attribute."""
        prev_path = self.ac.cache_dir
        try:
            self.ac.cache_dir = prev_path / 'foobar'
            self.assertEqual(self.ac.cache_dir, self.ac._par.CACHE_DIR)
            self.assertTrue(str(self.ac.cache_dir).endswith('foobar'))
        finally:
            self.ac._par = self.ac._par.set('CACHE_DIR', prev_path)

    def test_paginated_response(self):
        """Test the _PaginatedResponse class."""
        alyx = mock.Mock(spec_set=self.ac)
        N, lim = 2000, 250  # 2000 results, 250 records per page
        url = self.ac.base_url + f'/?foo=bar&offset={lim}&limit={lim}'
        res = {'count': N, 'next': url, 'previous': None, 'results': []}
        res['results'] = [{'id': i} for i in range(lim)]
        alyx._generic_request.return_value = res
        # Check initialization
        pg = wc._PaginatedResponse(alyx, res, cache_args=dict(clobber=True))
        self.assertEqual(pg.count, N)
        self.assertEqual(len(pg), N)
        self.assertEqual(pg.limit, lim)
        self.assertEqual(len(pg._cache), N)
        self.assertEqual(pg._cache[:lim], res['results'])
        self.assertTrue(not any(pg._cache[lim:]))
        self.assertIs(pg.alyx, alyx)

        # Check adding callbacks
        self.assertRaises(TypeError, pg.add_callback, None)
        wf = mock.Mock(spec_set=weakref.ref)
        cb1, cb2 = mock.MagicMock(), wf()
        pg.add_callback(cb1)
        pg.add_callback(wf)
        self.assertEqual(2, len(pg._callbacks))

        # Check fetching cached item with +ve int
        self.assertEqual({'id': 1}, pg[1])
        alyx._generic_request.assert_not_called()
        for cb in [cb1, cb2]:
            cb.assert_not_called()
        # Check fetching cached item with +ve slice
        self.assertEqual([{'id': 1}, {'id': 2}], pg[1:3])
        alyx._generic_request.assert_not_called()
        for cb in [cb1, cb2]:
            cb.assert_not_called()
        # Check fetching cached item with -ve int
        self.assertEqual({'id': 100}, pg[-1900])
        alyx._generic_request.assert_not_called()
        # Check fetching cached item with -ve slice
        self.assertEqual([{'id': 100}, {'id': 101}], pg[-1900:-1898])
        alyx._generic_request.assert_not_called()
        # Check fetching uncached item with +ve int
        n = offset = lim
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        assert not any(pg._cache[offset : offset + lim])
        self.assertEqual({'id': lim}, pg[n])
        self.assertEqual(res['results'], pg._cache[offset : offset + lim])
        alyx._generic_request.assert_called_once_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        for cb in [cb1, cb2]:
            cb.assert_called_once_with(res['results'])
        # Check that dead weakreaf will be removed from the list on next call
        wf.return_value = None
        # Check fetching uncached item with -ve int
        offset = lim * 3
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        n = offset - N + 2
        assert not any(pg._cache[offset : offset + lim])
        self.assertEqual({'id': N + n}, pg[n])
        self.assertEqual(res['results'], pg._cache[offset : offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        self.assertEqual(1, len(pg._callbacks), 'failed to remove weakref callback')
        # Check fetching uncached item with +ve slice
        offset = lim * 5
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        n = offset + 20
        assert not any(pg._cache[offset : offset + lim])
        self.assertEqual([{'id': n}, {'id': n + 1}], pg[n : n + 2])
        self.assertEqual(res['results'], pg._cache[offset : offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # Check fetching uncached item with -ve slice
        offset = N - lim
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        assert not any(pg._cache[offset : offset + lim])
        self.assertEqual([{'id': N - 2}, {'id': N - 1}], pg[-2:])
        self.assertEqual(res['results'], pg._cache[offset : offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # At this point, there should be a certain number of None values left
        self.assertEqual(expected_calls := 4, alyx._generic_request.call_count)
        self.assertEqual((expected_calls + 1) * lim, sum(list(map(bool, pg._cache))))

        # Check callbacks cleared when cache fully populated
        self.assertTrue(all(map(bool, pg)))
        self.assertEqual(0, len(pg._callbacks))

    def _check_get_query(self, call_args, limit, offset):
        """Check URL get query contains the expected limit and offset params."""
        (_, url), _ = call_args
        self.assertTrue(url.startswith(self.ac.base_url))
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        expected = {'foo': ['bar'], 'offset': [str(offset)], 'limit': [str(limit)]}
        self.assertDictEqual(query, expected)

    def test_json_encoder(self):
        """Test that the JSONEncoder subclass serializes UUID objects."""
        uid = uuid.uuid4()
        # Check encoder subclass behaviour for UUID objects
        self.assertEqual(str(uid), wc._JSONEncoder().default(uid))
        # Encoder should still raise for other object types
        self.assertRaises(TypeError, wc._JSONEncoder().default, b'foo')
        # Using json dumps
        data = {'foo': 12, 'bar': uid}
        # First check that the default encoder raises;
        # python could add support for UUID objects in the future
        self.assertRaises(TypeError, json.dumps, data)
        serialized = json.dumps(data, cls=wc._JSONEncoder)
        expected = '{"foo": 12, "bar": "' + str(uid) + '"}'
        self.assertEqual(expected, serialized)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
