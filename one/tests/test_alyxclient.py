"""Unit tests for the one.webclient module"""
import unittest
from unittest import mock
import urllib.parse
import random
import os
import one.webclient as wc
import one.params
import tempfile
import shutil
import requests
import json
import logging
from datetime import datetime, timedelta

from iblutil.io import hashfile
import iblutil.io.params as iopar

from . import OFFLINE_ONLY, TEST_DB_1, TEST_DB_2
from . import util

par = one.params.get(silent=True)

# Init connection to the database
ac = wc.AlyxClient(**TEST_DB_1)
# Remove /public from data server url
if 'public' in ac._par.HTTP_DATA_SERVER:
    ac._par = ac._par.set('HTTP_DATA_SERVER', ac._par.HTTP_DATA_SERVER.rsplit('/', 1)[0])


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestAuthentication(unittest.TestCase):
    """Tests for AlyxClient authentication, token storage, login/out methods and user prompts"""
    def setUp(self) -> None:
        self.ac = wc.AlyxClient(**TEST_DB_2)

    def test_authentication(self):
        """Test for AlyxClient.authenticate and AlyxClient.is_logged_in property"""
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
        ac._par = iopar.from_dict({k: v for k, v in ac._par.as_dict().items()
                                   if k not in login_keys})
        with mock.patch('builtins.input') as mock_input:
            ac.authenticate(TEST_DB_2['username'], TEST_DB_2['password'], cache_token=False)
            mock_input.assert_not_called()
        # Check token not saved in cache
        cached_token = getattr(one.params.get(TEST_DB_2['base_url']), 'TOKEN', {})
        self.assertFalse(TEST_DB_2['username'] in cached_token)
        # Test user prompts
        ac.logout()
        ac.silent = False
        with mock.patch('builtins.input', return_value=TEST_DB_2['username']), \
             mock.patch('one.webclient.getpass', return_value=TEST_DB_2['password']):
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
        """Test behaviour when calling AlyxClient._generic_request when logged out"""
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
        """Test behaviour when authentication fails"""
        self.ac.logout()  # Make sure logged out
        with self.assertRaises(requests.HTTPError) as ex:
            self.ac.authenticate(password='wrong_pass')
            self.assertTrue('user = intbrainlab' in str(ex))
            self.assertFalse('wrong_pass' in str(ex))
        # Check behaviour when connection error occurs (should mention firewall settings)
        with mock.patch('one.webclient.requests.post', side_effect=requests.ConnectionError), \
             self.assertRaises(ConnectionError) as ex:
            self.ac.authenticate()
            self.assertTrue('firewall' in str(ex))
        # Check behaviour when server error occurs
        rep = requests.Response()
        rep.status_code = 500
        with mock.patch('one.webclient.requests.post', return_value=rep), \
             self.assertRaises(requests.HTTPError):
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
        sessions = [self.ac.rest('sessions', 'create', data={
            'subject': name,
            'start_time': datetime.isoformat(datetime.now()),
            'number': random.randint(1, 999),
            'type': 'Experiment',
            'users': [TEST_DB_1['username']],
        }) for _ in range(2)]

        self.eids = [x['url'].split('/')[-1] for x in sessions]
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
        """Test for AlyxClient.json_field* methods (write, update, remove_key and delete)"""
        self._json_field_write()
        self._json_field_update()
        self._json_field_remove_key()
        self._json_field_delete()

    def test_empty(self):
        """Test for AlyxClient.json_field* methods when JSON field is empty"""
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

    def tearDown(self):
        self.ac.rest('subjects', 'delete', id=self.subj['nickname'])


class TestRestCache(unittest.TestCase):
    """Tests for REST caching system, the cache decorator and cache flags"""
    def setUp(self):
        util.setup_test_params()  # Ensure test alyx set up
        util.setup_rest_cache(ac.cache_dir)  # Copy rest cache fixtures
        self.query = '/insertions/b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        self.tempdir = util.set_up_env()
        self.addCleanup(self.tempdir.cleanup)
        one.webclient.datetime = _FakeDateTime
        _FakeDateTime._now = None
        self.cache_dir = ac.cache_dir.joinpath('.rest')
        self.default_expiry = ac.default_expiry
        self.cache_mode = ac.cache_mode

    def test_loads_cached(self):
        """Test for one.webclient._cache_response decorator, checks returns cached result"""
        # Check returns cache
        wrapped = wc._cache_response(lambda *args: self.assertTrue(False))
        client = ac  # Bunch({'base_url': 'https://test.alyx.internationalbrainlab.org'})
        res = wrapped(client, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

    def test_expired_cache(self):
        """Test behaviour when cached REST query is expired"""
        # Checks expired
        wrapped = wc._cache_response(lambda *args: 'called')
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')
        res = wrapped(ac, requests.get, self.query)
        self.assertTrue(res == 'called')

    def test_caches_response(self):
        """Test caches query response before returning"""
        # Default expiry time
        ac.default_expiry = timedelta(minutes=1)
        wrapped = wc._cache_response(lambda *args: 'called')
        _FakeDateTime._now = datetime(2021, 5, 13)  # Freeze time
        res = wrapped(ac, requests.get, '/endpoint?id=5')
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
        """Test for AlyxClient.cache_mode property"""
        # With cache mode off, wrapped method should be called even in presence of valid cache
        ac.cache_mode = None  # cache nothing
        wrapped = wc._cache_response(lambda *args: 'called')
        res = wrapped(ac, requests.get, self.query)
        self.assertTrue(res == 'called')

    def test_expiry_param(self):
        """Test for expires kwarg in one.webclient._cache_response decorator"""
        # Check expiry param
        wrapped = wc._cache_response(lambda *args: '123')
        res = wrapped(ac, requests.get, '/endpoint?id=5', expires=True)
        self.assertTrue(res == '123')

        # A second call should yield a new response as cache immediately expired
        wrapped = wc._cache_response(lambda *args: '456')
        res = wrapped(ac, requests.get, '/endpoint?id=5', expires=False)
        self.assertTrue(res == '456')

        # With clobber=True the cache should be overwritten
        wrapped = wc._cache_response(lambda *args: '789')
        res = wrapped(ac, requests.get, '/endpoint?id=5', clobber=True)
        self.assertTrue(res == '789')

    def test_cache_returned_on_error(self):
        """Test behaviour when connection error occurs and cached response exists"""
        func = mock.Mock(side_effect=requests.ConnectionError())
        wrapped = wc._cache_response(func)
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')  # Expired
        with self.assertWarns(RuntimeWarning):
            res = wrapped(ac, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

        # With clobber=True exception should be raised
        with self.assertRaises(requests.ConnectionError):
            wrapped(ac, requests.get, self.query, clobber=True)

    def test_clear_cache(self):
        """Test for AlyxClient.clear_rest_cache"""
        assert any(self.cache_dir.glob('*'))
        ac.clear_rest_cache()
        self.assertFalse(any(self.cache_dir.glob('*')))

    def tearDown(self) -> None:
        ac.cache_mode = self.cache_mode
        ac.default_expiry = self.default_expiry


class _FakeDateTime(datetime):
    _now = None

    @staticmethod
    def now(*args, **kwargs):
        return _FakeDateTime._now or datetime.now(*args, **kwargs)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestDownloadHTTP(unittest.TestCase):

    def setUp(self):
        self.ac = ac
        self.test_data_uuid = '40af4a49-1b9d-45ec-b443-a151c010ea3c'  # OpenAlyx dataset

    def test_download_datasets_with_api(self):
        ac_public = wc.AlyxClient(**TEST_DB_2)
        cache_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(cache_dir))

        # Test 1: empty dir, dict mode
        dset = ac_public.get('/datasets/' + self.test_data_uuid)
        urls = wc.dataset_record_to_url(dset)
        url = [u for u in urls if u.startswith('https://ibl.flatiron')]
        file_name, = ac_public.download_file(url, target_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 2: empty dir, list mode
        dset = ac_public.get('/datasets?id=' + self.test_data_uuid)
        urls = wc.dataset_record_to_url(dset)
        url = [u for u in urls if u.startswith('https://ibl.flatiron')]
        file_name, = ac_public.download_file(url, target_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 3: Log unauthorized error with url (using test alyx)
        url = next(x['data_url'] for x in ac.get('/datasets?exists=True')[0]['file_records'])
        old_par = ac._par
        ac._par = ac._par.set('HTTP_DATA_SERVER_PWD', 'foobar')
        with self.assertLogs(logging.getLogger('one.webclient'), logging.ERROR) as log:
            raised = False
            try:
                ac.download_file(url, target_dir=cache_dir)
                self.assertTrue(url in log.output[-1])
            except Exception as ex:
                # Check error message mentions the HTTP_DATA_SERVER params
                self.assertTrue('HTTP_DATA_SERVER_PWD' in str(ex))
                raised = True
            finally:
                self.assertTrue(raised)
                ac._par = old_par

    def test_download_datasets(self):
        # test downloading a single file
        full_link_to_file = (
            'https://ibl.flatironinstitute.org/public/mrsicflogellab/Subjects/SWC_038/'
            '2020-07-29/001/alf/probes.description.f67570ac-1e54-4ce1-be5d-de2017a42116.json'
        )
        file_name, md5 = wc.http_download_file(full_link_to_file,
                                               return_md5=True, clobber=True)
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        self.assertTrue(len(data) > 0)
        self.assertTrue(hashfile.md5(file_name) == md5)

        self.assertFalse(wc.http_download_file('', clobber=True))

        # test downloading a list of files
        links = [full_link_to_file,
                 r'https://ibl.flatironinstitute.org/public/hoferlab/Subjects/SWC_043/'
                 r'2020-09-21/001/alf/probes.description.c4df1eea-c92c-479f-a907-41fa6e770094.json'
                 ]
        file_list = wc.http_download_file_list(links, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD)
        for file in file_list:
            with open(file, 'r') as json_file:
                data = json.load(json_file)
            self.assertTrue(len(data) > 0)


class TestMisc(unittest.TestCase):
    def test_update_url_params(self):
        """Test for one.webclient.update_url_params"""
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
        """Test for AlyxClient._validate_file_url"""
        # Should assert that domain matches data server parameter
        with self.assertRaises(AssertionError):
            ac._validate_file_url('https://webserver.net/path/to/file')
        # Should check that the domain is equal and return same URL
        expected = ac._par.HTTP_DATA_SERVER + '/path/to/file.ext'
        self.assertEqual(ac._validate_file_url(expected), expected)
        # Should prepend data server URL
        self.assertEqual(ac._validate_file_url('/path/to/file.ext'), expected)

    def test_no_cache_context_manager(self):
        """Test for one.webclient.no_cache function"""
        assert ac.cache_mode is not None
        with wc.no_cache(ac):
            self.assertIsNone(ac.cache_mode)
        self.assertIsNotNone(ac.cache_mode)

    def test_cache_dir_setter(self):
        """Tests setter for AlyxClient.cache_dir attribute."""
        prev_path = ac.cache_dir
        try:
            ac.cache_dir = prev_path / 'foobar'
            self.assertEqual(ac.cache_dir, ac._par.CACHE_DIR)
            self.assertTrue(str(ac.cache_dir).endswith('foobar'))
        finally:
            ac._par = ac._par.set('CACHE_DIR', prev_path)

    def test_paginated_response(self):
        """Test the _PaginatedResponse class."""
        alyx = mock.Mock(spec_set=ac)
        N, lim = 2000, 250  # 2000 results, 250 records per page
        url = ac.base_url + f'/?foo=bar&offset={lim}&limit={lim}'
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

        # Check fetching cached item with +ve int
        self.assertEqual({'id': 1}, pg[1])
        alyx._generic_request.assert_not_called()
        # Check fetching cached item with +ve slice
        self.assertEqual([{'id': 1}, {'id': 2}], pg[1:3])
        alyx._generic_request.assert_not_called()
        # Check fetching cached item with -ve int
        self.assertEqual({'id': 100}, pg[-1900])
        alyx._generic_request.assert_not_called()
        # Check fetching cached item with -ve slice
        self.assertEqual([{'id': 100}, {'id': 101}], pg[-1900:-1898])
        alyx._generic_request.assert_not_called()
        # Check fetching uncached item with +ve int
        n = offset = lim
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        assert not any(pg._cache[offset:offset + lim])
        self.assertEqual({'id': lim}, pg[n])
        self.assertEqual(res['results'], pg._cache[offset:offset + lim])
        alyx._generic_request.assert_called_once_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # Check fetching uncached item with -ve int
        offset = lim * 3
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        n = offset - N + 2
        assert not any(pg._cache[offset:offset + lim])
        self.assertEqual({'id': N + n}, pg[n])
        self.assertEqual(res['results'], pg._cache[offset:offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # Check fetching uncached item with +ve slice
        offset = lim * 5
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        n = offset + 20
        assert not any(pg._cache[offset:offset + lim])
        self.assertEqual([{'id': n}, {'id': n + 1}], pg[n:n + 2])
        self.assertEqual(res['results'], pg._cache[offset:offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # Check fetching uncached item with -ve slice
        offset = N - lim
        res['results'] = [{'id': i} for i in range(offset, offset + lim)]
        assert not any(pg._cache[offset:offset + lim])
        self.assertEqual([{'id': N - 2}, {'id': N - 1}], pg[-2:])
        self.assertEqual(res['results'], pg._cache[offset:offset + lim])
        alyx._generic_request.assert_called_with(requests.get, mock.ANY, clobber=True)
        self._check_get_query(alyx._generic_request.call_args, lim, offset)
        # At this point, there should be a certain number of None values left
        self.assertEqual(expected_calls := 4, alyx._generic_request.call_count)
        self.assertEqual((expected_calls + 1) * lim, sum(list(map(bool, pg._cache))))

    def _check_get_query(self, call_args, limit, offset):
        """Check URL get query contains the expected limit and offset params."""
        (_, url), _ = call_args
        self.assertTrue(url.startswith(ac.base_url))
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        expected = {'foo': ['bar'], 'offset': [str(offset)], 'limit': [str(limit)]}
        self.assertDictEqual(query, expected)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
