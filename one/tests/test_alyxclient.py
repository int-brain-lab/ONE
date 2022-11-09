"""Unit tests for the one.webclient module"""
import unittest
from unittest import mock
import random
import os
import io
import one.webclient as wc
import one.params
import tempfile
import shutil
import requests
import json
import logging
from datetime import datetime, timedelta
from uuid import UUID

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
        # Test using input args
        ac._par = iopar.from_dict({k: v for k, v in ac._par.as_dict().items()
                                   if k not in login_keys})
        ac.logout()
        with mock.patch('builtins.input') as mock_input:
            ac.authenticate(TEST_DB_2['username'], TEST_DB_2['password'], cache_token=False)
            mock_input.assert_not_called()
        # Check token not saved in cache
        cached_token = getattr(one.params.get(TEST_DB_2['base_url']), 'TOKEN', {})
        self.assertFalse(TEST_DB_2['username'] in cached_token)
        # Test user prompts
        ac.logout()
        ac.silent = False
        with mock.patch('builtins.input', return_value=TEST_DB_2['username']),\
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
        # Check non-silent double logout
        ac.logout()
        ac.logout()  # Shouldn't complain

    def test_auth_methods(self):
        """Test behaviour when calling AlyxClient._generic_request when logged out"""
        # Check that authentication happens when making a logged out request
        self.ac.logout()
        assert not self.ac.is_logged_in
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
        assert not self.ac.is_logged_in
        url = self.ac.get('cache/info')['location']
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
        with mock.patch('one.webclient.requests.post', side_effect=requests.ConnectionError),\
             self.assertRaises(ConnectionError) as ex:
            self.ac.authenticate()
            self.assertTrue('firewall' in str(ex))
        # Check behaviour when server error occurs
        rep = requests.Response()
        rep.status_code = 500
        with mock.patch('one.webclient.requests.post', return_value=rep),\
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

    def test_paginated_request(self):
        """Check that paginated response object is returned upon making large queries"""
        rep = self.ac.rest('datasets', 'list')
        self.assertTrue(isinstance(rep, one.webclient._PaginatedResponse))
        self.assertTrue(len(rep) > 250)
        # This fails when new records are added/removed from the remote db while iterating
        # self.assertTrue(len([_ for _ in rep]) == len(rep))

        # Test what happens when list changes between paginated requests
        name = '0A' + str(random.randint(0, 10000))
        # Add subject between calls
        rep = self.ac.rest('subjects', 'list', limit=5, no_cache=True)
        s = self.ac.rest('subjects', 'create', data={'nickname': name, 'lab': 'cortexlab'})
        self.addCleanup(self.ac.rest, 'subjects', 'delete', id=s['nickname'])
        with self.assertWarns(RuntimeWarning):
            _ = rep[10]

    def test_update_url_params(self):
        url = f'{self.ac.base_url}/sessions?param1=foo&param2=&limit=5&param3=bar'
        expected = f'{self.ac.base_url}/sessions?param1=foo&limit=10&param3=bar&param4=baz'
        self.assertEqual(expected, wc.update_url_params(url, {'limit': 10, 'param4': 'baz'}))
        # Without pars
        url = url.split('?')[0]
        self.assertEqual(url, wc.update_url_params(url, {}))
        # With lists
        expected = f'{url}?foo=bar&foo=baz'
        self.assertEqual(expected, wc.update_url_params(url, {'foo': ['bar', 'baz']}))

    def test_generic_request(self):
        a = self.ac.get('/labs')
        b = self.ac.get('labs')
        self.assertEqual(a, b)

    def test_rest_endpoint_write(self):
        # test object creation and deletion with weighings
        wa = {'subject': 'flowers',
              'date_time': '2018-06-30T12:34:57',
              'weight': 22.2,
              'user': 'olivier'
              }
        a = self.ac.rest('weighings', 'create', data=wa)
        b = self.ac.rest('weighings', 'read', id=a['url'])
        self.assertEqual(a, b)
        self.ac.rest('weighings', 'delete', id=a['url'])
        # test patch object with subjects
        data = {'birth_date': '2018-04-01',
                'death_date': '2018-09-10'}
        sub = self.ac.rest('subjects', 'partial_update', id='flowers', data=data)
        self.assertEqual(sub['birth_date'], data['birth_date'])
        self.assertEqual(sub['death_date'], data['death_date'])
        data = {'birth_date': '2018-04-02',
                'death_date': '2018-09-09'}
        sub = self.ac.rest('subjects', 'partial_update', id='flowers', data=data)
        self.assertEqual(sub['birth_date'], data['birth_date'])
        self.assertEqual(sub['death_date'], data['death_date'])

    def test_rest_endpoint_read_only(self):
        """Test AlyxClient.rest method with 'list' and 'read' actions"""
        # tests that non-existing endpoints /actions are caught properly
        with self.assertRaises(ValueError):
            self.ac.rest(url='turlu', action='create')
        with self.assertRaises(ValueError):
            self.ac.rest(url='sessions', action='turlu')
        # test with labs : get
        a = self.ac.rest('labs', 'list')
        self.assertTrue(len(a) >= 3)
        b = self.ac.rest('/labs', 'list')
        self.assertTrue(a == b)
        # test with labs: read
        c = self.ac.rest('labs', 'read', 'mainenlab')
        self.assertTrue([lab for lab in a if
                         lab['name'] == 'mainenlab'][0] == c)
        # test read with UUID object
        dset = self.ac.rest('datasets', 'read', id=UUID('738eca6f-d437-40d6-a9b8-a3f4cbbfbff7'))
        self.assertEqual(dset['name'], '_iblrig_videoCodeFiles.raw.zip')
        # Test with full URL
        d = self.ac.rest(
            'labs', 'read',
            f'{TEST_DB_1["base_url"]}/labs/mainenlab')
        self.assertEqual(c, d)
        # test a more complex endpoint with a filter and a selection
        sub = self.ac.rest('subjects/flowers', 'list')
        sub1 = self.ac.rest('subjects?nickname=flowers', 'list')
        self.assertTrue(len(sub1) == 1)
        self.assertEqual(sub['nickname'], sub1[0]['nickname'])
        # also make sure the action is overriden on a filter query
        sub2 = self.ac.rest('/subjects?nickname=flowers')
        self.assertEqual(sub1, sub2)

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

    def test_rest_all_actions(self):
        """Test for AlyxClient.rest method using subjects endpoint"""
        # randint reduces conflicts with parallel tests
        nickname = f'foobar_{random.randint(0, 10000)}'
        newsub = {
            'nickname': nickname,
            'responsible_user': 'olivier',
            'birth_date': '2019-06-15',
            'death_date': None,
            'lab': 'cortexlab',
        }
        # look for the subject, create it if necessary
        sub = self.ac.get(f'/subjects?&nickname={nickname}', expires=True)
        if sub:
            self.ac.rest('subjects', 'delete', id=nickname)
        self.ac.rest('subjects', 'create', data=newsub)
        # partial update and full update
        newsub = self.ac.rest('subjects', 'partial_update',
                              id=nickname, data={'description': 'hey'})
        self.assertEqual(newsub['description'], 'hey')
        newsub['description'] = 'hoy'
        newsub = self.ac.rest('subjects', 'update', id=nickname, data=newsub)
        self.assertEqual(newsub['description'], 'hoy')
        # read
        newsub_ = self.ac.rest('subjects', 'read', id=nickname)
        self.assertEqual(newsub, newsub_)
        # list with filter
        sub = self.ac.rest('subjects', 'list', nickname=nickname)
        self.assertEqual(sub[0]['nickname'], newsub['nickname'])
        self.assertTrue(len(sub) == 1)
        # delete
        self.ac.rest('subjects', 'delete', id=nickname)
        self.ac.clear_rest_cache()  # Make sure we hit db
        sub = self.ac.get(f'/subjects?&nickname={nickname}', expires=True)
        self.assertFalse(sub)

    def test_endpoints_docs(self):
        """Test for AlyxClient.list_endpoints method and AlyxClient.rest"""
        # Test endpoint documentation and validation
        endpoints = self.ac.list_endpoints()
        self.assertTrue('auth-token' not in endpoints)
        # Check that calling rest method with no args prints endpoints
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.ac.rest()
            self.assertTrue(k in stdout.getvalue() for k in endpoints)
        # Same but with no action
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.ac.rest('sessions')
            actions = self.ac.rest_schemes['sessions'].keys()
            self.assertTrue(all(k in stdout.getvalue() for k in actions))
        # Check logs warning when no id provided
        with self.assertLogs(logging.getLogger('one.webclient'), logging.WARNING):
            self.assertIsNone(self.ac.rest('sessions', 'read'))
        # Check logs warning when creating record with missing data
        with self.assertLogs(logging.getLogger('one.webclient'), logging.WARNING):
            self.assertIsNone(self.ac.rest('sessions', 'create'))
        with self.assertRaises(ValueError) as e:
            self.ac.json_field_write('foobar')
        self.assertTrue(k in str(e.exception) for k in endpoints)


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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
