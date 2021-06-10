import unittest
from unittest import mock
from pathlib import Path
import random
import numpy as np
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
from . import OFFLINE_ONLY
from . import util

par = one.params.get()

# Init connection to the database
ac = wc.AlyxClient(
    username='test_user', password='TapetesBloc18',
    base_url='https://test.alyx.internationalbrainlab.org')


class TestSingletonPattern(unittest.TestCase):
    def setUp(self):
        self.ac = ac
        self.sameac = wc.AlyxClient(
            username='test_user',
            password='TapetesBloc18',
            base_url='https://test.alyx.internationalbrainlab.org')
        self.sameac2 = wc.AlyxClient(
            username='test_user',
            password='TapetesBloc18',
            base_url='https://test.alyx.internationalbrainlab.org')

    def test_multiple_singletons(self):
        self.assertTrue(id(self.ac) == id(self.sameac))
        self.assertTrue(id(self.ac) == id(self.sameac2))


class TestJsonFieldMethods(unittest.TestCase):
    def setUp(self):
        self.ac = ac
        self.eid1 = '242f2929-faaf-4e7c-ae3f-4a935c6d8da5'
        self.eid2 = 'dfe99506-b873-45db-bc93-731f9362e304'
        self.endpoint = 'sessions'
        self.field_name = 'extended_qc'
        self.data_dict = {'some': 0, 'data': 1}
        self.ac.json_field_delete(self.endpoint, self.eid1, self.field_name)
        self.ac.json_field_delete(self.endpoint, self.eid2, self.field_name)
        self.eid1_eqc = None
        self.eid2_eqc = None
        self.ac.clear_rest_cache()

    def _json_field_write(self):
        written1 = self.ac.json_field_write(
            self.endpoint, self.eid1, self.field_name, self.data_dict
        )
        written2 = self.ac.json_field_write(
            self.endpoint, self.eid2, self.field_name, self.data_dict
        )
        self.assertTrue(written1 == written2)
        self.assertTrue(written1 == self.data_dict)
        url = f'/{self.endpoint}?&extended_qc=some__lt,0.5'
        sess_dict = self.ac.get(url, expires=True)
        self.assertTrue(len(sess_dict) == 2)

    def _json_field_update(self):
        modified = self.ac.json_field_update(
            self.endpoint, self.eid1, self.field_name, {'some': 0.6}
        )
        self.assertTrue('data' in modified)
        self.assertTrue('some' in modified)
        url = f'/{self.endpoint}?&extended_qc=some__lt,0.5'
        self.assertTrue(len(self.ac.get(url, expires=True)) == 1)

    def _json_field_remove_key(self):
        url = f'/{self.endpoint}?&extended_qc=data__gte,0.5'
        pre_delete = self.ac.get(url, expires=True)
        self.assertTrue(len(pre_delete) == 2)
        deleted = self.ac.json_field_remove_key(self.endpoint, self.eid2, self.field_name, 'data')
        self.assertTrue('data' not in deleted)
        post_delete = self.ac.get(url, expires=True)
        self.assertTrue(len(post_delete) == 1)

    def _json_field_delete(self):
        deleted = self.ac.json_field_delete(self.endpoint, self.eid2, self.field_name)
        self.assertTrue(deleted is None)
        ses = self.ac.get(f'/{self.endpoint}?&extended_qc=data__gte,0.5', expires=True)
        self.assertTrue(len(ses) == 1)

    def test_json_methods(self):
        self._json_field_write()
        self._json_field_update()
        self._json_field_remove_key()
        self._json_field_delete()

    def tearDown(self):
        # Delete any dict created by this test
        url = f'/{self.endpoint}?&extended_qc=some__lt,0.5'
        for x in self.ac.get(url, expires=True):
            self.ac.json_field_delete(self.endpoint, x['url'][-36:], self.field_name)
        # Restore whatever was there in the first place
        self.ac.json_field_write(self.endpoint, self.eid1, self.field_name, self.eid1_eqc)
        self.ac.json_field_write(self.endpoint, self.eid2, self.field_name, self.eid2_eqc)


class TestRestCache(unittest.TestCase):
    def setUp(self):
        util.setup_test_params()  # Ensure test alyx set up
        util.setup_rest_cache()  # Copy rest cache fixtures
        self.query = '/insertions/b529f2d8-cdae-4d59-aba2-cbd1b5572e36'
        self.tempdir = util.set_up_env()
        self.addCleanup(self.tempdir.cleanup)
        one.webclient.datetime = _FakeDateTime
        _FakeDateTime._now = None
        path_parts = ('.rest', 'test.alyx.internationalbrainlab.org', 'https')
        self.cache_dir = Path(one.params.get_params_dir()).joinpath(*path_parts)

    def test_loads_cached(self):
        # Check returns cache
        wrapped = wc.cache_response(lambda *args: self.assertTrue(False))
        client = ac  # Bunch({'base_url': 'https://test.alyx.internationalbrainlab.org'})
        res = wrapped(client, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

    def test_expired_cache(self):
        # Checks expired
        wrapped = wc.cache_response(lambda *args: 'called')
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')
        res = wrapped(ac, requests.get, self.query)
        self.assertTrue(res == 'called')

    def test_caches_response(self):
        # Default expiry time
        dt = timedelta(minutes=1)
        wrapped = wc.cache_response(lambda *args: 'called', default_expiry=dt)
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

    def test_expiry_param(self):
        # Check expiry param
        wrapped = wc.cache_response(lambda *args: '123')
        res = wrapped(ac, requests.get, '/endpoint?id=5', expires=True)
        self.assertTrue(res == '123')

        # A second call should yield a new response as cache immediately expired
        wrapped = wc.cache_response(lambda *args: '456')
        res = wrapped(ac, requests.get, '/endpoint?id=5', expires=False)
        self.assertTrue(res == '456')

        # With clobber=True the cache should be overwritten
        wrapped = wc.cache_response(lambda *args: '789')
        res = wrapped(ac, requests.get, '/endpoint?id=5', clobber=True)
        self.assertTrue(res == '789')

    def test_cache_returned_on_error(self):
        func = mock.Mock(side_effect=requests.ConnectionError())
        wrapped = wc.cache_response(func)
        _FakeDateTime._now = datetime.fromisoformat('3001-01-01')  # Expired
        with self.assertWarns(RuntimeWarning):
            res = wrapped(ac, requests.get, self.query)
        self.assertEqual(res['id'], self.query.split('/')[-1])

        # With clobber=True exception should be raised
        with self.assertRaises(requests.ConnectionError):
            wrapped(ac, requests.get, self.query, clobber=True)

    def test_clear_cache(self):
        assert any(self.cache_dir.glob('*'))
        ac.clear_rest_cache()
        self.assertFalse(any(self.cache_dir.glob('*')))


class _FakeDateTime(datetime):
    _now = None

    @staticmethod
    def now(*args, **kwargs):
        return _FakeDateTime._now or datetime.now(*args, **kwargs)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestDownloadHTTP(unittest.TestCase):

    def setUp(self):
        self.ac = ac
        self.test_data_uuid = '3ddd45be-7d24-4fc7-9dd3-a98717342af6'

    def test_paginated_request(self):
        rep = self.ac.rest('datasets', 'list')
        self.assertTrue(isinstance(rep, one.webclient._PaginatedResponse))
        self.assertTrue(len(rep) > 250)
        self.assertTrue(len([d['hash'] for d in rep]) == len(rep))

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
        sub = self.ac.rest('subjects', 'read', id='flowers')
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
        d = self.ac.rest(
            'labs', 'read',
            'https://test.alyx.internationalbrainlab.org/labs/mainenlab')
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
        ac = self.ac  # easier to debug in console
        test_data_uuid = self.test_data_uuid
        cache_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(cache_dir))

        # Test 1: empty dir, dict mode
        dset = ac.get('/datasets/' + test_data_uuid)
        url = wc.dataset_record_to_url(dset)
        file_name, = ac.download_file(url, cache_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 2: empty dir, list mode
        dset = ac.get('/datasets?id=' + test_data_uuid)
        url = wc.dataset_record_to_url(dset)
        file_name, = ac.download_file(url, cache_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name))
        os.unlink(file_name)

        # Test 3: Log unauthorized error with url
        old_par = ac._par
        ac._par = ac._par.set('HTTP_DATA_SERVER_PWD', 'foobar')
        with self.assertLogs(logging.getLogger('one.webclient'), logging.ERROR) as log:
            try:
                ac.download_file(url, cache_dir=cache_dir)
                self.assertTrue(url[0] in log.output[-1])
            except Exception:
                pass
            finally:
                ac._par = old_par

    def test_download_datasets(self):
        # test downloading a single file
        full_link_to_file = r'http://ibl.flatironinstitute.org/mainenlab/Subjects/clns0730'\
                            '/2018-08-24/1/licks.times.51852a2f-c76e-4c0c-95cb-9c7ba54be0f9.npy'
        file_name, md5 = wc.http_download_file(full_link_to_file,
                                               username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               return_md5=True, clobber=True)
        a = np.load(file_name)
        self.assertTrue(hashfile.md5(file_name) == md5)
        self.assertTrue(len(a) > 0)

        # test downloading a list of files
        links = [r'http://ibl.flatironinstitute.org/mainenlab/Subjects/clns0730'
                 '/2018-08-24/1/licks.times.51852a2f-c76e-4c0c-95cb-9c7ba54be0f9.npy',
                 r'http://ibl.flatironinstitute.org/mainenlab/Subjects/clns0730'
                 '/2018-08-24/1/probes.sitePositions.3ddd45be-7d24-4fc7-9dd3-a98717342af6.npy'
                 ]
        file_list = wc.http_download_file_list(links, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD)
        a = np.load(file_list[0])
        b = np.load(file_list[1])
        self.assertTrue(len(a) > 0)
        self.assertTrue(len(b) > 0)

    def test_rest_all_actions(self):
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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
