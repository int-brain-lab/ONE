"""Unit tests for Alyx REST queries using the AlyxClient.rest method."""
from pathlib import Path
import unittest
import unittest.mock
import random
import string
from uuid import UUID
import io
from logging import WARNING

import numpy as np

from one.webclient import AlyxClient, _PaginatedResponse
from one.tests import TEST_DB_1, OFFLINE_ONLY


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestREST(unittest.TestCase):
    """Tests for AlyxClient.rest method and remote Alyx REST interactions."""
    EID = 'cf264653-2deb-44cb-aa84-89b82507028a'
    EID_EPHYS = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
    alyx = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.alyx = AlyxClient(**TEST_DB_1)

    def test_paginated_request(self):
        """Check that paginated response object is returned upon making large queries."""
        rep = self.alyx.rest('datasets', 'list')
        self.assertTrue(isinstance(rep, _PaginatedResponse))
        self.assertTrue(len(rep) > 250)
        # This fails when new records are added/removed from the remote db while iterating
        # self.assertTrue(len([_ for _ in rep]) == len(rep))

        # Test what happens when list changes between paginated requests
        name = '0A' + str(random.randint(0, 10000))
        # Add subject between calls
        rep = self.alyx.rest('subjects', 'list', limit=5, no_cache=True)
        s = self.alyx.rest('subjects', 'create', data={'nickname': name, 'lab': 'cortexlab'})
        self.addCleanup(self.alyx.rest, 'subjects', 'delete', id=s['nickname'])
        with self.assertWarns(RuntimeWarning):
            _ = rep[10]

    def test_generic_request(self):
        """Test AlyxClient.get method."""
        a = self.alyx.get('/labs')
        b = self.alyx.get('labs')
        self.assertEqual(a, b)

    def test_rest_endpoint_write(self):
        """Test create and delete actions in AlyxClient.rest method."""
        # test object creation and deletion with weighings
        wa = {'subject': 'flowers',
              'date_time': '2018-06-30T12:34:57',
              'weight': 22.2,
              'user': 'olivier'
              }
        a = self.alyx.rest('weighings', 'create', data=wa)
        b = self.alyx.rest('weighings', 'read', id=a['url'])
        self.assertEqual(a, b)
        self.alyx.rest('weighings', 'delete', id=a['url'])
        # test patch object with subjects
        data = {'birth_date': '2018-04-01',
                'death_date': '2018-09-10'}
        sub = self.alyx.rest('subjects', 'partial_update', id='flowers', data=data)
        self.assertEqual(sub['birth_date'], data['birth_date'])
        self.assertEqual(sub['death_date'], data['death_date'])
        data = {'birth_date': '2018-04-02',
                'death_date': '2018-09-09'}
        sub = self.alyx.rest('subjects', 'partial_update', id='flowers', data=data)
        self.assertEqual(sub['birth_date'], data['birth_date'])
        self.assertEqual(sub['death_date'], data['death_date'])

    def test_rest_endpoint_read_only(self):
        """Test list and read actions in AlyxClient.rest method."""
        # tests that non-existing endpoints /actions are caught properly
        with self.assertRaises(ValueError):
            self.alyx.rest(url='turlu', action='create')
        with self.assertRaises(ValueError):
            self.alyx.rest(url='sessions', action='turlu')
        # test with labs : get
        a = self.alyx.rest('labs', 'list')
        self.assertTrue(len(a) >= 3)
        b = self.alyx.rest('/labs', 'list')
        self.assertTrue(a == b)
        # test with labs: read
        c = self.alyx.rest('labs', 'read', 'mainenlab')
        self.assertTrue([lab for lab in a if lab['name'] == 'mainenlab'][0] == c)
        # test read with UUID object
        dset = self.alyx.rest('datasets', 'read', id=UUID('738eca6f-d437-40d6-a9b8-a3f4cbbfbff7'))
        self.assertEqual(dset['name'], '_iblrig_videoCodeFiles.raw.zip')
        # Test with full URL
        d = self.alyx.rest(
            'labs', 'read',
            f'{TEST_DB_1["base_url"]}/labs/mainenlab')
        self.assertEqual(c, d)
        # test a more complex endpoint with a filter and a selection
        sub = self.alyx.rest('subjects/flowers', 'list')
        sub1 = self.alyx.rest('subjects?nickname=flowers', 'list')
        self.assertTrue(len(sub1) == 1)
        self.assertEqual(sub['nickname'], sub1[0]['nickname'])
        # also make sure the action is overriden on a filter query
        sub2 = self.alyx.rest('/subjects?nickname=flowers')
        self.assertEqual(sub1, sub2)

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
        sub = self.alyx.get(f'/subjects?&nickname={nickname}', expires=True)
        if sub:
            self.alyx.rest('subjects', 'delete', id=nickname)
        self.alyx.rest('subjects', 'create', data=newsub)
        # partial update and full update
        newsub = self.alyx.rest('subjects', 'partial_update',
                                id=nickname, data={'description': 'hey'})
        self.assertEqual(newsub['description'], 'hey')
        newsub['description'] = 'hoy'
        newsub = self.alyx.rest('subjects', 'update', id=nickname, data=newsub)
        self.assertEqual(newsub['description'], 'hoy')
        # read
        newsub_ = self.alyx.rest('subjects', 'read', id=nickname)
        self.assertEqual(newsub, newsub_)
        # list with filter
        sub = self.alyx.rest('subjects', 'list', nickname=nickname)
        self.assertEqual(sub[0]['nickname'], newsub['nickname'])
        self.assertTrue(len(sub) == 1)
        # delete
        self.alyx.rest('subjects', 'delete', id=nickname)
        self.alyx.clear_rest_cache()  # Make sure we hit db
        sub = self.alyx.get(f'/subjects?&nickname={nickname}', expires=True)
        self.assertFalse(sub)

    def test_endpoints_docs(self):
        """Test for AlyxClient.list_endpoints method and AlyxClient.rest"""
        # Test endpoint documentation and validation
        endpoints = self.alyx.list_endpoints()
        self.assertTrue('auth-token' not in endpoints)
        # Check that calling rest method with no args prints endpoints
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.alyx.rest()
            self.assertTrue(k in stdout.getvalue() for k in endpoints)
        # Same but with no action
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.assertIsNone(self.alyx.rest('sessions'))
            actions = self.alyx.rest_schemes['sessions'].keys()
            self.assertTrue(all(k in stdout.getvalue() for k in actions))
            expected = "['list', 'create', 'read', 'update', 'partial_update', 'delete']\n"
            self.assertEqual(expected, stdout.getvalue()[:65])
        # Check raises when endpoint invalid
        self.assertRaises(ValueError, self.alyx.rest, 'foobar')
        # Check logs warning when no id provided
        with self.assertLogs('one.webclient', WARNING):
            self.assertIsNone(self.alyx.rest('sessions', 'read'))
        # Check logs warning when creating record with missing data
        with self.assertLogs('one.webclient', WARNING):
            self.assertIsNone(self.alyx.rest('sessions', 'create'))
        with self.assertRaises(ValueError) as e:
            self.alyx.json_field_write('foobar')
        self.assertTrue(k in str(e.exception) for k in endpoints)

    def test_print_endpoint_info(self):
        """Test endpoint query params are printed when calling AlyxClient.rest without action."""
        # Check behaviour when endpoint invalid
        endpoint = 'foobar'
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            self.assertIsNone(self.alyx.print_endpoint_info(endpoint))
            self.assertRegex(stdout.getvalue(), f'"{endpoint}" does not exist')
        # Check returns endpoint info as well as printing
        endpoint = 'subjects'
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            info = self.alyx.print_endpoint_info(endpoint)
            self.assertEqual(self.alyx.rest_schemes[endpoint], info)
            self.assertIsNot(self.alyx.rest_schemes[endpoint], info)  # Ensure copy returned
            self.assertTrue(stdout.getvalue().strip(), 'failed to print endpoint info')
        # Check action input
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            info = self.alyx.print_endpoint_info(endpoint, 'create')
            self.assertEqual(self.alyx.rest_schemes[endpoint]['create'], info)
            self.assertIsNot(self.alyx.rest_schemes[endpoint]['create'], info)  # Ensure copy
            self.assertTrue(stdout.getvalue().strip(), 'failed to print endpoint info')
            self.assertEqual("'create'\n\t", stdout.getvalue().strip()[:10])

    """Specific Alyx REST endpoint tests"""
    def test_water_restriction(self):
        """
        Test listing water-restriction endpoint.

        Examples of how to list all water restrictions and water-restriction for a given
        subject.
        """
        # get all the water restrictions from start
        all_wr = self.alyx.rest('water-restriction', 'list')
        # 2 different ways to  get water restrictions for one subject
        wr_sub2 = self.alyx.rest('water-restriction', 'list', subject='algernon')  # recommended
        # enforce test logic
        expected = {'end_time', 'reference_weight', 'start_time', 'subject', 'water_type'}
        self.assertTrue(expected >= set(all_wr[0].keys()))
        self.assertTrue(len(all_wr) > len(wr_sub2))

    def test_list_pk_query(self):
        """
        Test REST list with id keyword argument.

        It's a bit stupid but the REST endpoint can't forward a direct query of the uuid via
        the pk keyword. The AlyxClient has already an id parameter, which on the list method
        is used as a pk identifier. This special case is tested here.
        """
        # Sessions returned sorted: take last session as new sessions constantly added and
        # removed by parallel test runs
        ses = self.alyx.rest('sessions', 'list')[-1]
        eid = UUID(ses['url'][-36:])  # Should work with UUID object
        ses_ = self.alyx.rest('sessions', 'list', id=eid)[-1]
        self.assertEqual(ses, ses_)
        # Check works with django query arg
        query = f'start_time__date,{ses["start_time"][:10]}'
        ses_ = self.alyx.rest('sessions', 'list', id=eid, django=query)[-1]
        self.assertEqual(ses, ses_)

    def test_note_with_picture_upload(self):
        """Test adding session note with attached picture."""
        my_note = {'user': 'olivier',
                   'content_type': 'session',
                   'object_id': self.EID,
                   'text': 'gnagnagna'}

        png = Path(__file__).parent.joinpath('fixtures', 'test_img.png')
        with open(png, 'rb') as img_file:
            files = {'image': img_file}
            ar_note = self.alyx.rest('notes', 'create', data=my_note, files=files)

        self.assertTrue(len(ar_note['image']))
        self.assertTrue(ar_note['content_type'] == 'actions.session')
        self.alyx.rest('notes', 'delete', id=ar_note['id'])

    def test_channels(self):
        """Test creation of insertion, trajectory and channels."""
        # need to build insertion + trajectory + channels to test the serialization of a
        # record array in the channel endpoint
        name = ''.join(random.choices(string.ascii_letters, k=5))
        # Find any existing insertions with this name and delete (unlikely to find any)
        probe_insertions = self.alyx.rest('insertions', 'list',
                                          session=self.EID_EPHYS, name=name, no_cache=True)
        for pi in probe_insertions:
            self.alyx.rest('insertions', 'delete', pi['id'])
        # Create new insertion with this name and add teardown hook to delete it
        probe_insertion = self.alyx.rest(
            'insertions', 'create', data={'session': self.EID_EPHYS, 'name': name})
        self.addCleanup(self.alyx.rest, 'insertions', 'delete', id=probe_insertion['id'])
        trajectory = self.alyx.rest('trajectories', 'create', data={
            'probe_insertion': probe_insertion['id'],
            'x': 1500,
            'y': -2000,
            'z': 0,
            'depth': 4500,
            'phi': 0,
            'theta': 0,
            'provenance': 'Histology track',
        })
        channel_records = []
        for _ in np.arange(3):
            channel_records.append({
                'x': np.random.randint(-2000, 2000),
                'y': np.random.randint(-2000, 2000),
                'z': np.random.randint(-2000, 2000),
                'axial': np.random.rand() * 800,
                'lateral': np.random.rand() * 8,
                'brain_region': 889,
                'trajectory_estimate': trajectory['id']
            })
        channels = self.alyx.rest('channels', 'create', data=channel_records)
        self.assertTrue(len(channels) == 3)


if __name__ == '__main__':
    unittest.main(exit=False)
