"""Unit tests for Alyx REST queries using the AlyxClient.rest method"""
from pathlib import Path
import unittest
import random
import string
from uuid import UUID

import numpy as np

from one.api import ONE
from one.tests import TEST_DB_1, OFFLINE_ONLY


one = ONE(**TEST_DB_1)

EID = 'cf264653-2deb-44cb-aa84-89b82507028a'
EID_EPHYS = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class Tests_REST(unittest.TestCase):
    """Specific Alyx REST endpoint tests"""
    def test_water_restriction(self):
        """
        Test listing water-restriction endpoint.

        Examples of how to list all water restrictions and water-restriction for a given
        subject.
        """
        # get all the water restrictions from start
        all_wr = one.alyx.rest('water-restriction', 'list')
        # 2 different ways to  get water restrictions for one subject
        wr_sub2 = one.alyx.rest('water-restriction', 'list', subject='algernon')  # recommended
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
        ses = one.alyx.rest('sessions', 'list')[-1]
        eid = UUID(ses['url'][-36:])  # Should work with UUID object
        ses_ = one.alyx.rest('sessions', 'list', id=eid)[-1]
        self.assertEqual(ses, ses_)
        # Check works with django query arg
        query = f'start_time__date,{ses["start_time"][:10]}'
        ses_ = one.alyx.rest('sessions', 'list', id=eid, django=query)[-1]
        self.assertEqual(ses, ses_)

    def test_note_with_picture_upload(self):
        """Test adding session note with attached picture"""
        my_note = {'user': 'olivier',
                   'content_type': 'session',
                   'object_id': EID,
                   'text': 'gnagnagna'}

        png = Path(__file__).parent.joinpath('fixtures', 'test_img.png')
        with open(png, 'rb') as img_file:
            files = {'image': img_file}
            ar_note = one.alyx.rest('notes', 'create', data=my_note, files=files)

        self.assertTrue(len(ar_note['image']))
        self.assertTrue(ar_note['content_type'] == 'actions.session')
        one.alyx.rest('notes', 'delete', id=ar_note['id'])

    def test_channels(self):
        """Test creation of insertion, trajectory and channels"""
        # need to build insertion + trajectory + channels to test the serialization of a
        # record array in the channel endpoint
        name = ''.join(random.choices(string.ascii_letters, k=5))
        # Find any existing insertions with this name and delete (unlikely to find any)
        probe_insertions = one.alyx.rest('insertions', 'list',
                                         session=EID_EPHYS, name=name, no_cache=True)
        for pi in probe_insertions:
            one.alyx.rest('insertions', 'delete', pi['id'])
        # Create new insertion with this name and add teardown hook to delete it
        probe_insertion = one.alyx.rest(
            'insertions', 'create', data={'session': EID_EPHYS, 'name': name})
        self.addCleanup(one.alyx.rest, 'insertions', 'delete', id=probe_insertion['id'])
        trajectory = one.alyx.rest('trajectories', 'create', data={
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
        channels = one.alyx.rest('channels', 'create', data=channel_records)
        self.assertTrue(len(channels) == 3)


if __name__ == "__main__":
    unittest.main(exit=False)
