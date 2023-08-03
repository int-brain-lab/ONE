"""Unit tests for the one.alf.files module."""
import unittest
from pathlib import Path
import uuid

import one.alf.files as files


class TestAlfParse(unittest.TestCase):
    """Tests for ALF parsing methods"""
    def test_filename_parts(self):
        """Test for one.alf.files.filename_parts"""
        verifiable = files.filename_parts('_namespace_obj.times_timescale.extra.foo.ext')
        expected = ('namespace', 'obj', 'times', 'timescale', 'extra.foo', 'ext')
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('spikes.clusters.npy', as_dict=True)
        expected = {
            'namespace': None,
            'object': 'spikes',
            'attribute': 'clusters',
            'timescale': None,
            'extra': None,
            'extension': 'npy'}
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('spikes.times_ephysClock.npy')
        expected = (None, 'spikes', 'times', 'ephysClock', None, 'npy')
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('_iblmic_audioSpectrogram.frequencies.npy')
        expected = ('iblmic', 'audioSpectrogram', 'frequencies', None, None, 'npy')
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('_spikeglx_ephysData_g0_t0.imec.wiring.json')
        expected = ('spikeglx', 'ephysData_g0_t0', 'imec', None, 'wiring', 'json')
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('_spikeglx_ephysData_g0_t0.imec0.lf.bin')
        expected = ('spikeglx', 'ephysData_g0_t0', 'imec0', None, 'lf', 'bin')
        self.assertEqual(expected, verifiable)

        verifiable = files.filename_parts('_ibl_trials.goCue_times_bpod.csv')
        expected = ('ibl', 'trials', 'goCue_times', 'bpod', None, 'csv')
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            files.filename_parts('badfile')
        verifiable = files.filename_parts('badfile', assert_valid=False)
        self.assertFalse(any(verifiable))

    def test_rel_path_parts(self):
        """Test for one.alf.files.rel_path_parts"""
        alf_str = Path('collection/#revision#/_namespace_obj.times_timescale.extra.foo.ext')
        verifiable = files.rel_path_parts(alf_str)
        expected = ('collection', 'revision', 'namespace', 'obj', 'times',
                    'timescale', 'extra.foo', 'ext')
        self.assertEqual(expected, verifiable)

        # Check as_dict
        verifiable = files.rel_path_parts('spikes.clusters.npy', as_dict=True)
        expected = {
            'collection': None,
            'revision': None,
            'namespace': None,
            'object': 'spikes',
            'attribute': 'clusters',
            'timescale': None,
            'extra': None,
            'extension': 'npy'}
        self.assertEqual(expected, verifiable)

        # Check assert valid
        with self.assertRaises(ValueError):
            files.rel_path_parts('bad/badfile')
        verifiable = files.rel_path_parts('bad/badfile', assert_valid=False)
        self.assertFalse(any(verifiable))

    def test_session_path_parts(self):
        """Test for one.alf.files.session_path_parts"""
        session_path = '/home/user/Data/labname/Subjects/subject/2020-01-01/001/alf'
        parsed = files.session_path_parts(session_path, as_dict=True)
        expected = {
            'lab': 'labname',
            'subject': 'subject',
            'date': '2020-01-01',
            'number': '001'}
        self.assertEqual(expected, parsed)
        parsed = files.session_path_parts(session_path, as_dict=False)
        self.assertEqual(tuple(expected.values()), parsed)
        # Check Path as input
        self.assertTrue(any(files.session_path_parts(Path(session_path))))
        # Check parse fails
        session_path = '/home/user/Data/labname/2020-01-01/alf/001/'
        with self.assertRaises(ValueError):
            files.session_path_parts(session_path, assert_valid=True)
        parsed = files.session_path_parts(session_path, assert_valid=False, as_dict=True)
        expected = dict.fromkeys(expected.keys())
        self.assertEqual(expected, parsed)
        parsed = files.session_path_parts(session_path, assert_valid=False, as_dict=False)
        self.assertEqual(tuple([None] * 4), parsed)

    def test_folder_parts(self):
        """Test for one.alf.files.folder_parts"""
        path = Path('/home/user/Data/labname/Subjects/subject/2020-01-01/001/'
                    'collection/#revision#/')
        out = files.folder_parts(path)
        expected_values = ('labname', 'subject', '2020-01-01', '001', 'collection', 'revision')
        self.assertEqual(expected_values, out)

        path = '/home/user/Data/labname/Subjects/subject/2020-01-01/001'
        expected_values = ('labname', 'subject', '2020-01-01', '001', None, None)
        self.assertEqual(expected_values, files.folder_parts(path))

    def test_full_path_parts(self):
        """Test for one.alf.files.full_path_parts"""
        fullpath = Path(
            '/home/user/Data/labname/Subjects/subject/2020-01-01/001/'
            'collection/#revision#/_namespace_obj.times_timescale.extra.foo.ext'
        )
        # As dict
        out = files.full_path_parts(fullpath, as_dict=True)
        expected_keys = (
            'lab', 'subject', 'date', 'number', 'collection', 'revision',
            'namespace', 'object', 'attribute', 'timescale', 'extra', 'extension'
        )
        self.assertIsInstance(out, dict)
        self.assertEqual(expected_keys, tuple(out.keys()))

        # As tuple
        out = files.full_path_parts(fullpath, as_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(expected_keys), len(out))
        self.assertTrue(all(out))

        # Folders only
        out = files.full_path_parts(fullpath.parent, as_dict=False)
        self.assertTrue(all(out[:6]) and not any(out[6:]))

        # Filename only
        out = files.full_path_parts(fullpath.name, as_dict=False)
        self.assertTrue(not any(out[:6]) and all(out[6:]))

    def test_isdatetime(self):
        """Test for one.alf.files._isdatetime"""
        inp = ['açsldfkça', '12312', '2020-01-01', '01-01-2020', '2020-12-32']
        out = [False, False, True, False, False]
        for i, o in zip(inp, out):
            self.assertEqual(o, files._isdatetime(i))

    def test_add_uuid(self):
        """Test for one.alf.files.add_uuid_string."""
        _uuid = uuid.uuid4()

        file_with_uuid = f'/titi/tutu.part1.part1.{_uuid}.json'
        inout = [(file_with_uuid, Path(file_with_uuid)),
                 ('/tutu/tata.json', Path(f'/tutu/tata.{_uuid}.json')),
                 ('/tutu/tata.part1.json', Path(f'/tutu/tata.part1.{_uuid}.json'))]
        for tup in inout:
            self.assertEqual(tup[1], files.add_uuid_string(tup[0], _uuid))
            self.assertEqual(tup[1], files.add_uuid_string(tup[0], str(_uuid)))

        with self.assertRaises(ValueError):
            files.add_uuid_string('/foo/bar.npy', 'fake')

    def test_remove_uuid(self):
        """Test for one.alf.files.remove_uuid_string."""
        # First test with full file
        file_path = '/tmp/Subjects/CSHL063/2020-09-12/001/raw_ephys_data/probe00/' \
                    '_spikeglx_sync.channels.probe00.89c861ea-66aa-4729-a808-e79f84d08b81.npy'
        desired_output = Path(file_path).with_name('_spikeglx_sync.channels.probe00.npy')
        files.remove_uuid_string(file_path)
        self.assertEqual(desired_output, files.remove_uuid_string(file_path))
        self.assertEqual(desired_output, files.remove_uuid_string(desired_output))

        # Test with just file name
        file_path = 'toto.89c861ea-66aa-4729-a808-e79f84d08b81.npy'
        desired_output = Path('toto.npy')
        self.assertEqual(desired_output, files.remove_uuid_string(file_path))


class TestALFGet(unittest.TestCase):
    """Tests for path extraction functions"""
    def test_get_session_folder(self):
        """Test for one.alf.files.get_session_folder"""
        inp = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/raw_behavior_data/'
                    '_iblrig_micData.raw.wav'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/raw_behavior_data'
               '/_iblrig_micData.raw.wav',
               '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001',)
        out = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),)
        for i, o in zip(inp, out):
            self.assertEqual(o, files.get_session_path(i))
        # Test if None is passed
        no_out = files.get_session_path(None)
        self.assertTrue(no_out is None)

    def test_get_alf_path(self):
        """Test for one.alf.files.get_alf_path"""
        path = Path('/mnt/s0/Data/Subjects/'
                    'ZM_1368/2019-04-19/001/raw_behavior_data/_iblrig_micData.raw.wav')
        out = files.get_alf_path(path)
        self.assertEqual(out, '/'.join(path.parts[-7:]))
        path = 'collection/trials.intervals_bpod.npy'
        self.assertEqual(files.get_alf_path(path), path)
        path = '/trials.intervals_bpod.npy'
        self.assertEqual(files.get_alf_path(path), 'trials.intervals_bpod.npy')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
