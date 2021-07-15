import unittest
import tempfile
from pathlib import Path
import shutil
import re
import uuid

import one.alf.files as files
from one.alf.spec import FILE_SPEC, regex


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_filter_by(self):
        spec_idx_map = regex(FILE_SPEC).groupindex
        file_names = [
            'noalf.file',
            '_ibl_trials.intervals.npy',
            '_ibl_trials.intervals_bpod.csv',
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_namespace_obj.attr_timescale.raw.v12.ext']

        for f in file_names:
            (self.tmpdir / f).touch()

        # Test filter with None; should return files with no non-standard timescale
        alf_files, _ = files.filter_by(self.tmpdir, timescale=None)
        expected = [
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_ibl_trials.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with None attribute')

        # Test filtering by object; should return only 'wheel' ALF objects
        alf_files, parts = files.filter_by(self.tmpdir, object='wheel')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter by object')
        self.assertEqual(len(alf_files), len(parts))

        # Test wildcards; should return 'wheel' and 'wheelMoves' ALF objects
        alf_files, _ = files.filter_by(self.tmpdir, object='wh*')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy', 'wheelMoves.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with wildcard')

        # Test filtering by specific timescale; test parts returned
        alf_files, parts = files.filter_by(self.tmpdir, timescale='bpod')
        expected = ['_ibl_trials.intervals_bpod.csv']
        self.assertEqual(alf_files, expected, 'failed to filter by timescale')
        expected = ('ibl', 'trials', 'intervals', 'bpod', None, 'csv')
        self.assertTupleEqual(parts[0], expected)
        self.assertEqual(len(parts[0]), len(spec_idx_map))
        self.assertEqual(parts[0][spec_idx_map['timescale'] - 1], 'bpod')

        # Test filtering multiple attributes; should return only trials intervals
        alf_files, _ = files.filter_by(self.tmpdir, attribute='intervals', object='trials')
        expected = ['_ibl_trials.intervals.npy', '_ibl_trials.intervals_bpod.csv']
        self.assertCountEqual(alf_files, expected, 'failed to filter by multiple attribute')

        # Test returning only ALF files
        alf_files, _ = files.filter_by(self.tmpdir)
        self.assertCountEqual(alf_files, file_names[1:], 'failed to return ALF files')

        # Test return empty
        out = files.filter_by(self.tmpdir, object=None)
        self.assertEqual(out, ([], []))

        # Test extras
        alf_files, _ = files.filter_by(self.tmpdir, extra='v12')
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes')

        alf_files, _ = files.filter_by(self.tmpdir, extra=['v12', 'raw'])
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes as list')

        alf_files, _ = files.filter_by(self.tmpdir, extra=['foo', 'v12'])
        self.assertEqual(alf_files, [], 'failed to filter extra attributes')

        # Assert kwarg validation; should raise TypeError
        with self.assertRaises(TypeError):
            files.filter_by(self.tmpdir, unknown=None)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestAlfParse(unittest.TestCase):
    def test_filename_parts(self):
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
        alf_str = 'collection/#revision#/_namespace_obj.times_timescale.extra.foo.ext'
        verifiable = files.rel_path_parts(alf_str)
        expected = ('collection', 'revision', 'namespace', 'obj', 'times',
                    'timescale', 'extra.foo', 'ext')
        self.assertEqual(expected, verifiable)

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

        with self.assertRaises(ValueError):
            files.rel_path_parts('bad/badfile')
        verifiable = files.filename_parts('bad/badfile', assert_valid=False)
        self.assertFalse(any(verifiable))

    def test_session_path_parts(self):
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
        # Check parse fails
        session_path = '/home/user/Data/labname/2020-01-01/alf/001/'
        with self.assertRaises(ValueError):
            files.session_path_parts(session_path, assert_valid=True)
        parsed = files.session_path_parts(session_path, assert_valid=False, as_dict=True)
        expected = dict.fromkeys(expected.keys())
        self.assertEqual(expected, parsed)
        parsed = files.session_path_parts(session_path, assert_valid=False, as_dict=False)
        self.assertEqual(tuple([None] * 4), parsed)


class TestALFGet(unittest.TestCase):
    def test_get_session_folder(self):
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
        path = Path('/mnt/s0/Data/Subjects/'
                    'ZM_1368/2019-04-19/001/raw_behavior_data/_iblrig_micData.raw.wav')
        out = files.get_alf_path(path)
        self.assertEqual(out, '/'.join(path.parts[-7:]))
        path = 'collection/trials.intervals_bpod.npy'
        self.assertEqual(files.get_alf_path(path), path)
        path = '/trials.intervals_bpod.npy'
        self.assertEqual(files.get_alf_path(path), 'trials.intervals_bpod.npy')

    def test_isdatetime(self):
        inp = ['açsldfkça', '12312', '2020-01-01', '01-01-2020', '2020-12-32']
        out = [False, False, True, False, False]
        for i, o in zip(inp, out):
            self.assertEqual(o, files._isdatetime(i))

    def test_add_uuid(self):
        _uuid = uuid.uuid4()

        file_with_uuid = f'/titi/tutu.part1.part1.{_uuid}.json'
        inout = [(file_with_uuid, Path(file_with_uuid)),
            ('/tutu/tata.json', Path(f'/tutu/tata.{_uuid}.json')),
            ('/tutu/tata.part1.json', Path(f'/tutu/tata.part1.{_uuid}.json')), ]
        for tup in inout:
            self.assertEqual(tup[1], files.add_uuid_string(tup[0], _uuid))
            self.assertEqual(tup[1], files.add_uuid_string(tup[0], str(_uuid)))

        with self.assertRaises(ValueError):
            files.add_uuid_string('/foo/bar.npy', 'fake')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
