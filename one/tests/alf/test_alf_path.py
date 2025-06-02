"""Unit tests for the one.alf.path module."""
import unittest
import tempfile
from datetime import datetime
from types import GeneratorType
from uuid import uuid4
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath

from one.alf import path
from one.alf.path import ALFPath, PureALFPath, ensure_alf_path
from one.alf.exceptions import ALFInvalid


class TestALFParse(unittest.TestCase):
    """Tests for ALF parsing methods."""

    def test_filename_parts(self):
        """Test for one.alf.path.filename_parts."""
        verifiable = path.filename_parts('_namespace_obj.times_timescale.extra.foo.ext')
        expected = ('namespace', 'obj', 'times', 'timescale', 'extra.foo', 'ext')
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('spikes.clusters.npy', as_dict=True)
        expected = {
            'namespace': None,
            'object': 'spikes',
            'attribute': 'clusters',
            'timescale': None,
            'extra': None,
            'extension': 'npy'}
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('spikes.times_ephysClock.npy')
        expected = (None, 'spikes', 'times', 'ephysClock', None, 'npy')
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('_iblmic_audioSpectrogram.frequencies.npy')
        expected = ('iblmic', 'audioSpectrogram', 'frequencies', None, None, 'npy')
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('_spikeglx_ephysData_g0_t0.imec.wiring.json')
        expected = ('spikeglx', 'ephysData_g0_t0', 'imec', None, 'wiring', 'json')
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('_spikeglx_ephysData_g0_t0.imec0.lf.bin')
        expected = ('spikeglx', 'ephysData_g0_t0', 'imec0', None, 'lf', 'bin')
        self.assertEqual(expected, verifiable)

        verifiable = path.filename_parts('_ibl_trials.goCue_times_bpod.csv')
        expected = ('ibl', 'trials', 'goCue_times', 'bpod', None, 'csv')
        self.assertEqual(expected, verifiable)

        with self.assertRaises(ValueError):
            path.filename_parts('badfile')
        verifiable = path.filename_parts('badfile', assert_valid=False)
        self.assertFalse(any(verifiable))

    def test_rel_path_parts(self):
        """Test for one.alf.path.rel_path_parts."""
        alf_str = Path('collection/#revision#/_namespace_obj.times_timescale.extra.foo.ext')
        verifiable = path.rel_path_parts(alf_str)
        expected = ('collection', 'revision', 'namespace', 'obj', 'times',
                    'timescale', 'extra.foo', 'ext')
        self.assertEqual(expected, verifiable)

        # Check as_dict
        verifiable = path.rel_path_parts('spikes.clusters.npy', as_dict=True)
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
            path.rel_path_parts('bad/badfile')
        verifiable = path.rel_path_parts('bad/badfile', assert_valid=False)
        self.assertFalse(any(verifiable))

    def test_session_path_parts(self):
        """Test for one.alf.path.session_path_parts."""
        session_path = '/home/user/Data/labname/Subjects/subject/2020-01-01/001/alf'
        parsed = path.session_path_parts(session_path, as_dict=True)
        expected = {
            'lab': 'labname',
            'subject': 'subject',
            'date': '2020-01-01',
            'number': '001'}
        self.assertEqual(expected, parsed)
        parsed = path.session_path_parts(session_path, as_dict=False)
        self.assertEqual(tuple(expected.values()), parsed)
        # Check Path as input
        self.assertTrue(any(path.session_path_parts(Path(session_path))))
        # Check parse fails
        session_path = '/home/user/Data/labname/2020-01-01/alf/001/'
        with self.assertRaises(ValueError):
            path.session_path_parts(session_path, assert_valid=True)
        parsed = path.session_path_parts(session_path, assert_valid=False, as_dict=True)
        expected = dict.fromkeys(expected.keys())
        self.assertEqual(expected, parsed)
        parsed = path.session_path_parts(session_path, assert_valid=False, as_dict=False)
        self.assertEqual(tuple([None] * 4), parsed)

    def test_folder_parts(self):
        """Test for one.alf.path.folder_parts."""
        alfpath = Path(
            '/home/user/Data/labname/Subjects/subject/2020-01-01/001/collection/#revision#/')
        out = path.folder_parts(alfpath)
        expected_values = ('labname', 'subject', '2020-01-01', '001', 'collection', 'revision')
        self.assertEqual(expected_values, out)

        alfpath = '/home/user/Data/labname/Subjects/subject/2020-01-01/001'
        expected_values = ('labname', 'subject', '2020-01-01', '001', None, None)
        self.assertEqual(expected_values, path.folder_parts(alfpath))

    def test_full_path_parts(self):
        """Test for one.alf.path.full_path_parts."""
        fullpath = Path(
            '/home/user/Data/labname/Subjects/subject/2020-01-01/001/'
            'collection/#revision#/_namespace_obj.times_timescale.extra.foo.ext'
        )
        # As dict
        out = path.full_path_parts(fullpath, as_dict=True)
        expected_keys = (
            'lab', 'subject', 'date', 'number', 'collection', 'revision',
            'namespace', 'object', 'attribute', 'timescale', 'extra', 'extension'
        )
        self.assertIsInstance(out, dict)
        self.assertEqual(expected_keys, tuple(out.keys()))

        # As tuple
        out = path.full_path_parts(fullpath, as_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(expected_keys), len(out))
        self.assertTrue(all(out))

        # Folders only
        out = path.full_path_parts(fullpath.parent, as_dict=False)
        self.assertTrue(all(out[:6]) and not any(out[6:]))

        # Filename only
        out = path.full_path_parts(fullpath.name, as_dict=False)
        self.assertTrue(not any(out[:6]) and all(out[6:]))

    def test_isdatetime(self):
        """Test for one.alf.path._isdatetime."""
        inp = ['açsldfkça', '12312', '2020-01-01', '01-01-2020', '2020-12-32']
        out = [False, False, True, False, False]
        for i, o in zip(inp, out):
            self.assertEqual(o, path._isdatetime(i))

    def test_add_uuid(self):
        """Test for one.alf.path.add_uuid_string."""
        _uuid = uuid4()

        file_with_uuid = f'/titi/tutu.part1.part1.{_uuid}.json'
        inout = [(file_with_uuid, Path(file_with_uuid)),
                 ('/tutu/tata.json', Path(f'/tutu/tata.{_uuid}.json')),
                 ('/tutu/tata.part1.json', Path(f'/tutu/tata.part1.{_uuid}.json'))]
        for tup in inout:
            self.assertEqual(tup[1], path.add_uuid_string(tup[0], _uuid))
            self.assertEqual(tup[1], path.add_uuid_string(tup[0], str(_uuid)))

        _uuid2 = uuid4()
        with self.assertLogs(path.__name__, level=10) as cm:
            expected = Path(f'/titi/tutu.part1.part1.{_uuid2}.json')
            self.assertEqual(expected, path.add_uuid_string(file_with_uuid, _uuid2))
            self.assertRegex(cm.output[0], 'Replacing [a-f0-9-]+ with [a-f0-9-]+')

        with self.assertRaises(ValueError):
            path.add_uuid_string('/foo/bar.npy', 'fake')

    def test_remove_uuid(self):
        """Test for one.alf.path.remove_uuid_string."""
        # First test with full file
        file_path = '/tmp/Subjects/CSHL063/2020-09-12/001/raw_ephys_data/probe00/' \
                    '_spikeglx_sync.channels.probe00.89c861ea-66aa-4729-a808-e79f84d08b81.npy'
        desired_output = Path(file_path).with_name('_spikeglx_sync.channels.probe00.npy')
        path.remove_uuid_string(file_path)
        self.assertEqual(desired_output, path.remove_uuid_string(file_path))
        self.assertEqual(desired_output, path.remove_uuid_string(desired_output))

        # Test with just file name
        file_path = 'toto.89c861ea-66aa-4729-a808-e79f84d08b81.npy'
        desired_output = Path('toto.npy')
        self.assertEqual(desired_output, path.remove_uuid_string(file_path))

    def test_padded_sequence(self):
        """Test for one.alf.path.padded_sequence."""
        # Test with pure path file input
        filepath = PureWindowsPath(r'F:\ScanImageAcquisitions\subject\2023-01-01\1\foo\bar.baz')
        expected = PureWindowsPath(r'F:\ScanImageAcquisitions\subject\2023-01-01\001\foo\bar.baz')
        self.assertEqual(path.padded_sequence(filepath), expected)

        # Test with str input session path
        session_path = '/mnt/s0/Data/Subjects/subject/2023-01-01/001'
        expected = Path('/mnt/s0/Data/Subjects/subject/2023-01-01/001')
        self.assertEqual(path.padded_sequence(session_path), expected)

        # Test invalid ALF session path
        self.assertRaises(ValueError, path.padded_sequence, '/foo/bar/baz')


class TestALFGet(unittest.TestCase):
    """Tests for path extraction functions."""

    def test_get_session_folder(self):
        """Test for one.alf.path.get_session_folder."""
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
            self.assertEqual(o, path.get_session_path(i))
        # Test if None is passed
        no_out = path.get_session_path(None)
        self.assertTrue(no_out is None)

    def test_get_alf_path(self):
        """Test for one.alf.path.get_alf_path."""
        alfpath = Path(
            '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/'
            'raw_behavior_data/_iblrig_micData.raw.wav')
        out = path.get_alf_path(alfpath)
        self.assertEqual(out, '/'.join(alfpath.parts[-7:]))
        alfpath = 'collection/trials.intervals_bpod.npy'
        self.assertEqual(path.get_alf_path(alfpath), alfpath)
        alfpath = '/trials.intervals_bpod.npy'
        self.assertEqual(path.get_alf_path(alfpath), 'trials.intervals_bpod.npy')

    def test_without_revision(self):
        """Test for one.alf.path.without_revision function."""
        alfpath = '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/alf/#2020-01-01#/obj.attr.ext'
        out = path.without_revision(alfpath)
        expected = Path(alfpath.replace('/#2020-01-01#', ''))
        self.assertIsInstance(out, Path)
        self.assertEqual(expected, out, 'failed to remove revision folder')
        self.assertEqual(expected, path.without_revision(out))  # should do nothing to path
        with self.assertRaises(ALFInvalid):
            path.without_revision('foo/bar/baz.npy')


class TestALFPath(unittest.TestCase):
    """Tests for ALFPath class methods."""

    def setUp(self):
        self.alfpath = ALFPath(Path.home().joinpath(
            'foo', 'labname', 'Subjects', 'subject', '1900-01-01', '001',
            'alf', '#2020-01-01#', 'obj.attr.ext'
        ))

    def test_is_valid_alf(self):
        """Test for PureALFPath.is_valid_alf and ALFPath.is_valid_alf methods."""
        self.assertTrue(self.alfpath.is_valid_alf())
        self.assertTrue(PureALFPath.is_valid_alf(str(self.alfpath)))
        self.assertFalse(PureALFPath.is_valid_alf(self.alfpath.with_name('foo.npy')))
        self.assertFalse(ALFPath.is_valid_alf(self.alfpath.with_name('foo.npy')))
        # A session path with invalid subject name should return False
        self.assertFalse(PureALFPath.is_valid_alf('abc-@/2020-01-01/001'))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_session = ALFPath(
                tmp, 'foo', 'labname', 'Subjects', 'subject', '1900-01-01', '001')
            # An ostensibly valid file that is actually a folder should be invalid
            (fake_file := tmp_session.joinpath('obj.attr.ext')).mkdir(parents=True)
            self.assertFalse(fake_file.is_valid_alf())
            self.assertTrue(PureALFPath.is_valid_alf(str(fake_file)))
            # An ostensibly valid folder that is actually a file should be invalid
            (fake_folder := tmp_session.joinpath('#2020-01-01#')).touch()
            self.assertFalse(ALFPath.is_valid_alf(str(fake_folder)))
            self.assertTrue(PureALFPath(fake_folder).is_valid_alf())
        # If it doesn't exist it should still be considered valid
        self.assertTrue(tmp_session.is_valid_alf())

    def test_is_dataset(self):
        """Test for PureALFPath.is_dataset method."""
        self.assertTrue(self.alfpath.is_dataset())
        self.assertFalse(self.alfpath.parent.is_dataset())

    def test_session_path(self):
        """Test for PureALFPath.session_path method."""
        expected = self.alfpath.parents[2]
        self.assertEqual(expected, self.alfpath.session_path())

    def test_without_revision(self):
        """Test for PureALFPath.without_revision method."""
        # Test with dataset
        expected = self.alfpath.parents[1] / self.alfpath.name
        self.assertEqual(expected, self.alfpath.without_revision())
        # Test with revision folder
        expected = self.alfpath.parents[1]
        self.assertEqual(expected, self.alfpath.parent.without_revision())
        # Test with other folder
        expected = self.alfpath.parents[2]
        self.assertEqual(expected, self.alfpath.parents[2].without_revision())
        # Test with invalid path
        alfpath = self.alfpath.parent.joinpath('foo.npy')
        self.assertRaises(ALFInvalid, alfpath.without_revision)

    def test_with_revision(self):
        """Test for PureALFPath.with_revision method."""
        # Test dataset with revision
        expected = self.alfpath.parents[1] / '#bar#' / self.alfpath.name
        self.assertEqual(expected, self.alfpath.with_revision('bar'))
        # Test dataset without revision
        expected = self.alfpath.parents[1] / '#baz#' / self.alfpath.name
        alfpath = self.alfpath.parents[1] / self.alfpath.name
        self.assertEqual(expected, alfpath.with_revision('baz'))
        # Test revision folder
        expected = self.alfpath.parents[1] / '#bar#'
        self.assertEqual(expected, self.alfpath.parent.with_revision('bar'))
        # Test non-revision folder
        expected = self.alfpath.parents[1] / '#bar#'
        self.assertEqual(expected, self.alfpath.parents[1].with_revision('bar'))
        # Test path relative to session (currently not supported due to spec ambiguity)
        alfpath = self.alfpath.relative_to_session()
        self.assertRaises(ALFInvalid, alfpath.with_revision, 'bar')

    def test_with_padded_sequence(self):
        """Test for PureALFPath.with_padded_sequence method."""
        # Test already padded
        self.assertEqual(self.alfpath, self.alfpath.with_padded_sequence())
        # Test not padded
        alfpath = self.alfpath.parents[3].joinpath('1', *self.alfpath.parts[-3:])
        self.assertEqual(self.alfpath, alfpath.with_padded_sequence())

    def test_relative_to_session(self):
        """Test for PureALFPath.relative_to_session method."""
        expected = ALFPath(*self.alfpath.parts[-3:])
        self.assertEqual(expected, self.alfpath.relative_to_session())
        self.assertRaises(ValueError, expected.relative_to_session)

    def test_session_path_short(self):
        """Test for PureALFPath.session_path_short method."""
        expected = 'subject/1900-01-01/001'
        self.assertEqual(expected, self.alfpath.session_path_short())
        expected = 'labname/subject/1900-01-01/001'
        self.assertEqual(expected, self.alfpath.session_path_short(include_lab=True))

    def test_without_lab(self):
        """Test for PureALFPath.without_lab method."""
        # Test with lab
        expected = ALFPath(self.alfpath.as_posix().replace('labname/Subjects/', ''))
        self.assertEqual(expected, self.alfpath.without_lab())
        # Test without lab
        self.assertEqual(expected, expected.without_lab())

    def test_relative_to_lab(self):
        """Test ALFPath.relative_to_lab method."""
        # Test with lab
        expected = ALFPath(*self.alfpath.parts[-6:])
        self.assertEqual(expected, self.alfpath.relative_to_lab())
        # Test without lab
        self.assertRaises(ValueError, expected.relative_to_lab)

    def test_without_uuid(self):
        """Test for PureALFPath.without_uuid method."""
        # Test file without uuid
        self.assertEqual(self.alfpath, self.alfpath.without_uuid())
        # Test file with uuid
        alfpath = self.alfpath.parent / f'obj.attr.{uuid4()}.ext'
        self.assertEqual(self.alfpath, alfpath.without_uuid())
        # Test folder
        self.assertEqual(self.alfpath.parent, alfpath.parent.without_uuid())

    def test_with_uuid(self):
        """Test for PureALFPath.with_uuid method."""
        # Test file without uuid
        uuid = uuid4()
        expected = self.alfpath.parent / f'obj.attr.{uuid}.ext'
        self.assertEqual(expected, self.alfpath.with_uuid(uuid))
        # Test file with uuid
        uuid = uuid4()
        alfpath = expected.with_uuid(uuid)
        expected = self.alfpath.parent / f'obj.attr.{uuid}.ext'
        self.assertEqual(expected, alfpath)
        # Test folder
        self.assertRaises(ALFInvalid, alfpath.parent.with_uuid, uuid)

    def test_is_session_path(self):
        """Test PureALFPath and ALFPath.is_session_path methods."""
        # Check PureALFPath w/o system calls
        self.assertFalse(self.alfpath.is_session_path())
        self.assertTrue(self.alfpath.parents[2].is_session_path())
        self.assertTrue(PureALFPath(self.alfpath.parents[2]).is_session_path())
        # Check ALFPath method with system call
        with tempfile.TemporaryDirectory() as tmp:
            tmp_session = ALFPath(
                tmp, 'foo', 'labname', 'Subjects', 'subject', '1900-01-01', '001')
            self.assertTrue(tmp_session.is_session_path())
            # An ostensibly valid session path that is actually a file should be invalid
            tmp_session.parent.mkdir(parents=True)
            tmp_session.touch()
            self.assertFalse(tmp_session.is_session_path())

    def test_iter_datasets(self):
        """Test ALFPath.iter_datasets method."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_session = ALFPath(
                tmp, 'foo', 'labname', 'Subjects', 'subject', '1900-01-01', '001')
            tmp_session.mkdir(parents=True)
            for file in ('foo.bar', 'obj.attr.ext', 'bar.baz.foo', 'alf/foo.baz.bar'):
                if file.startswith('alf'):
                    tmp_session.joinpath(file).parent.mkdir()
                tmp_session.joinpath(file).touch()
            dsets = tmp_session.iter_datasets()
            self.assertIsInstance(dsets, GeneratorType)
            dsets = list(dsets)
            expected = [tmp_session / f for f in ('bar.baz.foo', 'obj.attr.ext')]
            self.assertEqual(expected, dsets)  # NB: Order important here
            # Check recursive
            dsets = list(tmp_session.iter_datasets(recursive=True))
            self.assertEqual(3, len(dsets))
            self.assertEqual(tmp_session / 'alf/foo.baz.bar', dsets[0])

    def test_with_object(self):
        """Test for PureALFPath.with_object method."""
        # Test without namespace
        expected = self.alfpath.with_name('foo.attr.ext')
        self.assertEqual(expected, self.alfpath.with_object('foo'))
        # Test with namespace
        alfpath = self.alfpath.with_name('_ns_obj.attr.ext')
        expected = self.alfpath.with_name('_ns_bar.attr.ext')
        self.assertEqual(expected, alfpath.with_object('bar'))
        self.assertRaises(ALFInvalid, alfpath.with_stem('foo').with_object, 'obj')

    def test_with_namespace(self):
        """Test for PureALFPath.with_namespace method."""
        # Test without namespace
        expected = self.alfpath.with_name('_ns_obj.attr.ext')
        self.assertEqual(expected, self.alfpath.with_namespace('ns'))
        # Test with namespace
        alfpath = self.alfpath.with_name('_foo_obj.attr.ext')
        self.assertEqual(expected, alfpath.with_namespace('ns'))
        # Test removing namespace
        self.assertEqual(self.alfpath, alfpath.with_namespace(''))
        self.assertRaises(ALFInvalid, alfpath.with_stem('foo').with_namespace, 'ns')

    def test_with_attribute(self):
        """Test for PureALFPath.with_attribute method."""
        # Test without timescale
        expected = self.alfpath.with_name('obj.foo.ext')
        self.assertEqual(expected, self.alfpath.with_attribute('foo'))
        # Test with timescale
        alfpath = self.alfpath.with_name('obj.attr_times_barClock.ext')
        expected = self.alfpath.with_name('obj.foo_barClock.ext')
        self.assertEqual(expected, alfpath.with_attribute('foo'))
        self.assertRaises(ALFInvalid, alfpath.with_stem('foo').with_attribute, 'attr')

    def test_with_timescale(self):
        """Test for PureALFPath.with_timescale method."""
        # Test without timescale
        expected = self.alfpath.with_name('obj.attr_foo.ext')
        self.assertEqual(expected, self.alfpath.with_timescale('foo'))
        # Test with timescale
        alfpath = self.alfpath.with_name('obj.attr_times_barClock.ext')
        expected = self.alfpath.with_name('obj.attr_times_foo.ext')
        self.assertEqual(expected, alfpath.with_timescale('foo'))
        # Test removing timescale
        expected = self.alfpath.with_name('obj.attr_times.ext')
        self.assertEqual(expected, alfpath.with_timescale(''))
        self.assertRaises(ALFInvalid, alfpath.with_stem('foo').with_timescale, 'bpod')

    def test_with_extra(self):
        """Test for PureALFPath.with_extra method."""
        # Test without extra
        expected = self.alfpath.with_name('obj.attr.extra.ext')
        self.assertEqual(expected, self.alfpath.with_extra('extra'))
        # Test with extra
        alfpath = expected
        expected = self.alfpath.with_name('obj.attr.foo.ext')
        self.assertEqual(expected, alfpath.with_extra('foo'))
        # Test append
        alfpath = expected
        expected = self.alfpath.with_name('obj.attr.foo.extra.ext')
        self.assertEqual(expected, alfpath.with_extra('extra', append=True))
        # Test list
        self.assertEqual(expected, alfpath.with_extra(['foo', 'extra']))
        # Test removing extra
        self.assertEqual(self.alfpath, alfpath.with_extra(''))
        self.assertRaises(ALFInvalid, alfpath.with_stem('foo').with_extra, 'extra')

    def test_with_extension(self):
        """Test for PureALFPath.with_extension method."""
        expected = self.alfpath.with_suffix('.npy')
        self.assertEqual(expected, self.alfpath.with_extension('npy'))
        self.assertRaises(ValueError, self.alfpath.with_extension, '')
        self.assertRaises(ALFInvalid, self.alfpath.with_stem('foo').with_extension, 'ext')

    def test_with_lab(self):
        """Test for PureALFPath.with_lab method."""
        # Test with lab
        expected = ALFPath(*self.alfpath.parts[:-8], 'newlab', *self.alfpath.parts[-7:])
        self.assertEqual(expected, self.alfpath.with_lab('newlab'))
        # Test without lab
        alfpath = ALFPath(*self.alfpath.parts[:-8], *self.alfpath.parts[-6:])
        self.assertEqual(expected, alfpath.with_lab('newlab'))
        # Test strict
        self.assertEqual(expected, self.alfpath.with_lab('newlab', strict=True))
        self.assertRaises(ALFInvalid, alfpath.with_lab, 'newlab', strict=True)
        # Test validation
        self.assertRaises(ValueError, self.alfpath.with_lab, '')
        self.assertRaises(ValueError, self.alfpath.with_lab, None)
        self.assertRaises(ValueError, self.alfpath.with_lab, '#s!@#')
        self.assertRaises(ALFInvalid, self.alfpath.relative_to_session().with_lab, 'lab')

    def test_with_subject(self):
        """Test for PureALFPath.with_subject method."""
        # Test with subject
        expected = ALFPath(*self.alfpath.parts[:-6], 'foo', *self.alfpath.parts[-5:])
        self.assertEqual(expected, self.alfpath.with_subject('foo'))
        # Test without lab (should not depend on Subjects folder)
        alfpath = ALFPath(*self.alfpath.parts[:-8], *self.alfpath.parts[-6:])
        expected = ALFPath(*alfpath.parts[:-6], 'foo', *alfpath.parts[-5:])
        self.assertEqual(expected, alfpath.with_subject('foo'))
        # Test validation
        self.assertRaises(ValueError, self.alfpath.with_subject, '')
        self.assertRaises(ValueError, self.alfpath.with_subject, None)
        self.assertRaises(ValueError, self.alfpath.with_subject, '#s!@#')
        self.assertRaises(ALFInvalid, self.alfpath.relative_to_session().with_subject, 'subject')

    def test_with_date(self):
        """Test for PureALFPath.with_date method."""
        # Test with date
        expected = ALFPath(*self.alfpath.parts[:-5], '2020-01-02', *self.alfpath.parts[-4:])
        self.assertEqual(expected, self.alfpath.with_date('2020-01-02'))
        # Test with datetime object
        date = datetime.fromisoformat('2020-01-02T00:00:00')
        self.assertEqual(expected, self.alfpath.with_date(date))
        # Test validation
        self.assertRaises(ValueError, self.alfpath.with_date, '')
        self.assertRaises(ValueError, self.alfpath.with_date, None)
        self.assertRaises(ValueError, self.alfpath.with_date, '6/1/2020')
        self.assertRaises(ALFInvalid, self.alfpath.relative_to_session().with_date, '2020-01-02')

    def test_with_sequence(self):
        """Test for PureALFPath.with_sequence method."""
        # Test with number
        expected = ALFPath(*self.alfpath.parts[:-4], '002', *self.alfpath.parts[-3:])
        self.assertEqual(expected, self.alfpath.with_sequence(2))
        self.assertEqual(expected, self.alfpath.with_sequence('002'))
        # Test with zero
        self.assertEqual('000', self.alfpath.with_sequence(0).parts[-4])
        # Test validation
        self.assertRaises(ValueError, self.alfpath.with_sequence, '')
        self.assertRaises(ValueError, self.alfpath.with_sequence, None)
        self.assertRaises(ValueError, self.alfpath.with_sequence, 'foo')
        self.assertRaises(ValueError, self.alfpath.with_sequence, 1e4)
        self.assertRaises(ALFInvalid, self.alfpath.relative_to_session().with_sequence, 2)

    def test_parts_properties(self):
        """Test the PureALFPath ALF dataset part properties."""
        # Namespace
        self.assertEqual('', self.alfpath.namespace)
        self.assertEqual('ns', self.alfpath.with_stem('_ns_obj.attr').namespace)
        self.assertEqual('', self.alfpath.with_stem('_ns_foo').namespace)
        # Object
        self.assertEqual('obj', self.alfpath.object)
        self.assertEqual('', self.alfpath.with_stem('foo').object)
        # Attribute
        self.assertEqual('attr', self.alfpath.attribute)
        self.assertEqual('', self.alfpath.with_stem('foo').attribute)
        # Timescale
        self.assertEqual('', self.alfpath.timescale)
        self.assertEqual('bpod', self.alfpath.with_stem('obj.attr_times_bpod').timescale)
        self.assertEqual('', self.alfpath.with_stem('foo').timescale)
        # Extra
        self.assertEqual('', self.alfpath.extra)
        self.assertEqual('foo.bar', self.alfpath.with_stem('obj.att.foo.bar').extra)
        self.assertEqual('', self.alfpath.with_stem('foo').extra)
        # dataset_name_parts
        self.assertEqual(('', 'obj', 'attr', '', '', 'ext'), self.alfpath.dataset_name_parts)
        alfpath = self.alfpath.with_name('_ns_obj.attr_times_bpod.foo.bar.ext')
        expected = ('ns', 'obj', 'attr_times', 'bpod', 'foo.bar', 'ext')
        self.assertEqual(expected, alfpath.dataset_name_parts)
        # Lab
        self.assertEqual('labname', self.alfpath.lab)
        self.assertEqual('', self.alfpath.relative_to_session().lab)
        # Subject
        self.assertEqual('subject', self.alfpath.subject)
        self.assertEqual('', self.alfpath.relative_to_session().subject)
        # Date
        self.assertEqual('1900-01-01', self.alfpath.date)
        self.assertEqual('', self.alfpath.relative_to_session().date)
        # Number
        self.assertEqual('001', self.alfpath.sequence)
        self.assertEqual('', self.alfpath.relative_to_session().sequence)
        # session_parts
        self.assertEqual(('labname', 'subject', '1900-01-01', '001'), self.alfpath.session_parts)
        alfpath = ALFPath(*self.alfpath.parts[5:])
        self.assertEqual(('', 'subject', '1900-01-01', '001'), alfpath.session_parts)
        # alf_parts
        alfpath = self.alfpath.with_name('_ns_obj.attr_times_bpod.foo.bar.ext')
        expected = ('labname', 'subject', '1900-01-01', '001', 'alf', '2020-01-01',
                    'ns', 'obj', 'attr_times', 'bpod', 'foo.bar', 'ext')
        self.assertEqual(expected, alfpath.alf_parts)
        expected = ('', '', '', '', '', '', '', '', '', '', '', '')
        self.assertEqual(expected, ALFPath('foo').alf_parts)

    def test_parse_alf_path(self):
        """Test PureALFPath.parse_alf_path method."""
        parsed = self.alfpath.parse_alf_path()
        self.assertIsInstance(parsed, dict)
        expected = dict(
            lab='labname', subject='subject', date='1900-01-01', number='001', collection='alf',
            revision='2020-01-01', namespace=None, object='obj', attribute='attr', timescale=None,
            extra=None, extension='ext')
        # NB: We assertEqual instead of assertDictEqual because the order must always be correct
        self.assertEqual(expected, parsed)
        # With session path
        parsed = self.alfpath.session_path().parse_alf_path()
        _expected = {**expected, **{k: None for k in list(expected.keys())[4:]}}
        self.assertEqual(_expected, parsed)
        # With dataset name
        parsed = PureALFPath(self.alfpath.name).parse_alf_path()
        _expected = {**expected, **{k: None for k in list(expected.keys())[:6]}}
        self.assertEqual(_expected, parsed)
        # With invalid path
        parsed = PureALFPath(ALFPath('foo/bar/Subjects/baz.pie')).parse_alf_path()
        _expected = dict.fromkeys(expected)
        self.assertEqual(_expected, parsed)

    def test_parse_alf_name(self):
        """Test PureALFPath.parse_alf_name method."""
        # With dataset name
        parsed = self.alfpath.parse_alf_name()
        self.assertIsInstance(parsed, dict)
        expected = dict(
            namespace=None, object='obj', attribute='attr',
            timescale=None, extra=None, extension='ext')
        # NB: We assertEqual instead of assertDictEqual because the order must always be correct
        self.assertEqual(expected, parsed)
        # With invalid dataset path
        parsed = PureALFPath(ALFPath('foo/bar/Subjects/baz.pie')).parse_alf_name()
        _expected = dict.fromkeys(expected)
        self.assertEqual(_expected, parsed)

    def test_ensure_alf_path(self):
        """Test for one.alf.path.ensure_alf_path function."""
        # Check str -> ALFPath
        alfpath = ensure_alf_path(str(self.alfpath))
        self.assertIsInstance(alfpath, ALFPath, 'failed to cast str to ALFPath')
        # Check ALFPath -> ALFPath
        alfpath = ensure_alf_path(self.alfpath)
        self.assertIs(alfpath, self.alfpath, 'expected identity behaviour')
        # Check PureALFPath -> PureALFPath
        alfpath = ensure_alf_path(PureALFPath(self.alfpath))
        self.assertIsInstance(alfpath, PureALFPath)
        self.assertNotIsInstance(alfpath, ALFPath)
        # Check PureWindowsPath -> PureWindowsALFPath
        alfpath = ensure_alf_path(PureWindowsPath(self.alfpath))
        self.assertIsInstance(alfpath, PureALFPath)
        self.assertIsInstance(alfpath, PureWindowsPath)
        self.assertNotIsInstance(alfpath, ALFPath)
        # Check PurePosixPath -> PurePosixALFPath
        alfpath = ensure_alf_path(PurePosixPath(self.alfpath))
        self.assertIsInstance(alfpath, PureALFPath)
        self.assertIsInstance(alfpath, PurePosixPath)
        self.assertNotIsInstance(alfpath, ALFPath)
        # Check arbitrary PurePath -> PureALFPath

        class ArbitraryPurePath(PurePath):
            @classmethod
            def _parse_args(cls, args):
                return self.alfpath._flavour.parse_parts(args[0].parts)
        alfpath = ensure_alf_path(ArbitraryPurePath(self.alfpath))
        self.assertIsInstance(alfpath, PureALFPath)
        # Check Path -> ALFPath
        alfpath = ensure_alf_path(Path(self.alfpath))
        self.assertIsInstance(alfpath, ALFPath)
        # Check operation on list
        alfpaths = ensure_alf_path([str(self.alfpath)])
        self.assertEqual([self.alfpath], alfpaths)
        # Check assertions
        self.assertRaises(TypeError, ensure_alf_path, 20)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
