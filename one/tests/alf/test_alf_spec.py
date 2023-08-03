"""Unit tests for the one.alf.spec module."""
import unittest.mock
import re
import io
from pathlib import Path
import uuid

import one.alf.spec as alf_spec
from one.alf.exceptions import ALFError


class TestALFSpec(unittest.TestCase):
    """Tests for ALF specification"""
    def test_regex(self):
        """Test for one.alf.spec.regex"""
        # Should return the full regex with named capture groups by default
        verifiable = alf_spec.regex()
        expected = ('((?P<lab>\\w+)/Subjects/)?'
                    '(?P<subject>[\\w.-]+)/(?P<date>\\d{4}-\\d{2}-\\d{2})/(?P<number>\\d{1,3})/'
                    '((?P<collection>[\\w./-]+)/)?(#(?P<revision>[\\w.-]+)#/)?'
                    '_?(?P<namespace>(?<=_)[a-zA-Z0-9]+)?_?(?P<object>\\w+)\\.'
                    '(?P<attribute>(?:_[a-z]+_)?[a-zA-Z0-9]+'
                    '(?:_times(?=[_.])|_intervals(?=[_.]))?)'
                    '(?:_(?P<timescale>\\w+))?(?:\\.(?P<extra>[.\\w-]+))*\\.(?P<extension>\\w+)$')
        self.assertEqual(expected, verifiable.pattern)

        # Should return only the filename regex pattern
        verifiable = alf_spec.regex(spec=alf_spec.FILE_SPEC)
        expected = expected[expected.index('_?(?P<namespace>'):]
        self.assertEqual(expected, verifiable.pattern)

        # Should replace the collection pattern
        verifiable = alf_spec.regex(spec=alf_spec.COLLECTION_SPEC, collection=r'probe\d{2}')
        expected = '((?P<collection>probe\\d{2})/)?(#(?P<revision>[\\w.-]+)#/)?'
        self.assertEqual(expected, verifiable.pattern)

        # Should raise a key error
        with self.assertRaises(KeyError):
            alf_spec.regex(foo=r'[bar]+')

    def test_named_group(self):
        """Test for one.alf.spec._named"""
        verifiable = alf_spec._named(r'[bar]+', 'foo')
        self.assertEqual(verifiable, '(?P<foo>[bar]+)')

    def test_patterns(self):
        """Tests for various spec strings"""
        # Test matching of some real paths
        filename = '_ibl_passivePeriods.intervalsTable_bpod.csv'
        expected = {
            'namespace': 'ibl',
            'object': 'passivePeriods',
            'attribute': 'intervalsTable',
            'timescale': 'bpod',
            'extra': None,
            'extension': 'csv'
        }
        verifiable = alf_spec.regex(alf_spec.FILE_SPEC).match(filename).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Match collection with filename
        spec = f'{alf_spec.COLLECTION_SPEC}{alf_spec.FILE_SPEC}'
        rel_path = 'alf/_ibl_trials.contrastRight.npy'
        expected = {
            'collection': 'alf',
            'revision': None,
            'namespace': 'ibl',
            'object': 'trials',
            'attribute': 'contrastRight',
            'timescale': None,
            'extra': None,
            'extension': 'npy'
        }
        verifiable = re.match(alf_spec.regex(spec), rel_path).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Test with revision
        rel_path = 'raw_ephys_data/probe00/#foobar#/_iblqc_ephysTimeRmsAP.timestamps.npy'
        expected = {
            'collection': 'raw_ephys_data/probe00',
            'revision': 'foobar',
            'namespace': 'iblqc',
            'object': 'ephysTimeRmsAP',
            'attribute': 'timestamps',
            'timescale': None,
            'extra': None,
            'extension': 'npy'
        }
        verifiable = re.match(alf_spec.regex(spec), rel_path).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Test a full path
        full = (
            'angelakilab/Subjects/NYU-40/2021-04-12/001/'
            'spike_sorters/ks2_matlab/probe01/'
            '_kilosort_raw.output.e8c9d765764778b7ee5bda08c982037f8f07e690.tar'
        )
        expected = {
            'lab': 'angelakilab',
            'subject': 'NYU-40',
            'date': '2021-04-12',
            'number': '001',
            'collection': 'spike_sorters/ks2_matlab/probe01',
            'revision': None,
            'namespace': 'kilosort',
            'object': 'raw',
            'attribute': 'output',
            'timescale': None,
            'extra': 'e8c9d765764778b7ee5bda08c982037f8f07e690',
            'extension': 'tar'
        }
        verifiable = alf_spec.regex().match(full).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Test a full path with no collection, no Subjects and no number padding
        full = (
            'NYU-40/2021-04-12/1/_kilosort_raw.output.e8c9d765764778b7ee5bda08c982037f8f07e690.tar'
        )
        expected.update(collection=None, number='1', lab=None)
        verifiable = re.match(alf_spec.regex(), full).groupdict()
        self.assertCountEqual(verifiable, expected)

    def test_dromedary(self):
        """Test for one.alf.spec._dromedary function."""
        self.assertEqual(alf_spec._dromedary('Hello world'), 'helloWorld')
        self.assertEqual(alf_spec._dromedary('motion_energy'), 'motionEnergy')
        self.assertEqual(alf_spec._dromedary('FooBarBaz'), 'fooBarBaz')
        self.assertEqual(alf_spec._dromedary('passive_RFM'), 'passiveRFM')
        self.assertEqual(alf_spec._dromedary('ROI Motion Energy'), 'ROIMotionEnergy')
        self.assertEqual(alf_spec._dromedary(''), '')

    def test_readable_ALF(self):
        """Test for one.alf.spec.readableALF function."""
        self.assertEqual(alf_spec.readableALF('DAQData'), 'DAQ data')
        self.assertEqual(alf_spec.readableALF('ROIMotion'), 'ROI motion')
        self.assertEqual(alf_spec.readableALF('blueChipTime'), 'blue chip time')
        self.assertEqual(alf_spec.readableALF('someROIDataset'), 'some ROI dataset')
        self.assertEqual(alf_spec.readableALF('fooBAR'), 'foo BAR')
        self.assertEqual(alf_spec.readableALF('fooBAR', capitalize=True), 'Foo BAR')

    def test_is_session_folder(self):
        """Test for one.alf.spec.is_session_folder"""
        inp = [(Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04'), False),
               ('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04', False),
               (Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001'), True),
               (Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001/tutu'), False),
               ('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001/', True)]
        for i in inp:
            self.assertEqual(alf_spec.is_session_path(i[0]), i[1], str(i[0]))

    def test_is_uuid_string(self):
        """Test for one.alf.spec.is_uuid_string"""
        testins = [
            None,
            'some_string',
            'f6ffe25827-06-425aaa-f5-919f70025835',
            'f6ffe258-2706-425a-aaf5-919f70025835']
        expected = [False, False, False, True]
        for i, e in zip(testins, expected):
            self.assertTrue(alf_spec.is_uuid_string(i) == e, i)

    def test_is_uuid(self):
        """Test for one.alf.spec.is_uuid"""
        hex_uuid = 'f6ffe25827-06-425aaa-f5-919f70025835'
        uuid_obj = uuid.UUID(hex_uuid)
        # Check valid inputs
        for valid in (hex_uuid, hex_uuid.replace('-', ''), uuid_obj.bytes, uuid_obj.int, uuid_obj):
            self.assertTrue(alf_spec.is_uuid(valid), f'{valid} is a valid uuid')
        # Check bad inputs
        for fake in (None, 54323, 'dddd-aaa-eeee'):
            self.assertFalse(alf_spec.is_uuid(fake), f'{fake} is not a valid uuid')

    def test_is_valid(self):
        """Test for one.alf.spec.is_valid"""
        self.assertTrue(alf_spec.is_valid('trials.feedbackType.npy'))
        self.assertTrue(alf_spec.is_valid(
            '_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.metadata.json'))
        self.assertFalse(alf_spec.is_valid('spike_train.npy'))
        self.assertTrue(alf_spec.is_valid('channels._phy_ids.csv'))

    def test_to_alf(self):
        """Test for one.alf.spec.to_alf"""
        filename = alf_spec.to_alf('spikes', 'times', 'ssv')
        self.assertEqual(filename, 'spikes.times.ssv')
        filename = alf_spec.to_alf('spikes', 'times', 'ssv', namespace='ibl')
        self.assertEqual(filename, '_ibl_spikes.times.ssv')
        filename = alf_spec.to_alf('spikes', 'times', 'ssv',
                                   namespace='ibl', timescale='ephysClock')
        self.assertEqual(filename, '_ibl_spikes.times_ephysClock.ssv')
        filename = alf_spec.to_alf('spikes', 'times', 'npy',
                                   namespace='ibl', timescale='ephysClock', extra='raw')
        self.assertEqual(filename, '_ibl_spikes.times_ephysClock.raw.npy')
        filename = alf_spec.to_alf('spikes', 'times', 'ssv',
                                   namespace='ibl', timescale=('ephys clock', 'minutes'))
        self.assertEqual(filename, '_ibl_spikes.times_ephysClock_minutes.ssv')
        filename = alf_spec.to_alf('wheel', 'timestamps', '.npy', 'ibl', 'bpod', ('raw', 'v12'))
        self.assertEqual(filename, '_ibl_wheel.timestamps_bpod.raw.v12.npy')

        with self.assertRaises(TypeError):
            alf_spec.to_alf('spikes', 'times', '')
        with self.assertRaises(ValueError):
            alf_spec.to_alf('spikes', 'foo_bar', 'npy')
        with self.assertRaises(ValueError):
            alf_spec.to_alf('spikes.times', 'fooBar', 'npy')
        with self.assertRaises(ValueError):
            alf_spec.to_alf('spikes', 'times', 'npy', namespace='_usr_')
        with self.assertRaises(ValueError):
            alf_spec.to_alf('_usr_spikes', 'times', 'npy')

    def test_path_pattern(self):
        """Test for one.alf.spec.path_pattern"""
        pattern = alf_spec.path_pattern()
        parts = alf_spec.regex(alf_spec.FULL_SPEC).groupindex.keys()
        self.assertTrue(all(x in pattern for x in parts))
        self.assertTrue('{' not in pattern)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_describe(self, sysout):
        """Test for one.alf.spec.describe"""
        alf_spec.describe('object', width=5)
        self.assertTrue('Every\nfile \ndescr' in sysout.getvalue())
        self.assertTrue(' ' + '^' * len('object') + ' ' in sysout.getvalue())
        self.assertTrue('EXTENSION' not in sysout.getvalue())
        alf_spec.describe()
        self.assertTrue(x.upper() in sysout.getvalue() for x in alf_spec.SPEC_DESCRIPTION.keys())
        with self.assertRaises(ValueError):
            alf_spec.describe('dimensions', width=5)


class TestALFErr(unittest.TestCase):
    """Tests for one.alf.exceptions"""
    def test_ALFError(self):
        """Test for one.alf.exceptions.ALFError"""
        # Check message
        self.assertEqual(str(ALFError()), '')
        self.assertTrue('foobar' in str(ALFError('foobar')))
        self.assertTrue(str(ALFError([0, 1, 2])).startswith('"[0, 1, 2]" '))
        ALFError.explanation = 'bazbazbaz'
        self.assertTrue(ALFError.explanation in str(ALFError('foobar')))
        self.assertFalse(ALFError.explanation in str(ALFError('foobar', terse=True)))
        with self.assertRaises(ALFError):
            raise ALFError()

    def tearDown(self) -> None:
        ALFError.explanation = ''


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
