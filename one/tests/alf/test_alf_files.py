import unittest
import tempfile
from pathlib import Path
import shutil
import re

import alf.files


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_filter_by(self):
        files = [
            'noalf.file',
            '_ibl_trials.intervals.npy',
            '_ibl_trials.intervals_bpod.csv',
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_namespace_obj.attr_timescale.raw.v12.ext']

        for f in files:
            (self.tmpdir / f).touch()

        # Test filter with None; should return files with no non-standard timescale
        alf_files, _ = alf.files.filter_by(self.tmpdir, timescale=None)
        expected = [
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_ibl_trials.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with None attribute')

        # Test filtering by object; should return only 'wheel' ALF objects
        alf_files, parts = alf.files.filter_by(self.tmpdir, object='wheel')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter by object')
        self.assertEqual(len(alf_files), len(parts))

        # Test wildcards; should return 'wheel' and 'wheelMoves' ALF objects
        alf_files, _ = alf.files.filter_by(self.tmpdir, object='wh*')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy', 'wheelMoves.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with wildcard')

        # Test filtering by specific timescale; test parts returned
        alf_files, parts = alf.files.filter_by(self.tmpdir, timescale='bpod')
        expected = ['_ibl_trials.intervals_bpod.csv']
        self.assertEqual(alf_files, expected, 'failed to filter by timescale')
        expected = ('ibl', 'trials', 'intervals', 'bpod', None, 'csv')
        self.assertTupleEqual(parts[0], expected)
        self.assertEqual(len(parts[0]), len(alf.files.ALF_EXP.groupindex))
        self.assertEqual(parts[0][alf.files.ALF_EXP.groupindex['timescale'] - 1], 'bpod')

        # Test filtering multiple attributes; should return only trials intervals
        alf_files, _ = alf.files.filter_by(self.tmpdir, attribute='intervals', object='trials')
        expected = ['_ibl_trials.intervals.npy', '_ibl_trials.intervals_bpod.csv']
        self.assertCountEqual(alf_files, expected, 'failed to filter by multiple attribute')

        # Test returning only ALF files
        alf_files, _ = alf.files.filter_by(self.tmpdir)
        self.assertCountEqual(alf_files, files[1:], 'failed to return ALF files')

        # Test return empty
        out = alf.files.filter_by(self.tmpdir, object=None)
        self.assertEqual(out, ([], []))

        # Test extras
        alf_files, _ = alf.files.filter_by(self.tmpdir, extra='v12')
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes')

        alf_files, _ = alf.files.filter_by(self.tmpdir, extra=['v12', 'raw'])
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes as list')

        alf_files, _ = alf.files.filter_by(self.tmpdir, extra=['foo', 'v12'])
        self.assertEqual(alf_files, [], 'failed to filter extra attributes')

        # Assert kwarg validation; should raise TypeError
        with self.assertRaises(TypeError):
            alf.files.filter_by(self.tmpdir, unknown=None)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestALFSpec(unittest.TestCase):
    def test_regex(self):
        # Should return the full regex with named capture groups by default
        verifiable = alf.files.regex()
        expected = ('(?P<lab>\\w+)/(Subjects/)?'
                    '(?P<subject>[\\w-]+)/(?P<date>\\d{4}-\\d{2}-\\d{2})/(?P<number>\\d{1,3})/'
                    '(?P<collection>[\\w/]+)(/#(?P<revision>[\\w-]+)#)?/'
                    '_?(?P<namespace>(?<=_)[a-zA-Z0-9]+)?_?(?P<object>\\w+)\\.'
                    '(?P<attribute>[a-zA-Z0-9]+(?:_times(?=[_\\b.])|_intervals(?=[_\\b.]))?)_?'
                    '(?P<timescale>(?:_?)\\w+)*\\.?(?P<extra>[.\\w-]+)*\\.(?P<extension>\\w+)$')
        self.assertEqual(expected, verifiable)

        # Should return only the filename regex pattern
        verifiable = alf.files.regex(spec=alf.files.FILE_SPEC)
        expected = expected[expected.index('_?(?P<namespace>'):]
        self.assertEqual(expected, verifiable)

        # Should replace the collection pattern
        verifiable = alf.files.regex(spec=alf.files.COLLECTION_SPEC, collection=r'probe\d{2}')
        self.assertEqual('(?P<collection>probe\\d{2})(/#(?P<revision>[\\w-]+)#)?', verifiable)

        # Should raise a key error
        with self.assertRaises(KeyError):
            alf.files.regex(foo=r'[bar]+')

    def test_named_group(self):
        verifiable = alf.files._named(r'[bar]+', 'foo')
        self.assertEqual(verifiable, '(?P<foo>[bar]+)')

    def test_patterns(self):
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
        verifiable = re.match(alf.files.regex(alf.files.FILE_SPEC), filename).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Match collection with filename
        spec = f'{alf.files.COLLECTION_SPEC}/{alf.files.FILE_SPEC}'
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
        verifiable = re.match(alf.files.regex(spec), rel_path).groupdict()
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
        verifiable = re.match(alf.files.regex(spec), rel_path).groupdict()
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
        verifiable = re.match(alf.files.regex(), full).groupdict()
        self.assertCountEqual(verifiable, expected)

    def test_dromedary(self):
        self.assertEqual(alf.files._dromedary('Hello world'), 'helloWorld')
        self.assertEqual(alf.files._dromedary('motion_energy'), 'motionEnergy')
        self.assertEqual(alf.files._dromedary('FooBarBaz'), 'fooBarBaz')
        self.assertEqual(alf.files._dromedary('passive_RFM'), 'passiveRFM')
        self.assertEqual(alf.files._dromedary('ROI Motion Energy'), 'ROIMotionEnergy')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
