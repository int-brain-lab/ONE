import unittest
import re

import one.alf.spec as alf_spec


class TestALFSpec(unittest.TestCase):
    def test_regex(self):
        # Should return the full regex with named capture groups by default
        verifiable = alf_spec.regex()
        expected = ('(?P<lab>\\w+)/(Subjects/)?'
                    '(?P<subject>[\\w-]+)/(?P<date>\\d{4}-\\d{2}-\\d{2})/(?P<number>\\d{1,3})/'
                    '((?P<collection>[\\w/]+)/)?(#(?P<revision>[\\w-]+)#/)?'
                    '_?(?P<namespace>(?<=_)[a-zA-Z0-9]+)?_?(?P<object>\\w+)\\.'
                    '(?P<attribute>[a-zA-Z0-9]+(?:_times(?=[_\\b.])|_intervals(?=[_\\b.]))?)_?'
                    '(?P<timescale>(?:_?)\\w+)*\\.?(?P<extra>[.\\w-]+)*\\.(?P<extension>\\w+)$')
        self.assertEqual(expected, verifiable)

        # Should return only the filename regex pattern
        verifiable = alf_spec.regex(spec=alf_spec.FILE_SPEC)
        expected = expected[expected.index('_?(?P<namespace>'):]
        self.assertEqual(expected, verifiable)

        # Should replace the collection pattern
        verifiable = alf_spec.regex(spec=alf_spec.COLLECTION_SPEC, collection=r'probe\d{2}')
        self.assertEqual('((?P<collection>probe\\d{2})/)?(#(?P<revision>[\\w-]+)#/)?', verifiable)

        # Should raise a key error
        with self.assertRaises(KeyError):
            alf_spec.regex(foo=r'[bar]+')

    def test_named_group(self):
        verifiable = alf_spec._named(r'[bar]+', 'foo')
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
        verifiable = re.match(alf_spec.regex(alf_spec.FILE_SPEC), filename).groupdict()
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
        verifiable = re.match(alf_spec.regex(), full).groupdict()
        self.assertCountEqual(verifiable, expected)

        # Test a full path with no collection, no Subjects and no number padding
        full = (
            'angelakilab/NYU-40/2021-04-12/1/'
            '_kilosort_raw.output.e8c9d765764778b7ee5bda08c982037f8f07e690.tar'
        )
        expected.update(collection=None, number='1')
        verifiable = re.match(alf_spec.regex(), full).groupdict()
        self.assertCountEqual(verifiable, expected)

    def test_dromedary(self):
        self.assertEqual(alf_spec._dromedary('Hello world'), 'helloWorld')
        self.assertEqual(alf_spec._dromedary('motion_energy'), 'motionEnergy')
        self.assertEqual(alf_spec._dromedary('FooBarBaz'), 'fooBarBaz')
        self.assertEqual(alf_spec._dromedary('passive_RFM'), 'passiveRFM')
        self.assertEqual(alf_spec._dromedary('ROI Motion Energy'), 'ROIMotionEnergy')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
