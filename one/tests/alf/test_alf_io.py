"""Unit tests for the one.alf.io module."""
import logging
import unittest
import unittest.mock
import tempfile
from pathlib import Path
import shutil
import json
import uuid
import yaml

import numpy as np
import numpy.testing
import pandas as pd

from iblutil.io import jsonable

import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import FILE_SPEC, regex

try:
    import sparse
    SKIP_SPARSE = False
except ModuleNotFoundError:
    SKIP_SPARSE = True


class TestAlfBunch(unittest.TestCase):

    def test_to_dataframe_scalars(self):
        simple = alfio.AlfBunch({'titi': np.random.rand(500), 'toto': np.random.rand(500)})
        df = simple.to_df()
        self.assertTrue(np.all(df['titi'].values == simple.titi))
        self.assertTrue(np.all(df['toto'].values == simple.toto))
        self.assertTrue(len(df.columns) == 2)
        simple['titi'] = np.random.rand(50)
        with self.assertRaises(ValueError):
            simple.to_df()
        simple['toto'] = np.random.rand(50, 10, 5)
        with self.assertLogs(logging.getLogger('one.alf.io'), logging.WARNING):
            self.assertTrue('toto' not in simple.to_df().columns)

    def test_to_dataframe_vectors(self):
        vectors = alfio.AlfBunch({'titi': np.random.rand(500, 1),
                                  'toto': np.random.rand(500),
                                  'tata': np.random.rand(500, 12)})
        df = vectors.to_df()
        self.assertTrue(np.all(df['titi'].values == vectors.titi[:, 0]))
        self.assertTrue(np.all(df['toto'].values == vectors.toto))
        self.assertTrue(np.all(df['tata_0'].values == vectors.tata[:, 0]))
        self.assertTrue(np.all(df['tata_1'].values == vectors.tata[:, 1]))
        self.assertTrue(len(df.columns) == 12)
        self.assertEqual(10, len(df.filter(regex=r'tata_\d+', axis=1).columns),
                         'failed to truncate columns')

    def test_from_dataframe(self):
        """Tests for AlfBunch.from_df method"""
        cols = ['foo_0', 'foo_1', 'bar_0', 'bar_1', 'baz']
        df = pd.DataFrame(np.random.rand(10, 5), columns=cols)
        a = alfio.AlfBunch.from_df(df)
        self.assertIsInstance(a, alfio.AlfBunch)
        self.assertCountEqual(['foo', 'bar', 'baz'], a.keys())
        numpy.testing.assert_array_equal(df['foo_0'], a['foo'][:, 0])

    def test_append_numpy(self):
        a = alfio.AlfBunch({'titi': np.random.rand(500), 'toto': np.random.rand(500)})
        b = alfio.AlfBunch({})
        # test with empty elements
        self.assertTrue(np.all(np.equal(a.append({})['titi'], a['titi'])))
        self.assertTrue(np.all(np.equal(b.append(a)['titi'], a['titi'])))
        self.assertEqual(b.append({}), {})
        # test with numpy arrays
        b = alfio.AlfBunch({'titi': np.random.rand(250),
                            'toto': np.random.rand(250)})
        c = a.append(b)
        t = np.all(np.equal(c['titi'][0:500], a['titi']))
        t &= np.all(np.equal(c['toto'][0:500], a['toto']))
        t &= np.all(np.equal(c['titi'][500:], b['titi']))
        t &= np.all(np.equal(c['toto'][500:], b['toto']))
        self.assertTrue(t)
        a.append(b, inplace=True)
        self.assertTrue(np.all(np.equal(c['toto'], a['toto'])))
        self.assertTrue(np.all(np.equal(c['titi'], a['titi'])))
        # test warning thrown when uneven append occurs
        with self.assertLogs('one.alf.io', logging.WARNING):
            a.append({
                'titi': np.random.rand(10),
                'toto': np.random.rand(4)
            })

    def test_append_list(self):
        # test with lists
        a = alfio.AlfBunch({'titi': [0, 1, 3], 'toto': ['a', 'b', 'c']})
        b = alfio.AlfBunch({'titi': [1, 2, 4], 'toto': ['d', 'e', 'f']})
        c = a.append(b)
        self.assertTrue(len(c['toto']) == 6)
        self.assertTrue(len(a['toto']) == 3)
        c = c.append(b)
        self.assertTrue(len(c['toto']) == 9)
        self.assertTrue(len(a['toto']) == 3)
        c.append(b, inplace=True)
        self.assertTrue(len(c['toto']) == 12)
        self.assertTrue(len(a['toto']) == 3)
        with self.assertRaises(NotImplementedError):
            a.append(alfio.AlfBunch({'foobar': [8, 9, 10]}))
        a['foobar'] = '123'
        with self.assertLogs(logging.getLogger('one.alf.io'), logging.WARNING) as log:
            a.append({'titi': [5], 'toto': [8], 'foobar': 'd'})
            self.assertTrue('str' in log.output[0])

    def test_check_dimensions(self):
        a = alfio.AlfBunch({'titi': np.array([0, 1, 3]), 'toto': np.array(['a', 'b', 'c'])})
        self.assertFalse(a.check_dimensions)
        a['titi'] = np.append(a['titi'], 4)
        self.assertTrue(a.check_dimensions)


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_npy_parts_and_file_filters(self):
        a = {'riri': np.random.rand(100), 'fifi': np.random.rand(100)}
        alfio.save_object_npy(self.tmpdir, a, 'neuveux', parts='tutu')
        alfio.save_object_npy(self.tmpdir, a, 'neuveux', parts='tutu', timescale='toto')
        self.assertTrue(alfio.exists(self.tmpdir, 'neuveux'))

        b = alfio.load_object(self.tmpdir, 'neuveux', short_keys=True)

        # Should include timescale in keys
        self.assertCountEqual(list(b.keys()), ['fifi', 'fifi_toto', 'riri', 'riri_toto'])
        for k in a:
            self.assertTrue(np.all(a[k] == b[k]))

        # Test load with extra filter
        b = alfio.load_object(self.tmpdir, 'neuveux', timescale='toto', short_keys=True)
        self.assertCountEqual(list(b.keys()), ['fifi_toto', 'riri_toto'])
        with self.assertRaises(ALFObjectNotFound):
            alfio.load_object(self.tmpdir, 'neuveux', timescale='toto', namespace='baz')

        # also test file filters through wildcard
        self.assertTrue(alfio.exists(self.tmpdir, 'neu*'))
        c = alfio.load_object(self.tmpdir, 'neuveux', timescale='to*', short_keys=True)
        self.assertEqual(set(c.keys()), set([k for k in c.keys() if k.endswith('toto')]))

        # test with the long keys
        b = alfio.load_object(self.tmpdir, 'neuveux', short_keys=False)
        expected = ['fifi.tutu', 'fifi_toto.tutu', 'riri.tutu', 'riri_toto.tutu']
        self.assertCountEqual(list(b.keys()), expected)

        # Test duplicate attributes
        alfio.save_object_npy(self.tmpdir, a, 'neuveux', parts=['tutu', 'titi'])
        with self.assertRaises(AssertionError):
            alfio.load_object(self.tmpdir, 'neuveux', short_keys=True)
        # Restricting by extra parts and using long keys should succeed
        alfio.load_object(self.tmpdir, 'neuveux', extra=['tutu', 'titi'])
        alfio.load_object(self.tmpdir, 'neuveux', short_keys=False)

    def test_filter_by(self):
        """Test for one.alf.io.filter_by"""
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
        alf_files, _ = alfio.filter_by(self.tmpdir, timescale=None)
        expected = [
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_ibl_trials.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with None attribute')

        # Test filtering by object; should return only 'wheel' ALF objects
        alf_files, parts = alfio.filter_by(self.tmpdir, object='wheel')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter by object')
        self.assertEqual(len(alf_files), len(parts))

        # Test wildcards; should return 'wheel' and 'wheelMoves' ALF objects
        alf_files, _ = alfio.filter_by(self.tmpdir, object='wh*')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy', 'wheelMoves.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with wildcard')

        # Test wildcard arrays
        alf_files, _ = alfio.filter_by(self.tmpdir, object='wh*', attribute=['time*', 'pos*'])
        expected = ['wheel.position.npy', 'wheel.timestamps.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with wildcard')

        # Test filtering by specific timescale; test parts returned
        alf_files, parts = alfio.filter_by(self.tmpdir, timescale='bpod')
        expected = ['_ibl_trials.intervals_bpod.csv']
        self.assertEqual(alf_files, expected, 'failed to filter by timescale')
        expected = ('ibl', 'trials', 'intervals', 'bpod', None, 'csv')
        self.assertTupleEqual(parts[0], expected)
        self.assertEqual(len(parts[0]), len(spec_idx_map))
        self.assertEqual(parts[0][spec_idx_map['timescale'] - 1], 'bpod')

        # Test filtering multiple attributes; should return only trials intervals
        alf_files, _ = alfio.filter_by(self.tmpdir, attribute='intervals', object='trials')
        expected = ['_ibl_trials.intervals.npy', '_ibl_trials.intervals_bpod.csv']
        self.assertCountEqual(alf_files, expected, 'failed to filter by multiple attribute')

        # Test returning only ALF files
        alf_files, _ = alfio.filter_by(self.tmpdir)
        self.assertCountEqual(alf_files, file_names[1:], 'failed to return ALF files')

        # Test return empty
        out = alfio.filter_by(self.tmpdir, object=None)
        self.assertEqual(out, ([], []))

        # Test extras
        alf_files, _ = alfio.filter_by(self.tmpdir, extra='v12')
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes')

        alf_files, _ = alfio.filter_by(self.tmpdir, extra=['v12', 'raw'])
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes as list')

        alf_files, _ = alfio.filter_by(self.tmpdir, extra=['foo', 'v12'])
        self.assertEqual(alf_files, [], 'failed to filter extra attributes')

        # Assert kwarg validation; should raise TypeError
        with self.assertRaises(TypeError):
            alfio.filter_by(self.tmpdir, unknown=None)

        # Check regular expression search
        alf_files, _ = alfio.filter_by(self.tmpdir, object='^wheel.*', wildcards=False)
        expected = ['wheel.position.npy', 'wheel.timestamps.npy', 'wheelMoves.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter by regex')
        # Should work with lists
        alf_files, _ = alfio.filter_by(self.tmpdir, object=['^wheel$', '.*Moves'], wildcards=False)
        self.assertCountEqual(alf_files, expected, 'failed to filter by regex')

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestsAlf(unittest.TestCase):
    def setUp(self) -> None:
        # riri, fifi and loulou are huey, duey and louie in French (Donald nephews for ignorants)
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)
        self.vfile = self.tmpdir / 'toto.titi.npy'
        self.tfile = self.tmpdir / 'toto.timestamps.npy'
        self.object_files = [self.tmpdir / 'neuveu.riri.npy',
                             self.tmpdir / 'neuveu.fifi.npy',
                             self.tmpdir / 'neuveu.loulou.npy',
                             self.tmpdir / 'object.attribute.part1.part2.npy',
                             self.tmpdir / 'object.attribute.part1.npy',
                             self.tmpdir / 'neuveu.foobar_matlab.npy']
        for f in self.object_files:
            shape = (5, 1) if 'matlab' in str(f) else (5,)
            np.save(file=f, arr=np.random.rand(*shape))
        self.object_files.append(self.tmpdir / 'neuveu.timestamps.npy')
        np.save(file=self.object_files[-1], arr=np.ones((2, 2)))
        # Save an obj.data pqt file
        self.object_files.append(self.tmpdir / 'obj.table.pqt')
        cols = ['foo_0', 'foo_1', 'bar_0', 'bar_1', 'baz']
        pd.DataFrame(np.random.rand(10, 5), columns=cols).to_parquet(self.object_files[-1])

    def test_exists(self):
        """Test for one.alf.io.exists"""
        self.assertFalse(alfio.exists(self.tmpdir, 'asodiujfas'))
        self.assertTrue(alfio.exists(self.tmpdir, 'neuveu'))
        # test with attribute string only
        self.assertTrue(alfio.exists(self.tmpdir, 'neuveu', attributes='riri'))
        # test with list of attributes
        self.assertTrue(alfio.exists(self.tmpdir, 'neuveu', attributes=['riri', 'fifi']))
        self.assertFalse(alfio.exists(self.tmpdir, 'neuveu', attributes=['riri', 'fifiasdf']))
        # test with extras
        self.assertTrue(alfio.exists(self.tmpdir, 'object', extra='part2'))
        self.assertTrue(alfio.exists(self.tmpdir, 'object', extra=['part1', 'part2']))
        self.assertTrue(alfio.exists(self.tmpdir, 'neuveu', extra=None))
        # test with wildcards
        self.assertTrue(alfio.exists(self.tmpdir, 'neu*', attributes='riri'))
        # globing with list: an empty part should return true as well
        self.assertTrue(alfio.exists(self.tmpdir, 'object', extra=['']))

    def test_metadata_columns(self):
        # simple test with meta data to label columns
        file_alf = self.tmpdir / '_ns_object.attribute.npy'
        data = np.random.rand(500, 4)
        cols = ['titi', 'tutu', 'toto', 'tata']
        np.save(file_alf, data)
        np.save(self.tmpdir / '_ns_object.gnagna.npy', data[:, -1])
        alfio.save_metadata(file_alf, {'columns': cols})
        dread = alfio.load_object(self.tmpdir, 'object', namespace='ns', short_keys=False)
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))
        self.assertTrue(np.all(dread['gnagna'] == data[:, -1]))
        # add another field to the metadata
        alfio.save_metadata(file_alf, {'columns': cols, 'unit': 'potato'})
        dread = alfio.load_object(self.tmpdir, 'object', namespace='ns', short_keys=False)
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))
        self.assertTrue(dread['attributemetadata']['unit'] == 'potato')
        self.assertTrue(np.all(dread['gnagna'] == data[:, -1]))

    def test_metadata_columns_UUID(self):
        data = np.random.rand(500, 4)
        # test with UUID extra field
        file_alf = self.tmpdir / '_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.npy'
        np.save(file_alf, data)
        cols = ['titi', 'tutu', 'toto', 'tata']
        file_meta = file_alf.parent / (file_alf.stem + '.metadata.json')
        with open(file_meta, 'w+') as fid:
            fid.write(json.dumps({'columns': cols}, indent=1))
        dread = alfio.load_object(self.tmpdir, 'obj', namespace='ns', short_keys=False)
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))

    def test_read_ts(self):
        """Test for one.alf.io.read_ts"""
        # simplest test possible with one column in each file
        t = np.arange(0, 10)
        d = np.random.rand(10)
        np.save(self.vfile, d)
        np.save(self.tfile, t)
        t_, d_ = alfio.read_ts(self.vfile)
        self.assertTrue(np.all(t_ == t))
        self.assertTrue(np.all(d_ == d))

        # Test expands timeseries and deals with single column 2D vectors
        t = np.array([[0, 10], [0.3, 0.4]]).T
        d = np.random.rand(10, 1)
        np.save(self.vfile, d)
        np.save(self.tfile, t)
        t_, d_ = alfio.read_ts(str(self.vfile))
        self.assertEqual(d_.ndim, 1)
        expected = np.around(np.arange(t[0, 1], t[1, 1], .01)[:-1], 2)
        np.testing.assert_array_equal(t_, expected)

        self.tfile.unlink()
        with self.assertRaises(FileNotFoundError):
            alfio.read_ts(self.vfile)

    def test_load_object(self):
        """Test for one.alf.io.load_object"""
        # first usage of load object is to provide one of the files belonging to the object
        expected_keys = {'riri', 'fifi', 'loulou', 'foobar_matlab', 'timestamps'}
        obj = alfio.load_object(self.object_files[0])
        self.assertTrue(obj.keys() == expected_keys)
        # Check flattens single column 2D vectors
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # the second usage is to provide a directory and the object name
        obj = alfio.load_object(self.tmpdir, 'neuveu')
        self.assertTrue(obj.keys() == expected_keys)
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # providing directory without object will return all ALF files
        with self.assertRaises(ValueError) as context:
            alfio.load_object(self.tmpdir)
        self.assertTrue('object name should be provided too' in str(context.exception))
        # Check key conflicts
        np.save(file=str(self.tmpdir / 'neuveu.loulou.extra.npy'), arr=np.random.rand(5,))
        obj = alfio.load_object(self.tmpdir, 'neuveu', short_keys=False)
        self.assertTrue('loulou.extra' in obj)
        with self.assertRaises(AssertionError):
            alfio.load_object(self.tmpdir, 'neuveu', short_keys=True)
        # the third usage is to provide file list
        obj = alfio.load_object(self.object_files[:3], short_keys=False)
        self.assertEqual(3, len(obj))
        # Check dimension mismatch
        data = np.random.rand(list(obj.values())[0].size + 1)
        np.save(file=str(self.object_files[0]), arr=data)  # Save a new size
        with self.assertLogs(logging.getLogger('one.alf.io'), logging.WARNING) as log:
            alfio.load_object(self.tmpdir, 'neuveu', short_keys=False)
        self.assertIn(str(data.shape), log.output[0])
        # Check loading of 'table' attribute
        obj = alfio.load_object(self.tmpdir, 'obj')
        self.assertIsInstance(obj, alfio.AlfBunch)
        self.assertCountEqual(obj.keys(), ['foo', 'bar', 'baz'])
        self.assertEqual(obj['foo'].shape, (10, 2))
        self.assertEqual(obj['bar'].shape, (10, 2))
        self.assertEqual(obj['baz'].shape, (10,))
        # Check behaviour on conflicting keys
        np.save(self.tmpdir.joinpath('obj.baz.npy'), np.arange(len(obj['foo'])))
        new_obj = alfio.load_object(self.tmpdir, 'obj')
        self.assertNotIn('table', new_obj)
        np.testing.assert_array_equal(new_obj['baz'], obj['baz'],
                                      'Table attribute should take precedent')
        # Check behaviour loading table with long keys / extra ALF parts
        table_file = next(self.tmpdir.glob('*table*'))
        new_name = table_file.stem + '_clock.extra' + table_file.suffix
        table_file.rename(table_file.parent.joinpath(new_name))
        new_obj = alfio.load_object(self.tmpdir, 'obj')
        expected = ['baz', 'baz_clock.extra', 'bar_clock.extra', 'foo_clock.extra']
        self.assertCountEqual(expected, new_obj.keys())

    def test_ls(self):
        """Test for one.alf.io._ls"""
        # Test listing all ALF files in a directory
        alf_files, _ = alfio._ls(self.tmpdir)
        self.assertIsInstance(alf_files[0], Path)
        self.assertEqual(8, len(alf_files))

        # Test with filepath
        alf_files, parts = alfio._ls(sorted(alf_files)[0])
        self.assertEqual(5, len(alf_files))
        self.assertTrue(all(x[1] == 'neuveu') for x in parts)

        # Test non-existent
        with self.assertRaises(ALFObjectNotFound):
            alfio._ls(self.tmpdir.joinpath('foobar'))

    def test_save_npy(self):
        """Test for one.alf.io.save_npy"""
        # test with straight vectors
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100)}
        alfio.save_object_npy(self.tmpdir, a, 'neuveux')
        # read after write
        b = alfio.load_object(self.tmpdir, 'neuveux')
        for k in a:
            self.assertTrue(np.all(a[k] == b[k]))
        # test with more exotic shapes, still valid
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100, 2),
             'loulou': np.random.rand(1, 2)}
        alfio.save_object_npy(self.tmpdir, a, 'neuveux')
        # read after write
        b = alfio.load_object(self.tmpdir, 'neuveux')
        for k in a:
            self.assertTrue(np.all(a[k] == b[k]))
        # test with non allowed shape
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100, 2),
             'loulou': np.random.rand(5, 2)}
        with self.assertRaises(Exception) as context:
            alfio.save_object_npy(self.tmpdir, a, 'neuveux')
        self.assertTrue('Dimensions are not consistent' in str(context.exception))

    def test_check_dimensions(self):
        """Test for one.alf.io.check_dimensions"""
        a = {'a': np.ones([10, 10]), 'b': np.ones([10, 2]), 'c': np.ones([10])}
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([10, 10]), 'b': np.ones([10, 1]), 'c': np.ones([10])}
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([10, 15]), 'b': np.ones([1, 15]), 'c': np.ones([10])}
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([9, 10]), 'b': np.ones([10, 1]), 'c': np.ones([10])}
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 1)
        # test for timestamps which is an exception to the rule
        a = {'a': np.ones([10, 15]), 'b': np.ones([1, 15]), 'c': np.ones([10])}
        a['timestamps'] = np.ones([2, 2])
        a['timestamps_titi'] = np.ones([10, 1])
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 0)
        a['timestamps'] = np.ones([2, 4])
        status = alfio.check_dimensions(a)
        self.assertTrue(status == 1)

    def test_ts2vec(self):
        """Test for one.alf.io.ts2vec"""
        n = 10
        # Test interpolate
        ts = np.array([[0, 10], [0, 100]]).T
        ts_ = alfio.ts2vec(ts, n)
        np.testing.assert_array_equal(ts_.astype(int), np.arange(0, 100, 10, dtype=int))
        # Test flatten
        ts = np.ones((n, 1))
        ts_ = alfio.ts2vec(ts, n)
        np.testing.assert_array_equal(ts_, np.ones(n))
        # Test identity
        np.testing.assert_array_equal(ts_, alfio.ts2vec(ts_, n))
        # Test ValueError
        with self.assertRaises(ValueError):
            alfio.ts2vec(np.empty((n, 2, 3)), n)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestsLoadFile(unittest.TestCase):
    """Tests for one.alf.io.load_fil_content function."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.empty = Path(self.tmpdir.name) / 'foo.bar.npy'
        self.empty.touch()
        self.npy = Path(self.tmpdir.name) / 'foo.baz.npy'
        np.save(file=self.npy, arr=np.random.rand(5))
        self.csv = Path(self.tmpdir.name) / 'foo.baz.csv'
        self.csvuids = Path(self.tmpdir.name) / 'uuids.csv'
        with open(self.csv, 'w') as f:
            f.write('a,b,c\n1,2,3')
        with open(self.csvuids, 'w') as f:
            f.write('\n'.join(['uuids'] + [str(uuid.uuid4()) for _ in range(6)]))
        self.ssv = Path(self.tmpdir.name) / 'foo.baz.ssv'
        with open(self.ssv, 'w') as f:
            f.write('a b c\n1 2 3')
        self.tsv = Path(self.tmpdir.name) / 'foo.baz.tsv'
        with open(self.tsv, 'w') as f:
            f.write('a\tb\tc\n1\t2\t3')
        self.json1 = Path(self.tmpdir.name) / 'foo.baz.json'
        with open(self.json1, 'w') as f:
            json.dump({'a': [1, 2, 3], 'b': [4, 5, 6]}, f)
        self.json2 = Path(self.tmpdir.name) / '_broken_foo.baz.json'
        with open(self.json2, 'w') as f:
            f.write('{"a": [1, 2, 3],"b": [4, 5 6]}')
        self.json3 = Path(self.tmpdir.name) / 'foo.baz.jsonable'
        jsonable.write(self.json3, {'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.yaml = Path(self.tmpdir.name) / 'foo.baz.yaml'
        with open(self.yaml, 'w') as f:
            yaml.dump({'a': [1, 2, 3], 'b': [4, 5, 6]}, f)
        self.xyz = Path(self.tmpdir.name) / 'foo.baz.xyz'
        with open(self.xyz, 'wb') as f:
            f.write(b'\x00\x00')

    def test_load_file_content(self):
        """Test for one.alf.io.load_file_content"""
        self.assertIsNone(alfio.load_file_content(self.empty))
        # csv / ssv / tsv files
        self.assertIsInstance(alfio.load_file_content(self.npy), np.ndarray)
        for file in (self.csv, self.ssv, self.tsv):
            with self.subTest('Loading text files', delim=file.suffix):
                loaded = alfio.load_file_content(file)
                self.assertEqual(3, loaded.size)
                self.assertCountEqual(loaded.columns, ['a', 'b', 'c'])
        # a single column file should be squeezed
        loaded = alfio.load_file_content(self.csvuids)
        self.assertEqual(loaded.shape, (6, ))
        loaded = alfio.load_file_content(self.json1)
        self.assertCountEqual(loaded.keys(), ['a', 'b'])
        self.assertIsNone(alfio.load_file_content(self.json2))
        loaded = alfio.load_file_content(self.json3)
        self.assertCountEqual(loaded, ['a', 'b'])
        # Load a parquet file
        pqt = next(Path(__file__).parents[1].joinpath('fixtures').glob('*.pqt'))
        loaded = alfio.load_file_content(pqt)
        self.assertIsInstance(loaded, pd.DataFrame)
        # Unknown file should return Path
        file = alfio.load_file_content(str(self.xyz))
        self.assertEqual(file, self.xyz)
        self.assertIsNone(alfio.load_file_content(None))
        # Load YAML file
        loaded = alfio.load_file_content(str(self.yaml))
        self.assertCountEqual(loaded.keys(), ['a', 'b'])

    def tearDown(self) -> None:
        self.tmpdir.cleanup()


@unittest.skipIf(SKIP_SPARSE, 'pydata sparse package not installed')
class TestsLoadFileNonStandard(unittest.TestCase):
    """Tests for one.alf.io.load_fil_content function with non-standard libraries."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.sparse_npz = Path(self.tmpdir.name) / 'foo.baz.sparse_npz'
        with open(self.sparse_npz, 'wb') as fp:
            sparse.save_npz(fp, sparse.random((2, 2, 2)))

    def test_load_sparse_npz(self):
        loaded = alfio.load_file_content(str(self.sparse_npz))
        self.assertIsInstance(loaded, sparse.COO)
        with unittest.mock.patch('sparse.load_npz', side_effect=ModuleNotFoundError), \
                self.assertWarns(UserWarning):
            loaded = alfio.load_file_content(str(self.sparse_npz))
            self.assertEqual(loaded, self.sparse_npz)


class TestUUID_Files(unittest.TestCase):

    def test_remove_uuid(self):
        with tempfile.TemporaryDirectory() as dir:
            f1 = Path(dir).joinpath('tutu.part1.part1.30c09473-4d3d-4f51-9910-c89a6840096e.json')
            f2 = Path(dir).joinpath('tata.part1.part1.json')
            f3 = Path(dir).joinpath('toto.json')
            f1.touch()
            f2.touch()
            f2.touch()
            self.assertTrue(alfio.remove_uuid_file(f1) ==
                            Path(dir).joinpath('tutu.part1.part1.json'))
            self.assertTrue(alfio.remove_uuid_file(f2) ==
                            Path(dir).joinpath('tata.part1.part1.json'))
            self.assertTrue(alfio.remove_uuid_file(f3) ==
                            Path(dir).joinpath('toto.json'))
            self.assertTrue(alfio.remove_uuid_file(str(f3)) ==
                            Path(dir).joinpath('toto.json'))

    def test_remove_uuid_recusive(self):
        uuid = '30c09473-4d3d-4f51-9910-c89a6840096e'
        with tempfile.TemporaryDirectory() as dir:
            f1 = Path(dir).joinpath(f'tutu.part1.part1.{uuid}.json')
            f2 = Path(dir).joinpath('tata.part1.part1.json')
            f3 = Path(dir).joinpath('toto.json')
            f4 = Path(dir).joinpath('collection', f'tutu.part1.part1.{uuid}.json')
            f1.touch()
            f2.touch()
            f2.touch()
            f3.touch()
            f4.parent.mkdir()
            f4.touch()
            alfio.remove_uuid_recursive(Path(dir))
            self.assertFalse(len(list(Path(dir).rglob(f'*{uuid}*'))))


class TestALFFolders(unittest.TestCase):
    tempdir = None
    session_path = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.session_path = (Path(cls.tempdir.name)
                            .joinpath('fakelab', 'Subjects', 'fakemouse', '1900-01-01', '001'))
        cls.session_path.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def tearDown(self) -> None:
        for obj in reversed(sorted(Path(self.session_path).rglob('*'))):
            obj.unlink() if obj.is_file() else obj.rmdir()

    def test_next_num_folder(self):
        """Test for one.alf.io.next_num_folder"""
        self.session_path.rmdir()  # Remove '001' folder
        next_num = alfio.next_num_folder(self.session_path.parent)
        self.assertEqual('001', next_num)

        self.session_path.parent.rmdir()  # Remove date folder
        next_num = alfio.next_num_folder(self.session_path.parent)
        self.assertEqual('001', next_num)

        self.session_path.parent.joinpath(next_num).mkdir(parents=True)  # Add '001' folder
        next_num = alfio.next_num_folder(self.session_path.parent)
        self.assertEqual('002', next_num)

        self.session_path.parent.joinpath('053').mkdir()  # Add '053' folder
        next_num = alfio.next_num_folder(self.session_path.parent)
        self.assertEqual('054', next_num)

        self.session_path.parent.joinpath('099').mkdir()  # Add '099' folder
        next_num = alfio.next_num_folder(self.session_path.parent)
        self.assertEqual('100', next_num)

        self.session_path.parent.joinpath('999').mkdir()  # Add '999' folder
        with self.assertRaises(AssertionError):
            alfio.next_num_folder(self.session_path.parent)

    def test_remove_empty_folders(self):
        """Test for one.alf.io.remove_empty_folders"""
        root = Path(self.tempdir.name) / 'glob_dir'
        root.mkdir()
        root.joinpath('empty0').mkdir(exist_ok=True)
        root.joinpath('full0').mkdir(exist_ok=True)
        root.joinpath('full0', 'file.txt').touch()
        self.assertTrue(len(list(root.glob('*'))) == 2)
        alfio.remove_empty_folders(root)
        self.assertTrue(len(list(root.glob('*'))) == 1)

    def test_iter_sessions(self):
        """Test for one.alf.io.iter_sessions"""
        # Create invalid session folder
        self.session_path.parent.parent.joinpath('bad_session').mkdir()

        valid_sessions = alfio.iter_sessions(self.tempdir.name)
        self.assertEqual(next(valid_sessions), self.session_path)
        self.assertFalse(next(valid_sessions, False))
        # makes sure that the session path returns itself on the iterator
        self.assertEqual(self.session_path, next(alfio.iter_sessions(self.session_path)))

    def test_iter_datasets(self):
        """Test for one.alf.io.iter_datasets"""
        # Create valid dataset
        dset = self.session_path.joinpath('collection', 'object.attribute.ext')
        dset.parent.mkdir()
        dset.touch()
        # Create invalid dataset
        self.session_path.joinpath('somefile.txt').touch()

        ses_files = list(alfio.iter_datasets(self.session_path))
        self.assertEqual([Path(*dset.parts[-2:])], ses_files)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
