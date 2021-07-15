import unittest
import tempfile
from pathlib import Path
import shutil
import json

import numpy as np

import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound


class TestAlfBunch(unittest.TestCase):

    def test_to_dataframe_scalars(self):
        simple = alfio.AlfBunch({'titi': np.random.rand(500), 'toto': np.random.rand(500)})
        df = simple.to_df()
        self.assertTrue(np.all(df['titi'].values == simple.titi))
        self.assertTrue(np.all(df['toto'].values == simple.toto))
        self.assertTrue(len(df.columns) == 2)

    def test_to_dataframe_vectors(self):
        vectors = alfio.AlfBunch({'titi': np.random.rand(500, 1),
                                  'toto': np.random.rand(500),
                                  'tata': np.random.rand(500, 2)})
        df = vectors.to_df()
        self.assertTrue(np.all(df['titi'].values == vectors.titi[:, 0]))
        self.assertTrue(np.all(df['toto'].values == vectors.toto))
        self.assertTrue(np.all(df['tata_0'].values == vectors.tata[:, 0]))
        self.assertTrue(np.all(df['tata_1'].values == vectors.tata[:, 1]))
        self.assertTrue(len(df.columns) == 4)

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

    def test_exists(self):
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

    def test_load_object(self):
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

    def test_save_npy(self):
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


class TestALFFolders(unittest.TestCase):
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.session_path = (Path(cls.tempdir.name)
                            .joinpath('fakelab', 'Subjects', 'fakemouse', '1900-01-01', '001'))
        cls.session_path.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_next_num_folder(self):
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
        root = Path(self.tempdir.name) / 'glob_dir'
        root.mkdir()
        root.joinpath('empty0').mkdir(exist_ok=True)
        root.joinpath('full0').mkdir(exist_ok=True)
        root.joinpath('full0', 'file.txt').touch()
        self.assertTrue(len(list(root.glob('*'))) == 2)
        alfio.remove_empty_folders(root)
        self.assertTrue(len(list(root.glob('*'))) == 1)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
