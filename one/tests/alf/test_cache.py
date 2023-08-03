"""Unit tests for the one.alf.cache module."""
import unittest
import tempfile
from pathlib import Path
import shutil
import datetime

import pandas as pd
from pandas.testing import assert_frame_equal

from iblutil.io import parquet
import one.alf.cache as apt
from one.tests.util import revisions_datasets_table


class TestsONEParquet(unittest.TestCase):
    """Tests for the make_parquet_db function and its helpers"""
    rel_ses_path = 'mylab/Subjects/mysub/2021-02-28/001/'
    ses_info = {
        'lab': 'mylab',
        'subject': 'mysub',
        'date': datetime.date.fromisoformat('2021-02-28'),
        'number': int('001'),
        'projects': '',
        'task_protocol': '',
        'id': 'mylab/mysub/2021-02-28/001',
    }
    rel_ses_files = [Path('alf/spikes.clusters.npy'), Path('alf/spikes.times.npy')]

    def setUp(self) -> None:
        pd.set_option('display.max_columns', 12)

        # root path:
        self.tmpdir = Path(tempfile.gettempdir()) / 'pqttest'
        self.tmpdir.mkdir(exist_ok=True)
        # full session path:
        self.full_ses_path = self.tmpdir / self.rel_ses_path
        (self.full_ses_path / 'alf').mkdir(exist_ok=True, parents=True)

        self.file_path = self.full_ses_path / 'alf/spikes.times.npy'
        self.file_path.write_text('mock')

        sc = self.full_ses_path / 'alf/spikes.clusters.npy'
        sc.write_text('mock2')

        # Create a second session containing an invalid dataset
        second_session = self.tmpdir.joinpath(self.rel_ses_path).parent.joinpath('002')
        second_session.mkdir()
        second_session.joinpath('trials.intervals.npy').touch()
        second_session.joinpath('.invalid').touch()

    def test_parse(self):
        self.assertEqual(apt._get_session_info(self.rel_ses_path), self.ses_info)
        self.assertTrue(
            self.full_ses_path.as_posix().endswith(self.rel_ses_path[:-1]))

    def test_parquet(self):
        # Test data
        columns = ('colA', 'colB')
        rows = [('a1', 'b1'), ('a2', 'b2')]
        metadata = apt._metadata('dbname')
        filename = self.tmpdir.resolve() / 'mypqt.pqt'

        # Save parquet file.
        df = pd.DataFrame(rows, columns=columns)
        parquet.save(filename, df, metadata=metadata)

        # Load parquet file
        df2, metadata2 = parquet.load(filename)
        assert_frame_equal(df, df2)
        self.assertTrue(metadata == metadata2)

    def test_sessions_df(self):
        df = apt._make_sessions_df(self.tmpdir)
        print('Sessions dataframe')
        print(df)
        self.assertEqual(df.loc[0].to_dict(), self.ses_info)

    def test_datasets_df(self):
        df = apt._make_datasets_df(self.tmpdir)
        print('Datasets dataframe')
        print(df)
        dset_info = df.loc[0].to_dict()
        self.assertEqual(dset_info['session_path'], self.rel_ses_path[:-1])
        self.assertEqual(dset_info['rel_path'], self.rel_ses_files[0].as_posix())
        self.assertTrue(dset_info['file_size'] > 0)
        self.assertFalse(df.rel_path.str.contains('invalid').any())

    def tests_db(self):
        fn_ses, fn_dsets = apt.make_parquet_db(self.tmpdir, hash_ids=False)
        metadata_exp = apt._metadata(self.tmpdir.resolve())

        df_ses, metadata = parquet.load(fn_ses)

        # Check sessions dataframe.
        self.assertEqual(metadata, metadata_exp)
        self.assertEqual(df_ses.loc[0].to_dict(), self.ses_info)

        # Check datasets dataframe.
        df_dsets, metadata2 = parquet.load(fn_dsets)
        self.assertEqual(metadata2, metadata_exp)
        dset_info = df_dsets.loc[0].to_dict()
        self.assertEqual(dset_info['session_path'], self.rel_ses_path[:-1])
        self.assertEqual(dset_info['rel_path'], self.rel_ses_files[0].as_posix())

        # Check behaviour when no files found
        with tempfile.TemporaryDirectory() as tdir:
            with self.assertWarns(RuntimeWarning):
                fn_ses, fn_dsets = apt.make_parquet_db(tdir, hash_ids=False)
            self.assertTrue(parquet.load(fn_ses)[0].empty)
            self.assertTrue(parquet.load(fn_dsets)[0].empty)

        # Check labname arg
        with self.assertRaises(AssertionError):
            apt.make_parquet_db(self.tmpdir, hash_ids=False, lab='another')

        # Create some more datasets in a session folder outside of a lab directory
        dsets = revisions_datasets_table()
        with tempfile.TemporaryDirectory() as tdir:
            for session_path, rel_path in dsets[['session_path', 'rel_path']].values:
                filepath = Path(tdir).joinpath(session_path, rel_path)
                filepath.parent.mkdir(exist_ok=True, parents=True)
                filepath.touch()
            fn_ses, _ = apt.make_parquet_db(tdir, hash_ids=False, lab='another')
            df_ses, _ = parquet.load(fn_ses)
            self.assertTrue((df_ses['lab'] == 'another').all())

    def test_hash_ids(self):
        # Build and load caches with int UUIDs
        (ses, _), (dsets, _) = map(parquet.load, apt.make_parquet_db(self.tmpdir, hash_ids=True))
        # Check ID fields in both dataframes
        self.assertTrue(ses.index.nlevels == 1 and ses.index.name == 'id')
        self.assertTrue(dsets.index.nlevels == 2 and tuple(dsets.index.names) == ('eid', 'id'))

    def test_remove_missing_datasets(self):
        # Add a session that will only contains missing datasets
        ghost_session = self.tmpdir.joinpath('lab', 'Subjects', 'sub', '2021-01-30', '001')
        ghost_session.mkdir(parents=True)

        tables = {
            'sessions': apt._make_sessions_df(self.tmpdir),
            'datasets': apt._make_datasets_df(self.tmpdir)
        }

        # Touch some files and folders for deletion
        empty_missing_session = self.tmpdir.joinpath(self.rel_ses_path).parent.joinpath('003')
        empty_missing_session.mkdir()
        missing_dataset = self.tmpdir.joinpath(self.rel_ses_path).joinpath('foo.bar.npy')
        missing_dataset.touch()
        ghost_dataset = ghost_session.joinpath('foo.bar.npy')
        ghost_dataset.touch()

        # Test dry
        to_remove = apt.remove_missing_datasets(
            self.tmpdir, tables=tables, dry=True, remove_empty_sessions=False
        )
        self.assertTrue(all(map(Path.exists, to_remove)), 'Removed files during dry run!')
        self.assertTrue(all(map(Path.is_file, to_remove)), 'Failed to ignore empty folders')
        self.assertNotIn(empty_missing_session, to_remove, 'Failed to ignore empty folders')
        self.assertNotIn(next(self.tmpdir.rglob('.invalid')), to_remove, 'Removed non-ALF file')

        # Test removal of files and folders
        removed = apt.remove_missing_datasets(
            self.tmpdir, tables=tables, dry=False, remove_empty_sessions=True
        )
        self.assertTrue(sum(map(Path.exists, to_remove)) == 0, 'Failed to remove all files')
        self.assertIn(empty_missing_session, removed, 'Failed to remove empty session folder')
        self.assertIn(missing_dataset, removed, 'Failed to remove missing dataset')
        self.assertIn(ghost_dataset, removed, 'Failed to remove missing dataset')
        self.assertNotIn(ghost_session, removed, 'Removed empty session that was in session table')

        # Check without tables input
        apt.make_parquet_db(self.tmpdir, hash_ids=False)
        removed = apt.remove_missing_datasets(self.tmpdir, dry=False)
        self.assertTrue(len(removed) == 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == '__main__':
    unittest.main(exit=False)
