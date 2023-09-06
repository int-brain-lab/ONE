"""Unit tests for the one.remote.base module."""
import unittest
from unittest import mock
from pathlib import Path
import shutil
from functools import partial
from tempfile import TemporaryDirectory
import json

from iblutil.io import params as iopar

from one.tests.util import get_file, setup_rest_cache
from one.tests import TEST_DB_1
from one.webclient import AlyxClient
from one.remote import base


class TestBase(unittest.TestCase):
    """Tests for the one.remote.base module."""

    """unittest.mock._patch: Mock object for setting parameter location as temporary directory."""
    path_mock = None
    """tempfile.TemporaryDirectory: The temporary location of remote parameters file."""
    tempdir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = TemporaryDirectory()
        cls.path_mock = mock.patch('one.remote.base.iopar.getfile',
                                   new=partial(get_file, cls.tempdir.name))

    def setUp(self) -> None:
        self.path_mock.start()

    def test_load_client_params(self):
        """Tests for one.remote.base.load_client_params function."""
        # Check behaviour when no parameters file exists
        with self.assertRaises(FileNotFoundError):
            base.load_client_params(assert_present=True)
        self.assertIsNone(base.load_client_params(assert_present=False))

        # Check behaviour with parameters file
        iopar.write(base.PAR_ID_STR, {'foo': {'bar': 'baz'}})
        p = base.load_client_params(assert_present=True)
        self.assertIn('foo', p.as_dict())

        # Check behaviour when existing key provided
        p = base.load_client_params('foo', assert_present=True)
        self.assertEqual(p.bar, 'baz')

        with self.assertRaises(AttributeError):
            base.load_client_params('bar', assert_present=True)
        self.assertIsNone(base.load_client_params('bar', assert_present=False))

        # Loading a sub-key
        iopar.write(base.PAR_ID_STR, {'globus': {'default': {'par1': 'par2'}}})
        p = base.load_client_params('globus.default')
        self.assertIn('par1', p.as_dict())

    def test_save_client_params(self):
        """Tests for one.remote.base.save_client_params function."""
        # Check behaviour when saving all params
        expected = {'foo': {'bar': 'baz'}}
        base.save_client_params(expected)
        par_path = next(Path(self.tempdir.name).rglob('*remote'), None)
        self.assertIsNotNone(par_path)
        with open(par_path, 'r') as f:
            p = json.load(f)
        self.assertEqual(p, expected)

        # Check validation
        with self.assertRaises(ValueError):
            base.save_client_params({'foo': 'bar'})

        # Check behaviour when saving into client key
        base.save_client_params({'new': {'par1': 1}})
        with open(par_path, 'r') as f:
            p = json.load(f)
        self.assertIn('new', p)
        self.assertEqual(1, p['new']['par1'])

    def test_repo_from_alyx(self):
        """Test for DownloadClient.repo_from_alyx method."""
        ac = AlyxClient(**TEST_DB_1)
        setup_rest_cache(ac.cache_dir)  # Copy REST cache fixtures to temp dir
        record = base.DownloadClient.repo_from_alyx('mainenlab', ac)
        self.assertEqual('mainenlab', record['name'])

    def tearDown(self) -> None:
        par_path = Path(iopar.getfile('.one'))
        assert str(par_path).startswith(self.tempdir.name)
        shutil.rmtree(par_path)
        self.path_mock.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main(exit=False)
