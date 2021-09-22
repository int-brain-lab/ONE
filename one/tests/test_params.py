"""Unit tests for the one.params module"""
import unittest
from unittest import mock
from pathlib import Path
from functools import partial

import one.params as params
from . import util


class TestONEParamUtil(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_key_from_url(self):
        key = params._key_from_url('https://sub.domain.org/')
        self.assertEqual(key, 'sub.domain.org')

        key = params._key_from_url('http://www.domain.org/db/?rest=True')
        self.assertEqual(key, 'www.domain.org_db__rest_true')

    def test_get_params_dir(self):
        par_dir = Path('path', 'to', 'params')
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, par_dir)):
            path = params.get_params_dir()
        self.assertIsInstance(path, Path)
        self.assertEqual('path/to/params/.one', path.as_posix())

    def test_get_rest_dir(self):
        par_dir = Path('path', 'to', 'params')
        url = 'https://sub.domain.net/'
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, par_dir)):
            path1 = params.get_rest_dir()
            path2 = params.get_rest_dir(url)

        expected = ('path', 'to', 'params', '.one', '.rest')
        self.assertCountEqual(expected, path1.parts)

        expected = (*expected, 'sub.domain.net', 'https')
        self.assertCountEqual(expected, path2.parts)

    def test_get_default_client(self):
        temp_dir = util.set_up_env()
        self.addCleanup(temp_dir.cleanup)
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, temp_dir.name)):
            self.assertIsNone(params.get_default_client())
            # Copy over caches fixture
            params.setup(silent=True)
            self.assertEqual(params.get_default_client(), 'openalyx.internationalbrainlab.org')


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
