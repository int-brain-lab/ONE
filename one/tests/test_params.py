"""Unit tests for the one.params module.

NB: `setup` function also tested with TestOneSetup class in one.tests.test_one.
"""
import unittest
from unittest import mock
from pathlib import Path
from functools import partial
import tempfile

import one.params
import one.params as params
from . import util
from . import TEST_DB_1


class TestParamSetup(unittest.TestCase):
    """Test for one.params.setup function"""
    def setUp(self) -> None:
        self.par_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.par_dir.cleanup)
        self.url = TEST_DB_1['base_url'][8:]  # URL without schema
        # Change the location of the parameters to our temp dir
        get_file = partial(util.get_file, self.par_dir.name)
        self.get_file_mock = mock.patch('iblutil.io.params.getfile', new=get_file)
        self.get_file_mock.start()
        self.addCleanup(self.get_file_mock.stop)

    def _mock_input(self, prompt, **kwargs):
        """Stub function for builtins.input"""
        if prompt.lower().startswith('warning'):
            return 'n'
        elif 'url' in prompt.lower():
            return self.url
        else:
            for k, v in kwargs.items():
                if k in prompt:
                    return v
            return ''

    @mock.patch('one.params.getpass', return_value='mock_pwd')
    def test_setup(self, _):
        """Test fresh setup with default args"""
        with mock.patch('one.params.input', new=self._mock_input):
            cache = one.params.setup()
        # Check client map contains our url sans schema
        self.assertTrue(self.url in cache.CLIENT_MAP)
        # Check default download location includes client key
        self.assertTrue(cache.CLIENT_MAP[self.url].endswith(self.url))
        # Check default
        self.assertEqual(cache.DEFAULT, self.url)
        # Check that it added the schema to the URL
        par = one.params.get(self.url, silent=True)
        self.assertEqual('https://' + self.url, par.ALYX_URL)
        self.assertEqual('mock_pwd', par.HTTP_DATA_SERVER_PWD)

        # Check verification prompt
        resp_map = {'ALYX_LOGIN': 'mistake', 'settings correct?': 'N'}
        with mock.patch('one.params.input', new=partial(self._mock_input, **resp_map)):
            cache = one.params.setup()
            self.assertNotEqual(cache.ALYX_LOGIN, 'mistake')

        # Check that raises ValueError when bad URL provided
        self.url = 'ftp://'
        with self.assertRaises(ValueError), mock.patch('one.params.input', new=self._mock_input):
            one.params.setup()


class TestONEParamUtil(unittest.TestCase):
    """Test class for one.params utility functions"""
    def setUp(self) -> None:
        pass

    def test_key_from_url(self):
        """Test for one.params._key_from_url"""
        key = params._key_from_url('https://sub.domain.org/')
        self.assertEqual(key, 'sub.domain.org')

        key = params._key_from_url('http://www.domain.org/db/?rest=True')
        self.assertEqual(key, 'www.domain.org_db__rest_true')

    def test_get_params_dir(self):
        """Test for one.params.get_params_dir"""
        par_dir = Path('path', 'to', 'params')
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, par_dir)):
            path = params.get_params_dir()
        self.assertIsInstance(path, Path)
        self.assertEqual('path/to/params/.one', path.as_posix())

    def test_get_default_client(self):
        """Test for one.params.get_default_client"""
        temp_dir = util.set_up_env()
        self.addCleanup(temp_dir.cleanup)
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, temp_dir.name)):
            self.assertIsNone(params.get_default_client())
            # Copy over caches fixture
            params.setup(silent=True)
            client = params.get_default_client()
            self.assertEqual(client, 'https://openalyx.internationalbrainlab.org')
            # Test with include_schema=False
            client = params.get_default_client(include_schema=False)
            self.assertEqual(client, 'openalyx.internationalbrainlab.org')

    def test_get_cache_dir(self):
        """Test for one.params.get_cache_dir"""
        temp_dir = util.set_up_env()
        cache_dir = Path(temp_dir.name) / 'download_cache'
        assert not cache_dir.exists()
        self.addCleanup(temp_dir.cleanup)
        with mock.patch('iblutil.io.params.getfile', new=partial(util.get_file, temp_dir.name)):
            util.setup_test_params(cache_dir=cache_dir)
            out = params.get_cache_dir()
        self.assertEqual(cache_dir, out)
        self.assertTrue(cache_dir.exists())


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)
