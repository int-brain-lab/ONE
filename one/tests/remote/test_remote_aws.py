"""Tests for the one.remote.aws module."""
import json
import logging
import unittest
from unittest import mock
import tempfile
from pathlib import Path
from functools import partial

from one.tests import TEST_DB_1, OFFLINE_ONLY, util
from one.webclient import AlyxClient
try:
    import one.remote.aws as aws
except ModuleNotFoundError:
    raise unittest.SkipTest('boto3 module not installed')


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestAWSPublic(unittest.TestCase):
    """
    This test relies on downloading the cache folder for open alyx as it is rather small
    The assertions may change if the content does change.
    If the folder grows bigger, we could create a dedicated test folder for this test
    """
    source = 'caches/openalyx/cache_info.json'

    def test_download_file(self):
        """Test for one.aws.remote.s3_download_file function."""
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td).joinpath('caches/unit_test/cache_info.json')
            local_file = aws.s3_download_file(self.source, destination)
            with open(local_file, 'r') as fid:
                js = json.load(fid)
            self.assertEqual(5, len(js.keys()))
            # Test skip re-download
            with self.assertLogs('one.remote.aws', logging.DEBUG) as lg:
                aws.s3_download_file(self.source, destination)
                self.assertIn('exists', lg.output[-1])
            # Test file not found
            with self.assertLogs('one.remote.aws', logging.ERROR) as lg:
                self.assertIsNone(aws.s3_download_file('foo/bar/baz.BAT', destination))
                self.assertIn('not found', lg.output[-1])

    def test_download_folder(self):
        """Test for one.remote.aws.s3_download_public_folder function."""
        source = 'caches/openalyx'
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td).joinpath('caches/unit_test')
            local_files = aws.s3_download_folder(source, destination)
            self.assertEqual(2, len(local_files))
            # Test not folder assert
            destination = Path(td).joinpath('file')
            destination.touch()
            with self.assertRaises(AssertionError):
                aws.s3_download_folder(source, destination)


class TestAWS(unittest.TestCase):
    """Tests for AlyxClient authentication, token storage, login/out methods and user prompts"""

    """Data for our repo fixture."""
    repo = None

    tempdir = None
    one = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = util.set_up_env()
        with mock.patch('one.params.iopar.getfile', new=partial(util.get_file, cls.tempdir.name)):
            # util.setup_test_params(token=True)
            cls.alyx = AlyxClient(
                **TEST_DB_1,
                cache_dir=cls.tempdir.name
            )

    def test_credentials(self):
        """Test for one.remote.aws.get_aws_access_keys function."""
        cred, bucket_name = aws.get_aws_access_keys(self.alyx, 'aws_cortexlab')
        expected = {
            'aws_access_key_id': 'ABCDEF',
            'aws_secret_access_key': 'shhh',
            'region_name': None
        }
        self.assertDictEqual(cred, expected)
        self.assertEqual(bucket_name, 's3_bucket')


class TestUtils(unittest.TestCase):
    """Tests for one.remote.aws utility functions"""

    def test_get_s3_virtual_host(self):
        """Tests for one.remote.aws.get_s3_virtual_host function"""
        expected = 'https://my-s3-bucket.s3.eu-east-1.amazonaws.com/'
        url = aws.get_s3_virtual_host('s3://my-s3-bucket', 'eu-east-1')
        self.assertEqual(expected, url)

        # NB slight diff in behaviour: with no scheme provided output URL has no trailing slash.
        url = aws.get_s3_virtual_host('my-s3-bucket/', 'eu-east-1')
        self.assertEqual(expected.strip('/'), url)
        expected = 'https://my-s3-bucket.s3.eu-east-1.amazonaws.com/path/to/file'
        url = aws.get_s3_virtual_host('s3://my-s3-bucket/path/to/file', 'eu-east-1')
        self.assertEqual(expected, url)

        with self.assertRaises(AssertionError):
            aws.get_s3_virtual_host('s3://my-s3-bucket/path/to/file', 'wrong-foo-4')
