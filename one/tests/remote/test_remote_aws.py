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
    from botocore import UNSIGNED
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
            # Test other client errors re-raised
            with mock.patch('one.remote.aws.get_s3_public') as s3_mock:
                m = mock.MagicMock()
                s3_mock.return_value = (m, m)
                m.Object.side_effect = aws.ClientError(mock.MagicMock(), mock.MagicMock())
                self.assertRaises(aws.ClientError, aws.s3_download_file, '', destination)

    def test_download_folder(self):
        """Test for one.remote.aws.s3_download_public_folder function."""
        source = 'caches/openalyx/2021_Q2_Varol_et_al'
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

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = util.set_up_env()
        cls.expected_credentials = {
            'aws_access_key_id': 'ABCDEF',
            'aws_secret_access_key': 'shhh',
            'region_name': None}
        with mock.patch('one.params.iopar.getfile', new=partial(util.get_file, cls.tempdir.name)):
            # util.setup_test_params(token=True)
            cls.alyx = AlyxClient(
                **TEST_DB_1,
                cache_dir=cls.tempdir.name
            )

    def test_credentials(self):
        """Test for one.remote.aws.get_aws_access_keys function."""
        cred, bucket_name = aws.get_aws_access_keys(self.alyx, 'aws_cortexlab')
        self.assertDictEqual(cred, self.expected_credentials)
        self.assertEqual(bucket_name, 's3_bucket')

    @mock.patch('boto3.Session')
    def test_get_s3_from_alyx(self, session_mock):
        """Tests for one.remote.aws.get_s3_from_alyx function"""
        s3, bucket_name = aws.get_s3_from_alyx(self.alyx, 'aws_cortexlab')
        self.assertEqual(bucket_name, 's3_bucket')
        session_mock.assert_called_once_with(**self.expected_credentials)
        resource = session_mock().resource
        resource.assert_called_once_with('s3', config=None)
        self.assertIs(s3, resource())

        # Assert that resource is unsigned when no credentials are returned
        session_mock.reset_mock()
        repo_json = {'json': {'bucket_name': 'public_foo'}}
        with mock.patch.object(self.alyx, 'rest', return_value=repo_json):
            s3, _ = aws.get_s3_from_alyx(self.alyx)
        _, kwargs = resource.call_args
        self.assertIs(kwargs['config'].signature_version, UNSIGNED)
        # If the bucket does not have 'public' in the name, no assumptions should be made about
        # the credentials
        session_mock.reset_mock()
        repo_json['json']['bucket_name'] = 'private_foo'
        with mock.patch.object(self.alyx, 'rest', return_value=repo_json):
            s3, _ = aws.get_s3_from_alyx(self.alyx)
        resource.assert_called_once_with('s3', config=None)


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


if __name__ == '__main__':
    unittest.main(exit=False)
