"""Tests for the one.remote.aws module."""
import json
import unittest
import tempfile
from pathlib import Path

from one.tests import TEST_DB_1, OFFLINE_ONLY
from one.api import ONE
import one.remote.aws as aws

one = ONE(**TEST_DB_1)


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
        assert(len(js.keys()) == 5)

    def test_download_folder(self):
        """Test for one.remote.aws.s3_download_public_folder function."""
        source = 'caches/openalyx'
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td).joinpath('caches/unit_test')
            local_files = aws.s3_download_folder(source, destination)
        assert(len(local_files) == 2)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestAWS(unittest.TestCase):
    """Tests for AlyxClient authentication, token storage, login/out methods and user prompts"""

    """Data for our repo fixture."""
    repo = None

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestAWS, cls).setUpClass()

        cls.repo = {
            'name': 'aws_test_gnagnagna',
            'timezone': 'America/New_York',
            'globus_path': '',
            'hostname': '',
            'data_url': 'http://whatever.com/',
            'repository_type': 'Fileserver',
            'globus_endpoint_id': None,
            'globus_is_personal': False,
            'json': {'bucket_name': 'ibl-brain-wide-map-public',
                     'Access key ID': 'key',
                     'Secret access key': 'secret_access'}
        }

        repo_test = one.alyx.rest('data-repository', 'list', name=cls.repo['name'], no_cache=True)
        if len(repo_test) > 0:
            one.alyx.rest('data-repository', 'delete', id=repo_test[0]['name'])
        one.alyx.rest('data-repository', 'create', data=cls.repo)

    @classmethod
    def tearDownClass(cls) -> None:
        one.alyx.rest('data-repository', 'delete', id=cls.repo['name'])

    def test_credentials(self):
        """Test for one.remote.aws.get_aws_access_keys function."""
        cred, bucket_name = aws.get_aws_access_keys(one, self.repo['name'])
        expected = {'aws_access_key_id': 'key', 'aws_secret_access_key': 'secret_access'}
        self.assertDictEqual(cred, expected)
        self.assertEqual(bucket_name, self.repo['json']['bucket_name'])
