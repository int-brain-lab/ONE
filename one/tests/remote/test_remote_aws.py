import unittest

from one.tests import TEST_DB_1, OFFLINE_ONLY
from one.api import ONE
import one.remote.aws as aws

one = ONE(**TEST_DB_1)


@unittest.skipIf(OFFLINE_ONLY, 'online only test')
class TestAWS(unittest.TestCase):
    """Tests for AlyxClient authentication, token storage, login/out methods and user prompts"""
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
        cred, bucket_name = aws.get_aws_access_keys(one, self.repo['name'])
        assert cred == {'aws_access_key_id': 'key', 'aws_secret_access_key': 'secret_access'}
        assert bucket_name == self.repo['json']['bucket_name']
