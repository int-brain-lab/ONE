""" Backend for AWS"""
from pathlib import Path
import logging
from tqdm import tqdm

import boto3

from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

_logger = logging.getLogger(__name__)

S3_BUCKET_IBL = 'ibl-brain-wide-map-public'
REGION_NAME = 'us-east-1'


def _callback_hook(t):
    # https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
    # For example that uses actual file size:
    # boto3.amazonaws.com/v1/documentation/api/latest/_modules/boto3/s3/transfer.html
    def inner(bytes_amount):
        t.update(bytes_amount)
    return inner


def get_aws_access_keys(one, repo_name='aws_cortexlab'):
    """
    Query alyx database to get credentials in the json field of an aws repository
    :param one:
    :param repo_name:
    :return: dictionary of access keys to use in boto3, bucket name
    """
    repo_json = one.alyx.rest('data-repository', 'read', id=repo_name)['json']
    bucket_name = repo_json['bucket_name']
    session_keys = {
        'aws_access_key_id': repo_json.get('Access key ID', None),
        'aws_secret_access_key': repo_json.get('Secret access key', None)
    }
    return session_keys, bucket_name


def get_s3_public():
    session = boto3.Session(region_name=REGION_NAME)
    s3 = session.resource('s3', config=Config(signature_version=UNSIGNED))
    return s3, S3_BUCKET_IBL


def get_s3_from_alyx(one, region_name=REGION_NAME):
    """
    Create a s3 resource instance by getting credentials off the oneAlyx instance
    :param one: one alyx instance
    :param region_name: aws region name
    :return: s3 object, bucket name
    """
    session_keys, bucket_name = get_aws_access_keys(one)
    session = boto3.Session(**session_keys, region_name=region_name)
    s3 = session.resource('s3')
    return s3, bucket_name


def s3_download_file(source, destination, s3=None, bucket_name=None, overwrite=True):
    """
    :param source: relative path of file" 'atlas/dorsal_cortex_50.nrrd'
    :param destination: full file path on local machine '/usr/ibl/dorsal_cortex_50.nrrd'
    :param s3: s3 resource
    :param bucket_name: aws bucket name
    :param overwrite: defaults to True: force rewrite even if existing file with matching size
    :return: local file path, None if not found
    """
    destination = Path(destination)
    if s3 is None:
        s3, bucket_name = get_s3_public()
    try:
        file_object = s3.Object(bucket_name, Path(source).as_posix())
        filesize = file_object.content_length
        if not overwrite and destination.exists() and filesize == destination.stat().st_size:
            _logger.debug(f"{destination} exists and match size -- skipping")
            return destination
        with tqdm(total=filesize, unit='B',
                  unit_scale=True, desc=str(destination)) as t:
            file_object.download_file(Filename=str(destination), Callback=_callback_hook(t))
    except (NoCredentialsError, PartialCredentialsError) as ex:
        raise ex  # Credentials need updating in Alyx
    except ClientError as ex:
        if ex.response.get('Error', {}).get('Code', None) == '404':
            _logger.error(f'File {source} not found on {bucket_name}')
            return None
        else:
            raise ex
    return destination


def s3_download_public_folder(source, destination, s3=None, bucket_name=S3_BUCKET_IBL, overwrite=False):
    """
    downloads a public folder content to a local folder
    :param source: relative path within the bucket, for example: 'spikesorting/benchmark
    :param destination: local folder path
    :param bucket_name: if not specified, 'ibl-brain-wide-map-public'
    :param s3: s3 resource, if not specified, public IBL repository
    :param overwrite: (default False) if set to True, will re-download files even if already exists and size matches
    :return:
    """
    if s3 is None:
        s3, bucket_name = get_s3_public()
    response = s3.list_objects_v2(Prefix=source, Bucket=bucket_name)
    local_files = []
    for item in response.get('Contents', []):
        object = item['Key']
        if object.endswith('/') and item['Size'] == 0:  # skips folder
            continue
        lf = s3_download_file(source, destination, s3=s3, bucket_name=bucket_name, overwrite=overwrite)
        local_files.append(lf)
