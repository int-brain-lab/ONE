"""A backend to download IBL data from AWS Buckets.

Examples
--------
Without any credentials, to download a public file from the IBL public bucket:

>>> from one.remote import aws
... source = 'caches/unit_test/cache_info.json'
... destination = '/home/olivier/scratch/cache_info.json'
... aws.s3_download_file(source, destination)

For a folder, the following:

>>> source = 'caches/unit_test'
>>> destination = '/home/olivier/scratch/caches/unit_test'
>>> local_files = aws.s3_download_folder(source, destination)
"""
import re
from pathlib import Path
import logging
import urllib.parse

from tqdm import tqdm
import boto3

from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

_logger = logging.getLogger(__name__)

REPO_DEFAULT = 'aws_cortexlab'
S3_BUCKET_IBL = 'ibl-brain-wide-map-public'
REGION_NAME = 'us-east-1'


def _callback_hook(t):
    """A callback hook for boto3.download_file to update the progress bar.

    Parameters
    ----------
    t : tqdm.tqdm
        An tqdm instance used as the progress bar.

    See Also
    --------
    https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
    For example that uses actual file size:
    https://boto3.amazonaws.com/v1/documentation/api/latest/_modules/boto3/s3/transfer.html

    """
    def inner(bytes_amount):
        t.update(bytes_amount)
    return inner


def get_s3_virtual_host(uri, region) -> str:
    """
    Convert a given bucket URI to a generic Amazon virtual host URL.
    URI may be the bucket (+ path) or a full URI starting with 's3://'

    .. _S3 documentation
       https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-bucket-intro.html#virtual-host-style-url-ex

    Parameters
    ----------
    uri : str
        The bucket name or full path URI.
    region : str
        The region, e.g. eu-west-1.

    Returns
    -------
    str
        The Web URL (virtual host name and https scheme).
    """
    assert region and re.match(r'\w{2}-\w+-[1-3]', region), 'Invalid region'
    parsed = urllib.parse.urlparse(uri)  # remove scheme if necessary
    key = parsed.path.strip('/').split('/')
    bucket = parsed.netloc or key.pop(0)
    hostname = f"{bucket}.{parsed.scheme or 's3'}.{region}.amazonaws.com"
    return 'https://' + '/'.join((hostname, *key))


def is_folder(obj_summery) -> bool:
    """
    Given an S3 ObjectSummery instance, returns true if the associated object is a directory.

    Parameters
    ----------
    obj_summery : s3.ObjectSummery
        An S3 ObjectSummery instance to test.

    Returns
    -------
    bool
        True if object is a directory.
    """
    return obj_summery.key.endswith('/') and obj_summery.size == 0


def get_aws_access_keys(alyx, repo_name=REPO_DEFAULT):
    """
    Query Alyx database to get credentials in the json field of an aws repository.

    Parameters
    ----------
    alyx : one.webclient.AlyxInstance
        An instance of alyx.
    repo_name : str
        The data repository name in Alyx from which to fetch the S3 access keys.

    Returns
    -------
    dict
        The API access keys and region name to use with boto3.
    str
        The name of the S3 bucket associated with the Alyx data repository.
    """
    repo_json = alyx.rest('data-repository', 'read', id=repo_name)['json']
    bucket_name = repo_json['bucket_name']
    session_keys = {
        'aws_access_key_id': repo_json.get('Access key ID', None),
        'aws_secret_access_key': repo_json.get('Secret access key', None),
        'region_name': repo_json.get('region_name', None)
    }
    return session_keys, bucket_name


def get_s3_public():
    """
    Retrieve the IBL public S3 service resource.

    Returns
    -------
    s3.ServiceResource
        An S3 ServiceResource instance with the provided.
    str
        The name of the S3 bucket.
    """
    session = boto3.Session(region_name=REGION_NAME)
    s3 = session.resource('s3', config=Config(signature_version=UNSIGNED))
    return s3, S3_BUCKET_IBL


def get_s3_from_alyx(alyx, repo_name=REPO_DEFAULT):
    """
    Create an S3 resource instance using credentials from an Alyx data repository.

    Parameters
    ----------
    alyx : one.webclient.AlyxInstance
        An instance of alyx.
    repo_name : str
        The data repository name in Alyx from which to fetch the S3 access keys.

    Returns
    -------
    s3.ServiceResource
        An S3 ServiceResource instance with the provided.
    str
        The name of the S3 bucket.

    Notes
    -----
    - If no credentials are present in the database, boto3 will use environment config or default
      AWS profile settings instead.
    - If there are no credentials for the bucket and the bucket has 'public' in the name, the
      returned resource will use an unsigned signature.
    """
    session_keys, bucket_name = get_aws_access_keys(alyx, repo_name)
    no_creds = not any(filter(None, (v for k, v in session_keys.items() if 'key' in k.lower())))
    session = boto3.Session(**session_keys)
    if no_creds and 'public' in bucket_name.lower():
        config = Config(signature_version=UNSIGNED)
    else:
        config = None
    s3 = session.resource('s3', config=config)
    return s3, bucket_name


def s3_download_file(source, destination, s3=None, bucket_name=None, overwrite=False):
    """
    Downloads a file from an S3 instance to a local folder.

    Parameters
    ----------
    source : str, pathlib.Path, pathlib.PurePosixPath
        Relative path (key) within the bucket, for example: 'atlas/dorsal_cortex_50.nrrd'.
    destination : str, pathlib.Path
        The full file path on local machine.
    s3 : s3.ServiceResource
        An S3 ServiceResource instance.  Defaults to the IBL public instance.
    bucket_name : str
        The name of the bucket to access.  Defaults to the public IBL repository.
    overwrite : bool
        If True, will re-download files even if the file sizes match.

    Returns
    -------
    pathlib.Path
        The local file path of the downloaded file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        raise ex  # Credentials need updating in Alyx # pragma: no cover
    except ClientError as ex:
        if ex.response.get('Error', {}).get('Code', None) == '404':
            _logger.error(f'File {source} not found on {bucket_name}')
            return None
        else:
            raise ex
    return destination


def s3_download_folder(source, destination, s3=None, bucket_name=S3_BUCKET_IBL, overwrite=False):
    """
    Downloads S3 folder content to a local folder.

    Parameters
    ----------
    source : str
        Relative path (key) within the bucket, for example: 'spikesorting/benchmark'.
    destination : str, pathlib.Path
        Local folder path.  Note: The contents of the source folder will be downloaded to
        `destination`, not the folder itself.
    s3 : s3.ServiceResource
        An S3 ServiceResource instance.  Defaults to the IBL public instance.
    bucket_name : str
        The name of the bucket to access.  Defaults to the public IBL repository.
    overwrite : bool
        If True, will re-download files even if the file sizes match.

    Returns
    -------
    list of pathlib.Path
        The local file paths.
    """
    destination = Path(destination)
    if destination.exists():
        assert destination.is_dir(), 'destination must be a folder'
    if s3 is None:
        s3, bucket_name = get_s3_public()
    local_files = []
    objects = s3.Bucket(name=bucket_name).objects.filter(Prefix=source)
    for obj_summary in filter(lambda x: not is_folder(x), objects):
        local_file = Path(destination).joinpath(Path(obj_summary.key).relative_to(source))
        lf = s3_download_file(obj_summary.key, local_file, s3=s3, bucket_name=bucket_name,
                              overwrite=overwrite)
        local_files.append(lf)
    return local_files
