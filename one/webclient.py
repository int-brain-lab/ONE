"""API for interacting with a remote Alyx instance through REST
The AlyxClient class contains methods for making remote Alyx REST queries and downloading remote
files through Alyx.

Examples:
    alyx = AlyxClient(
        username='test_user', password='TapetesBloc18',
        base_url='https://test.alyx.internationalbrainlab.org')

    # List subjects
    subjects = alyx.rest('subjects', 'list')

    # Create a subject
    record = {
        'nickname': nickname,
        'responsible_user': 'olivier',
        'birth_date': '2019-06-15',
        'death_date': None,
        'lab': 'cortexlab',
    }
    new_subj = alyx.rest('subjects', 'create', data=record)

    # Download a remote file, given a local path
    url = 'zadorlab/Subjects/flowers/2018-07-13/1/channels.probe.npy'
    local_path = self.alyx.download_file(url)

TODO Move converters to another module
TODO Add logout method to delete the active token
"""
import json
import logging
import math
import os
import re
import functools
import urllib.request
from urllib.error import HTTPError
from collections.abc import Mapping
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import warnings
import hashlib
import zipfile
import tempfile

import requests
from tqdm import tqdm

from pprint import pprint
import one.params
from iblutil.io import hashfile
import one.alf.io as alfio

SDSC_ROOT_PATH = PurePosixPath('/mnt/ibl')  # FIXME Move to ibllib, or get from Alyx
_logger = logging.getLogger(__name__)


def _cache_response(method):
    """
    Decorator for the generic request method; caches the result of the query and on subsequent
    calls, returns cache instead of hitting the database
    :param method: function to wrap (i.e. AlyxClient._generic_request)
    :return: handle to wrapped method
    """

    @functools.wraps(method)
    def wrapper_decorator(alyx_client, *args, expires=None, clobber=False, **kwargs):
        """
        REST caching wrapper
        :param alyx_client: an instance of the AlyxClient class
        :param args: positional arguments for applying to wrapped function
        :param expires: an optional timedelta for how long cached response is valid.  If True,
        the cached response will not be used on subsequent calls.  If None, the default expiry
        is applied.
        :param clobber: If True any existing cached response is overwritten
        :param kwargs: keyword arguments for applying to wrapped function
        :return: the REST response JSON either from cached file or directly from remote
        endpoint
        """
        expires = expires or alyx_client.default_expiry
        mode = (alyx_client.cache_mode or '').lower()
        if args[0].__name__ != mode and mode != '*':
            return method(alyx_client, *args, **kwargs)
        # Check cache
        rest_cache = one.params.get_rest_dir(alyx_client.base_url)
        sha1 = hashlib.sha1()
        sha1.update(bytes(args[1], 'utf-8'))
        name = sha1.hexdigest()
        # Reversible but length may exceed 255 chars
        # name = base64.urlsafe_b64encode(args[2].encode('UTF-8')).decode('UTF-8')
        files = list(rest_cache.glob(name))
        cached = None
        if len(files) == 1 and not clobber:
            _logger.debug('loading REST response from cache')
            with open(files[0], 'r') as f:
                cached, when = json.load(f)
            if datetime.fromisoformat(when) > datetime.now():
                return cached
        try:
            response = method(alyx_client, *args, **kwargs)
        except requests.exceptions.ConnectionError as ex:
            if cached and not clobber:
                warnings.warn('Failed to connect, returning cached response', RuntimeWarning)
                return cached
            raise ex  # No cache and can't connect to database; re-raise

        # Save response into cache
        rest_cache.mkdir(exist_ok=True, parents=True)
        _logger.debug('caching REST response')
        expiry_datetime = datetime.now() + (timedelta() if expires is True else expires)
        with open(rest_cache / name, 'w') as f:
            json.dump((response, expiry_datetime.isoformat()), f)
        return response

    return wrapper_decorator


class _PaginatedResponse(Mapping):
    """
    This class allows to emulate a list from a paginated response.
    Provides cache functionality
    PaginatedResponse(alyx, response)
    """

    def __init__(self, alyx, rep):
        self.alyx = alyx
        self.count = rep['count']
        self.limit = len(rep['results'])
        # warning: the offset and limit filters are not necessarily the last ones
        lquery = [q for q in rep['next'].split('&')
                  if not (q.startswith('offset=') or q.startswith('limit='))]
        self.query = '&'.join(lquery)
        # init the cache, list with None with count size
        self._cache = [None for _ in range(self.count)]
        # fill the cache with results of the query
        for i in range(self.limit):
            self._cache[i] = rep['results'][i]

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        if self._cache[item] is None:
            offset = self.limit * math.floor(item / self.limit)
            query = f'{self.query}&limit={self.limit}&offset={offset}'
            res = self.alyx._generic_request(requests.get, query)
            for i, r in enumerate(res['results']):
                self._cache[i + offset] = res['results'][i]
        return self._cache[item]

    def __iter__(self):
        for i in range(self.count):
            yield self.__getitem__(i)


def sdsc_globus_path_from_dataset(dset):
    """
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    Returns SDSC globus file path from a dset record or a list of dsets records from REST
    """
    return _path_from_dataset(dset, root_path=PurePosixPath('/'), repository=None, uuid=True)


def globus_path_from_dataset(dset, repository=None, uuid=False):
    """
    Returns local one file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param repository: (optional) repository name of the file record (if None, will take
     the first filerecord with an URL)
    """
    return _path_from_dataset(dset, root_path=PurePosixPath('/'), repository=repository, uuid=uuid)


def one_path_from_dataset(dset, one_cache):
    """
    Returns local one file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param one_cache: the one cache directory
    """
    return _path_from_dataset(dset, root_path=one_cache, uuid=False)


def sdsc_path_from_dataset(dset, root_path=SDSC_ROOT_PATH):
    """
    Returns sdsc file path from a dset record or a list of dsets records from REST
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param root_path: (optional) the prefix path such as one download directory or sdsc root
    """
    return _path_from_dataset(dset, root_path=root_path, uuid=True)


def _path_from_dataset(dset, root_path=None, repository=None, uuid=False):
    """
    returns the local file path from a dset record from a REST query
    :param dset: dset dictionary or list of dictionaries from ALyx rest endpoint
    :param root_path: (optional) the prefix path such as one download directory or sdsc root
    :param repository:
    :param uuid: (optional bool) if True, will add UUID before the file extension
    :return: Path or list of Path
    """
    if isinstance(dset, list):
        return [_path_from_dataset(d) for d in dset]
    if repository:
        fr = next((fr for fr in dset['file_records'] if fr['data_repository'] == repository))
    else:
        fr = next((fr for fr in dset['file_records'] if fr['data_url']))
    uuid = dset['url'][-36:] if uuid else None
    return _path_from_filerecord(fr, root_path=root_path, uuid=uuid)


def _path_from_filerecord(fr, root_path=SDSC_ROOT_PATH, uuid=None):  # FIXME change default to '/'
    """
    Returns a data file Path constructed from an Alyx file record.  The Path type returned
    depends on the type of root_path: If root_path is a string a Path object is returned,
    otherwise if the root_path is a PurePath, the same path type is returned.
    :param fr: An Alyx file record dict
    :param root_path: An optional root path
    :param uuid: An optional UUID to add to the file name
    :return: A filepath as a pathlib object
    """
    if isinstance(fr, list):
        return [_path_from_filerecord(f) for f in fr]
    repo_path = fr['data_repository_path']
    repo_path = repo_path[repo_path.startswith('/'):]  # remove starting / if any
    # repo_path = (p := fr['data_repository_path'])[p[0] == '/':]  # py3.8 Remove slash at start
    file_path = PurePosixPath(repo_path, fr['relative_path'])
    if root_path:
        # NB: By checking for string we won't cast any PurePaths
        if isinstance(root_path, str):
            root_path = Path(root_path)
        file_path = root_path / file_path
    if uuid:
        file_path = alfio.add_uuid_string(file_path, uuid)
    return file_path


def http_download_file_list(links_to_file_list, **kwargs):
    """
    Downloads a list of files from the flat Iron from a list of links.
    Same options behaviour as http_download_file

    :param links_to_file_list: list of http links to files.
    :type links_to_file_list: list

    :return: (list) a list of the local full path of the downloaded files.
    """
    file_names_list = []
    for link_str in links_to_file_list:
        file_names_list.append(http_download_file(link_str, **kwargs))
    return file_names_list


def http_download_file(full_link_to_file, chunks=None, *, clobber=False, silent=False,
                       username='', password='', cache_dir='', return_md5=False, headers=None):
    """
    :param full_link_to_file: http link to the file.
    :type full_link_to_file: str
    :param chunks: chunks to download
    :type chunks: tuple of ints
    :param clobber: [False] If True, force overwrite the existing file.
    :type clobber: bool
    :param username: [''] authentication for password protected file server.
    :type username: str
    :param password: [''] authentication for password protected file server.
    :type password: str
    :param cache_dir: [''] directory in which files are cached; defaults to user's
     Download directory.
    :type cache_dir: str
    :param return_md5: if true an MD5 hash of the file is additionally returned
    :type return_md5: bool
    :param: headers: [{}] additional headers to add to the request (auth tokens etc..)
    :type headers: dict
    :param: silent: [False] suppress download progress bar
    :type silent: bool

    :return: (str) a list of the local full path of the downloaded files.
    """
    if not full_link_to_file:
        return ''

    # default cache directory is the home dir
    if not cache_dir:
        cache_dir = str(Path.home().joinpath('Downloads'))

    # This is the local file name
    file_name = str(cache_dir) + os.sep + os.path.basename(full_link_to_file)

    # do not overwrite an existing file unless specified
    if not clobber and os.path.exists(file_name):
        return (file_name, hashfile.md5(file_name)) if return_md5 else file_name

    # This should be the base url you wanted to access.
    baseurl = os.path.split(str(full_link_to_file))[0]

    # Create a password manager
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    if username and password:
        manager.add_password(None, baseurl, username, password)

    # Create an authentication handler using the password manager
    auth = urllib.request.HTTPBasicAuthHandler(manager)

    # Create an opener that will replace the default urlopen method on further calls
    opener = urllib.request.build_opener(auth)
    urllib.request.install_opener(opener)

    # Support for partial download.
    req = urllib.request.Request(full_link_to_file)
    if chunks is not None:
        first_byte, n_bytes = chunks
        req.add_header('Range', 'bytes=%d-%d' % (first_byte, first_byte + n_bytes - 1))

    # add additional headers
    if headers is not None:
        for k in headers:
            req.add_header(k, headers[k])

    # Open the url and get the length
    try:
        u = urllib.request.urlopen(req)
    except HTTPError as e:
        _logger.error(f'{str(e)} {full_link_to_file}')
        raise e

    file_size = int(u.getheader('Content-length'))
    if not silent:
        print(f'Downloading: {file_name} Bytes: {file_size}')
    file_size_dl = 0
    block_sz = 8192 * 64 * 8

    md5 = hashlib.md5()
    f = open(file_name, 'wb')
    with tqdm(total=file_size, disable=silent) as pbar:
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if return_md5:
                md5.update(buffer)
            pbar.update(file_size_dl)
    f.close()

    return (file_name, md5.hexdigest()) if return_md5 else file_name


def file_record_to_url(file_records):
    """
    Translate a Json dictionary to an usable http url for downloading files.

    :param file_records: json containing a 'data_url' field
    :type file_records: dict

    :return: urls: (list) a list of strings representing full data urls
    """
    urls = []
    for fr in file_records:
        if fr['data_url'] is not None:
            urls.append(fr['data_url'])
    return urls


def dataset_record_to_url(dataset_record):
    """
    Extracts a list of files urls from a list of dataset queries.

    :param dataset_record: dataset Json from a rest request.
    :type dataset_record: list

    :return: (list) a list of strings representing files urls corresponding to the datasets records
    """
    urls = []
    if isinstance(dataset_record, dict):
        dataset_record = [dataset_record]
    for ds in dataset_record:
        urls += file_record_to_url(ds['file_records'])
    return urls


class UniqueSingletons(type):
    _instances: list = []

    def __call__(cls, *args, **kwargs):
        for inst in UniqueSingletons._instances:
            if cls in inst and inst.get(cls, None).get('args') == (args, kwargs):
                return inst[cls].get('instance')

        new_instance = super(UniqueSingletons, cls).__call__(*args, **kwargs)
        # Optional rerun of constructor
        # new_instance.__init__(*args, **kwargs)
        new_instance_record = {
            cls: {'args': (args, kwargs), 'instance': new_instance}
        }
        UniqueSingletons._instances.append(new_instance_record)

        return new_instance


class AlyxClient(metaclass=UniqueSingletons):
    """
    Class that implements simple GET/POST wrappers for the Alyx REST API
    http://alyx.readthedocs.io/en/latest/api.html  # FIXME old link
    """
    _token = None
    _headers = None  # Headers for REST requests only
    user = None
    base_url = None

    def __init__(self, base_url=None, username=None, password=None,
                 cache_dir=None, silent=False, cache_rest='GET'):
        """
        Create a client instance that allows to GET and POST to the Alyx server
        For oneibl, constructor attempts to authenticate with credentials in params.py
        For standalone cases, AlyxClient(username='', password='', base_url='')

        :param username: Alyx database user
        :param password: Alyx database password
        :param base_url: Alyx server address, including port and protocol
        :param cache_rest: which type of http method to apply cache to; if '*', all requests are
        cached.
        """
        self.silent = silent
        self._par = one.params.get(client=base_url, silent=self.silent)
        # TODO Pass these to `params.get` and have it deal with setup defaults
        self._par = self._par.set('ALYX_LOGIN', username or self._par.ALYX_LOGIN)
        self._par = self._par.set('ALYX_PWD', password or self._par.ALYX_PWD)
        self.base_url = base_url or self._par.ALYX_URL
        self._par = self._par.set('CACHE_DIR', cache_dir or self._par.CACHE_DIR)
        self.authenticate()
        self._rest_schemes = None
        # the mixed accept application may cause errors sometimes, only necessary for the docs
        self._headers['Accept'] = 'application/json'
        # REST cache parameters
        # The default length of time that cache file is valid for,
        # The default expiry is overridden by the `expires` kwarg.  If False, the caching is
        # turned off.
        self.default_expiry = timedelta(days=1)
        self.cache_mode = cache_rest
        self._obj_id = id(self)

    @property
    def rest_schemes(self):
        """Delayed fetch of rest schemes speeds up instantiation"""
        if not self._rest_schemes:
            self._rest_schemes = self.get('/docs', expires=timedelta(weeks=1))
        return self._rest_schemes

    @property
    def cache_dir(self):
        return Path(self._par.CACHE_DIR)

    def is_logged_in(self):
        return self._token and self.user

    def list_endpoints(self):
        """
        Return a list of available REST endpoints
        :return: List of REST endpoint strings
        """
        EXCLUDE = ('_type', '_meta', '', 'auth-token')
        return sorted(x for x in self.rest_schemes.keys() if x not in EXCLUDE)

    @_cache_response
    def _generic_request(self, reqfunction, rest_query, data=None, files=None):
        # makes sure the base url is the one from the instance
        rest_query = rest_query.replace(self.base_url, '')
        if not rest_query.startswith('/'):
            rest_query = '/' + rest_query
        _logger.debug(f"{self.base_url + rest_query}, headers: {self._headers}")
        headers = self._headers.copy()
        if files is None:
            data = json.dumps(data) if isinstance(data, dict) or isinstance(data, list) else data
            headers['Content-Type'] = 'application/json'
        if rest_query.startswith('/docs'):
            # the mixed accept application may cause errors sometimes, only necessary for the docs
            headers['Accept'] = 'application/coreapi+json'
        r = reqfunction(self.base_url + rest_query, stream=True, headers=headers,
                        data=data, files=files)
        if r and r.status_code in (200, 201):
            return json.loads(r.text)
        elif r and r.status_code == 204:
            return
        else:
            if not self.silent:
                _logger.error(self.base_url + rest_query)
                _logger.error(r.text)
            raise (requests.HTTPError(r))

    def authenticate(self, cache_token=True, force=False):
        """
        Gets a security token from the Alyx REST API to create requests headers.
        Credentials are loaded via oneibl.params
        """
        if getattr(self._par, 'TOKEN', False) and not force:
            self._token = self._par.TOKEN
            self._headers = {
                'Authorization': f'Token {list(self._token.values())[0]}',
                'Accept': 'application/json'}
            self.user = self._par.ALYX_LOGIN
            return
        try:
            credentials = {'username': self._par.ALYX_LOGIN, 'password': self._par.ALYX_PWD}
            rep = requests.post(self.base_url + '/auth-token', data=credentials)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Can't connect to {self.base_url}.\n" +
                "IP addresses are filtered on IBL database servers. \n" +
                "Are you connecting from an IBL participating institution ?"
            )
        # Assign token or raise exception on internal server error
        self._token = rep.json() if rep.ok else rep.raise_for_status()
        if not (list(self._token.keys()) == ['token']):
            _logger.error(rep)
            raise Exception('Alyx authentication error. Check your credentials')
        self._headers = {
            'Authorization': 'Token {}'.format(list(self._token.values())[0]),
            'Accept': 'application/json'}
        if cache_token:
            par = one.params.get(client=self.base_url, silent=True).set('TOKEN', self._token)
            one.params.save(par, self.base_url)
        self.user = self._par.ALYX_LOGIN
        if not self.silent:
            print(f"Connected to {self.base_url} as {self.user}")

    def delete(self, rest_query):
        """
        Sends a DELETE request to the Alyx server. Will raise an exception on any status_code
        other than 200, 201.

        :param rest_query: examples:
         '/weighings/c617562d-c107-432e-a8ee-682c17f9e698'
         'https://test.alyx.internationalbrainlab.org/weighings/c617562d-c107-432e-a8ee-682c17f9e698'.
        :type rest_query: str

        :return: (dict/list) json interpreted dictionary from response
        """
        return self._generic_request(requests.delete, rest_query)

    def download_file(self, url, **kwargs):
        """
        Downloads a file on the Alyx server from a file record REST field URL
        :param url: full url(s) of the file(s)
        :param kwargs: webclient.http_download_file parameters
        :return: local path(s) of downloaded file(s)
        """
        if isinstance(url, str):
            url = self._validate_file_url(url)
            download_fcn = http_download_file
        else:
            url = (self._validate_file_url(x) for x in url)
            download_fcn = http_download_file_list
        pars = dict(
            silent=kwargs.pop('silent', self.silent),
            cache_dir=kwargs.pop('cache_dir', self._par.CACHE_DIR),
            username=self._par.HTTP_DATA_SERVER_LOGIN,
            password=self._par.HTTP_DATA_SERVER_PWD,
            **kwargs
        )
        return download_fcn(url, **pars)

    def download_cache_tables(self):
        """
        TODO Document
        :return: List of parquet table file paths
        """
        # query the database for the latest cache; expires=None overrides cached response
        self.cache_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as tmp:
            file = http_download_file(f'{self.base_url}/cache.zip',
                                      username=self._par.ALYX_LOGIN,
                                      password=self._par.ALYX_PWD,
                                      headers=self._headers,
                                      silent=self.silent,
                                      cache_dir=tmp,
                                      clobber=True)
            with zipfile.ZipFile(file, 'r') as zipped:
                files = zipped.namelist()
                zipped.extractall(self.cache_dir)
        return [Path(self.cache_dir, table) for table in files]

    def _validate_file_url(self, url):
        """
        TODO Document
        :param url:
        :return:
        """
        if url.startswith('http'):
            assert url.startswith(self._par.HTTP_DATA_SERVER), \
                ('remote protocol and/or hostname does not match HTTP_DATA_SERVER parameter:\n' +
                 f'"{url[:40]}..." should start with "{self._par.HTTP_DATA_SERVER}"')
        elif not url.startswith(self._par.HTTP_DATA_SERVER):
            url = self.rel_path2url(url)
        return url

    def rel_path2url(self, path):
        """
        TODO Document
        :param path:
        :return:
        """
        path = str(path).strip('/')
        assert not path.startswith('http')
        return f'{self._par.HTTP_DATA_SERVER}/{path}'

    def get(self, rest_query, **kwargs):
        """
        Sends a GET request to the Alyx server. Will raise an exception on any status_code
        other than 200, 201.
        For the dictionary contents and list of endpoints, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: example: '/sessions?user=Hamish'.
        :type rest_query: str

        :return: (dict/list) json interpreted dictionary from response
        """
        rep = self._generic_request(requests.get, rest_query, **kwargs)
        _logger.debug(rest_query)
        if isinstance(rep, dict) and list(rep.keys()) == ['count', 'next', 'previous', 'results']:
            if len(rep['results']) < rep['count']:
                rep = _PaginatedResponse(self, rep)
            else:
                rep = rep['results']
        return rep

    def patch(self, rest_query, data=None, files=None):
        """
        Sends a PATCH request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: json encoded string or dictionary (cf.requests)
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.patch, rest_query, data=data, files=files)

    def post(self, rest_query, data=None, files=None):
        """
        Sends a POST request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: dictionary or json encoded string
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.post, rest_query, data=data, files=files)

    def put(self, rest_query, data=None, files=None):
        """
        Sends a PUT request to the Alyx server.
        For the dictionary contents, refer to:
        https://alyx.internationalbrainlab.org/docs

        :param rest_query: (required)the endpoint as full or relative URL
        :type rest_query: str
        :param data: dictionary or json encoded string
        :type data: None, dict or str
        :param files: dictionary / tuple (cf.requests)

        :return: response object
        """
        return self._generic_request(requests.put, rest_query, data=data, files=files)

    def rest(self, url=None, action=None, id=None, data=None, files=None,
             no_cache=False, **kwargs):
        """
        alyx_client.rest(): lists endpoints
        alyx_client.rest(endpoint): lists actions for endpoint
        alyx_client.rest(endpoint, action): lists fields and URL

        Example REST endpoint with all actions:

            client.rest('subjects', 'list')
            client.rest('subjects', 'list', field_filter1='filterval')
            client.rest('subjects', 'create', data=sub_dict)
            client.rest('subjects', 'read', id='nickname')
            client.rest('subjects', 'update', id='nickname', data=sub_dict)
            client.rest('subjects', 'partial_update', id='nickname', data=sub_dict)
            client.rest('subjects', 'delete', id='nickname')
            client.rest('notes', 'create', data=nd, files={'image': open(image_file, 'rb')})

        :param url: endpoint name
        :param action: 'list', 'create', 'read', 'update', 'partial_update', 'delete'
        :param id: lookup string for actions 'read', 'update', 'partial_update', and 'delete'
        :param data: data dictionary for actions 'update', 'partial_update' and 'create'
        :param files: if file upload
        :param no_cache: if true the `list` and `read` actions are performed without caching
        :param ``**kwargs``: filter as per the Alyx REST documentation
            cf. https://alyx.internationalbrainlab.org/docs/
        :return: list of queried dicts ('list') or dict (other actions)
        """
        # if endpoint is None, list available endpoints
        if not url:
            pprint(self.list_endpoints())
            return
        # remove beginning slash if any
        if url.startswith('/'):
            url = url[1:]
        # and split to the next slash or question mark
        endpoint = re.findall("^/*[^?/]*", url)[0].replace('/', '')
        # make sure the queried endpoint exists, if not throw an informative error
        if endpoint not in self.rest_schemes.keys():
            av = [k for k in self.rest_schemes.keys() if not k.startswith('_') and k]
            raise ValueError('REST endpoint "' + endpoint + '" does not exist. Available ' +
                             'endpoints are \n       ' + '\n       '.join(av))
        endpoint_scheme = self.rest_schemes[endpoint]
        # on a filter request, override the default action parameter
        if '?' in url:
            action = 'list'
        # if action is None, list available actions for the required endpoint
        if not action:
            pprint(list(endpoint_scheme.keys()))
            return
        # make sure the the desired action exists, if not throw an informative error
        if action not in endpoint_scheme:
            raise ValueError('Action "' + action + '" for REST endpoint "' + endpoint + '" does ' +
                             'not exist. Available actions are: ' +
                             '\n       ' + '\n       '.join(endpoint_scheme.keys()))
        # the actions below require an id in the URL, warn and help the user
        if action in ['read', 'update', 'partial_update', 'delete'] and not id:
            _logger.warning('REST action "' + action + '" requires an ID in the URL: ' +
                            endpoint_scheme[action]['url'])
            return
        # the actions below require a data dictionary, warn and help the user with fields list
        if action in ['create', 'update', 'partial_update'] and not data:
            pprint(endpoint_scheme[action]['fields'])
            for act in endpoint_scheme[action]['fields']:
                print("'" + act['name'] + "': ...,")
            _logger.warning('REST action "' + action + '" requires a data dict with above keys')
            return

        # clobber=True means remote request always made, expires=True means response is not cached
        cache_args = {'clobber': no_cache, 'expires': no_cache}
        if action == 'list':
            # list doesn't require id nor
            assert (endpoint_scheme[action]['action'] == 'get')
            # add to url data if it is a string
            if id:
                # this is a special case of the list where we query an uuid. Usually read is better
                if 'django' in kwargs.keys():
                    kwargs['django'] = kwargs['django'] + ','
                else:
                    kwargs['django'] = ""
                kwargs['django'] = f"{kwargs['django']}pk,{id}"
            # otherwise, look for a dictionary of filter terms
            if kwargs:
                url += '?'
                for k in sorted(kwargs.keys()):
                    if isinstance(kwargs[k], str):
                        query = kwargs[k]
                    elif isinstance(kwargs[k], list):
                        query = ','.join(kwargs[k])
                    else:
                        query = str(kwargs[k])
                    url = url + f"&{k}=" + query
            return self.get('/' + url, **cache_args)
        if action == 'read':
            assert (endpoint_scheme[action]['action'] == 'get')
            return self.get('/' + endpoint + '/' + id.split('/')[-1], **cache_args)
        elif action == 'create':
            assert (endpoint_scheme[action]['action'] == 'post')
            return self.post('/' + endpoint, data=data, files=files)
        elif action == 'delete':
            assert (endpoint_scheme[action]['action'] == 'delete')
            return self.delete('/' + endpoint + '/' + id.split('/')[-1])
        elif action == 'partial_update':
            assert (endpoint_scheme[action]['action'] == 'patch')
            return self.patch('/' + endpoint + '/' + id.split('/')[-1], data=data, files=files)
        elif action == 'update':
            assert (endpoint_scheme[action]['action'] == 'put')
            return self.put('/' + endpoint + '/' + id.split('/')[-1], data=data, files=files)

    # JSON field interface convenience methods
    def _check_inputs(self, endpoint: str) -> None:
        # make sure the queryied endpoint exists, if not throw an informative error
        if endpoint not in self.rest_schemes.keys():
            av = [k for k in self.rest_schemes.keys() if not k.startswith('_') and k]
            raise ValueError('REST endpoint "' + endpoint + '" does not exist. Available ' +
                             'endpoints are \n       ' + '\n       '.join(av))
        return

    def json_field_write(
            self,
            endpoint: str = None,
            uuid: str = None,
            field_name: str = None,
            data: dict = None
    ) -> dict:
        """json_field_write [summary]
        Write data to WILL NOT CHECK IF DATA EXISTS
        NOTE: Destructive write!

        :param endpoint: Valid alyx endpoint, defaults to None
        :type endpoint: str, optional
        :param uuid: uuid or lookup name for endpoint
        :type uuid: str, optional
        :param field_name: Valid json field name, defaults to None
        :type field_name: str, optional
        :param data: data to write to json field, defaults to None
        :type data: dict, optional
        :return: Written data dict
        :rtype: dict
        """
        self._check_inputs(endpoint)
        # Prepare data to patch
        patch_dict = {field_name: data}
        # Upload new extended_qc to session
        ret = self.rest(endpoint, "partial_update", id=uuid, data=patch_dict)
        return ret[field_name]

    def json_field_update(
            self,
            endpoint: str = None,
            uuid: str = None,
            field_name: str = 'json',
            data: dict = None
    ) -> dict:
        """json_field_update
        Non destructive update of json field of endpoint for object
        Will update the field_name of the object with pk = uuid of given endpoint
        If data has keys with the same name of existing keys it will squash the old
        values (uses the dict.update() method)

        Example:
        one.alyx.json_field_update("sessions", "eid_str", "extended_qc", {"key": value})

        :param endpoint: endpoint to hit
        :type endpoint: str
        :param uuid: uuid or lookup name of object
        :type uuid: str
        :param field_name: name of the json field
        :type field_name: str
        :param data: dictionary with fields to be updated
        :type data: dict
        :return: new patched json field contents
        :rtype: dict
        """
        self._check_inputs(endpoint)
        # Load current json field contents
        current = self.rest(endpoint, "read", id=uuid)[field_name]
        if current is None:
            current = {}

        if not isinstance(current, dict):
            _logger.warning(
                f"Current json field {field_name} does not contains a dict, aborting update"
            )
            return current

        # Patch current dict with new data
        current.update(data)
        # Prepare data to patch
        patch_dict = {field_name: current}
        # Upload new extended_qc to session
        ret = self.rest(endpoint, "partial_update", id=uuid, data=patch_dict)
        return ret[field_name]

    def json_field_remove_key(
            self,
            endpoint: str = None,
            uuid: str = None,
            field_name: str = 'json',
            key: str = None
    ) -> Optional[dict]:
        """json_field_remove_key
        Will remove inputted key from json field dict and reupload it to Alyx.
        Needs endpoint, uuid and json field name

        :param endpoint: endpoint to hit, defaults to None
        :type endpoint: str, optional
        :param uuid: uuid or lookup name for endpoint
        :type uuid: str, optional
        :param field_name: json field name of object, defaults to None
        :type field_name: str, optional
        :param key: key name of dictionary inside object, defaults to None
        :type key: str, optional
        :return: returns new content of json field
        :rtype: dict
        """
        self._check_inputs(endpoint)
        current = self.rest(endpoint, "read", id=uuid)[field_name]
        # If no contents, cannot remove key, return
        if current is None:
            return current
        # if contents are not dict, cannot remove key, return contents
        if isinstance(current, str):
            _logger.warning(f"Cannot remove key {key} content of json field is of type str")
            return None
        # If key not present in contents of json field cannot remove key, return contents
        if current.get(key, None) is None:
            _logger.warning(
                f"{key}: Key not found in endpoint {endpoint} field {field_name}"
            )
            return current
        _logger.info(f"Removing key from dict: '{key}'")
        current.pop(key)
        # Re-write contents without removed key
        written = self.json_field_write(
            endpoint=endpoint, uuid=uuid, field_name=field_name, data=current
        )
        return written

    def json_field_delete(
            self, endpoint: str = None, uuid: str = None, field_name: str = None
    ) -> None:
        self._check_inputs(endpoint)
        _ = self.rest(endpoint, "partial_update", id=uuid, data={field_name: None})
        return _[field_name]

    def clear_rest_cache(self):
        for file in one.params.get_rest_dir(self.base_url).glob('*'):
            file.unlink()
