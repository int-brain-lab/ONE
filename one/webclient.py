import json
import logging
import math
import os
import re
import functools
import urllib.request
from urllib.error import HTTPError
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import warnings
import hashlib
import shutil
import zipfile
import tempfile

import requests
from tqdm import tqdm

from pprint import pprint
import one.params
from one.lib.io import hashfile, spikeglx
import one.alf.io as alfio

SDSC_ROOT_PATH = PurePosixPath('/mnt/ibl')
_logger = logging.getLogger('ibllib')


def cache_response(func, mode='get', default_expiry=timedelta(days=1)):
    @functools.wraps(func)
    def wrapper_decorator(*args, expires=None, **kwargs):
        expires = expires or default_expiry
        if args[1].__name__ != mode:
            return func(*args, **kwargs)
        # Check cache
        proc, loc = args[0].base_url.replace(':/', '').split('/')
        rest_cache = Path(one.params.get_params_dir(), '.rest', loc, proc)
        sha1 = hashlib.sha1()
        sha1.update(bytes(args[2], 'utf-8'))
        name = sha1.hexdigest()
        # Reversible but length may exceed 255 chars
        # name = base64.urlsafe_b64encode(args[2].encode('UTF-8')).decode('UTF-8')
        files = list(rest_cache.glob(name))
        cached = None
        if len(files) == 1:
            _logger.debug('loading REST response from cache')
            with open(files[0], 'r') as f:
                cached, when = json.load(f)
            if datetime.fromisoformat(when) > datetime.now():
                return cached
        try:
            response = func(*args, **kwargs)
        except requests.exceptions.ConnectionError as ex:
            if cached:
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


def _path_from_filerecord(fr, root_path=SDSC_ROOT_PATH, uuid=None):
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
        cache_dir = str(Path.home().joinpath("Downloads"))

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
        req.add_header("Range", "bytes=%d-%d" % (first_byte, first_byte + n_bytes - 1))

    # add additional headers
    if headers is not None:
        for k in headers:
            req.add_header(k, headers[k])

    # Open the url and get the length
    try:
        u = urllib.request.urlopen(req)
    except HTTPError as e:
        _logger.error(f"{str(e)} {full_link_to_file}")
        raise e

    file_size = int(u.getheader('Content-length'))
    if not silent:
        print(f"Downloading: {file_name} Bytes: {file_size}")
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


def file_record_to_url(file_records, urls=[]):
    """
    Translate a Json dictionary to an usable http url for downlading files.

    :param file_records: json containing a 'data_url' field
    :type file_records: dict
    :param urls: a list of strings containing previous data_urls on which new urls
     will be appended
    :type urls: list

    :return: urls: (list) a list of strings representing full data urls
    """
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
    if type(dataset_record) is dict:
        dataset_record = [dataset_record]
    for ds in dataset_record:
        urls = file_record_to_url(ds['file_records'], urls)
    return urls


class UniqueSingletons(type):  # TODO Perhaps make ONE the singleton
    _instances: list = []

    def __call__(cls, *args, **kwargs):
        # print('args', args, '\nkwargs', kwargs)
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
    http://alyx.readthedocs.io/en/latest/api.html
    """
    _token = None
    _headers = None

    def __init__(self, base_url=None, username=None, password=None, cache_dir=None, silent=False):
        """
        Create a client instance that allows to GET and POST to the Alyx server
        For oneibl, constructor attempts to authenticate with credentials in params.py
        For standalone cases, AlyxClient(username='', password='', base_url='')

        :param username: Alyx database user
        :type username: str
        :param password: Alyx database password
        :type password: str
        :param base_url: Alyx server address, including port and protocol
        :type base_url: str
        """
        self.silent = silent
        self._par = one.params.get(client=base_url, silent=self.silent)
        # TODO Pass these to `get` and have it deal with setup defaults
        self._par = self._par.set('ALYX_LOGIN', username or self._par.ALYX_LOGIN)
        self._par = self._par.set('ALYX_PWD', password or self._par.ALYX_PWD)
        self._par = self._par.set('ALYX_URL', base_url or self._par.ALYX_URL)
        self._par = self._par.set('CACHE_DIR', cache_dir or self._par.CACHE_DIR)
        self.authenticate()
        self._rest_schemes = None
        # the mixed accept application may cause errors sometimes, only necessary for the docs
        self._headers['Accept'] = 'application/json'
        self._obj_id = id(self)

    @property
    def rest_schemes(self):
        """Delayed fetch of rest schemes speeds up instantiation"""
        if not self._rest_schemes:
            self._rest_schemes = self.get('/docs')
        return self._rest_schemes

    @property
    def cache_dir(self):
        return self._par.CACHE_DIR

    @property
    def base_url(self):
        return self._par.ALYX_URL

    def list_endpoints(self):
        """
        Return a list of available REST endpoints
        :return: List of REST endpoint strings
        """
        EXCLUDE = ('_type', '_meta', '', 'auth-token')
        return sorted(x for x in self.rest_schemes.keys() if x not in EXCLUDE)

    @cache_response
    def _generic_request(self, reqfunction, rest_query, data=None, files=None):
        # makes sure the base url is the one from the instance
        rest_query = rest_query.replace(self._par.ALYX_URL, '')
        if not rest_query.startswith('/'):
            rest_query = '/' + rest_query
        _logger.debug(f"{self._par.ALYX_URL + rest_query}, headers: {self._headers}")
        headers = self._headers.copy()
        if files is None:
            data = json.dumps(data) if isinstance(data, dict) or isinstance(data, list) else data
            headers['Content-Type'] = 'application/json'
        if rest_query.startswith('/docs'):
            # the mixed accept application may cause errors sometimes, only necessary for the docs
            headers['Accept'] = 'application/coreapi+json'
        r = reqfunction(self._par.ALYX_URL + rest_query, stream=True, headers=headers,
                        data=data, files=files)
        if r and r.status_code in (200, 201):
            return json.loads(r.text)
        elif r and r.status_code == 204:
            return
        else:
            _logger.error(self._par.ALYX_URL + rest_query)
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
            return
        try:
            credentials = {'username': self._par.ALYX_LOGIN, 'password': self._par.ALYX_PWD}
            rep = requests.post(self._par.ALYX_URL + '/auth-token', data=credentials)
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Can't connect to {self._par.ALYX_URL}.\n" +
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
        if not self.silent:
            print(f"Connected to {self._par.ALYX_URL} as {self._par.ALYX_LOGIN}")

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
            headers=self._headers,
            silent=kwargs.pop('silent', self.silent),
            cache_dir=kwargs.pop('cache_dir', self._par.CACHE_DIR),
            username=self._par.HTTP_DATA_SERVER_LOGIN,
            password=self._par.HTTP_DATA_SERVER_PWD,
            **kwargs
        )
        return download_fcn(url, **pars)

    def download_raw_partial(self, url_cbin, url_ch, first_chunk=0, last_chunk=0):
        """
        TODO Document; move into ibllib
        :param url_cbin:
        :param url_ch:
        :param first_chunk:
        :param last_chunk:
        :return:
        """
        import warnings
        warnings.warn('This method will soon be moved to ibllib.io.one', DeprecationWarning)
        assert str(url_cbin).endswith('.cbin')
        assert str(url_ch).endswith('.ch')

        relpath = Path(url_cbin.replace(self._par.HTTP_DATA_SERVER, '.')).parents[0]
        target_dir = Path(self.cache_dir, relpath)
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        # First, download the .ch file if necessary
        if isinstance(url_ch, Path):
            ch_file = url_ch
        else:
            ch_file = Path(self.download_file(
                url_ch, cache_dir=target_dir, clobber=True, return_md5=False))
            ch_file = alfio.remove_uuid_file(ch_file)
        ch_file_stream = ch_file.with_suffix('.stream.ch')

        # Load the .ch file.
        with open(ch_file, 'r') as f:
            cmeta = json.load(f)

        # Get the first byte and number of bytes to download.
        i0 = cmeta['chunk_bounds'][first_chunk]
        ns_stream = cmeta['chunk_bounds'][last_chunk + 1] - i0

        # if the cached version happens to be the same as the one on disk, just load it
        if ch_file_stream.exists():
            with open(ch_file_stream, 'r') as f:
                cmeta_stream = json.load(f)
            if (cmeta_stream.get('chopped_first_sample', None) == i0 and
                    cmeta_stream.get('chopped_total_samples', None) == ns_stream):
                return spikeglx.Reader(ch_file_stream.with_suffix('.cbin'))
        else:
            shutil.copy(ch_file, ch_file_stream)
        assert ch_file_stream.exists()

        # prepare the metadata file
        cmeta['chunk_bounds'] = cmeta['chunk_bounds'][first_chunk:last_chunk + 2]
        cmeta['chunk_bounds'] = [_ - i0 for _ in cmeta['chunk_bounds']]
        assert len(cmeta['chunk_bounds']) >= 2
        assert cmeta['chunk_bounds'][0] == 0

        first_byte = cmeta['chunk_offsets'][first_chunk]
        cmeta['chunk_offsets'] = cmeta['chunk_offsets'][first_chunk:last_chunk + 2]
        cmeta['chunk_offsets'] = [_ - first_byte for _ in cmeta['chunk_offsets']]
        assert len(cmeta['chunk_offsets']) >= 2
        assert cmeta['chunk_offsets'][0] == 0
        n_bytes = cmeta['chunk_offsets'][-1]
        assert n_bytes > 0

        # Save the chopped chunk bounds and offsets.
        cmeta['sha1_compressed'] = None
        cmeta['sha1_uncompressed'] = None
        cmeta['chopped'] = True
        cmeta['chopped_first_sample'] = i0
        cmeta['chopped_total_samples'] = ns_stream

        with open(ch_file_stream, 'w') as f:
            json.dump(cmeta, f, indent=2, sort_keys=True)

        # Download the requested chunks
        cbin_local_path = self.download_file(
            url_cbin, chunks=(first_byte, n_bytes),
            cache_dir=target_dir, clobber=True, return_md5=False)
        cbin_local_path = alfio.remove_uuid_file(cbin_local_path)
        cbin_local_path_renamed = cbin_local_path.with_suffix('.stream.cbin')
        cbin_local_path.rename(cbin_local_path_renamed)
        assert cbin_local_path_renamed.exists()

        shutil.copy(cbin_local_path.with_suffix('.meta'),
                    cbin_local_path_renamed.with_suffix('.meta'))
        reader = spikeglx.Reader(cbin_local_path_renamed)
        return reader

    def download_cache_tables(self):
        """
        TODO Document
        :return: List of parquet table file paths
        """
        # query the database for the latest cache; expires=None overrides cached response
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

    def rest(self, url=None, action=None, id=None, data=None, files=None, **kwargs):
        """
        alyx_client.rest(): lists endpoints
        alyx_client.rest(endpoint): lists actions for endpoint
        alyx_client.rest(endpoint, action): lists fields and URL

        Example with a rest endpoint with all actions

        >>> alyx.client.rest('subjects', 'list')
            alyx.client.rest('subjects', 'list', field_filter1='filterval')
            alyx.client.rest('subjects', 'create', data=sub_dict)
            alyx.client.rest('subjects', 'read', id='nickname')
            alyx.client.rest('subjects', 'update', id='nickname', data=sub_dict)
            alyx.client.rest('subjects', 'partial_update', id='nickname', data=sub_dict)
            alyx.client.rest('subjects', 'delete', id='nickname')
            alyx.client.rest('notes', 'create', data=nd, files={'image': open(image_file, 'rb')})

        :param url: endpoint name
        :param action: 'list', 'create', 'read', 'update', 'partial_update', 'delete'
        :param id: lookup string for actions 'read', 'update', 'partial_update', and 'delete'
        :param data: data dictionary for actions 'update', 'partial_update' and 'create'
        :param files: if file upload
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
        if action not in endpoint_scheme.keys():
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
                for k in kwargs.keys():
                    if isinstance(kwargs[k], str):
                        query = kwargs[k]
                    elif isinstance(kwargs[k], list):
                        query = ','.join(kwargs[k])
                    else:
                        query = str(kwargs[k])
                    url = url + f"&{k}=" + query
            return self.get('/' + url)
        if action == 'read':
            assert (endpoint_scheme[action]['action'] == 'get')
            return self.get('/' + endpoint + '/' + id.split('/')[-1])
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
    ) -> dict:
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
            return current
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
