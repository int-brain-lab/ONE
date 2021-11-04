import os
import re
import sys
import logging
from pathlib import Path, PurePosixPath, PurePath, PureWindowsPath
import warnings

import globus_sdk
from globus_sdk import TransferAPIError, GlobusAPIError, NetworkError, GlobusTimeoutError, \
    GlobusConnectionError, GlobusConnectionTimeoutError
from iblutil.io import params as iopar

from one.alf.spec import is_uuid, is_uuid_string
from one.alf.io import remove_uuid_file
import one.params
from one.webclient import AlyxClient
from .base import DownloadClient, load_client_params, save_client_params

_logger = logging.getLogger(__name__)
CLIENT_KEY = 'globus'
DEFAULT_PAR = {'GLOBUS_CLIENT_ID': None, 'local_endpoint': None, 'local_path': None}


def setup(par_id=None):
    """
    Sets up Globus as a backend for ONE functions.
    In order to use this function you need:

    1. The Client ID of an existing Globus Client, or to create one
       (https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html).
    2. Set up Global Connect on your local device (https://www.globus.org/globus-connect-personal).
    3. Register your local device as an Endpoint in your Globus Client (https://app.globus.org/).

    Parameters
    ----------
    par_id : str
        Parameter profile name to set up e.g. 'default', 'admin'.

    """

    print('Setting up Globus parameter file. See docstring for help.')
    if not par_id:
        default_par_id = 'default'
        par_id = input(
            f'Enter name for this client or press Enter to keep value "{default_par_id}": '
        )
        par_id = par_id.strip() or default_par_id

    # Read existing globus params if present
    globus_pars = iopar.as_dict(load_client_params(CLIENT_KEY, assert_present=False) or {})
    pars = {**DEFAULT_PAR, **globus_pars.get(par_id, {})}

    # Set GLOBUS_CLIENT_ID
    current_id = pars['GLOBUS_CLIENT_ID']
    if current_id:
        prompt = (f'Found Globus client ID in parameter file ({current_id}). '
                  'Press Enter to keep it, or enter a new ID here: ')
        pars['GLOBUS_CLIENT_ID'] = input(prompt).strip() or current_id
    else:
        new_id = input(f'Please enter the Globus client ID: ').strip()
        if not new_id:
            raise ValueError('Globus client ID is a required field')
        pars['GLOBUS_CLIENT_ID'] = new_id

    # Find and set local ID
    message = 'Please enter the local endpoint ID'
    try:
        default_endpoint = pars['local_endpoint'] or get_local_endpoint_id()
        message += f' (default: {default_endpoint})'
    except AssertionError:
        default_endpoint = ''
        warnings.warn(
            'Cannot find local endpoint ID. Beware that this might mean that Globus Connect '
            'is not set up properly.')
    pars['local_endpoint'] = input(message + ':').strip() or default_endpoint
    if not pars['local_endpoint']:
        raise ValueError('Globus local endpoint ID is a required field')

    # Check for local path
    message = 'Please enter the local endpoint path'
    local_path = pars['local_path'] or one.params.get(silent=True).CACHE_DIR
    message += f' (default: {local_path})'
    pars['local_path'] = input(message + ':').strip() or local_path

    # Log in manually and get refresh token to avoid having to login repeatedly
    client = globus_sdk.NativeAppAuthClient(pars['GLOBUS_CLIENT_ID'])
    client.oauth2_start_flow(refresh_tokens=True)
    authorize_url = client.oauth2_get_authorize_url()
    print('To get a new token, go to this URL and login: {0}'.format(authorize_url))
    auth_code = input('Enter the code you get after login here (press "c" to cancel): ').strip()
    if auth_code and auth_code.lower() != 'c':
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)
        globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']
        for par in ['refresh_token', 'access_token', 'expires_at_seconds']:
            pars[par] = globus_transfer_data[par]

    globus_pars[par_id] = pars
    save_client_params(globus_pars, client_key=CLIENT_KEY)
    print('Finished setup.')


def create_globus_client(client_name='default'):
    """
    Creates a Globus transfer client based on existing parameter file.

    Parameters
    ----------
    client_name : str
        Defines the parameter name to use (globus.client_name), e.g. 'default', 'admin'.

    Returns
    -------
    globus_sdk.TransferClient
        Globus transfer client instance
    """
    try:
        globus_pars = load_client_params(f'{CLIENT_KEY}.{client_name}')
    except (AttributeError, FileNotFoundError):
        setup(client_name)
        globus_pars = load_client_params(f'{CLIENT_KEY}.{client_name}', assert_present=False) or {}
    required_fields = {'refresh_token', 'GLOBUS_CLIENT_ID'}
    if not (globus_pars and required_fields.issubset(iopar.as_dict(globus_pars))):
        raise ValueError('No token in client parameter file. Run one.globus.setup first')
    client = globus_sdk.NativeAppAuthClient(globus_pars.GLOBUS_CLIENT_ID)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus_sdk.RefreshTokenAuthorizer(globus_pars.refresh_token, client)
    return globus_sdk.TransferClient(authorizer=authorizer)


def get_local_endpoint_id():
    """
    Extracts the ID of the local Globus Connect endpoint.

    Returns
    -------
    str
        The local Globus endpoint ID
    """
    msg = ('Cannot find local endpoint ID, check if Globus Connect is set up correctly, '
           '{} exists and contains a UUID.')
    if sys.platform in ('win32', 'cygwin'):
        id_path = Path(os.environ['LOCALAPPDATA']).joinpath('Globus Connect')
    else:
        id_path = Path.home().joinpath('.globusonline', 'lta')

    id_file = id_path.joinpath('client-id.txt')
    assert id_file.exists(), msg.format(id_file)
    local_id = id_file.read_text().strip()
    assert isinstance(local_id, str), msg.format(id_file)
    print(f'Found local endpoint ID in Globus Connect settings {local_id}')
    return local_id


def get_local_endpoint_paths():
    """
    Extracts the local endpoint paths accessible by Globus Connect.  NB: This is only supported
    on Linux.

    Returns
    -------
    list of pathlib.Path
        Local endpoint paths set in Globus Connect.
    """
    if sys.platform in ('win32', 'cygwin'):
        print('On windows the local Globus path needs to be entered manually')
        return []
    else:
        path_file = Path.home().joinpath('.globusonline', 'lta', 'config-paths')
        if path_file.exists():
            local_paths = map(Path, filter(None, path_file.read_text().strip().split(',')))
            _logger.debug(f'Found local endpoint paths in Globus Connect settings')
        else:
            msg = ('Cannot find local endpoint path, check if Globus Connect is set up correctly, '
                   '{} exists and contains a valid path.')
            warnings.warn(msg.format(path_file))
            local_paths = []
        return list(local_paths)


def get_lab_from_endpoint_id(endpoint=None, alyx=None):
    """
    Extracts lab name given an endpoint id root path given a repository name that is registered in the
    database accessed by ONE.

    Parameters
    ----------
    endpoint : uuid.UUID, str
        Endpoint UUID, optional if not given will get attempt to find local endpoint UUID.
    alyx : one.webclient.AlyxClient
        An instance of AlyxClient to use

    Returns
    -------
    str
        Lab name associated with the endpoint UUID
    """

    alyx = alyx or AlyxClient(silent=True)
    if not endpoint:
        endpoint = get_local_endpoint_id()
    lab = alyx.rest('labs', 'list', django=f'repositories__globus_endpoint_id,{endpoint}')
    if len(lab):
        lab_names = [la['name'] for la in lab]
        assert len(set(lab_names)) == 1, f'Multiple labs associated with endpoint UUID {endpoint}'
        return lab_names[0]


def as_globus_path(path):
    """
    Convert a path into one suitable for the Globus TransferClient.  NB:


    Parameters
    ----------
    path : pathlib.Path, pathlib.PurePath, str
        A path to convert to a Globus-complient path string

    Returns
    -------
    str
        A formatted path string

    Notes
    -----
    - If using tilda in path, the home folder of your Globus Connect instance must be the same as
      the OS home dir.
    - If validating a path for another system ensure the input path is a PurePath, in particular,
    on a Linux computer a remote Windows should first be made into a PureWindowsPath.

    Examples
    --------
    A Windows path

    >>> as_globus_path('E:\\FlatIron\\integration')
    '/E/FlatIron/integration'

    A relative POSIX path

    >>> as_globus_path('../data/integration')
    '/mnt/data/integration'

    A globus path

    >>> as_globus_path('/E/FlatIron/integration')
    '/E/FlatIron/integration'
    """
    is_pure_path = isinstance(path, PurePath)
    is_win = sys.platform in ('win32', 'cygwin') or isinstance(path, PureWindowsPath)
    if isinstance(path, str):
        path = Path(path)
    if (
        re.match(r'/[A-Z]($|/)', path.as_posix())
        if is_win
        else path.is_absolute()
    ):
        return path.as_posix()
    if not is_pure_path:
        path = path.resolve()
    if path.drive:
        path = '/' + str(path.as_posix().replace(':', '', 1))
    return str(path)


class Globus(DownloadClient):
    """Wrapper for managing files on Globus endpoints."""
    def __init__(self, client_name='default'):
        # Setting up transfer client
        super().__init__()
        self.client = create_globus_client(client_name=client_name)
        self.pars = load_client_params(f'{CLIENT_KEY}.{client_name}')
        # Try adding local endpoint
        self.endpoints = {'local': {'id': self.pars.local_endpoint}}
        _logger.info('Adding local endpoint.')
        self.endpoints['local']['root_path'] = self.pars.local_path

    def to_address(self, data_path, endpoint):
        pass

    def download_file(self, file_address):
        pass

    @staticmethod
    def setup(*args, **kwargs):
        pass

    def add_endpoint(self, endpoint, label=None, root_path=None, overwrite=False, alyx=None):
        """
        Add an endpoint to the Globus instance to be used by other functions.

        Parameters
        ----------
        endpoint : uuid.UUID, str
            The endpoint UUID or database repository name of the endpoint.
        label : str
            Label to access the endpoint. If endpoint is UUID this has to be set, otherwise is
            optional.
        root_path : str, pathlib.Path, pathlib.PurePath
            File path to be accessed by Globus on the endpoint.
        overwrite : bool
            Whether existing endpoint with the same label should be replaced.
        alyx : webclient.AlyxClient
            An AlyxClient instance for looking up repository information.
        """
        if is_uuid(endpoint, versions=(1,)):  # MAC address UUID
            if label is None:
                raise ValueError('If "endpoint" is a UUID, "label" cannot be None.')
            endpoint_id = str(endpoint)
        else:
            repo = self.repo_from_alyx(endpoint, alyx=alyx)
            endpoint_id = repo['globus_endpoint_id']
            root_path = root_path or repo['globus_path']
            label = label or endpoint
        if label in self.endpoints.keys() and overwrite is False:
            _logger.error(f'An endpoint called "{label}" already exists. Choose a different label '
                          'or set overwrite=True')
        else:
            self.endpoints[label] = {'id': endpoint_id}
            if root_path:
                self.endpoints[label]['root_path'] = root_path

    @staticmethod
    def _endpoint_path(path, root_path=None):
        """
        Given an absolute path or relative path with a root path, return a Globus path str.
        Note: Paths must be POSIX or Globus-compliant paths.  In other words for Windows systems
        the input root_path or absolute path must be passed through `as_globus_path` before
        calling this method.

        TODO include globus_path_from_dataset

        Parameters
        ----------
        path : Path, PurePath, str
            An absolute or relative POSIX path
        root_path : Path, PurePath, str
            A root path to prepend.  Optional if `path` is absolute.

        Returns
        -------
        str
            A path string formatted for Globus.

        See Also
        --------
        as_globus_path

        Raises
        ------
        ValueError
            Path was not absolute and no root path was given.  An absolute path must start with
            a slash on *nix systems.
        """
        if isinstance(path, str):
            path = PurePosixPath(path)
        if root_path and not str(path).startswith(str(root_path)):
            path = PurePosixPath(root_path) / path
        if not path.is_absolute():
            raise ValueError(f'{path} is relative and no root_path defined')
        return as_globus_path(path)

    def _endpoint_id_root(self, endpoint):
        root_path = None
        if endpoint in self.endpoints.keys():
            endpoint_id = self.endpoints[endpoint]['id']
            if 'root_path' in self.endpoints[endpoint].keys():
                root_path = str(self.endpoints[endpoint]['root_path'])
            return endpoint_id, root_path
        elif is_uuid(endpoint, range(1, 5)):
            endpoint_id = endpoint
            return endpoint_id, None
        else:
            raise ValueError(
                '"endpoint" must be a UUID or the label of an endpoint registered in this '
                'Globus instance. You can add endpoints via the add_endpoints method')

    def ls(self, endpoint, path, remove_uuid=False, return_size=False, max_retries=1):
        """
        Return the list of (filename, filesize) in a given endpoint directory.

        Parameters
        ----------
        endpoint
        path
        remove_uuid
        return_size

        Returns
        -------

        """
        # Check if endpoint is a UUID, if not try to get UUID from registered endpoints
        endpoint_id, root_path = self._endpoint_id_root(endpoint)
        # Check if root_path should be added and if path is absolute
        path = self._endpoint_path(path, root_path)
        # Do the actual listing
        out = []
        response = []
        for i in range(max_retries):
            try:
                response = self.client.operation_ls(endpoint_id, path=path)
                break
            except (GlobusConnectionError, GlobusAPIError) as ex:
                if i == max_retries - 1:
                    raise ex
        for entry in response:
            fn = remove_uuid_file(entry['name'], dry=True) if remove_uuid else entry['name']
            if return_size:
                size = entry['size'] if entry['type'] == 'file' else None
                out.append((fn, size))
            else:
                out.append(fn)

        return out

    # TODO: allow to move all content of a directory with 'recursive' keyword in add_item
    def mv(self, source_endpoint, target_endpoint, source_paths, target_paths, timeout=None):
        """
        Move files from one endpoint to another.
        """
        source_endpoint, source_root = self._endpoint_id_root(source_endpoint)
        target_endpoint, target_root = self._endpoint_id_root(target_endpoint)
        source_paths = [str(self._endpoint_path(path, source_root)) for path in source_paths]
        target_paths = [str(self._endpoint_path(path, target_root)) for path in target_paths]

        tdata = globus_sdk.TransferData(self.client, source_endpoint, target_endpoint,
                                        verify_checksum=True, sync_level='checksum',
                                        label=f'ONE globus')
        for source_path, target_path in zip(source_paths, target_paths):
            tdata.add_item(source_path, target_path)

        def wrapper():
            response = self.client.submit_transfer(tdata)
            task_id = response.get('task_id', None)
            return task_id

        return self.run_task(wrapper, time_out=timeout)

    def run_task(self, globus_func, retries=3, time_out=None, skip_source_errors=False):
        """
        Block until a Globus task finishes and retry upon Network or REST Errors.
        globus_func needs to submit a task to the client and return a task_id


        Parameters
        ----------
        globus_func
        retries
        time_out
        skip_source_errors

        Returns
        -------

        """
        try:
            task_id = globus_func()
            print(f'Waiting for Globus task {task_id} to complete')
            # While the task with task is active, print a dot every second. Timeout after timeout
            i = 0
            while not self.client.task_wait(task_id, timeout=5, polling_interval=1):
                print('.', end='')
                i += 1
                if time_out and i >= time_out:
                    task = self.client.get_task(task_id)
                    raise IOError(f'Globus task {task_id} timed out after {time_out} seconds, '
                                  f'with task status {task["status"]}')
            task = self.client.get_task(task_id)
            if task['status'] == 'SUCCEEDED':
                # Sometime Globus sets the status to SUCCEEDED but doesn't truly finish.
                # Handle error thrown when querying task_successful_transfers too early
                try:
                    successful = self.client.task_successful_transfers(task_id)
                    skipped = self.client.task_skipped_errors(task_id)
                    print(f'\nGlobus task {task_id} completed.'
                          f'\nSkipped transfers: {len(list(skipped))}'
                          f'\nSuccessful transfers: {len(list(successful))}')
                    for info in successful:
                        _logger.debug(f'{info["source_path"]} -> {info["destination_path"]}')
                except TransferAPIError:
                    _logger.warning(f'\nGlobus task {task_id} SUCCEEDED but querying transfers was'
                                    f'unsuccessful')
            else:
                raise IOError(f'Globus task finished unsucessfully with status {task["status"]}')
            return task_id
        except (GlobusAPIError, NetworkError, GlobusTimeoutError, GlobusConnectionError,
                GlobusConnectionTimeoutError) as e:
            if retries < 1:
                _logger.error(f'\nMax retries exceeded.')
                raise e
            else:
                _logger.debug('\nGlobus experienced a network error', exc_info=True)
                # if we reach this point without returning or erring, retry
                _logger.warning('\nGlobus experienced a network error, retrying.')
                self.run_task(globus_func, retries=(retries - 1), time_out=time_out)
