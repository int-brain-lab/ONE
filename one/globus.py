import os
import re
import sys
import logging
from pathlib import Path

from iblutil.io import params as iopar
from one.alf.spec import is_uuid, is_uuid_string
from one.api import ONE

import globus_sdk
from globus_sdk import TransferAPIError, GlobusAPIError, NetworkError, GlobusTimeoutError, \
    GlobusConnectionError, GlobusConnectionTimeoutError

_logger = logging.getLogger(__name__)
_PAR_ID_STR = 'globus'
DEFAULT_PAR = {'GLOBUS_CLIENT_ID': None, 'local_endpoint': None, 'local_path': None}


def setup():
    """
    Sets up Globus as a backend for ONE functions.

    In order to use this function you need:
    1. Have the Client ID of an existing Globus Client, or create one
       (https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html)
    2. Set up Global Connect on your local device (https://www.globus.org/globus-connect-personal)
    3. Register your local device as an Enpoint in your Globus Client (https://app.globus.org/)
    """

    print("Setting up Globus parameter file. See docstring for help.")
    par_id = input("Enter name for this client or press Enter to keep value 'default': ")
    par_id = _PAR_ID_STR + '/default' if par_id == '' else _PAR_ID_STR + f'/{par_id.strip(".")}'

    # Read existing globus params if present
    pars = iopar.read(par_id, DEFAULT_PAR)

    # Set GLOBUS_CLIENT_ID
    current_id = getattr(pars, 'GLOBUS_CLIENT_ID')
    if current_id:
        new_id = input(f'Found Globus client ID in parameter file ({current_id}). Press Enter to '
                       f'keep it, or enter a new ID here: ').strip()
        new_id = current_id if not new_id else new_id
    else:
        new_id = input(f'Please enter the Globus client ID: ').strip()
    pars = pars.set('GLOBUS_CLIENT_ID', new_id)

    # Find and set local ID
    try:
        local_endpoint = get_local_endpoint_id()
    except AssertionError:
        try:
            local_endpoint = getattr(pars, 'local_endpoint')
            print(f"Found local endpoint ID in parameter file ({local_endpoint}).")
        except AttributeError:
            local_endpoint = input(
                "Cannot find local endpoint ID. Beware that this might mean that Globus Connect "
                "is not set up properly. You can enter the local endpoint ID manually here: ")
    pars = pars.set('local_endpoint', local_endpoint)

    # Check for local path
    try:
        local_path = get_local_endpoint_path()
    except AssertionError:
        try:
            local_path = getattr(pars, 'local_path')
            print(f"Found local endpoint path in parameter file ({local_path}).")
        except AttributeError:
            local_path = input(
                "Cannot find local endpoint path accessible by Globys. Beware that this might mean"
                "that Globus Connect is not set up properly. Press Enter to leave this field "
                "empty, or enter path here: ")

    pars = pars.set('local_path', local_path)

    # Log in manually and get refresh token to avoid having to login repeatedly
    client = globus_sdk.NativeAppAuthClient(pars.GLOBUS_CLIENT_ID)
    client.oauth2_start_flow(refresh_tokens=True)
    authorize_url = client.oauth2_get_authorize_url()
    print('To get a new token, go to this URL and login: {0}'.format(authorize_url))
    auth_code = input('Enter the code you get after login here (press "c" to cancel): ').strip()
    if auth_code in ['c', 'C', '']:
        return

    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']
    for par in ['refresh_token', 'access_token', 'expires_at_seconds']:
        pars = pars.set(par, globus_transfer_data[par])

    iopar.write(par_id, pars)
    print("Finished setup.")


def create_globus_client(client_name='default'):
    """
    Creates a Globus transfer client based on existing parameter file.

    :param client_name: str, defines the parameter file to use (~/.globus/.client_name)
    :return: Globus transfer client
    """
    par_id = _PAR_ID_STR + f'/{client_name}'
    try:
        globus_pars = iopar.read(par_id)
    except FileNotFoundError as err:
        _logger.error(f"{str(err)}. Choose a different client name or run one.globus.setup() "
                      f"first\n")
        return
    required_fields = {'refresh_token', 'GLOBUS_CLIENT_ID'}
    if not (globus_pars and required_fields.issubset(globus_pars.as_dict())):
        raise ValueError("No token in client parameter file. Run one.globus.setup first")
    client = globus_sdk.NativeAppAuthClient(globus_pars.GLOBUS_CLIENT_ID)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus_sdk.RefreshTokenAuthorizer(globus_pars.refresh_token, client)
    return globus_sdk.TransferClient(authorizer=authorizer)


def get_local_endpoint_id():
    """
    Extracts the ID of the local Globus Connect endpoint.
    :return: str, local endpoint ID
    """
    msg = ("Cannot find local endpoint ID, check if Globus Connect is set up correctly, "
           "{} exists and contains a UUID.")
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        id_path = Path(os.environ['LOCALAPPDATA']).joinpath("Globus Connect")
    else:
        id_path = Path.home().joinpath(".globusonline", "lta")

    id_file = id_path.joinpath("client-id.txt")
    assert (id_file.exists()), msg.format(id_file)
    local_id = id_file.read_text().strip()
    assert (isinstance(local_id, str)), msg.format(id_file)
    print(f"Found local endpoint ID in Globus Connect settings {local_id}")
    return local_id


def get_local_endpoint_path():
    """
    Extracts the local endpoint path accessible by Globus Connect.
    :return: str, local_path
    """
    msg = ("Cannot find local endpoint path, check if Globus Connect is set up correctly, "
           "{} exists and contains a valid path.")
    if sys.platform == 'win32':
        local_path_input = input(f'On windows the local globus path needs to be entered manually. Press Enter to use the same'
                                 f' path as the ONE cache dir or enter a path here: ').strip()
        local_path = ONE().cache_dir if not local_path_input else Path(local_path_input)
    else:
        path_file = Path.home().joinpath(".globusonline", "lta", "config-paths")
        if path_file.exists():
            local_path = Path(path_file.read_text().strip().split(',')[0])

    if local_path.exists():
        print(f"Found local endpoint path in Globus Connect settings {local_path}")
        return str(local_path)
    _logger.warning(msg.format(path_file))
    return None


def get_endpoint_info_from_name(endpoint=None, one=None):
    """
    Extracts Globus endpoint ID and root path given a repository name that is registered in the
    database accessed by ONE.

    :param endpoint: str, repository name as registered in database
    :param one: ONE instance, optional
    :return: tuple of str, (endpoint_id, root_path)
    """
    one = one or ONE()
    repos = one.alyx.rest('data-repository', 'list')
    repo_names = [r['name'] for r in repos]
    if not endpoint or endpoint not in repo_names:
        _logger.error("Choose one of the following endpoints from the current Alyx database or "
                      "pass an ONE instance with a different base_url:")
        _ = [print(n) for n in repo_names]
    else:
        endpoint_dict = [r for r in repos if r['name'] == endpoint][0]
        return endpoint_dict['globus_endpoint_id'], endpoint_dict['globus_path']


def get_lab_from_endpoint_id(endpoint=None, one=None):
    """
    Extracts lab name given an endpoint id root path given a repository name that is registered in the
    database accessed by ONE.

    :param endpoint: endpoint UUID, optional if not given will get attempt to find local endpoint id
    :param one: ONE instance, optional
    :return: list of str, [lab name]
    """

    one = one or ONE()
    if not endpoint:
        endpoint = get_local_endpoint_id()
    lab = one.alyx.rest('labs', 'list', django=f"repositories__globus_endpoint_id,{endpoint}")
    if len(lab):
        return [la['name'] for la in lab]


def _remove_uuid_from_filename(file_path):
    file_path = Path(file_path)
    name_parts = file_path.name.split('.')
    if len(name_parts) < 2 or not is_uuid_string(name_parts[-2]):
        return str(file_path)
    name_parts.pop(-2)
    return str(file_path.parent.joinpath('.'.join(name_parts)))


def as_globus_path(path):
    """
    Copied from ibblib.io.globus
    Convert a path into one suitable for the Globus TransferClient.  NB: If using tilda in path,
    the home folder of your Globus Connect instance must be the same as the OS home dir.

    :param path: A path str or Path instance
    :return: A formatted path string

    Examples:
        # A Windows path
        >>> as_globus_path('E:\\FlatIron\\integration')
        >>> '/E/FlatIron/integration'

        # A relative POSIX path
        >>> as_globus_path('../data/integration')
        >>> '/mnt/data/integration'

        # A globus path
        >>> as_globus_path('/E/FlatIron/integration')
        >>> '/E/FlatIron/integration'
    """
    path = str(path)
    if (
        re.match(r'/[A-Z]($|/)', path)
        if sys.platform in ('win32', 'cygwin')
        else Path(path).is_absolute()
    ):
        return path
    path = Path(path).resolve()
    if path.drive:
        path = '/' + str(path.as_posix().replace(':', '', 1))
    return str(path)


class Globus:
    """
    Wrapper for managing files on Globus endpoints.
    Adapted from https://github.com/int-brain-lab/ibllib/blob/ca8cd93ccac665efda6943f14047ba53503e8bb8/ibllib/io/globus.py
    """
    def __init__(self, client_name='default'):
        # Setting up transfer client
        self.client = create_globus_client(client_name=client_name)
        self.pars = iopar.read(_PAR_ID_STR + f'/{client_name}')
        # Try adding local endpoint
        if hasattr(self.pars, 'local_endpoint'):
            self.endpoints = {'local': {'id': self.pars.local_endpoint}}
            _logger.info("Adding local endpoint.")
            if hasattr(self.pars, 'local_path'):
                self.endpoints['local']['root_path'] = self.pars.local_path
        else:
            _logger.warning("Not adding local endpoint as information is missing from parameter "
                            "file. Add endpoints manually or run one.globus.setup() first.")
            self.endpoints = {}

    def add_endpoint(self, endpoint, label=None, root_path=None, overwrite=False, one=None):
        """
        Add an endpoint to the Globus instance to be used by other functions.

        :param endpoint: UUID or str, the endpoint UUID or database repository name of the endpoint
        :param label: str, label to access the endpoint. If endpoint is UUID this has to be set,
        otherwise optional
        :param root_path: str or Path, path to be accessed by globus on the endpoint.
        :param overwrite: bool, whether existing endpoint with the same label should be replaced
        :param one: ONE instance for extracting endpoint information from database. If endpoint is
        UUID this has no effect.
        :return:
        """
        if is_uuid(endpoint, range(5)):
            if label is None:
                _logger.error("If 'endpoint' is a UUID, 'label' cannot be None.")
            endpoint_id = str(endpoint)
        else:
            endpoint_id, globus_path = get_endpoint_info_from_name(endpoint, one=one)
            root_path = root_path or globus_path
            label = label or endpoint
        if label in self.endpoints.keys() and overwrite is False:
            _logger.error(f"An endpoint called '{label}' already exists. Choose a different label "
                          f"or set overwrite=True")
        else:
            self.endpoints[label] = {'id': endpoint_id}
            if root_path:
                self.endpoints[label]['root_path'] = root_path

    @staticmethod
    def _endpoint_path(path, root_path=None):
        path = Path(path)
        if root_path and not str(path).startswith(root_path):
            path = Path(root_path).joinpath(path)
        if not path.is_absolute():
            _logger.error("If there is no root_path for this endpoint in .endpoints, 'path' must "
                          "be an absolute path.")
        return path

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
            _logger.error(f"'endpoint' must be a UUID or the label of an endpoint registered in "
                          f"this Globus instance. You can add endpoints via the add_endpoints "
                          f"method")
            return

    def ls(self, endpoint, path, remove_uuid=False, return_size=False):
        """
        Return the list of (filename, filesize) in a given endpoint directory.
        """
        # Check if endpoint is a UUID, if not try to get UUID from registered endpoints
        endpoint_id, root_path = self._endpoint_id_root(endpoint)
        # Check if root_path should be added and if path is absolute
        path = str(self._endpoint_path(path, root_path))
        # Do the actual listing
        out = []
        try:
            for entry in self.client.operation_ls(endpoint_id, path=path):
                fn = _remove_uuid_from_filename(entry['name']) if remove_uuid else entry['name']
                if return_size:
                    size = entry['size'] if entry['type'] == 'file' else None
                    out.append((fn, size))
                else:
                    out.append(fn)
        except Exception as e:
            _logger.error(str(e))
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

    def run_task(self, globus_func, retries=3, time_out=None):
        """
        Block until a Globus task finishes and retry upon Network or REST Errors.
        globus_func needs to submit a task to the client and return a task_id
        """
        try:
            task_id = globus_func()
            print(f"Waiting for Globus task {task_id} to complete")
            # While the task with task is active, print a dot every second. Timeout after timeout
            i = 0
            while not self.client.task_wait(task_id, timeout=5, polling_interval=1):
                print('.', end='')
                i += 1
                if time_out and i >= time_out:
                    task = self.client.get_task(task_id)
                    raise IOError(f"Globus task {task_id} timed out after {time_out} seconds, "
                                  f"with task status {task['status']}")
            task = self.client.get_task(task_id)
            if task['status'] == 'SUCCEEDED':
                # Sometime Globus sets the status to SUCCEEDED but doesn't truly finish.
                # Handle error thrown when querying task_successful_transfers too early
                try:
                    successful = self.client.task_successful_transfers(task_id, None)
                    skipped = self.client.task_skipped_errors(task_id, None)
                    print(f"\nGlobus task {task_id} completed."
                          f"\nSkipped transfers: {len(list(skipped))}"
                          f"\nSuccessful transfers: {len(list(successful))}")
                    for info in successful:
                        _logger.debug(f"{info['source_path']} -> {info['destination_path']}")
                except TransferAPIError:
                    _logger.warning(f"\nGlobus task {task_id} SUCCEEDED but querying transfers was"
                                    f"unsuccessful")
            else:
                raise IOError(f"Globus task finished unsucessfully with status {task['status']}")
            return task_id
        except (GlobusAPIError, NetworkError, GlobusTimeoutError, GlobusConnectionError,
                GlobusConnectionTimeoutError) as e:
            if retries < 1:
                _logger.error(f"\nRetried too many times.")
                raise e
            else:
                _logger.debug("\nGlobus experienced a network error", exc_info=True)
                # if we reach this point without returning or erroring, retry
                _logger.warning("\nGlobus experienced a network error, retrying.")
                self.run_task(globus_func, retries=(retries - 1), time_out=time_out)
