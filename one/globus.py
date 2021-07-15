import logging
import globus_sdk
from pathlib import Path, PurePosixPath
from one.webclient import _path_from_dataset
from iblutil.io import params as iopar
from one.api import ONE

_logger = logging.getLogger(__name__)
_PAR_ID_STR = 'one/globus'
DEFAULT_PAR = {'CLIENT_ID': None, 'LOCAL_ID': None, 'LOCAL_PATH': None}


def setup():
    """
    Sets up Globus as a backend for ONE functions.

    In order to use this function you need:
    1. Have the Client ID of an existing Globus Client, or create one
       (https://globus-sdk-python.readthedocs.io/en/stable/tutorial.html)
    2. Set up Global Connect on your local device (https://www.globus.org/globus-connect-personal)
    3. Register your local device as an Enpoint in your Globus Client (https://app.globus.org/)
    """

    print("Setting up Globus. See docstring for help.")
    par_id = input("Enter name for this client or press Enter to keep value 'default': ")
    par_id = _PAR_ID_STR + '/default' if par_id == '' else _PAR_ID_STR + f'/{par_id.strip(".")}'

    # Read existing globus params if present
    pars = iopar.read(par_id, DEFAULT_PAR)

    # Set CLIENT_ID
    current_id = getattr(pars, 'CLIENT_ID')
    if current_id:
        new_id = input(f'Please enter Client ID or press Enter to use the current '
                       f'value ({current_id}): ').strip()
        new_id = current_id if not new_id else new_id
    else:
        new_id = input(f'Please enter the Client ID: ').strip()
    pars = pars.set('CLIENT_ID', new_id)

    # Find and set local path and ID
    local_id, local_path = get_local_endpoint()

    for found, attr, descr in [(local_id, 'LOCAL_ID', 'Local endpoint ID'),
                               (local_path, 'LOCAL_PATH', 'Local path to be accessed by Globus')]:
        current = getattr(pars, attr)
        if not current or current == found:
            new = input(f'{descr} set to {found}. Press Enter to keep this value, or enter new '
                        f'value here: ')
            new = new if new else found
        else:
            new = input(f'{descr} set to {current}, but found {found} in Globus settings. '
                        f'Please enter the value you want to use: ').strip()
        pars = pars.set(attr, new)

    # Log in manually and get refresh token to avoid having to login repeatedly
    client = globus_sdk.NativeAppAuthClient(pars.CLIENT_ID)
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

    :param client_name: str, defines the parameter file to use (~/.one/.globus/.client_name)
    :return: Globus transfer client
    """
    par_id = _PAR_ID_STR + f'/{client_name}'
    try:
        globus_pars = iopar.read(par_id)
    except FileNotFoundError as err:
        _logger.error(f"{str(err)}. Choose a different client name or run one.globus.setup() "
                      f"first\n")
        return
    required_fields = {'refresh_token', 'CLIENT_ID'}
    if not (globus_pars and required_fields.issubset(globus_pars.as_dict())):
        raise ValueError("No token in client parameter file. Run one.globus.setup first")
    client = globus_sdk.NativeAppAuthClient(globus_pars.CLIENT_ID)
    client.oauth2_start_flow(refresh_tokens=True)
    authorizer = globus_sdk.RefreshTokenAuthorizer(globus_pars.refresh_token, client)
    return globus_sdk.TransferClient(authorizer=authorizer)


def get_local_endpoint():
    """
    Extracts the ID of the local Globus Connect endpoint and the local path accessible by Globus.

    :return: tuple of str, (local_id, local_path)
    """
    msg = ("Cannot find local endpoint {}, check if Globus Connect is set up correctly, "
           "{} exists and contains {}.")
    globus_path = Path.home().joinpath(".globusonline", "lta")
    id_file = globus_path.joinpath('client-id.txt')
    assert (id_file.exists()), msg.format('ID', id_file, 'the enpoint ID')
    local_id = id_file.read_text().strip()
    print(f"Found local endpoint ID {local_id}")
    path_file = globus_path.joinpath('config-paths')
    assert (path_file.exists()), msg.format('path', path_file, 'the path accessible by Globus')
    local_path = path_file.read_text().strip().split(',')[0]
    print(f"Found local endpoint path {local_path}")
    return local_id, local_path


def get_endpoint_id_from_name(endpoint=None, one=None):
    one = one or ONE()
    repos = one.alyx.rest('data-repository', 'list')
    repo_names = [r['name'] for r in repos]
    if not endpoint or endpoint not in repo_names:
        _logger.error("Choose one of the following endpoints from the current Alyx database or "
                      "pass an ONE instance with a different base_url:")
        _ = [print(n) for n in repo_names]
    else:
        endpoint_id = [r['globus_endpoint_id'] for r in repos if r['name'] == endpoint]
        return endpoint_id[0]



class Globus:
    """
    Wrapper for managing files on Globus endpoints.
    From: https://github.com/int-brain-lab/ibllib/blob/ca8cd93ccac665efda6943f14047ba53503e8bb8/ibllib/io/globus.py
    """
    def __init__(self):
        self._tc = create_globus_client()

    def ls(self, endpoint, path='', remove_uuid=False):
        """Return the list of (filename, filesize) in a given endpoint directory.
        NOTE: this function automatically removes the UUIDs from the filenames.
        """
        endpoint, root = ENDPOINTS.get(endpoint, (endpoint, ''))
        assert root
        path = _remote_path(root, path)
        out = []
        try:
            for entry in self._tc.operation_ls(endpoint, path=path):
                fn = entry['name']
                if remove_uuid:
                    fn = _remove_uuid_from_filename(fn)
                size = entry['size'] if entry['type'] == 'file' else None
                out.append((fn, size))
        except Exception as e:
            logger.error(str(e))
        return out

    def file_exists(self, endpoint, path, size=None, remove_uuid=False):
        """Return whether a given file exists on a given endpoint, optionally with a specified
        file size."""
        parent, filename = _split_file_path(path)
        existing = self.ls(endpoint, parent, remove_uuid=remove_uuid)
        return _filename_size_matches((filename, size), existing)

    def dir_contains_files(self, endpoint, dir_path, filenames, remove_uuid=False):
        """Return whether a directory contains a list of filenames. Returns a list of boolean,
        one for each input file."""
        files = self.ls(endpoint, dir_path, remove_uuid=remove_uuid)
        existing = [fn for fn, size in files]
        out = []
        for filename in filenames:
            out.append(filename in existing)
        return out

    def files_exist(self, endpoint, paths, sizes=None, remove_uuid=False):
        """Return whether a list of files exist on an endpoint, optionally with specified
        file sizes."""
        if not paths:
            return []
        parents = sorted(set(_split_file_path(path)[0] for path in paths))
        existing = []
        for parent in parents:
            filenames_sizes = self.ls(endpoint, parent, remove_uuid=remove_uuid)
            existing.extend([(parent + '/' + fn, size) for fn, size in filenames_sizes])
        if sizes is None:
            sizes = [None] * len(paths)
        return [_filename_size_matches((path, size), existing) for (path, size) in zip(paths, sizes)]

    def blocking(f):
        """Add two keyword arguments to a method, blocking (boolean) and timeout."""
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            blocking = kwargs.pop('blocking', None)
            timeout = kwargs.pop('timeout', None)
            task_id = f(self, *args, **kwargs)
            if blocking:
                return self.wait_task(task_id, timeout=timeout)
            else:
                return task_id
        return wrapped

    @blocking
    def rm(self, endpoint, path, blocking=False):
        """Delete a single file on an endpoint."""
        endpoint, root = ENDPOINTS.get(endpoint, (endpoint, ''))
        assert root
        path = _remote_path(root, path)

        ddata = globus.DeleteData(self._tc, endpoint, recursive=False)
        ddata.add_item(path)
        delete_result = self._tc.submit_delete(ddata)
        task_id = delete_result["task_id"]
        message = delete_result["message"]
        return task_id

    @blocking
    def move_files(
        self, source_endpoint, target_endpoint,
        source_paths, target_paths):
        """Move files from one endpoint to another."""
        source_endpoint, source_root = ENDPOINTS.get(source_endpoint, (source_endpoint, ''))
        target_endpoint, target_root = ENDPOINTS.get(target_endpoint, (target_endpoint, ''))

        source_paths = [_remote_path(source_root, str(_)) for _ in source_paths]
        target_paths = [_remote_path(target_root, str(_)) for _ in target_paths]

        tdata = globus.TransferData(
            self._tc, source_endpoint, target_endpoint, verify_checksum=True, sync_level='checksum',
        )
        for source_path, target_path in zip(source_paths, target_paths):
            tdata.add_item(source_path, target_path)
        response = self._tc.submit_transfer(tdata)
        task_id = response.get('task_id', None)
        message = response.get('message', None)
        return task_id

    def add_text_file(self, endpoint, path, contents):
        """Create a text file on a remote endpoint."""
        local = ENDPOINTS.get('local', None)
        if not local or not local[0]:
            raise IOError(
            "Can only add a text file on a remote endpoint "
            "if the current computer is a Globus endpoint")
        local_endpoint, local_root = local
        assert local_endpoint
        assert local_root
        local_root = Path(local_root.replace('/~', ''))
        assert local_root.exists()
        fn = '_tmp_text_file.txt'
        local_path = local_root / fn
        local_path.write_text(contents)
        task_id = self.move_files(local_endpoint, endpoint, [local_path], [path], blocking=True)
        os.remove(local_path)

    def wait_task(self, task_id, timeout=None):
        """Block until a Globus task finishes."""
        if timeout is None:
            timeout = 300
        i = 0
        while not self._tc.task_wait(task_id, timeout=1, polling_interval=1):
            print('.', end='')
            i += 1
            if i >= timeout:
                raise IOError(
                    "The task %s has not finished after %d seconds." % (task_id, timeout))



# TODO put this into the globus module
token = params.read('globus/default')
authorizer = globus_sdk.AccessTokenAuthorizer(token.access_token)
gtc = globus_sdk.TransferClient(authorizer=authorizer)
gt = globus_sdk.TransferData(gtc, FLAT_IRON_ENDPOINT, LOCAL_ENDPOINT, verify_checksum=True,
                             sync_level='checksum', label=f'one globus {one._par.ALYX_LOGIN}')

for dset in dsets:
    sdsc_gpath = wc.sdsc_globus_path_from_dataset(dset)
    destination_gpath = wc.one_path_from_dataset(dset, one_cache=one._par.CACHE_DIR)
    print(sdsc_gpath, destination_gpath)
    gt.add_item(sdsc_gpath, destination_gpath)

# TODO add the wait for transfer option in globus from the GLobus patcher
gtc.submit_transfer(data=gt)



def get_local_path(local_globus_path):
    if os.environ.get('WSL_DISTRO_NAME', None):
        local_path = str(Path('/mnt/c/').joinpath(*local_globus_path.split('/')[2:])) + '/'
    else:
        local_path = local_globus_path
    return local_path

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