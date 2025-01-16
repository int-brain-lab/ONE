"""Download methods common to all remote file download clients.

All supported access protocols are defined in ALYX_JSON.

Remote parameters:
    - All parameters are stored in the .remote JSON file.
    - The base keys are access protocols (e.g. 'globus').
    - Each contains a map of ID to parameters (e.g. keys such as 'default', 'admin').
    - load_client_params and save_client_params are used to read/write these params.

Includes the DownloadClient superclass.

TODO Currently record2url assumes all data are stored on a single HTTP data server. Data repos
 are not stored in the cache tables so should by default use HTTP data server but could get
 address by precedence.  NB: OneAlyx._download_dataset uses record2url then calls
 AlyxClient.download_file.
TODO Could have .one/.params file that stores ONE state, including whether files are distributed?
"""
from abc import abstractmethod
import logging

from iblutil.io import params as iopar

from one.params import _PAR_ID_STR

"""tuple: Default order of precedence for download protocol"""
PROC_PRECEDENCE = ('aws', 'http', 'globus', 'kachary')
ALYX_JSON = {
    'access_protocol': {
        ('aws', 'http', 'kachary', 'globus')
    }
}
"""str: Location of the remote download client parameters"""
PAR_ID_STR = f'{_PAR_ID_STR}/remote'
_logger = logging.getLogger(__name__)


def load_client_params(client_key=None, assert_present=True):
    """Load the parameters from the remote params file.

    If a client key is provided, only those client parameters are returned.
    NB: Remote param values are expected to all be dicts.

    Parameters
    ----------
    client_key : str
        An optional, specific client whose parameters to return.
    assert_present : bool
        If True, raise error if client parameters not found.

    Returns
    -------
    IBLParams, None
        Download client parameters or None if assert_present is False and no parameters found.

    Raises
    ------
    FileNotFoundError
        No one/remote JSON file found.
    AttributeError
        Provided client key not present in one/remote params.

    Examples
    --------
    Load all remote parameters

    >>> pars = load_client_params()

    Load all glogus parameters or return None if non-existent

    >>> pars = load_client_params('globus', assert_present=False)

    Load parameters for a specific globus profile

    >>> pars = load_client_params('globus.admin')

    """
    try:
        p = iopar.read(PAR_ID_STR)
        if not client_key:
            return p
        for k in client_key.split('.'):
            p = iopar.from_dict(getattr(p, k))
        return p
    except (FileNotFoundError, AttributeError) as ex:
        if assert_present:
            raise ex


def save_client_params(new_pars, client_key=None):
    """Save parameters into the remote params file.

    If a client key is provided, parameters are saved into this field.

    Parameters
    ----------
    new_pars : dict, IBLParams
        A set or subset or parameters to save.
    client_key : str
        An optional, specific client whose parameters to save.

    Raises
    ------
    ValueError
        If client_key is None, all parameter fields must hold dicts.

    """
    if not client_key:
        if not all(isinstance(x, dict) for x in iopar.as_dict(new_pars).values()):
            raise ValueError('Not all parameter fields contain dicts')
        return iopar.write(PAR_ID_STR, new_pars)  # Save all parameters
    # Save parameters into client key field
    pars = iopar.as_dict(iopar.read(PAR_ID_STR, {}) or {})
    pars[client_key] = iopar.as_dict(new_pars)
    iopar.write(PAR_ID_STR, pars)


class DownloadClient:
    """Data download handler base class."""

    def __init__(self):
        pass

    @abstractmethod
    def to_address(self, data_path, *args, **kwargs):
        """Returns the remote data URL for a given ALF path."""
        pass  # pragma: no cover

    @abstractmethod
    def download_file(self, file_address, *args, **kwargs):
        """Download an ALF dataset given its address."""
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def setup(*args, **kwargs):
        pass  # pragma: no cover

    @staticmethod
    def repo_from_alyx(name, alyx):
        """Return the data repository information for a given data repository."""
        return alyx.rest('data-repository', 'read', id=name)
