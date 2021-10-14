"""Functions for modifying, loading and saving ONE and Alyx database parameters.

Scenarios:

    - Load ONE with a cache dir: tries to load the Web client params from the dir
    - Load ONE with http address - gets cache dir from the URL map

The ONE params comprise two files: a caches file that contains a map of Alyx db URLs to cache
directories, and a separate parameter file for each url containing the client parameters.  The
caches file also sets the default client for when no url is provided.
"""
import re
from iblutil.io import params as iopar
from getpass import getpass
from pathlib import Path
from urllib.parse import urlsplit
import unicodedata

_PAR_ID_STR = 'one'
_CLIENT_ID_STR = 'caches'
CACHE_DIR_DEFAULT = str(Path.home() / "Downloads" / "ONE")
"""str: The default data download location"""


def default():
    """Default Web client parameters"""
    par = {"ALYX_URL": "https://openalyx.internationalbrainlab.org",
           "ALYX_LOGIN": "intbrainlab",
           "HTTP_DATA_SERVER": "https://ibl.flatironinstitute.org",
           "HTTP_DATA_SERVER_LOGIN": None,
           "HTTP_DATA_SERVER_PWD": None}
    return iopar.from_dict(par)


def _get_current_par(k, par_current):
    """
    Return the current parameter value or the default.

    Parameters
    ----------
    k : str
        The parameter key lookup
    par_current : IBLParams
        The current parameter set

    Returns
    -------
    any
        The current parameter value or default if None or not set
    """
    cpar = getattr(par_current, k, None)
    if cpar is None:
        cpar = getattr(default(), k, None)
    return cpar


def _key_from_url(url: str) -> str:
    """
    Convert a URL str to one valid for use as a file name or dict key.  URL Protocols are
    removed entirely.  The returned string will have characters in the set [a-zA-Z.-_].

    Parameters
    ----------
    url : str
        A URL string

    Returns
    -------
    str
        A filename-safe string

    Example
    -------
    >>> url = _key_from_url('http://test.alyx.internationalbrainlab.org/')
   'test.alyx.internationalbrainlab.org'
    """
    url = unicodedata.normalize('NFKC', url)  # Ensure ASCII
    url = re.sub('^https?://', '', url).strip('/')  # Remove protocol and trialing slashes
    url = re.sub(r'[^.\w\s-]', '_', url.lower())  # Convert non word chars to underscore
    return re.sub(r'[-\s]+', '-', url)  # Convert spaces to hyphens


def setup(client=None, silent=False, make_default=None):
    """
    Set up ONE parameters.  If a client (i.e. Alyx database URL) is provided, settings for
    that instance will be set.  If silent, the user will be prompted to input each parameter
    value.  Pressing return will use either current parameter or the default.

    Parameters
    ----------
    client : str
        An Alyx database URL. If None, the user will be prompted to input one.
    silent : bool
        If True, user is not prompted for any input.
    make_default : bool
        If True, client is set as the default and will be returned when calling `get` with no
        arguments.

    Returns
    -------
    IBLParams
        An updated cache map.
    """
    # First get default parameters
    par_default = default()
    client_key = _key_from_url(client or par_default.ALYX_URL)

    # If a client URL has been provided, set it as the default URL
    par_default = par_default.set('ALYX_URL', client or par_default.ALYX_URL)
    par_current = iopar.read(f'{_PAR_ID_STR}/{client_key}', par_default)

    # Load the db URL map
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {'CLIENT_MAP': dict()})
    cache_dir = cache_map.CLIENT_MAP.get(client_key, Path(CACHE_DIR_DEFAULT, client_key))

    if not silent:
        par = iopar.as_dict(par_default)
        for k in par.keys():
            cpar = _get_current_par(k, par_current)
            # Prompt for database URL; skip if client url already provided
            if k == 'ALYX_URL':
                if not client:
                    par[k] = input(f'Param {k}, current value is ["{str(cpar)}"]:') or cpar
                    if '://' not in par[k]:
                        par[k] = 'https://' + par[k]
                    url_parsed = urlsplit(par[k])
                    if not (url_parsed.netloc and re.match('https?', url_parsed.scheme)):
                        raise ValueError(f'{k} must be valid HTTP URL')
                    client = par[k]
            # Iterate through other non-password pars
            elif 'PWD' not in k:
                par[k] = input(f'Param {k}, current value is ["{str(cpar)}"]:') or cpar

        cpar = _get_current_par('HTTP_DATA_SERVER_PWD', par_current)
        prompt = f'Enter the FlatIron HTTP password for {par["HTTP_DATA_SERVER_LOGIN"]} '\
                 '(leave empty to keep current): '
        par['HTTP_DATA_SERVER_PWD'] = getpass(prompt) or cpar

        if 'ALYX_PWD' in par_current.as_dict():
            # Only store plain text password if user manually added it to params JSON file
            cpar = _get_current_par('ALYX_PWD', par_current)
            prompt = (f'Enter the Alyx password for {par["ALYX_LOGIN"]} '
                      '(leave empty to keep current):')
            par['ALYX_PWD'] = getpass(prompt) or cpar

        par = iopar.from_dict(par)

        # Prompt for cache directory
        client_key = _key_from_url(par.ALYX_URL)
        cache_dir = Path(CACHE_DIR_DEFAULT, client_key)
        prompt = f'Enter the location of the download cache, current value is ["{cache_dir}"]:'
        cache_dir = input(prompt) or cache_dir

        # Check if directory already used by another instance
        in_use = [v for k, v in cache_map.CLIENT_MAP.items() if k != client_key]
        while str(cache_dir) in in_use:
            answer = input(
                'Warning: the directory provided is already a cache for another URL.  '
                'This may cause conflicts.  Would you like to change the cache location? [Y/n]')
            if answer and answer[0].lower() == 'n':
                break
            cache_dir = input(prompt) or cache_dir  # Prompt for another directory

        if make_default is None and 'DEFAULT' not in cache_map.as_dict():
            answer = input('Would you like to set this URL as the default one? [Y/n]')
            make_default = True if not answer or answer[0].lower() == 'y' else False
    else:
        par = par_current

    # Update and save parameters
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    cache_map.CLIENT_MAP[client_key] = str(cache_dir)
    if make_default or 'DEFAULT' not in cache_map.as_dict():
        cache_map = cache_map.set('DEFAULT', client_key)

    iopar.write(f'{_PAR_ID_STR}/{client_key}', par)  # Client params
    iopar.write(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', cache_map)

    if not silent:
        print('ONE Parameter files location: ' + iopar.getfile(_PAR_ID_STR))
    return cache_map


def get(client=None, silent=False):
    """Returns the AlyxClient parameters

    Parameters
    ----------
    silent : bool
        If true, defaults are chosen if no parameters found
    client : str
        The database URL to retrieve parameters for.  If None, the default is loaded

    Returns
    -------
    IBLParams
        A Params object for the AlyxClient
    """
    client_key = _key_from_url(client) if client else None
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    if not cache_map:  # This can be removed in the future
        cache_map = _patch_params()
    # If there are no params for this client, run setup routine
    if not cache_map or (client_key and client_key not in cache_map.CLIENT_MAP):
        cache_map = setup(client=client, silent=silent)
    cache = cache_map.CLIENT_MAP[client_key or cache_map.DEFAULT]
    return iopar.read(f'{_PAR_ID_STR}/{client_key or cache_map.DEFAULT}').set('CACHE_DIR', cache)


def get_default_client(include_schema=True) -> str:
    """Returns the default AlyxClient URL, or None if no default is set

    Parameters
    ----------
    include_schema : bool
        When True, the URL schema is included (i.e. http(s)://).  Set to False to return the URL
        as a client key.

    Returns
    -------
    str
        The default database URL with or without the schema, or None if no default is set
    """
    cache_map = iopar.as_dict(iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})) or {}
    client_key = cache_map.get('DEFAULT', None)
    if not client_key or include_schema is False:
        return client_key
    return get(client_key).ALYX_URL


def save(par, client):
    """
    Save a set of parameters for a given client.

    Parameters
    ----------
    par : dict, IBLParams
        A set of Web client parameters to save
    client : str
        The Alyx URL that corresponds to these parameters
    """
    # Remove cache dir variable before saving
    par = {k: v for k, v in iopar.as_dict(par).items() if 'CACHE_DIR' not in k}
    iopar.write(f'{_PAR_ID_STR}/{_key_from_url(client)}', par)


def get_cache_dir(client=None) -> Path:
    """Return the download directory for a given client.

    If no client is set up, the default download location is returned.

    Parameters
    ----------
    client : str
        The client to return cache dir from.  If None, the default client is used.

    Returns
    -------
    pathlib.Path
        The download cache path
    """
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    client = _key_from_url(client) if client else cache_map.DEFAULT
    cache_dir = Path(cache_map.CLIENT_MAP[client] if cache_map else CACHE_DIR_DEFAULT)
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def get_params_dir() -> Path:
    """Return the path to the root ONE parameters directory

    Returns
    -------
    pathlib.Path
        The root ONE parameters directory
    """
    return Path(iopar.getfile(_PAR_ID_STR))


def get_rest_dir(client=None) -> Path:
    """Return path to REST cache directory

    Parameters
    ----------
    client : str
        Location of rest cache for a given database URL.  If None, the root REST cache directory is
        returned

    Returns
    -------
    pathlib.Path
        The REST cache directory path
    """
    rest_dir = get_params_dir() / '.rest'
    if client:
        scheme, loc, *_ = urlsplit(client)
        rest_dir /= Path(loc.replace(':', '_'), scheme)
    return rest_dir


def check_cache_conflict(cache_dir):
    """Asserts that a given directory is not currently used as a cache directory.
    This function checks whether a given directory is used as a cache directory for an Alyx
    Web client.  This function is called by the ONE factory to determine whether to return an
    OneAlyx object or not.  It is also used when setting up params for a new client.

    Parameters
    ----------
    cache_dir : str, pathlib.Path
        A directory to check.

    Raises
    ------
    AssertionError
        The directory is set as a cache for a Web client
    """
    cache_map = getattr(iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {}), 'CLIENT_MAP', None)
    if cache_map:
        assert not any(x == str(cache_dir) for x in cache_map.values())


def _patch_params():
    """
    Copy over old parameters to the new cache dir based format

    Returns
    -------
    IBLParams
        New parameters object containing the previous parameters
    """
    OLD_PAR_STR = 'one_params'
    old_par = iopar.read(OLD_PAR_STR, {})
    par = None
    if getattr(old_par, 'HTTP_DATA_SERVER_PWD', None):
        # Copy pars to new location
        assert old_par.CACHE_DIR
        cache_dir = Path(old_par.CACHE_DIR)
        cache_dir.mkdir(exist_ok=True)

        # Save web client parameters
        new_web_client_pars = {k: v for k, v in old_par.as_dict().items()
                               if k in default().as_dict() or k == 'ALYX_LOGIN'}
        cache_name = _key_from_url(old_par.ALYX_URL)
        iopar.write(f'{_PAR_ID_STR}/{cache_name}', new_web_client_pars)

        # Add to cache map
        cache_map = {
            'CLIENT_MAP': {
                cache_name: old_par.CACHE_DIR
            },
            'DEFAULT': cache_name
        }
        iopar.write(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', cache_map)
        par = iopar.from_dict(cache_map)

    # Remove the old parameters file
    old_path = Path(iopar.getfile(OLD_PAR_STR))
    if old_path.exists():
        old_path.unlink()

    return par
