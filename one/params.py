"""
Scenarios:
    - Load ONE with a cache dir: tries to load the Web client params from the dir
    - Load ONE with http address - gets cache dir from the address?

The ONE params comprise two files: a caches file that contains a map of Alyx db URLs to cache
directories, and a separate parameter file for each url containing the client parameters.  The
caches file also sets the default client for when no url is provided.
"""
import re
from one.lib.io import params as iopar
from getpass import getpass
from pathlib import Path
import unicodedata

_PAR_ID_STR = 'one'
_CLIENT_ID_STR = 'caches'
CACHE_DIR_DEFAULT = str(Path.home() / "Downloads" / "ONE")


def default():
    """Default WebClient parameters"""
    par = {"ALYX_LOGIN": "intbrainlab",
           "ALYX_PWD": "international",
           "ALYX_URL": "https://openalyx.internationalbrainlab.org",
           "HTTP_DATA_SERVER": "https://ibl.flatironinstitute.org",
           "HTTP_DATA_SERVER_LOGIN": "iblmember",
           "HTTP_DATA_SERVER_PWD": None,
           }
    return iopar.from_dict(par)


def _get_current_par(k, par_current):
    cpar = getattr(par_current, k, None)
    if cpar is None:
        cpar = getattr(default(), k, None)
    return cpar


def _key_from_url(url: str) -> str:
    """
    Convert a URL str to one valid for use as a file name or dict key.  URL Protocols are
    removed entirely.  The returned string will have characters in the set [a-zA-Z.-_].

    Example:
        url = _key_from_url('http://test.alyx.internationalbrainlab.org/')
        assert url == 'test.alyx.internationalbrainlab.org'

    :param url: A URL string
    :return: A file-name-safe string
    """
    url = unicodedata.normalize('NFKC', url)  # Ensure ASCII
    url = re.sub('^https?://', '', url).strip('/')  # Remove protocol and trialing slashes
    url = re.sub(r'[^.\w\s-]', '_', url.lower())  # Convert non word chars to underscore
    return re.sub(r'[-\s]+', '-', url)  # Convert spaces to hyphens


def setup(client=None, silent=False, make_default=None):
    # first get default parameters
    par_default = default()
    client_key = _key_from_url(client or par_default.ALYX_URL)

    # If a client URL has been provided, set it as the default URL
    par_default.set('ALYX_URL', client or par_default.ALYX_URL)
    par_current = iopar.read(f'{_PAR_ID_STR}/{client_key}', par_default)
    # Load the db URL map
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {'CLIENT_MAP': dict()})
    cache_dir = cache_map.CLIENT_MAP.get(client_key, Path(CACHE_DIR_DEFAULT, client_key))

    if not silent:
        par = iopar.as_dict(par_default)
        for k in par.keys():
            cpar = _get_current_par(k, par_current)
            # Iterate through non-password pars; skip url if client url already provided
            if 'PWD' not in k and not (client and k == 'ALYX_URL'):
                par[k] = input(f'Param {k}, current value is ["{str(cpar)}"]:') or cpar

        cpar = _get_current_par('ALYX_PWD', par_current)
        prompt = f'Enter the Alyx password for {par["ALYX_LOGIN"]} (leave empty to keep current):'
        par['ALYX_PWD'] = getpass(prompt) or cpar

        cpar = _get_current_par('HTTP_DATA_SERVER_PWD', par_current)
        prompt = f'Enter the FlatIron HTTP password for {par["HTTP_DATA_SERVER_LOGIN"]} '\
                 '(leave empty to keep current): '
        par['HTTP_DATA_SERVER_PWD'] = getpass(prompt) or cpar

        par = iopar.from_dict(par)

        # Prompt for cache directory
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
            answer = input('Would you like to set this URL as the default one? [y/N]')
            make_default = True if answer and answer[0].lower() == 'y' else False
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


def get(silent=False, client=None):
    client_key = _key_from_url(client) if client else None
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    if not cache_map:  # This can be removed in the future
        cache_map = _patch_params()
    # If there are no
    if not cache_map or (client_key and client_key not in cache_map.CLIENT_MAP):
        cache_map = setup(client=client, silent=silent)
    cache = cache_map.CLIENT_MAP[client_key or cache_map.DEFAULT]
    return iopar.read(f'{_PAR_ID_STR}/{client_key or cache_map.DEFAULT}').set('CACHE_DIR', cache)


def save(par, client):
    iopar.write(f'{_PAR_ID_STR}/{_key_from_url(client)}', par)


def get_cache_dir() -> Path:
    cache_map = iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {})
    cache_dir = Path(cache_map.CLIENT_MAP[cache_map.DEFAULT] if cache_map else CACHE_DIR_DEFAULT)
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def get_params_dir() -> Path:
    return iopar.getfile(_PAR_ID_STR)


def _check_cache_conflict(cache_dir):
    cache_map = getattr(iopar.read(f'{_PAR_ID_STR}/{_CLIENT_ID_STR}', {}), 'CLIENT_MAP', None)
    if cache_map:
        assert not any(x == str(cache_dir) for x in cache_map.values())


def _patch_params():
    """
    Copy over old parameters to the new cache dir based format
    :return: new parameters
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
                               if k in default().as_dict()}
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
    # TODO Restore when fully deprecated
    # old_path = Path(iopar.getfile(OLD_PAR_STR))
    # old_path.unlink(missing_ok=True)

    return par
