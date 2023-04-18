"""Utilities functions for setting up test fixtures."""
import tempfile
from pathlib import Path
import shutil
import json
from uuid import uuid4

import pandas as pd
from iblutil.io.parquet import str2np
from iblutil.io.params import set_hidden

import one.params


def set_up_env() -> tempfile.TemporaryDirectory:
    """
    Create a temporary directory and copy cache fixtures over.

    Returns
    -------
    tempfile.TemporaryDirectory
        The temporary directory containing the test ONE caches

    """
    fixture = Path(__file__).parent.joinpath('fixtures')
    tempdir = tempfile.TemporaryDirectory()
    # Copy cache files to temporary directory
    for cache_file in ('sessions', 'datasets'):
        filename = shutil.copy(fixture / f'{cache_file}.pqt', tempdir.name)
        assert Path(filename).exists()

    # Copy cached rest responses
    setup_rest_cache(tempdir.name)

    return tempdir


def setup_rest_cache(cache_dir):
    """Copy REST cache fixtures to the .one parameter directory.

    Parameters
    ----------
    cache_dir : str, pathlib.Path
        The location of the ONE cache directory (e.g. ~/Downloads/ONE/alyx.example.com)

    """
    fixture = Path(__file__).parent.joinpath('fixtures')
    rest_cache_dir = Path(cache_dir).joinpath('.rest')

    # Ensure empty
    shutil.rmtree(rest_cache_dir, ignore_errors=True)
    rest_cache_dir.mkdir(parents=True, exist_ok=True)
    set_hidden(rest_cache_dir, True)
    # Copy over fixtures
    for file in fixture.joinpath('rest_responses').glob('*'):
        filename = shutil.copy(file, rest_cache_dir)
        assert Path(filename).exists()


def create_file_tree(one):
    """Touch all the files in the datasets table.

    Parameters
    ----------
    one : one.api.One
        An instance of One containing cache tables to use.

    """
    # Create dset files from cache
    for session_path, rel_path in one._cache.datasets[['session_path', 'rel_path']].values:
        filepath = Path(one.cache_dir).joinpath(session_path, rel_path)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        filepath.touch()


def setup_test_params(token=False, cache_dir=None):
    """
    Copies cache parameter fixture to .one directory.

    Parameters
    ----------
    token : bool
        If true, save a token file so that client doesn't hit auth endpoint
    cache_dir : str, pathlib.Path
        The cache_dir to save

    """
    params_dir = Path(one.params.get_params_dir())
    params_dir.mkdir(exist_ok=True)
    fixture = Path(__file__).parent.joinpath('fixtures')
    test_pars = '.test.alyx.internationalbrainlab.org'
    if not next(params_dir.glob(test_pars), None):
        filename = shutil.copy(fixture / 'params' / test_pars, params_dir / test_pars)
        assert Path(filename).exists()

        # Add to cache map
        map_file = params_dir / '.caches'
        if not map_file.exists():
            shutil.copy(fixture / 'params' / '.caches', map_file)
            assert Path(filename).exists()
        with open(map_file, 'r+') as f:
            data = json.load(f)
            data['CLIENT_MAP'][test_pars[1:]] = str(cache_dir or '')
            f.seek(0)
            json.dump(data, f)
            f.truncate()

    # Add token to file so db not hit
    if token:
        pars = one.params.get(client=test_pars[1:])
        if not getattr(pars, 'TOKEN', False):
            one.params.save(pars.set('TOKEN', {'token': 'T0k3N'}), test_pars[1:])


def revisions_datasets_table(collections=('', 'alf/probe00', 'alf/probe01'),
                             revisions=('', '2020-01-08', '2021-07-06'),
                             object='spikes',
                             attributes=('times', 'waveforems')):
    """Returns a datasets cache DataFrame containing datasets with revision folders.

    As there are no revised datasets on the test databases, this function acts as a fixture for
    testing the filtering of datasets by a revision.

    Parameters
    ----------
    collections : tuple
        A list of collections
    revisions : tuple
        A list of revisions
    object : str
        An ALF object
    attributes : tuple
        A list of ALF attributes

    Returns
    -------
    pd.DataFrame
        A datasets cache table containing datasets made from the input names

    """
    rel_path = []
    for attr in attributes:
        for collec in collections:
            for rev in (f'#{x}#' if x else '' for x in revisions):
                rel_path.append('/'.join(x for x in (collec, rev, f'{object}.{attr}.npy') if x))
    d = {
        'rel_path': rel_path,
        'session_path': 'subject/1900-01-01/001',
        'file_size': None,
        'hash': None,
        'exists': True,
        'eid': str(uuid4()),
        'id': map(str, (uuid4() for _ in rel_path))
    }

    return pd.DataFrame(data=d).set_index(['eid', 'id'])


def create_schema_cache(param_dir=None):
    """Save REST cache file for docs/ endpoint.

    Ensures the database isn't hit when the rest_schemas property is accessed.

    Parameters
    ----------
    param_dir : str, pathlib.Path
        The location of the parameter directory.  If None, the default one is used.

    """
    actions = dict.fromkeys(['list', 'read', 'create', 'update', 'partial_update', 'delete'])
    endpoints = ['cache', 'dataset-types', 'datasets', 'downloads', 'insertions', 'sessions']
    path_parts = ('.rest', 'test.alyx.internationalbrainlab.org', 'https')
    rest_cache_dir = Path(param_dir or one.params.get_params_dir()).joinpath(*path_parts)
    with open(rest_cache_dir / '1baff95c2d0e31059720a3716ad5b5a34b61a207', 'r') as f:
        json.dump({k: actions for k in endpoints}, f)


def get_file(root: str, str_id: str) -> str:
    """
    A stub function for iblutil.io.params.getfile.  Allows the injection of a different param dir.

    Parameters
    ----------
    root : str, pathlib.Path
        The root directory of the new parameters
    str_id : str
        The parameter string identifier

    Returns
    -------
    str
        The parameter file path

    """
    parts = ['.' + p if not p.startswith('.') else p for p in Path(str_id).parts]
    pfile = Path(root, *parts).as_posix()
    return pfile


def caches_str2int(caches):
    """Convert str ids to int ids for cache tables.

    Parameters
    ----------
    caches : Bunch
        A bunch of cache tables (from One._cache)

    """
    for table in ('sessions', 'datasets'):
        index = caches[table].index
        names = (index.name,) if index.name else index.names
        cache = caches[table].reset_index()
        for name in names:
            for i, col in enumerate(str2np(cache.pop(name).values).T):
                cache[f'{name}_{i}'] = col
        int_cols = cache.filter(regex=r'_\d{1}$').columns.sort_values().tolist()
        caches[table] = cache.set_index(int_cols)
