import tempfile
from pathlib import Path
import shutil
import json

import one.params


def set_up_env() -> tempfile.TemporaryDirectory:
    """
    Create a temporary directory and copy cache fixtures over
    :return: TemporaryDirectory object
    """
    fixture = Path(__file__).parent.joinpath('fixtures')
    tempdir = tempfile.TemporaryDirectory()
    # Copy cache files to temporary directory
    for cache_file in ('sessions', 'datasets'):
        filename = shutil.copy(fixture / f'{cache_file}.pqt', tempdir.name)
        assert Path(filename).exists()

    # Copy cached rest responses
    setup_rest_cache(Path(tempdir.name) / '.one')

    return tempdir


def setup_rest_cache(param_dir=None):
    """Copy REST cache fixtures to the .one parameter directory"""
    fixture = Path(__file__).parent.joinpath('fixtures')
    path_parts = ('.rest', 'test.alyx.internationalbrainlab.org', 'https')
    rest_cache_dir = Path(param_dir or one.params.get_params_dir()).joinpath(*path_parts)

    # Ensure empty
    shutil.rmtree(rest_cache_dir, ignore_errors=True)
    rest_cache_dir.mkdir(parents=True, exist_ok=True)
    # Copy over fixtures
    for file in fixture.joinpath('rest_responses').glob('*'):
        filename = shutil.copy(file, rest_cache_dir)
        assert Path(filename).exists()


def create_file_tree(one):
    """Touch all the files in the datasets table"""
    # Create dset files from cache
    for session_path, rel_path in one._cache.datasets[['session_path', 'rel_path']].values:
        filepath = Path(one._cache_dir).joinpath(session_path, rel_path)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        filepath.touch()


def setup_test_params(token=False):
    params_dir = Path(one.params.get_params_dir())
    fixture = Path(__file__).parent.joinpath('fixtures')
    test_pars = '.test.alyx.internationalbrainlab.org'
    if not params_dir.glob(test_pars):
        filename = shutil.copy(fixture / test_pars, params_dir)
        assert Path(filename).exists()

        # Add to cache map
        with open(params_dir / '.caches', 'rw') as f:
            data = json.load(f)
            data['CLIENT_MAP'][test_pars[1:]] = None
            json.dump(data, f)

    # Add token to file so db not hit
    if token:
        pars = one.params.get(client=test_pars[1:])
        if not getattr(pars, 'TOKEN', False):
            one.params.save(pars.set('TOKEN', {'token': 'T0k3N'}), test_pars[1:])
