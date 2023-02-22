import os
import sys
import shutil
import argparse
from pathlib import Path
import logging
from scripts.execute_notebooks import process_notebooks

_logger = logging.getLogger('ONE')
root = Path.cwd()
scripts_path = root.joinpath('scripts')

nb_path = root.joinpath('notebooks')
nb_path_external = []


def make_documentation(execute, force, documentation, clean, specific):

    # Clean up any nblink files
    nb_external_files = root.joinpath('notebooks_external').glob('*')
    for file in nb_external_files:
        os.remove(file)

    status = 0
    # Case where we want to rebuild all examples
    if execute and not specific:
        # Execute notebooks in docs folder
        if nb_path:
            status += process_notebooks(nb_path, execute=True, force=force)
        # Execute notebooks in external folders
        for nb_path_ext in nb_path_external:
            status += process_notebooks(nb_path_ext, execute=True, force=force,
                                        link=True, filename_pattern='docs')
        _logger.info('Finished processing notebooks')

        if status != 0:
            # One or more examples returned an error
            sys.exit(1)
        else:
            # If no errors make the documentation
            _logger.info('Cleaning up previous documentation')
            os.system('make clean')
            _logger.info('Making documentation')
            os.system('make html')
            sys.exit(0)

    # Case where we only want to build specific examples
    if execute and specific:
        for nb in specific:
            if str(nb).startswith(str(root)):
                status += process_notebooks(nb, execute=True, force=force)
            else:
                status += process_notebooks(nb, execute=True, force=force, link=True)
            _logger.info('Finished processing notebooks')

        # Create the link files for the other notebooks in external paths that we haven't
        # executed. N.B this must be run after the above commands
        for nb_path_ext in nb_path_external:
            process_notebooks(nb_path_ext, execute=False, link=True, filename_pattern='docs')

        if status != 0:
            # One or more examples returned an error
            sys.exit(1)
        else:
            # If no errors make the documentation
            _logger.info('Cleaning up previous documentation')
            os.system('make clean')
            _logger.info('Making documentation')
            os.system('make html')
            sys.exit(0)

    if documentation:
        for nb_path_ext in nb_path_external:
            process_notebooks(nb_path_ext, execute=False, link=True, filename_pattern='docs')

        _logger.info('Cleaning up previous documentation')
        os.system('make clean')
        _logger.info('Making documentation')
        os.system('make html')
        sys.exit(0)

    # Clean up notebooks in directory if also specified
    if clean:
        _logger.info('Cleaning up notebooks')
        process_notebooks(nb_path, execute=False, cleanup=True)
        for nb_path_ext in nb_path_external:
            process_notebooks(nb_path_ext, execute=False, cleanup=True,
                              filename_pattern='docs')

        try:
            build_path = root.joinpath('_build')
            if build_path.exists():
                shutil.rmtree(build_path)
        except Exception as err:
            print(err)
            _logger.error('Could not remove _build directory in iblenv/docs_gh_pages, please '
                          'delete manually')
        try:
            autosummary_path = root.joinpath('_autosummary')
            if autosummary_path.exists():
                shutil.rmtree(autosummary_path)
        except Exception as err:
            print(err)
            _logger.error('Could not remove _autosummary directory in iblenv/docs_gh_pages, please'
                          ' delete manually')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make IBL documentation')

    parser.add_argument('-e', '--execute', default=False, action='store_true',
                        help='Execute notebooks')
    parser.add_argument('-f', '--force', default=False, action='store_true',
                        help='Force notebook execution even if already run')
    parser.add_argument('-d', '--documentation', default=False, action='store_true',
                        help='Make documentation')
    parser.add_argument('-s', '--specific', nargs='+', required=False,
                        help='List of specific files to execute')
    parser.add_argument('-c', '--cleanup', default=False, action='store_true',
                        help='Cleanup notebooks once documentation made')
    args = parser.parse_args()
    make_documentation(execute=args.execute, force=args.force, documentation=args.documentation,
                       clean=args.cleanup, specific=args.specific)
