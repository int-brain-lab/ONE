import os
import json
import time
from nbconvert.preprocessors import (ExecutePreprocessor, CellExecutionError,
                                     ClearOutputPreprocessor)
import nbformat
import re
import sphinx_gallery.notebook as sph_nb
import sphinx_gallery.gen_gallery as gg
import shutil
import logging
from pathlib import Path

"""
    Setup the logging environment
"""
_logger = logging.getLogger('ONE')

_logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
_logger.addHandler(stream_handler)

IPYTHON_VERSION = 4


class NotebookConverter(object):

    def __init__(self, nb_path, output_path=None, overwrite=True, kernel_name=None):
        """
        Parameters
        ----------
        nb_path : str
            Path to ipython notebook
        output_path: str, default=None
            Path to where executed notebook, rst file and colab notebook will be saved. Default is
            to save in same directory of notebook
        overwrite: bool, default=True
            Whether to save executed notebook as same filename as unexecuted notebook or create new
            file with naming convention 'exec_....'. Default is to write to same file
        kernel_name: str
            Kernel to use to run notebooks. If not specified defaults to 'python3'
        """
        self.nb_path = Path(nb_path).absolute()
        self.nb_link_path = Path(__file__).parent.parent.joinpath('notebooks_external')
        os.makedirs(self.nb_link_path, exist_ok=True)
        self.nb = self.nb_path.parts[-1]
        self.nb_dir = self.nb_path.parent
        self.nb_name = self.nb_path.stem
        self.overwrite = overwrite

        # If no output path is specified save everything into directory containing notebook
        if output_path is not None:
            self.output_path = Path(output_path).absolute()
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = self.nb_dir

        # If overwrite is True, write the executed notebook to the same name as the notebook
        if self.overwrite:
            self.executed_nb_path = self.output_path.joinpath(self.nb)
            self.temp_nb_path = self.output_path.joinpath(f'executed_{self.nb}')
        else:
            self.executed_nb_path = self.output_path.joinpath(f'executed_{self.nb}')

        if kernel_name is not None:
            self.execute_kwargs = dict(timeout=900, kernel_name=kernel_name, allow_errors=False)
        else:
            self.execute_kwargs = dict(timeout=900, kernel_name='python3', allow_errors=False)

    @staticmethod
    def py_to_ipynb(py_path):
        """
        Convert python script to ipython notebook
        Returns
        -------
        """
        nb_path = sph_nb.replace_py_ipynb(py_path)
        if not Path(nb_path).exists():
            file_conf, blocks = sph_nb.split_code_and_text_blocks(py_path)
            gallery_config = gg.DEFAULT_GALLERY_CONF
            gallery_config['first_notebook_cell'] = None
            example_nb = sph_nb.jupyter_notebook(blocks, gallery_config, nb_path)

            code = example_nb['cells'][1]['source'][0]
            # If using mayavi add in the notebook initialisation so that figures render properly
            if re.search("from mayavi import mlab", code):
                if not re.search("mlab.init_notebook()", code):
                    new_code = re.sub("from mayavi import mlab",
                                      "from mayavi import mlab\nmlab.init_notebook()", code)
                    example_nb['cells'][1]['source'][0] = new_code
            sph_nb.save_notebook(example_nb, nb_path)
        return nb_path

    def link(self):
        """
        Create nb_sphinx link file for notebooks external to the docs directory
        """
        link_path = os.path.relpath(self.nb_path, self.nb_link_path)
        link_dict = {"path": link_path}
        link_save_path = self.nb_link_path.joinpath(str(self.nb_name) + '.nblink')

        with open(link_save_path, 'w') as f:
            json.dump(link_dict, f)

    def execute(self, force=False):
        """
        Executes the specified notebook file, and writes the executed notebook to a
        new file.
        Parameters
        ----------
        force : bool, optional
            To force rerun notebook even if it has already been executed
        Returns
        -------
        executed_nb_path : str, ``None``
            The path to the executed notebook path, or ``None`` if ``write=False``.
        status: bool
            Whether the notebook executed without errors or not, 0 = ran without error, 1 = error
        """

        with open(self.nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=IPYTHON_VERSION)

        is_executed = nb['metadata'].get('docs_executed')

        if is_executed == 'executed' and not force:
            _logger.warning(f"Notebook {self.nb} in {self.nb_dir} already executed, skipping,"
                            f"to force execute, parse argument -f")
            status = 0
        else:

            # Execute the notebook
            _logger.info(f"Executing notebook {self.nb} in {self.nb_dir}")
            t0 = time.time()

            clear_executor = ClearOutputPreprocessor()
            executor = ExecutePreprocessor(**self.execute_kwargs)

            # First clean up the notebook and remove any cells that have been run
            clear_executor.preprocess(nb, {})

            try:
                executor.preprocess(nb, {'metadata': {'path': self.nb_dir}})
                execute_dict = {'docs_executed': 'executed'}
                nb['metadata'].update(execute_dict)
                status = 0
            except CellExecutionError as err:
                execute_dict = {'docs_executed': 'errored'}
                nb['metadata'].update(execute_dict)
                _logger.error(f"Error executing notebook {self.nb}")
                _logger.error(err)
                status = 1

            _logger.info(f"Finished running notebook ({time.time() - t0})")

            _logger.info(f"Writing executed notebook to {self.executed_nb_path}")
            # Makes sure original notebook isn't left blank in case of error during writing
            if self.overwrite:
                with open(self.temp_nb_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                shutil.copyfile(self.temp_nb_path, self.executed_nb_path)
                os.remove(self.temp_nb_path)
            else:
                with open(self.executed_nb_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)

        return self.executed_nb_path, status

    def unexecute(self, remove_gh=False):
        """
        Unexecutes the notebook i.e. removes all output cells. If remove_gh=True looks to see if
        notebook metadata contains an executed tag. If it doesn't it means the notebook either
        errored or was not run (for case when only specific notebooks chosen to build examples) and
        removes the notebooks so old ones can be used
        """
        _logger.info(f"Cleaning up notebook {self.nb} in {self.nb_dir}")
        if not self.executed_nb_path.exists():
            _logger.warning(f"{self.executed_nb_path} not found, nothing to clean")
            return

        with open(self.executed_nb_path, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=IPYTHON_VERSION)

        if not remove_gh:
            if nb['metadata'].get('docs_executed', None):
                nb['metadata'].pop('docs_executed')

            clear_executor = ClearOutputPreprocessor()
            clear_executor.preprocess(nb, {})

            with open(self.executed_nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

        elif remove_gh:
            executed_flag = nb['metadata'].get('docs_executed', None)
            if executed_flag != 'executed':
                _logger.warning(f"Notebook {self.nb} not executed or errored, "
                                f"version already on website will be used")
                os.remove(self.executed_nb_path)
                os.remove(self.output_path.joinpath(self.nb_name + '.html'))
            else:
                _logger.info(f"Notebook {self.nb} executed, "
                             f"new version will be uploaded to website")
                clear_executor = ClearOutputPreprocessor()
                clear_executor.preprocess(nb, {})

                with open(self.executed_nb_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)


def process_notebooks(nbfile_or_path, execute=True, force=False, link=False, cleanup=False,
                      filename_pattern='', remove_gh=False, **kwargs):
    """
    Execute and optionally convert the specified notebook file or directory of
    notebook files.
    Wrapper for `NotebookConverter` class that does all the file handling.
    Parameters
    ----------
    nbfile_or_path : str
        Either a single notebook filename or a path containing notebook files.
    execute : bool
        Whether or not to execute the notebooks
    link : bool, default = False
        Whether to create nbsphink link file
    cleanup : bool, default = False
        Whether to unexecute notebook and clean up files. To clean up must set this to True and
        execute argument to False
    filename_pattern: str, default = ''
        Filename pattern to look for in .py or .ipynb files to include in docs
    remove_gh: bool default = False
        Whether to remove notebook from build examples (in case where we want to use old version)
    **kwargs
        Other keyword arguments that are passed to the 'NotebookExecuter'
    """

    overall_status = 0
    if os.path.isdir(nbfile_or_path):
        # It's a path, so we need to walk through recursively and find any
        # notebook files
        for root, dirs, files in os.walk(nbfile_or_path):
            for name in files:

                _, ext = os.path.splitext(name)
                full_path = os.path.join(root, name)

                # skip checkpoints
                if 'ipynb_checkpoints' in full_path:
                    if cleanup:
                        os.remove(full_path)
                        continue
                    else:
                        continue

                # if file has 'ipynb' extension create the NotebookConverter object
                if ext == '.ipynb':
                    if re.search(filename_pattern, name):
                        nbc = NotebookConverter(full_path, **kwargs)
                        # Want to create the link file
                        if link:
                            nbc.link()
                        # Execute the notebook
                        if execute:
                            _, status = nbc.execute(force=force)
                            overall_status += status
                        # If cleanup is true and execute is false unexecute the notebook
                        if cleanup:
                            nbc.unexecute(remove_gh=remove_gh)

                # if file has 'py' extension convert to '.ipynb' and then execute
                elif ext == '.py':
                    if re.search(filename_pattern, name):
                        # See if the ipynb version already exists
                        ipy_path = sph_nb.replace_py_ipynb(full_path)
                        if Path(ipy_path).exists():
                            # If it does and we want to execute, skip as it would have been
                            # executed above already
                            if execute:
                                continue
                            # If cleanup then we want to delete this file
                            if cleanup:
                                os.remove(ipy_path)
                        else:
                            # If it doesn't exist, we need to make it
                            full_path = NotebookConverter.py_to_ipynb(full_path)
                            nbc = NotebookConverter(full_path, **kwargs)
                            if link:
                                nbc.link()
                            # Execute the notebook
                            if execute:
                                _, status = nbc.execute(force=force)
                                overall_status += status
                            # If cleanup then we want to delete this file
                            if cleanup:
                                os.remove(full_path)

    else:
        full_path = Path(nbfile_or_path)
        ext = full_path.suffix

        if ext == '.py':
            ipy_path = sph_nb.replace_py_ipynb(full_path)
            if not Path(ipy_path).exists():
                full_path = NotebookConverter.py_to_ipynb(full_path)
            else:
                full_path = ipy_path

        nbc = NotebookConverter(full_path, **kwargs)
        # Want to create the link file
        if link:
            nbc.link()
        # Execute the notebook
        if execute:
            _, status = nbc.execute(force=force)
            overall_status += status
        # If cleanup is true and execute is false, unexecute the notebook
        if cleanup:
            nbc.unexecute()
            if ext == '.py':
                os.remove(full_path)

    return overall_status
