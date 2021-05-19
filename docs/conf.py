# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ONE'
copyright = '2021, International Brain Lab'
author = 'International Brain Lab'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    # 'sphinx_copybutton',
    'nbsphinx',
    'nbsphinx_link',
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/style.css']


# -- Options for autosummary and autodoc ------------------------------------
autosummary_generate = True
# Don't add module names to function docs
add_module_names = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': False
}


# -- Options for nbsphinx ------------------------------------

# Only use nbsphinx for formatting the notebooks i.e never execute
nbsphinx_execute = 'never'
# Cancel compile on errors in notebooks
nbsphinx_allow_errors = False
# Add cell execution out number
nbsphinx_output_prompt = 'Out[%s]:'
# Configuration for images
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
plot_formats = [('png', 512)]

# Add extra prolog to beginning of each .ipynb file
# Add option to download notebook and link to github page
nbsphinx_prolog = r"""
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set nb_path = env.metadata[env.docname]['nbsphinx-link-target'] | dirname %}
{% set nb_name = env.metadata[env.docname]['nbsphinx-link-target'] | basename %}
{% else %}
{% set nb_name = env.doc2path(env.docname, base=None) | basename %}
{% set nb_path = env.doc2path(env.docname, base=None) | dirname %}
{% endif %}
.. raw:: html
      <a href="{{ nb_name }}"><button id="download">Download tutorial notebook</button></a>
      <a href="https://github.com/int-brain-lab/ibllib/tree/docsMayo/docs_gh_pages/{{ nb_path }}/
      {{ nb_name }}"><button id="github">Github link</button></a>
"""
