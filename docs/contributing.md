# Contributing to documentation
## Structure
The main contents file is found in `docs/index.rst`.  Some pages are written in markdown (.md), and 
are in the `docs/` folder.  The rest are Jupyter notebooks (.ipynb), placed in the `docs/notebooks/` folder.
The API reference is automatically generated from the docstrings in the code.  Docstrings should follow
the NumPy style.  Examples of the NumPy docstring format can be found [here](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).

## Committing code to GitHub repo
Commits to the 'docs' branch will trigger the documentation to compile and build automatically.  
External users should open a pull request to this branch.

## Running locally
To build the docs locally, first ensure all the requirements are installed:
```
pip install -r requirements-docs.txt
```

Then run the make-script.py file from within the `docs/` folder with the -d flag:
```
cd docs/
python ./make-script.py -d
```

The HTML files are placed in `docs/_build/html/`.

