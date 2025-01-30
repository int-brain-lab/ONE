# Contributing to code

Always branch off branch `main` before commiting changes, then push to remote and open a PR into `main`.
A developer will then approve the PR and release.

## Before making a pull request

### Commit history

Commit messages should reference a specific issue wherever possible.

### Linting and formatting

We use [ruff](https://docs.astral.sh/ruff/) to lint and format this repository. The ruff settings are defined in [pyproject.toml](./pyproject.toml).
You can check for linting errors by running `ruff check .`. This should be run within the ONE root directory for the pyproject configuration to be used.
The linter also checks for docstring formatting.

### Testing and coverage

You should ensure all tests are running locally.  We run tests using python unittest: `python -m unittest discover`. This should be run within the ONE root directory.  Coverage should also be checked as we keep coverage at 100%. Testing with coverage can be run with the following: `coverage run --omit=one/tests/* -m unittest discover`.

## Releasing (maintainers only)

Before merging to main, the CI checks that all unit tests pass and that coverage does not decrease.  Merges should not occur until all checks pass.
The Github testing workflow can be in [.github/workflows/main.yaml](.github/workflows/main.yaml) and is testing against Python version 3.10 and 3.12 on Ubuntu and Windows.
The divergent changes to the `docs` branch should be pulled into your release branch as the [.github/workflows/docs.yaml](.github/workflows/docs.yaml) workflow is triggered by both pushes to main and docs.

### Release notes and version
On a release or feature branch, update the [CHANGELOG.md](./CHANGELOG.md) file with a terse list of changes for the release, oriented for users.  The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  We organize changes by 'added' - new functions/modules/packages; 'modified' - changes to current functions and modules; 'removed' - deleted functions/modules/packages.  A short one or two sentence description below the version heading should describe the principle changes.  For example:

```markdown
## [Latest](https://github.com/int-brain-lab/ONE/commits/main) [X.X.X]
Short sentence describing major changes.

### Modified

- ...

### Added

- ...

### Removed

- ...

```

The ONE version should be iterated. The version is defined in [one/\_\_init\_\_.py](./one/__init__.py) and follows the
[Python packaging semver specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).

* Patches (X.X.1) are small or urgent bug fixes or changes that don't affect compatibility.
* Minor releases (X.1.X) are new features such as added functions or small changes that don't cause major compatibility issues.
* Major releases (1.X.X) are major new features or changes that break backward compatibility in a big way.

### Releasing to PyPi

Once merged to main you can make a release to PyPi using the [.github/workflows/python-publish.yaml](.github/workflows/python-publish.yaml) workflow:

1. Draft a [release on Github](https://github.com/int-brain-lab/ONE/releases).
2. Title the release as the version number, starting with 'v'.
3. Copy and paste the release notes from [CHANGELOG.md](./CHANGELOG.md).
4. Create a tag with the version. Note that in order to trigger a PyPi release the tag must begin with 'v', e.g. `v2.8.0`.


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
