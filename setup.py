from setuptools import setup, find_packages
import sys
from pathlib import Path

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 7)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of ONE requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]


def get_version(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError('Unable to find version string.')


setup(
    name='ONE-api',
    version=get_version(Path('one', '__init__.py')),
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    description='Open Neurophysiology Environment',
    license="MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='IBL Staff',
    url="https://github.com/int-brain-lab/ONE",
    packages=find_packages(exclude=['one.tests*']),  # same as name
    include_package_data=False,
    # external packages as dependencies
    install_requires=require,
    entry_points={},
    scripts={},
)
