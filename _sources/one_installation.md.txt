# ONE installation and setup

## 1. Installation
ONE can be installed as a standalone package with python 3.7 or later by running,
```
pip install ONE-api
```

```{note}
If you have installed the unified IBL environment, ONE will already have been automatically 
installed, so you can skip this step!
```

## 2. Setup
To start using ONE, we must first configure some settings that tell ONE whether it should connect to a database or 
use a local file system

### Connect to IBL Public database
By default ONE is configured to connect to the public IBL database, this can be setup by typing the following
```python
from one.api import ONE
ONE.setup(silent=True)  # silent means use default parameters for the IBL public database
one = ONE()
```

If you are having an issue at this stage, or need to re-configure ONE, please see [this FAQ page](/FAQ.md#i-made-a-mistake-during-setup-and-now-can-t-call-setup-how-do-i-fix-it).

### Using local folder structure
ONE can also be used independently of a database by using a local cache directory. This can be setup in the following way 

```python
from one.api import One
one = One(cache_dir='/home/user/downlaods/ONE/behavior_paper')
```

For more information about using ONE with a local cache directory please refer to this [section](notebooks/data_sharing).

### Connecting to specific database (relevant for IBL users)
To connect to a specific database, for example the internal IBL Alyx database, a base-url argument must be given
```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```
```{note}
When connecting to the IBL internal database you will be prompted to enter credentials. See the IBL Welcome Guide for more information.
```

```{Warning}
The internal IBL database is only open to IBL members, please connect to our public database to access our publically
available datasets
```

## 3. Post setup
Once you've setup the server you can initialise ONE in the following way and it will automatically connect to the
default database configured in your settings,
```python
from one.api import ONE
one = ONE()
```

To change your default database, or re-run the setup for a given database, you can use the following

```python
ONE.setup(base_url='https://test.alyx.internationalbrainlab.org', make_default=True)
```

## 4. Update
To update, open a shell terminal, activate the environment `iblenv` and launch the command:
```python
pip install -U ONE-api
```

You can check the API version with Python one of two ways:
```python
from one.api import ONE
print(ONE.version)

from one import __version__
print(__version__)
```
