# ONE installation and setup

## 1. Installation
ONE can be installed as a standalone package with python 3.8 or later by running,
```python
pip install ONE-api
```

```{note}
If you have installed the unified IBL environment, ONE will already have been automatically 
installed, so you can skip this step!
```

```{note}
If you want to access public data via ONE, please follow [these simplified steps](../public_docs/public_one) instead.
```

## 2. Setup
To start using ONE, we must first configure some settings that tell ONE whether it should connect to a database or 
use a local file system

### Connecting to specific database (relevant for IBL users)
To connect to a specific database, for example the internal IBL Alyx database, a base-url argument must be given
```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```

When connecting to the IBL internal database you will be prompted to enter credentials, please follow 
[these instructions](one_credentials.md) for help on configuring your credentials 

```{Warning}
The internal IBL database is only open to IBL members, please connect to our public database to access our publically
available datasets
```
If you are interested in accessing these publicly available datasets,
please visit this [section](../../08_public.html) for instructions.

### Connect to IBL Public database
By default ONE is configured to connect to the public IBL database, this can be setup by typing the following
```python
from one.api import ONE
one = ONE(silent=True)
```

### Using local folder structure
ONE can also be used independently of a database by using a local cache directory. This can be setup in the following way 

```python
from one.api import One
one = One(cache_dir='/home/user/downlaods/ONE/behavior_paper')
```

For more information about using ONE with a local cache directory please refer to this [section](../docs_external/one_offline.html).

## 3. Post setup
Once you've setup the server you can initialise ONE in the following way and it will automatically connect to the
default database configured in your settings,
```python
from one.api import ONE
one = ONE()
```

To change your default database, you can use the following
```python
from one.api import OneAlyx, ONE
OneAlyx.setup(client='https://test.alyx.internationalbrainlab.org', make_default=True)
one = ONE()  # Connected to https://test.alyx.internationalbrainlab.org
```
