# FAQ
## How do I release my own data with the ONE API?
First create a directory of [ALF datasets](notebooks/datasets_and_types), generate the
cache tables, then share the directory with others.  See [data sharing](notebooks/data_sharing)
guide for more information.

## How do I use ONE without connecting to my database?
If you temporarily lose access to your Alyx database, you can instantiate ONE in local mode:
```
one = ONE(base_url='https://openalyx.internationalbrainlab.org', mode='local')
```
Read more about ONE modes [here](notebooks/one_modes).

## Why are my recent data missing from my cache but present on Alyx?
After new data are acquired it may take time for it to be copied to an online server (it will
be marked as 'online' in Alyx).  Once the data is marked as existing and online, it should appear
in queries, and the remote cache tables next time they are generated.  The latency depends on the
[ONE mode](notebooks/one_modes) used.

**Remote mode (default)**
When running remote queries (anything that uses `one.alyx.rest(....)`),
ONE stores the results of these queries for 5 minutes, so that if you
repeatedly make the same query over and over you don't hit the database
each time but can use the local cached result.

To circumvent this, instantiate ONE with `cache_rest=None` or use the `one.webclient.no_cache`
context manager when calling ONE list, search and load methods. You can pass the `no_cache=True`
argument AlyxClient: `one.alyx.rest(..., no_cache=True)`.  More information can be found
[here](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html#REST-caching).

**Local mode**
Local cache tables may be used when ONE is in 'local' mode (or when `query_type='local'` is passed).
These table contain info about all sessions and their associated datasets and is used instead of querying
the database.
For the IBL Alyx, the tables are generated every 6 hours and can be downloaded using the `one.load_cache` method
to only download when new data are available.  More information, including increasing refresh frequency, can be found
[here](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html#Refreshing-the-cache).

## I made a mistake during setup and now can't call setup, how do I fix it?
Usually you can re-run your setup with the following command:
```python
from one.api import ONE
ONE.setup(base_url='https://alyx.example.com')
```
⚠️<span style="color: red;"> Note: 'alyx.example.com' is just an example URL - replace with your actual database URL</span>

## How do I reset my ONE parameters to use Open Alyx?
To reset your ONE configuration to use the public Open Alyx database with default settings:
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', make_default=True)
```
**Note**: The `make_default=True` argument saves these settings as your default configuration. This means future ONE instances will use these settings unless specified otherwise.

## How do I change my download (a.k.a. cache) directory?
To **permanently** change the directory, simply re-run the setup routine:
```python
from one.api import ONE
ONE.setup()  # Re-run setup for default database (takes effect next time you instantiate ONE)
```
When prompted ('Enter the location of the download cache') enter the absolute path of the new download location.

To **temporarily** change the download directory, use the cache_dir arg:
```python
from pathlib import Path
from one.api import ONE

one = ONE(base_url='https://alyx.example.com', cache_dir=Path.home() / 'new_download_dir')
```
⚠️<span style="color: red;"> Note: 'alyx.example.com' is just an example URL - replace with your actual database URL</span>

**Note**: This will (down)load the cache tables in the newly specified location.  To avoid this, specify the cache table location separately using the `tables_dir` kwarg.

## How do I load cache tables from a different location?
By default, the cache tables are in the cache_dir root.  You can load cache tables in a different location in the following two ways:
```python
from pathlib import Path
from one.api import ONE

# 1. Specify location upon instantiation
one = ONE(tables_dir=Path.home() / 'tables_dir')
# 2. Specify location after instantiation
one.load_cache(Path.home() / 'tables_dir')
```
**Note**: Avoid using the same location for different database cache tables: ONE will overwrite tables when `load_cache` is called in remote mode.

## How do check who I'm logged in as?
```python
from one.api import ONE
one = ONE()
if not one.offline:
    print(one.alyx.user)
    print(one.alyx.base_url)
```

## How do I log out, or temporarily log in as someone else?
To log out:
```python
from one.api import ONE
one = ONE()

one.alyx.logout()
```

To log in as someone else temporarily:
```python
one.alyx.authenticate(username='other_user', cache_token=False, force=True)
```

## What to do if I am seeing a certificate error?
If you are using the Windows platform, you may see a certificate error when initially trying to connect with ONE. The last few
lines of the traceback should like this:
```powershell
File "C:\Users\User\anaconda3\envs\ONE\lib\urllib\request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>
```
This has a relatively easy fix:
* Open `Microsoft Edge` or `Internet Explorer` and navigate to the URL https://alyx.internationalbrainlab.org, or whichever alyx
site you are attempting to access with ONE (no need to log in)
* Reattempt to run any ONE query or setup on the command line
  * Simply visiting the website with a Microsoft web browser should be enough to get the site's certificate to be stored properly.
This is a unique issue with the way that the Windows OS handles certificates.

## How do I download the datasets cache for a specific IBL paper release?
You can download cache tables containing datasets with a specific release tag.

```python
from one.api import ONE

one = ONE()
TAG = '2021_Q1_IBL_et_al_Behaviour'  # Release tag to download cache for
one.load_cache(tag=TAG)
```

To return to the full cache containing an index of all experiments:
```python
ONE.cache_clear()
one = ONE()
```

## How do I check which version of ONE I'm using within Python?
You can check your version with the following: `print(ONE.version)`.\
The latest version can be found in the CHANGELOG, [here](https://github.com/int-brain-lab/ONE/blob/main/CHANGELOG.md). \
To update to the latest available version run `pip install -U ONE-api`.

## How do I use ONE in a read-only environment?
To use ONE without any write access or internet access, simply instantiate in local mode:
```python
from one.api import ONE
one = ONE(cache_dir='/path/to/data/dir', mode='local')
assert one.offline
```

If you wish to make Alyx database REST requests in a read-only environment, provide a database URL
and set `cache_rest=None` to avoid saving REST responses on disk:
```python
from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', cache_rest=None, mode='local')
assert one.offline and one.alyx.cache_mode is None
```

## Why does the search return a LazyID object?
When in remote mode using one.search or one.search_insertions, a LazyID object is returned instead of a list.
It behaves exactly the same as a list (you can index, slice and get its length).  Instead of retrieving all the
values from the database query it will fetch only the items you index from the list.  This greatly speeds up
the function when there are large search results.

## How do I get information about a session from an experiment ID?
You can fetch a dictionary of experiment details from an experiment ID using the `get_details` method:
```python
details = ONE().get_details(eid)
```

## How do I search for sessions with the exact subject name (excluding partial matches)?
When not in remote mode you can use a [regular expression](notebooks/one_search/one_search.html#Advanced-searching)
to assert the start and end of the search string:
```python
one = ONE(mode='local', wildcards=True)  # Should be True by default
subject = 'FD_04'
eids = one.search(subject=f'^{subject}$')
```

When in remote mode you can use a [Django exact query](notebooks/useful_alyx_queries.html#exact):
```python
one = ONE(mode='remote')
subject = 'FD_04'
eids = one.search(django=f'subject__nickname__exact,{subject}')
```

## Why are my search results inconsistent and/or seem to change?
This may be caused by one of two things:

First, each day when connecting to the database you download an updated cache table. The data on the database
may simply have changed, or you are loading a different cache table from somewhere. This may be because you
are connecting to a different database (check `one.alyx.base_url`), providing a different cache location (check `one._tables_dir`),
or provided a different tag (see [this question](#how-do-i-download-the-datasets-cache-for-a-specific-ibl-paper-release)).

Second, there are minor differences between the default/local modes and remote mode. Namely that in remote mode
queries are generally case-insensitive.  See the 'gotcha' section of
'[Searching with ONE](notebooks/one_search/one_search.html#Gotchas)' for more information.

## How do I load datasets that pass quality control
You can first filter sessions by those that the supplied datasets with QC level WARNING or less:

```python
one = ONE()
# In local and remote mode
eids = one.search(datasets=['trials.table.pqt', 'spikes.times.npy'], dataset_qc_lte='WARNING')
```

You can then load the datasets with list_datasets and load_datasets:
```python
dsets = one.list_datasets(eid, qc='WARNING', ignore_qc_not_set=True)
data, info = one.load_datasets(eid, dsets)
```
