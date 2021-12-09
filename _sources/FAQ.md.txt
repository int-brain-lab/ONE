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
in the cache tables next time they are generated.  For the IBL Alyx, the ONE cache tables are
re-generated every 6 hours, however by default ONE will only download a new cache once per day.  To
force a download you can run `ONE().refresh_cache('remote')`.  More information, including
increasing refresh frequency, can be found [here](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html#Refreshing-the-cache).

Note: There are two different definitions of caches that are used in ONE2:
1. The cache table that stores info about all sessions and their associated datasets.
This is refreshed every night and uploaded to Flatiron and downloaded onto your computer
every 24hr (this is what the datetime object returned as output of the `ONE().refresh_cache('remote')`
command is showing, i.e. when this cache was last updated).
This table is used in all one.search, one.load, one.list functions. When doing 
`ONE().refresh_cache('remote')`, you are basically forcing ONE to re-download this table 
regardless of when it was last downloaded from Flatiron.

2. When running remote queries (anything that uses `one.alyx.rest(....)`), 
ONE stores the results of these queries for 24 hours, so that if you 
repeatedly make the same query over and over you don't hit the database 
each time but can use the local cached result.
A problem can arise if something on the Alyx database changes in between the same query:
    - For example, at time X a given query returns an empty result (e.g. no histology session for a given subject).
    At time X+1, data is registered onto Alyx.
    At time X+2, you run the same query again.
    Because you had already made the query earlier, ONE uses the local result that 
    it had previously and displays that there isn't a histology session. 
    To circumvent this, use the `no_cache=True` argument in `one.alyx.rest(..., no_cache=True)` or
    the `no_cache` web client context.  More information can be found [here](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html#REST-caching).
    Use this only if necessary, as these methods are not optimized.

## I made a mistake during setup and now can't call setup, how do I fix it?
Usually you can re-run your setup with the following command:
```python
from one.api import ONE
new_one = ONE().setup(base_url='https://alyx.example.com')
```

Sometimes if the settings are wrong, the call to `ONE()' raises an error before the setup method is
called.  To avoid this, run the following command instead:

```python
from one.api import OneAlyx
new_one = OneAlyx.setup(base_url='https://alyx.example.com')
```
