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

## Why are my recent data are missing from my cache but present on Alyx?
After new data are acquired it may take time for it to be copied to an online server (it will
be marked as 'online' in Alyx).  Once the data is marked as existing and online, it should appear
in the cache tables next time they are generated.  For the IBL Alyx, the ONE cache tables are
re-generated every 6 hours, however by default ONE will only download a new cache once per day.  To
force a download you can run `ONE().refresh_cache('remote')`.  More information, including
increasing refresh frequency, can be found [here](https://int-brain-lab.github.io/ONE/notebooks/one_modes.html#Refreshing-the-cache).

