"""
ONE can be instantiated in either the online and offline state.  When online, ONE can query the
Alyx database to retrieve information about dataset types <TODO link>, search with metadata such as
session QC, task performance, histology and session narrative.  Advanced queries may also be
made via REST <TODO link>.  Other online methods include `pid2eid` and `describe_revision`.

When the mode is not specified, it is usually set to 'auto', unless a cache_dir is specified for
which no database has been configured.
"""
from one.api import ONE

one = ONE()
one.mode  # 'auto'
assert not one.offline

one_offline = ONE(cache_dir=r'C:\data')
one_offline.mode  # 'local'
assert one.offline

"""
In 'auto' mode, the list, search and load methods will use the local cache tables and not 
connect to Alyx, however the option to use Alyx is always there.  If the cache tables can't be 
downloaded from Alyx, or authentication fails, the mode will fall back to 'local'.

If 'remote' mode is specified, ONE will only query the remote database and will not use the 
local cache tables.  Avoiding the database whenever possible is recommended as it doesn't rely 
on a stable internet connection and reduces the load on the remote Alyx.

While in 'remote' mode, the local cache may be used by providing the query_type='local' keyword 
argument to any method.  Likewise, in 'auto'/'local' mode, a remote query can be made by 
specifying query_type='remote':
"""
one.search(lab='cortexlab', query_type='remote')  # Search Alyx instead of the local cache
# NB: The 'remote' query type is not valid in offline mode as there is no database associated to
# the local cache directory.

"""
Refreshing the cache:

By default ONE will try to update the cache once every 24 hours.  This can be set by changing 
the 'cache_expiry' property:
"""
from datetime import timedelta
one.cache_expiry = timedelta(hours=1)  # Check for new remote cache every hour

# The time when the cache was generated can be found in the cache metadata:
print(one._cache._meta)
"""
Note that the cache won't be downloaded if the remote cache hasn't been updated since the last 
download.  The cache can be explicitly refreshed in two ways:
"""
one.refresh_cache('refresh')  # Explicitly refresh the cache
one.search(lab='cortexlab', query_type='refresh')  # Calls `refresh_cache` before searching
