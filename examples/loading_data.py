from one.api import ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)

"""
The datasets are organized into directory trees by subject, date and session number.  For a 
given session there are data files grouped by object (e.g. 'trials'), each with a specific 
attribute (e.g. 'rewardVolume').  The dataset name follows the pattern 'object.attribute', 
for example 'trials.rewardVolume.npy'.
"""

# To load all the data for a given object, use the load_object method:
eid = 'KS023/2019-12-10/001'  # folder session structure is subject/date/number
trials = one.load_object(eid, 'trials')  # Returns a dict-like object of numpy arrays
# The attributes of the returned object mirror the datasets:
print(trials.rewardVolume[:5])
# These can also be accessed with dictionary syntax:
print(trials['rewardVolume'][:5])
# All arrays in the object have the same length (the size of the first dimension) and can
# therefore be converted to a DataFrame:
trials.to_df().head()

"""
If the data don't exist locally, they will be downloaded, then loaded.  The *should* be 
consistent, however for some sessions there may be data extraction errors.  For analysis you can
assert that the dimensions match using the check_dimensions property:
"""
assert trials.check_dimensions == 0

"""
Datasets... TODO 
"""
reward_volume = one.load_session_dataset(eid, 'trials.rewardVolume')  # c.f. load_object, above

# To list the datasets available for a given session:
dsets = one.list_datasets(eid)['rel_path']

# To get information on a dataset # FIXME Currently doesn't work
dset_id = dsets.index[0]
one.describe_dataset(dset_id)

"""
Collections
"""

"""
Download only
"""

"""
Loading from file path
"""

"""
Loading with timeseries
"""
import one.alf.io as alfio
ts, values = alfio.read_ts()
