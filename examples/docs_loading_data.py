"""
Introduction to using ONE
=========================
"""

from one.api import ONE
import one.alf.io as alfio

one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)

"""
The datasets are organized into directory trees by subject, date and session number.  For a
given session there are data files grouped by object (e.g. 'trials'), each with a specific
attribute (e.g. 'rewardVolume').  The dataset name follows the pattern 'object.attribute',
for example 'trials.rewardVolume'.

An experiment ID (eid) is a string that uniquely identifies a session, for example a combination
of subject date and number (e.g. KS023/2019-12-10/001), a file path (e.g.
'C:\\Users\\Subjects\\KS023\\2019-12-10\\001'), or a UUID (aad23144-0e52-4eac-80c5-c4ee2decb198).

If the data don't exist locally, they will be downloaded, then loaded.
"""

# To load all the data for a given object, use the load_object method:
eid = 'KS023/2019-12-10/001'  # subject/date/number
trials = one.load_object(eid, 'trials')  # Returns a dict-like object of numpy arrays
# The attributes of the returned object mirror the datasets:
print(trials.rewardVolume[:5])
# These can also be accessed with dictionary syntax:
print(trials['rewardVolume'][:5])
# All arrays in the object have the same length (the size of the first dimension) and can
# therefore be converted to a DataFrame:
trials.to_df().head()

"""
Datasets can be individually downloaded using the load_dataset method.  This
function takes an experiment ID and a dataset name as positional args.
"""
reward_volume = one.load_dataset(eid, '_ibl_trials.rewardVolume.npy')  # c.f. load_object, above

# To list the datasets available for a given session:
dsets = one.list_datasets(eid, details=False)

# If connected to a remote database you can get documentation on a dataset
one.describe_dataset(dsets[0])  # alf/_ibl_trials.choice.npy
# e.g. prints 'which choice was made in choiceworld: -1 (turn CCW), +1 (turn CW), or 0 (nogo)'

"""
Collections

For any given session there may be multiple datasets with the same name that are organized into
separate subfolders called collections.  For example there may be spike times for two probes, one
in 'alf/probe00/spikes.times.npy', the other in 'alf/probe01/spikes.times.npy'.  In IBL, the 'alf'
directory (for ALyx Files) contains the main datasets that people use.  Raw data is in other
directories.

In this case you must specify the collection when multiple matching datasets are found:
"""
probe1_spikes = one.load_dataset(eid, 'spikes.times.npy', collection='alf/probe01')

"""
Loading with file name parts

You may also specify specific parts of the filename for even more specific filtering.  Here a
list of options will be treated as a logical OR
"""
dataset = dict(object='spikes', attribute='times', extension=['npy', 'bin'])
probe1_spikes = one.load_dataset(eid, dataset, collection='alf/probe01')

"""
Revisions

Revisions provide an optional way to organize data by version.  The version label is
arbitrary, however the folder must start and end with pound signs and is typically an ISO date,
e.g. "#2021-01-01#". Unlike collections, if a specified revision is not found, the previous
revision will be returned.  The revisions are ordered lexicographically.

probe1_spikes = one.load_dataset(eid, 'trials.intervals.npy', revision='2021-03-15a')
"""

"""
Download only

By default the load methods will download any missing data, then load and return the data.
When the 'download_only' kwarg is true, the data are not loaded.  Instead a list of file paths
are returned, and any missing datasets are represented by None.
TODO Revisions should always be dated
"""
files = one.load_object(eid, 'trials', download_only=True)

"""
You can load objects and datasets from a file path
"""
trials = one.load_object(files[0], 'trials')
contrast_left = one.load_dataset(files[0], files[0].name)

"""
Loading with timeseries

For loading a dataset along with its timestamps, alf.io.read_ts can be used. It requires a
filepath as input.
"""
files = one.load_object(eid, 'spikes', collection='alf/probe01', download_only=True)
ts, clusters = alfio.read_ts(files[1])

"""
Filtering attributes

To download and load only a subset of attributes, you can provide a list to the attribute kwarg.
"""
spikes = one.load_object(eid, 'spikes', collection='alf/probe01', attribute=['time*', 'clusters'])
assert 'amps' not in spikes


"""
For any given object the first dimension of every attribute should match in length.  For
analysis you can assert that the dimensions match using the check_dimensions property:
"""
assert trials.check_dimensions == 0

# Load spike times from a probe UUID
pid = 'b749446c-18e3-4987-820a-50649ab0f826'
session, probe = one.pid2eid(pid)
spikes_times = one.load_dataset(session, 'spikes.times.npy', collection=f'alf/{probe}')

# List all probes for a session
print([x for x in one.list_collections(session) if 'alf/probe' in x])

"""
Advanced loading:

The load methods typically require an exact match, therefore when loading '_ibl_wheel.position.npy'
`one.load_dataset(eid, 'wheel.position.npy')` will raise an exception because the namespace is
missing. Likewise `one.load_object(eid, 'trial')` will fail because 'trial' != 'trials'.

Loading can be done using unix shell style wildcards, allowing you to load objects and datasets
that match a particular pattern, e.g. `one.load_dataset(eid, '*wheel.position.npy')`.

By default wildcard mode is on.  In this mode, the extension may be omitted, e.g.
`one.load_dataset(eid, 'spikes.times')`. This is equivalent to 'spikes.times.*'. Note that an
exception will be raised if datasets with more than one extension are found (such as
'spikes.times.npy' and 'spikes.times.csv').  When loading a dataset with extra parts,
the extension (or wildcard) is explicitly required: 'spikes.times.part1.*'.

If you set the wildcards property of One to False, loading will be done using regular expressions,
allowing for more powerful pattern matching.

Below is table showing how to express unix style wildcards as a regular expression:

 Regex   |    Wildcard    |         Description        |    Example
-----------------------------------------------------------------------
   .*            *          Match zero or more chars     spikes.times.*
   .?            ?          Match one char               timestamps.?sv
   []            []         Match a range of chars       obj.attr.part[0-9].npy

NB: In regex '.' means 'any character'; to match '.' exactly, escape it with a backslash

Examples:
    spikes.times.* (regex), spikes.times* (wildcard) matches...
        spikes.times.npy
        spikes.times
        spikes.times_ephysClock.npy
        spikes.times.bin

    clusters.uuids..?sv (regex), clusters.uuids.?sv (wildcard) matches...
        clusters.uuids.ssv
        clusters.uuids.csv

    alf/probe0[0-5] (regex), alf/probe0[0-5] (wildcard) matches...
        alf/probe00
        alf/probe01
        [...]
        alf/probe05
"""

# More regex examples
one.wildcards = False

# Load specific attributes from an object ('|' represents a logical OR in regex)
spikes = one.load_object(eid, 'spikes', collection='alf/probe01', attribute='times|clusters')
assert 'amps' not in spikes

# Load a dataset ignoring any namespace or extension:
spike_times = one.load_dataset(eid, '.*spikes.times.*', collection='alf/probe01')

# List all datasets in any probe collection (matches 0 or more of any number)
dsets = one.list_datasets(eid, collection='alf/probe[0-9]*')

# Load object attributes that are not delimited text files (i.e. tsv, ssv, csv, etc.)
files = one.load_object(eid, 'clusters', extension='[^sv]*', download_only=True)
assert not any(str(x).endswith('csv') for x in files)
