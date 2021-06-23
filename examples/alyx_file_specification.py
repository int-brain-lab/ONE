"""
An ALyx File (ALF) is any file whose path matches a specific pattern.

The full spec is available in the `one.alf.spec module`.
"""
import one.alf.spec as alf_spec
from one.api import ONE

one = ONE()

# A valid ALF path includes the following parts (those in brackets are optional):
print(alf_spec.path_pattern())

# Details of each part can be obtained through the `one.alf.spec.describe` function:
alf_spec.describe('collection')

"""
When using `One.load_object` an object is passed to the method for loading.  Other specifiers
such as attributes, collection and revision may be passed.  For example given the following file
structure:

    subject/
    ├─ 2021-06-30/
    │  ├─ 001/
    │  │  ├─ alf/
    │  │  │  ├─ probe00/
    │  │  │  │  ├─ spikes.clusters.npy
    │  │  │  │  ├─ spikes.times.npy
    │  │  │  ├─ probe01/
    │  │  │  │  ├─ #2021-07-05#/
    │  │  │  │  │  ├─ spikes.clusters.npy
    │  │  │  │  │  ├─ spikes.times.npy
    │  │  │  │  ├─ spikes.clusters.npy
    │  │  │  │  ├─ spikes.times.npy
    │  │  ├─ probes.description.json

    subject/2021-06-01/001/probes.description.json
    subject/2021-06-01/001/alf/probe00/spikes.times.npy
    subject/2021-06-01/001/alf/probe00/spikes.clusters.npy
    subject/2021-06-01/001/alf/probe01/spikes.times.npy
    subject/2021-06-01/001/alf/probe01/spikes.clusters.npy
    subject/2021-06-01/001/alf/probe01/#2021-07-05#/spikes.times.npy
    subject/2021-06-01/001/alf/probe01/#2021-07-05#/spikes.clusters.npy
"""
# To list all the files in 'subject/2021-06-01/001'
datasets = one.list_datasets('subject/2021-06-01/001')
"""
    probes.description.json
    alf/probe00/spikes.times.npy
    alf/probe00/spikes.clusters.npy
    alf/probe01/spikes.times.npy
    alf/probe01/spikes.clusters.npy
    alf/probe01/#2021-07-05#/spikes.times.npy
    alf/probe01/#2021-07-05#/spikes.clusters.npy
"""

# To list all datasets in the 'alf/probe01' collection
datasets = one.list_datasets('subject/2021-06-01/001', collection='alf/probe01')
"""
    alf/probe01/spikes.times.npy
    alf/probe01/spikes.clusters.npy
    alf/probe01/#2021-07-05#/spikes.times.npy
    alf/probe01/#2021-07-05#/spikes.clusters.npy
"""

# To list all datasets not in a collection
datasets = one.list_datasets('subject/2021-06-01/001', collection='')
"""
    probes.description.json
"""

# To list all revisions for a given session
revisions = one.list_revisions('subject/2021-06-01/001')
"""
    None
    2021-07-05
"""

# To list all collections for a given session
collections = one.list_collections('subject/2021-06-01/001')
"""
    None
    alf
    alf/probe00
    alf/probe01
"""

# To load the 'spikes' object from the 'alf/probe00' collection
spikes = one.load_object('subject/2021-06-01/001', 'spikes', collection='alf/probe00')

# To load the 'spikes' object from the 'alf/probe01' collection, and the last revision before July
spikes = one.load_object('subject/2021-06-01/001', 'spikes',
                         collection='alf/probe01', revision='2021-07-01')

# To load 'spikes.times' from collection 'alf/probe00'
spike_times = one.load_dataset('subject/2021-06-01/001', 'spikes.times.npy',
                               collection='alf/probe00')
