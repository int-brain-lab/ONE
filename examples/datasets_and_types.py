"""
A dataset typically contains a single signal or data source, either values or times.  When
creating a new dataset, first familiarize yourself with the specification.
"""
from one.alf import spec
from one.alf.files import filename_parts

# Print information about ALF objects
spec.describe('object')

# Check the file name is ALF compliant
assert spec.is_valid('spikes.times.npy')

# Safely construct an ALF dataset using the 'to_alf' function.  This will ensure the correct
# case and format
filename = spec.to_alf('spikes', 'times', 'npy',
                       namespace='ibl', timescale='ephys clock', extra='raw')

# Parsing a new file into its constituent parts ensures the dataset is correct
parts = filename_parts('_ibl_spikes.times_ephysClock.raw.npy', as_dict=True, assert_valid=True)

"""
A dataset type includes wildcards in the name so that you can search over datasets with the same
content but different formats, etc. For example you could create a new dataset type called
'raw log' with the filename pattern *log.raw* When you register a file such as _rig1_log.raw.txt
or log.raw.rtf it will automatically be part of the 'raw log' dataset type. The main purpose of
this is to use the dataset type description field to document what the files are and how to work
with them. When registering files they must match exactly 1 dataset type.
"""
from one.api import ONE
one = ONE()
one.describe_dataset('spikes.times')  # Requires online version (an Alyx database connection)

one.dataset2type('_ibl_leftCamera.times.npy')  # Requires online version (an Alyx database
# connection)

one.type2datasets('camera.times')
