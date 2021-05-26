from one.alf import spec

# Print information about ALF objects
spec.describe('object')

#
spec.is_valid('spikes.times.npy')

filename = spec.to_alf('spikes', 'times', 'npy',
                       namespace='ibl', timescale='ephys clock', extra='raw')
