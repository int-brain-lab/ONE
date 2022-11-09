# ALyx Filenames (ALF)
This package is concerned with parsing and loading files that follow the Alyx file name specification.
Files should be organized in folders by subject name, date and session number, for example:
```text
mouse_001/2021-05-27/001
```

Optionally the lab may also be present in the folder structure, for example:
```text
lab_name/Subjects/mouse_001/2021-05-27/001
```

The file names themselves should have at least two components; an object and attribute, separated by a period.
For example a file called `trials.intervals` represents the trials object with an intervals attribute. 
The full file path would be:
```text
lab_name/Subjects/mouse_001/2021-05-27/001/trials.intervals
```

Objects and attributes should be in Haskell case for example `sparseNoise.xyPos` but supports 
acronyms, e.g. `RFMapStim.intervals`, `ROIMotionEnergy.position`.  Underscores, hyphens and spaces are 
not supported, except with 'times', 'timestamps' and 'intervals', which have a special meaning:

```text
trials.goCue_times
```
## Optional components
There are other optional parts to the file path that are used to convey other information.

### Collections
Within a session folder the data may be placed in any number of sub-folders, each one is referred to as a 
collection and these may be used to sort identical datasets by device or preprocessing software.  For 
example spikes collected on two different probes maybe in different numbered probe collections:
```text
mouse_001/2021-05-27/001/probe00/spikes.times.npy
mouse_001/2021-05-27/001/probe01/spikes.times.npy
```

Perhaps for analysis the spikes were sorted using two different spike sorters, one with Kilosort, the 
other with Yass:
```text
mouse_001/2021-05-27/001/probe00/ks2.1/spikes.times.npy
mouse_001/2021-05-27/001/probe01/yass/spikes.times.npy
```

### Revisions
If the data require pre-processing in a different manner, a revision folder may be used so that the 
original data is not overwritten.  This can be used as a form of versioning and should be a dated 
folder surrounded by pound signs, e.g.
```text
mouse_001/2021-05-27/001/#2021-06-01#/spikes.times.npy
```

Unlike collections these can be searched in lexicographical order such that a users can load a revision 
before or after a certain date.  If multiple revisions exist for a given date, letters may be appended 
to preserve ordering:
```text
mouse_001/2021-05-27/001/#2021-06-01#/spikes.times.npy
mouse_001/2021-05-27/001/#2021-06-01a#/spikes.times.npy
mouse_001/2021-05-27/001/#2021-06-01b#/spikes.times.npy
```

### Namespace
For datasets that are not intended to be standard in the community, a namespace may be added to the 
start of the filename.  This must be surrounded by underscores:
```text
_ibl_wheel.position
_ss_gratingID.laserOn.npy
```

### Timescale
Datasets containing timestamp data are expected to be in the same common timescale (usually seconds from
experiment start).  For datasets in a different timescale, the clock name should be appended to the 
attribute part with an underscore:

```text
spikes.times_ephysClock.npy
trials.intervals_bpod.ssv
```

### Extension
The extension should be self-explanatory.  Although they are optional in the ALF spec, it's preferable 
to include the format in the filename, and to use formats that are well supported in MATLAB and Python:

```text
spikes.times.npy
spikes.times.csv
spikes.times.mat
```

### Extra
Any number of extra parts, separated by periods, can be added after the attribute.  Examples include UUIDs
for ensuring the filename is unique or parts for splitting datasets into parts.  NB: The text after the final
period is expected to be the file extension.
```text
trials.intervals.9198edcd-e8a4-4e8a-994f-d68a2e300380.npy
2p.raw.part01.tiff
2p.raw.part02.tiff
```

### Relations
Alf objects can be related through their attributes. If the attribute name of one file matches the object name of a 
second, then the first file is guaranteed to contain integers referring to the rows of the second. For example, 
spikes.clusters.npy would contain integer references to the rows of clusters.brain_location.json and 
clusters.probes.npy; and clusters.probes.npy would contain integer references to probes.insertion.json.

## Glossary

### Dataset name
A filename with at least an object and attribute.  Some examples of valid ALF datasets:

```text
spikes.times
spikes.times.npy
_ibl_trials.goCue_times_bpodClock.csv
```

### Dataset type
In Alyx datasets are grouped by a type.  Datasets should belong to exactly one dataset type.  The 
group is determined by a filename pattern.  Dataset types group datasets with the same content but 
different formats, etc. and include a description of the dataset.  For example, the following datasets
belong to the '*spikes.times*' dataset type:
```text
spikes.times
_spikeglx_spikes.times_ephysClock.npy
spikes.times.9198edcd-e8a4-4e8a-994f-d68a2e300380.npy
spikes.times.cbin
``` 

### Session path
The part of the path that includes the subject name, date and number.  Optionally a lab name may also 
be part of the session path:

```text
mouse_001/2021-05-27/001
cortexlab/Subjects/mouse_001/2021-05-27/1
```

### Relative path
Everything that comes after the session path.  In other words the filename and optional collections
and revision folders:

```text
alf/probe00/spikes.times.npy
trials.intervals.npy
#2021-06-01#/trials.intervals.npy
```

### ALF path
The full file path, including the session path and relative path, e.g.
```text
cortexlab/Subjects/mouse_001/2021-05-27/1/alf/probe00/spikes.times.npy
```
