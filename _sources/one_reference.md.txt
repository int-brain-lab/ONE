# Introduction to ONE (Open Neurophysiology Environment)

Neurophysiology needs data standardization.
A scientist should be able to analyze data collected from multiple labs using a single analysis program,
without spending untold hours figuring out new file formats.
Substantial efforts have recently been put into developing neurodata file standards, 
with the most successful being the [Neurodata Without Borders](https://www.nwb.org/) (NWB) format. 
The NWB format has a comprehensive and careful design that allows one to store all data and metadata 
pertaining to a neurophysiology experiment.
Nevertheless, its comprehensive design also means a steep learning curve for potential users, which has limited its adoption.

## How the open neurophysiology environment works

Here we provide a solution to this problem: a set of four simple functions, 
that allow users to access and search data from multiple sources. 
To adopt the standard, data providers can use any format they like - 
all they need to do is implement these functions to fetch the data from their server and load it into Python or MATLAB. 
Users can then analyze this data with the same exact code as data from any other provider. 
The learning curve will be simple, and the loader functions would be enough for around 90% of common use cases.

By a *data provider* we mean an organization that hosts a set of neurophysiology data on an internet server (for example, the [International Brain Lab](https://www.internationalbrainlab.com/)). The Open Neurophysiology Environment (ONE) provides a way for scientists to analyze data from multiple data providers using the same analysis code. There is no need for the scientist to explicitly download data files or understand their format - this is all handled seamlessly by the ONE framework. The ONE protocol can also be used to access a scientist's own experiments stored on their personal computer, but we do not describe this use-case here.


### Data structure
Every experiment a data provider releases is identified by an *experiment ID* (eID) -- a small token that uniquely identifies a particular experiment. 
It is up to the data provider to specify the format of their eIDs. 

The data files **have to follow the Alyx file name specification** ([ALF](../alf)).
The ONE package is concerned with parsing and loading files that follow this specification.

### Accessing the API
When a user wants to analyze data released by a provider, they 
first import that provider's loader functions. In python, to analyze IBL data, they would type
```
from one.api import ONE
```

### Loading data

If a user already knows the eID of an experiment they are interested in, they can load data for the experiment using a command like:
```
st, sc, cbl = ONE().load_datasets(eID, ['spikes.times', 'spikes.clusters', 'clusters.brain_location'])
```
This command will (down)load three datasets containing the times and cluster assignments of all spikes recorded in that experiment, together with an estimate of the brain location of each cluster.

### Searching for experiments
Finally, a user needs to be able to search the data released by a provider, to obtain the eIDs of experiments they want to analyze. To do so they would run a command like:
```
eIDs = ONE().search(lab='CortexLabUCL', subject='hercules', dataset=['spikes.times', 'spikes.clusters','headTracking.xyPos'])
eIDs, eInfo = ONE().search(details=True, lab='CortexLabUCL', subject='hercules', dataset=['spikes.times', 'spikes.clusters','headTracking.xyPos'])
```
This would find the eIDs for all experiments collected in the specified lab for the specified experimental subject, for which all the required data is present. There are more metadata options to refine the search in online mode (e.g. dates, genotypes, experimenter), and additional metadata on each matching experiment is returned in `eInfo`. However, the existence of datasets is normally enough to find the data you want. For example, if you want to analyze electrophysiology in a particular behavior task, the experiments you want are exactly those with datasets describing the ephys recordings and that task's parameters.

## Standardization

The key to ONE's standardization is the concept of a "standard dataset type".
When a user requests one of these (such as `spikes.times`), they are guaranteed that each data provider will return them the same information, organized in the same way - in this case, the times of all extracellularly recorded spikes, measured in seconds relative to experiment start, and returned as a 1-dimensional column vector. It is guaranteed that any dataset types of the form `*.times` or `*.*_times` will be measured in seconds relative to experiment start. Furthermore, all dataset types differing only in their last word (e.g. `spikes.times` and `spikes.clusters`) will have the same number of rows, describing multiple attributes of the same objects. Finally, words matching across dataset types encode references: for example, `spikes.clusters` is guaranteed to contain integers, and to find the brain location of each of these one looks to the corresponding row of `clusters.brain_location`, counting from 0.

Not all data can be standardized, since each project will do unique experiments. Data providers can thereform add their own project-specific dataset types. The list of standard dataset types will be maintained centrally, and will start small but increase over time as the community converges on good ways to standardize more information. It is therefore important to distinguish dataset types agreed as universal standards from types specific to individual projects. To achieve this, names beginning with an underscore are guaranteed never to be standard. It is recommended that nonstandard names identify the group that produces them: for example the dataset types `_ibl_trials.stimulusContrast` and `clusters._ibl_task_modulation` could contain information specific to the IBL project.

## Versioning and subcollections

Data are often released in multiple versions. Most users will want to have the latest version, but sometimes a user working will want to continue working with a historical version even after it has been updated,
to maintain consistency with previous work.
To enable versioning, the ONE functions accept an optional argument of the form `revision='v1'`, which ensures this specific version is loaded. If the argument is not passed, the latest version will be loaded. 

Sometimes the data will contain multiple measurements of the same type, for example if recordings are made with multiple recording probes simultaneously. In these cases, the dataset types for probe 0 will have names like `probe00/spikes.times`, `probe00/spikes.clusters`, and `probe00/clusters.brain_location`; data for probe 1 will be `probe01/spikes.times`, etc. Encoding of references works within a subcollection: i.e. the entries of `probe00/spikes.clusters` point to the rows of `probe00/clusters.brain_location`, starting from 0, and independently of any datasets starting with `probe01/`.

## For data sharers

Data standards are only adopted when they are easy to use, for both providers and users of data. For users, the three ONE functions will be simple to learn, and will cover most common use cases.

For providers, a key advantage of this framework is its low barrier to entry. To share data with ONE, providers do not need to run and maintain a backend server, just to upload their data to a website. We provide a "ONE light" implementation of the ONE loader functions that searches, downloads and caches files from a web server. This will allow producers who do not have in-house computational staff two simple paths to achieve ONE compatibility. The first is to place the data in a directory, using a standard file-naming convention described below, in standard formats including `npy`, `csv`, `json`, and `tiff`. Next, the user runs a program in this directory, which uploads the files to a website or to figshare. Users can then access this data using ONE light. And example of ONE light data is [here](https://figshare.com/articles/Test1/9917741). The ONE light code is [here](https://github.com/int-brain-lab/ibllib/tree/onelight/oneibl#one-light).
