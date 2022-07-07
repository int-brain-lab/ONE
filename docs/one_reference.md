# Introduction to ONE (Open Neurophysiology Environment)

The Open Neurophysiology Environment is a protocol for standardizing, searching and sharing
neurophysiology data.

ONE defines a simple set of conventions for how to store and share neurophysiology data, including features such as cross-references between datasets, versioning, and time synchronization. The easiest way to share data with ONE is save it in files following the [ONE filename convention](./Open_Neurophysiology_Environment_Filename_Convention.pdf), using [standard filenames](https://docs.google.com/spreadsheets/d/13-3f1JE_wdSAqlO2xi_6XV8nQ5up-uaOIvWAYFBJ7q0/edit#gid=0) when applicable, and put them on a website. By following this convention, individual labs or small collaborations can enable users to easily load their data and understand how it is organized, without having to spend hours reading documentation.  

ONE also provides an API to search and load datasets. The same API commands can be used to access a few files kept on a user's local machine, or large collections of files stored on a remote server. By releasing data in remote mode, large collaborations can release data covering all aspects of their experiments while allowing users to find and download only the specific data items that they need. Using the same interface to access small and large collections makes it easy for scientists to share data using ONE as a standard, and to scale up as their projects grow.

The following documentation describes the ONE API, and how to use it to access data from the [International Brain Lab](https://www.internationalbrainlab.com/). 

## How the ONE API works

The API comprises three simple methods to search, list and load data. 
Because the data files follow a standard naming convention, the Open Neurophysiology Environment (ONE) 
provides a way for scientists to analyze data from multiple data providers using the same analysis code. 
There is no need for the scientist to explicitly download data files or understand their format - this 
is all handled seamlessly by the ONE framework. 

### Experiment IDs
Every experimental session is identified by an *experiment ID* (*eID*) -- a string that uniquely 
identifies a particular experiment. This may be a path fragment (i.e. subject/date/number) or [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier). 

For detailed information, see the [searching data](./notebooks/experiment_ids) guide.

### Searching for experiments
To obtain the eIDs of experiments a user can use the search method to filter experiments by a set of criteria:
```python
eids = ONE().search(
    lab='CortexLabUCL', 
    subject='hercules', 
    dataset=['spikes.times', 'spikes.clusters','headTracking.xyPos']
)
```
This would find the eIDs for all experiments collected in the specified lab for the specified
experimental subject, for which all the required data is present. There are more metadata options
to refine the search in online mode (e.g. dates, genotypes, experimenter), and additional metadata
can optionally be returned . The existence of datasets is normally enough to find the data you want. 
For example, if you want to analyze electrophysiology in a particular behavior task, the experiments 
you want are exactly those with datasets describing the ephys recordings and that task's parameters.

For detailed information, see the [searching data](./notebooks/one_search/one_search) guide.

### Datasets

The data for each experiment are organized into *datasets*, which are normally (but not always) 
numerical arrays. A *dataset name* is a string identifying a particular piece of data 
(such as `spikes.times`). When a user requests one of these, they are guaranteed to be returned the 
same information, organized in the same way - in this case, the times of all extracellularly 
recorded spikes, measured in seconds relative to experiment start, and returned as a 1-dimensional 
column vector. 

Dataset names have two parts, called the *object* and the *attribute*, which allow encoding of
relationships between datasets.  Datasets with the same object name (e.g. `spikes.times` and `spikes.clusters`)
describe multiple attributes of the same object analogously to a database table or data frame
(in this example the times and cluster assignments of each spike). Datasets with the same object
name will always have the same number of rows (or the same leading dimension size for high-dimensional arrays).

If the attribute of one dataset matches the object of another, this represents a cross-reference.
For example, `spikes.clusters` contains an integer cluster assignment for each spike (counting from 0),
while `clusters.waveforms` contains a 3d numerical array giving the mean waveform of these clusters. 
This convention therefore allows a basic relational model to be encoded in datasets. 

Any dataset name of the form `*.times` or `*.*_times` will a 1-column array of times measured in
seconds relative to experiment start. Any dataset name of the form `*.intervals` or `*.*_intervals` 
will be a two-column array of start and stop times measured in seconds relative to experiment start. 

Additionally datasets with a `table` attribute will be loaded and split into one key per column and
merged with any other data part of the same object.  Table columns will take precedent in the case
of duplicate attributes.  If a `*.*.metadata.*` file exists for a given attribute and specifies
column names, the loaded table/matrix will be split into said columns.

Datasets are organized into experiment folders by subject, date and sequence.  These session folders 
may optionally be organized by lab.

For detailed information, see the [datasets and types](./notebooks/datasets_and_types) guide.

### Collections and revisions
An experiment may contain multiple datasets of the same type. For example in an electrophysiology 
recording with multiple probes, for which the results of multiple spike sorting algorithms have been stored,
the user must be able to specify which version of `spikes.times` they want. In this case, the datasets 
belong to different *collections*. Collections are optional subdirectories within a session folder.
For example, datasets pertaining probe number 00, spike-sorted with kilosort 2.5 would belong to the 
collection `probe00\ks2.5`. 

Sometimes, datasets will be revised, for example if pre-processing software is rerun.
Nevertheless, users might prefer to keep using an older version of the datasets, for example if
finalizing a paper. To enable this, revised datasets are identified by a *revision*: a subdirectory
such as an ISO date like `2021-08-31`. If a user requests a particular revision, they will be 
returned the most recent previous revision (in lexicographical order). Thus, a user can "freeze" an 
analysis by specifying a single date, and thus be given a snapshot of what the most recent data on that day.

For detailed information, see the [full ALF specification](./alf_intro).

### Listing data
The second API method allows the user to list and filter the available datasets for an experiment.
To list the datasets for a given experiment and filter by a collection, the user would run
```python
datasets = ONE().list_datasets(eid, collection='*probe00')
```

Likewise, users can list collections and revisions for a given experiment, and all methods support wildcards.

For detailed information, see the [listing data](./notebooks/one_list/one_list) guide.

### Loading data

Finally the user can load data for the experiment using one of the load methods:
```
st, sc, cbl = ONE().load_datasets(eID, ['spikes.times', 'spikes.clusters', 'clusters.brain_location'])
```

This command will (down)load three datasets containing the times and cluster assignments of all 
spikes recorded in that experiment, together with an estimate of the brain location of each cluster.  

For detailed information, see the [loading data](./notebooks/one_load/one_load) guide.

### For data sharers

Data standards are only adopted when they are easy to use, for both providers and users of data. 
For users, the three ONE methods will be simple to learn, and will cover most common use cases.

For providers, a key advantage of this framework is its low barrier to entry. To share data with ONE,
providers do not need to run and maintain a backend server, just to upload their data to a website.

For detailed information, see the [data sharing](./notebooks/data_sharing) guide.
