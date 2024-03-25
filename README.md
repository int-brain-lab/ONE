# Open Neurophysiology Environment
[![Coverage Status](https://coveralls.io/repos/github/int-brain-lab/ONE/badge.svg?branch=main)](https://coveralls.io/github/int-brain-lab/ONE?branch=main)
![CI workflow](https://github.com/int-brain-lab/ONE/actions/workflows/main.yaml/badge.svg?branch=main)

The Open Neurophysiology Environment is a scheme for sharing neurophysiology data in a standardized manner. It is a Python API for searching and loading ONE-standardized data, stored either on a user's local machine or on a remote server.

Please [Click here](https://int-brain-lab.github.io/ONE/) for the main documentation page.  For a quick primer on the file naming convention we use, [click here](https://github.com/int-brain-lab/ONE/blob/main/docs/Open_Neurophysiology_Environment_Filename_Convention.pdf).

**NB**: The API and backend database are still under active development, for the best experience please regularly update the package by running `pip install -U ONE-api`. 

## Requirements
ONE runs on Python 3.8 or later, and is tested on the latest Ubuntu and Windows (3.8 and 3.11 only).

## Installing
Installing the package via pip typically takes a few seconds.  To install, run
```
pip install ONE-api
```

## Set up
For using ONE with a local cache directory:
```python
from one.api import One
one = One(cache_dir='/home/user/downlaods/ONE/behavior_paper')
```

To use the default setup settings that connect you to the [IBL public database](https://openalyx.internationalbrainlab.org):
```python
from one.api import ONE
ONE.setup(silent=True)  # Will use default information
one = ONE(password='international')
```

For setting up ONE for a given database e.g. internal IBL Alyx:
```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```

Once you've setup the API for the first time, subsequent calls will use the same parameters:
```python
from one.api import ONE
one = ONE()
```

To set up ONE for another database and make it the default:
```python
from one.api import ONE
ONE.setup(base_url='https://test.alyx.internationalbrainlab.org', make_default=True)
one = ONE()  # Connected to https://test.alyx.internationalbrainlab.org
```

## Using ONE
To search for sessions:
```python
from one.api import ONE
one = ONE()
print(one.search_terms())  # A list of search keyword arguments

# Search session with wheel timestamps from January 2021 onward
eids = one.search(date_range=['2021-01-01',], dataset='wheel.timestamps')
['d3372b15-f696-4279-9be5-98f15783b5bb']

# Search for project sessions with two probes
eids = one.search(data=['probe00', 'probe01'], project='brainwide')
```

To load data:
```python
from one.api import ONE
one = ONE()

# Load an ALF object
eid = 'a7540211-2c60-40b7-88c6-b081b2213b21'
wheel = one.load_object(eid, 'wheel')

# Load a specific dataset
eid = 'a7540211-2c60-40b7-88c6-b081b2213b21'
ts = one.load_dataset(eid, 'wheel.timestamps', collection='alf')

# Download, but not load, a dataset
filename = one.load_dataset(eid, 'wheel.timestamps', download_only=True)
```

To [share data](https://int-brain-lab.github.io/ONE/notebooks/data_sharing.html):
```python
from one.api import One
one = One.setup()  # Enter the location of the ALF datasets when prompted
```

Further examples and tutorials can be found in the [documentation](https://int-brain-lab.github.io/ONE/).
