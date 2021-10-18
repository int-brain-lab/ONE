# Open Neurophysiology Environment
[![Coverage Status](https://coveralls.io/repos/github/int-brain-lab/ONE/badge.svg?branch=main)](https://coveralls.io/github/int-brain-lab/ONE?branch=main)
![CI workflow](https://github.com/int-brain-lab/ONE/actions/workflows/main.yaml/badge.svg?branch=main)

[Click here](https://int-brain-lab.github.io/ONE/) for the main documentation page.

## Installing
For Python 3.7 or later, run
```
pip install ONE-api
```

## Set up
For using ONE with a local cache directory:
```python
from one.api import One
one = One(cache_dir='/home/user/downlaods/ONE/behavior_paper')
```

For setting up ONE for a given database e.g. internal IBL Alyx:
```python
from one.api import ONE
one = ONE(base_url='https://alyx.internationalbrainlab.org')
```

To use the default setup settings that connect you to the [IBL public database](https://openalyx.internationalbrainlab.org):
```python
from one.api import ONE
one = ONE(silent=True, password='international')  # Will use default information
```

Once you've setup the server, subsequent calls will use the same parameters:
```python
from one.api import ONE
one = ONE()
```

To set up ONE for another database and make it the default:
```python
from one.api import OneAlyx, ONE
OneAlyx.setup(base_url='https://test.alyx.internationalbrainlab.org', make_default=True)
one = ONE()  # Connected to https://test.alyx.internationalbrainlab.org
```

## Using ONE
To search for sessions:
```python
from one.api import ONE
one = ONE()
print(one.search_terms)  # A list of search keyword arguments

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

Further examples and tutorials can be found in the documentation.
