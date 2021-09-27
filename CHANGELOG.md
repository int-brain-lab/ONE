# Changelog
## [Latest](https://github.com/int-brain-lab/ONE/commits/main)  [1.4.0]

### Modified

- One and OneAlyx setup methods
- Old params files now cleaned up
- Removed ALYX_PWD prompt from setup
- Improved test coverage
- Docs formatting
- One list_* methods return list instead of np arrays
- get_details returns dict with 'date' key in remote mode
- eid2ref works in offline mode
- record2url now expects a pandas.Series object, iterates over DataFrame and returns list
- path2record now returns a pandas.Series instead of DataFrame

### Added

- tests for eid2ref
- load_collection method for loading a Bunch of ALF objects

## [1.3.0]

### Modified

- propagate down filter datasets filters
- OneAlyx list_* methods return list instead of np arrays
- cache module gracefully deals with empty repos
- cache module refactored to reuse ALF functions
- cache str ids now exclude 'Subjects' part
- session_path_parts now accepts pathlib objects
- one.params.check_cache_conflict now public function
- One.cache_dir now public property
- cache args preserved between calls to paginated response object
- runtime warning when remote list count changes

### Added
- alf.io function to return valid session paths within a directory 

## [1.2.1]

### Modified

- HOTFIX: correct exception raise when files missing in load_datasets

## [1.2.0]

### Added

- registration module with functions for posting sessions and file records to Alyx

### Modified

- bumped minimum pandas version in requirements
- REST cache supports URL with port
- describe revision now supported
- ALF spec now requires 'Subjects' folder in order to parse lab, i.e. .../lab/Subjects/subjects/...
- password prompt now specifies user
- clearer REST HTTP response error messages

## [1.1.0]

### Added
- extension may be omitted when loading dataset in wildcard mode if dataset doesn't have extra parts

## [1.0.0]

### Modified
- removed deprecated `_from_` converters
- removed walrus from test
- raise warning when fails to set dataset exists to False

## [0.5.3]

### Modified

- HOTFIX: error no longer raised when logging dimension mismatch in alf.io.load_object

## [0.5.2]

### Modified

- HOTFIX: ref2dj no longer raises error

## [0.5.1]

### Modified

- HOTFIX: handles case when file_size is None in _download_dataset

## [0.5.0]

### Modified

- consistent regex file pattern between functions
- unix shell style wildcards used by default
- limited support for attribute namespace for backward compatibility
- can filter with lists of parts, e.g. `extension=['npy', '?sv']`

## [0.4.0]

- alf package refactored for module consistency; removed alf.folders

## [0.3.0]

### Added

- function to convert datasets list endpoint to DataFrame
- added logout method; 'authenticate' now prompts for Alyx password if none provided and no token

### Modified

- fully adopted Numpy docstrings
- revisions now filtered using <= instead of <
- datasets_from_type now type2datasets; returns similar output to list_ methods
- removed ALYX_PWD from setup
- webclient functions sdsc_globus_path_from_dataset, sdsc_path_from_dataset and globus_path_from_dataset moved to ibllib

## [0.2.3]

### Modified

- HOTFIX: default query_mode now None, fixes missing instance mode
- HOTFIX: correct ses2rec parsing of data URLs (allow_fragments=False)

## [0.2.2]

### Modified

- HOTFIX: returns correct index when revision returns empty for dataset

## [0.2.1]

### Modified

- webclient encodes characters in URL query parameters
- docstrings added to ONE factory and classes
- HOTFIX: list_datasets passes eid in superclass call

## [0.2.0]

### Added

- list_collections and list_revisions methods
- added example scripts on experiment ids and ALF spec
- search_terms now has query type flag

### Modified

- removed walruses (support for python 3.7)
- fixed bugs in various converter methods
- fix for silent setup when base url provided
- deals with default_revision all false
- one.search works with remote alyx queries

## [0.1.5]

### Added

- configurable test databases
- configurable REST cache mode and default expiry

### Modified

- test image upload without matplotlib
- clearer One.search docstring
- make_parquet_db date column now datetime.date dtype

## [0.1.4]

### Added

- proper support for loading revisions in local mode
- separate function for filtering by collection
- tests for cache table filter functions

### Modified

- refactored load methods to use the same filter functions

## 0.1.3

### Added

- a load_datasets method for loading multiple datasets at once

## 0.1.2

### Modified

 - REST headers no longer used in download_file, fixes behaviour for 401 errors
 - fix to download_dataset error with alyx record; tests added
 - api utils moved to new module

## 0.1.1

### Modified

 - fix setting default ALYX_URL in setup
 - fix for updating missing records in cache

## 0.1.0

### Added

 - search cache tables without hitting db by default
 - onepqt module to build cache from local filesystem
 - silent mode now suppresses print statements
 - rest GET requests are now cached for 24 hours
 - alf.spec module for constructing, documenting, and validating ALyx Files
 
### Modified

 - removed load method
 - support for multiple configured databases in params
 - Alyx web token now cached between sessions
 - delayed loading of rest schemes
 - ONE factory cache for fast repeat instantiation
 - alf.io loaders now deal with MATLAB NPYs and expanding timeseries
