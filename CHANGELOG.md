# Changelog
## [Latest](https://github.com/int-brain-lab/ONE/commits/main) [1.12.1]

### Modified

- HOTFIX: default S3 repo set for get_s3_from_alyx

## [1.12.0]

### Added

- remote package for handling file operations with different protocols
- a Globus class for interfacing with the Globus SDK
- an abstract download manager class
- one.remote.aws module to provide low-level access to s3 download functions, both private and public

### Modified

- added clarification for generating cache with an Alyx database in documentation
- debug log of exceptions upon HTTP and connection errors during loading of cache
- added htsv file support in alf.io.load_file_content
- REST cache now stored in download cache directory
- JSON tests may now be run concurrently
- fix'd dimension check for DataFrame attributes in ALF objects
- dimension warning logged when dimensions don't match after appending to ALFBunch
- created_time now updated correctly in cache meta

## [1.11.0]

### Added

- One._update_cache_from_records method for adding remote Alyx records to the cache
- One.save_cache method to save modified cache

### Modified

- path2record now returns a session path when called with session path
- record2url now returns a session URL when called with a sessions record
- fix'd download bar
- attempt to re-authenticate upon 403 invalid token response
- set cache modified timestamp whenever cache tables are modified

## [1.10.0]

### Modified

- cache may be downloaded from a variable location set by the 'location' field returned by cache/info endpoint
- urllib exception now caught in OneAlyx._load_cache
- details dict in remote mode search now contains 'date' field
- fix tests relying on OpenAlyx
- warning instead of error raised when tag_mismatched_dataset REST query returns 403
- list_datasets and ses2records now better handle sessions with no datasets in remote mode

## [1.9.1]

### Added

- tests for OneAlyx._download_aws

### Modified

- HOTFIX: OneAlyx._download_aws now works with new cache index

## [1.9.0]

### Added

- method for recording and save the UUIDs of loaded datasets

### Modified

- fix order of records returned by One.load_datasets when datasets missing

## [1.8.1]

### Modified

- HOTFIX: OneAlyx._download_datasets deals gracefully with empty datasets frame
- removed try-assert-catch logic from One._download_datasets to improve error stack 

## [1.8.0]

### Added

- added `from_df` method to one.alf.io.AlfBunch
- added `__version__` variable
- added check for remote cache minimum API version
- user prompted to verify settings correct in setup

### Modified

- datasets cache table expected to have index of (eid, id).  NB: This changes the order of datasets returned by some functions
- multithreading moved from One._download_datasets to one.webclient.http_download_file_list
- cache_dir kwarg renamed to target_dir in one.webclient.http_download_file
- 'table' attribute now split into columns and merged
- when no username or password provided to constructor, AlyxClient init doesn't call authenticate
- 'stay_logged_in' kwarg removed from AlyxClient constructor; must manually call `authenticate` or remember to call `logout`
- user prompted whether to make url default in setup even if default already set

## [1.7.1]

### Modified

- HOTFIX: failed to return most recent revision; raised MultipleObjectsFound error instead

## [1.7.0]

### Added

- expires kwarg in AlyxClient.rest
- fix for AWS download location

## [1.6.3]

### Added

- ugly hack to download from aws instead of default http server

## [1.6.2]

### Modified

- more readable error message; raw JSON printed to debug logger

## [1.6.1]

### Modified

- rest command loging includes the whole json field on error
- added silent option to instantiate One on local files

## [1.6.0]

### Added

- no_cache function for temporarily deactivating the cache in a one-liner
- fix for setup where wrong client key used
- Alyx URL validation during setup and make default now yes by default

## [1.5.1]

### Modified

- HOTFIX: correct kwarg name in setup documentation; get_default_client includes schema in URL
- minor improvements to documentation and test coverage
- raise ValueError in register_session when lab doesn't match parsed session path

## [1.5.0]

### Modified

- fix bug where filters don't work in remote list_datasets
- change order of kwargs in list_datasets: filename now the first kwarg
- can now filter by list of filename strings functioning as a logical OR
- dataset kwarg renamed to filename in list_revisions
- fix ALF regular expression pattern: attribute, timescale and extension now parsed correctly
- can now filter datasets by timescale
- clearer error message auth errors raised
- alyx client is_logged_in method now a dependent property

## [1.4.0]

### Modified

- One and OneAlyx setup methods
- old params files now cleaned up
- removed ALYX_PWD prompt from setup
- improved test coverage
- docs formatting
- One list_* methods return list instead of np arrays
- get_details returns dict with 'date' key in remote mode
- eid2ref works in offline mode
- record2url now expects a pandas.Series object, iterates over DataFrame and returns list
- path2record now returns a pandas.Series instead of DataFrame
- _update_filesystem renamed to _check_filesystem
- _index_type now accepts DataFrame as table input
- better support for string indices in path2url and _download_dataset
- fix for _check_filesystem with datasets dict input

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
