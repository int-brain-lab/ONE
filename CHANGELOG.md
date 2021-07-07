# Changelog

## [Latest](https://github.com/int-brain-lab/ONE/commits/main) [0.2.1]

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
