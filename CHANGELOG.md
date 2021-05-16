# Changelog

## [Latest](https://github.com/int-brain-lab/ONE/commits/main) [beta]

### Added

 - search cache tables without hitting db by default
 - onepqt module to build cache from local filesystem
 - silent mode now suppresses print statements
 - rest GET requests are now cached for seven days
 - alf.spec module for constructing, documenting, and validating ALyx Files
 
### Modified

 - removed load method
 - support for multiple configured databases in params
 - Alyx web token now cached between sessions
 - delayed loading of rest schemes
 - ONE factory cache for fast repeat instantiation
 - alf.io loaders now deal with MATLAB NPYs and expanding timeseries
