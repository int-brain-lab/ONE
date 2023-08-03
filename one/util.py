"""Decorators and small standalone functions for api module."""
import logging
import urllib.parse
from functools import wraps
from typing import Sequence, Union, Iterable, Optional, List
from collections.abc import Mapping
import fnmatch
from datetime import datetime

import pandas as pd
from iblutil.io import parquet
import numpy as np
from packaging import version

import one.alf.exceptions as alferr
from one.alf.files import rel_path_parts, get_session_path, get_alf_path, remove_uuid_string
from one.alf.spec import FILE_SPEC, regex as alf_regex

logger = logging.getLogger(__name__)


def Listable(t):
    """Return a typing.Union if the input and sequence of input."""
    return Union[t, Sequence[t]]


def ses2records(ses: dict):
    """Extract session cache record and datasets cache from a remote session data record.

    Parameters
    ----------
    ses : dict
        Session dictionary from Alyx REST endpoint.

    Returns
    -------
    pd.Series
        Session record.
    pd.DataFrame
        Datasets frame.
    """
    # Extract session record
    eid = ses['url'][-36:]
    session_keys = ('subject', 'start_time', 'lab', 'number', 'task_protocol', 'projects')
    session_data = {k: v for k, v in ses.items() if k in session_keys}
    session = (
        pd.Series(data=session_data, name=eid).rename({'start_time': 'date'})
    )
    session['projects'] = ','.join(session.pop('projects'))
    session['date'] = datetime.fromisoformat(session['date']).date()

    # Extract datasets table
    def _to_record(d):
        rec = dict(file_size=d['file_size'], hash=d['hash'], exists=True)
        rec['id'] = d['id']
        rec['eid'] = session.name
        file_path = urllib.parse.urlsplit(d['data_url'], allow_fragments=False).path.strip('/')
        file_path = get_alf_path(remove_uuid_string(file_path))
        rec['session_path'] = get_session_path(file_path).as_posix()
        rec['rel_path'] = file_path[len(rec['session_path']):].strip('/')
        rec['default_revision'] = d['default_revision'] == 'True'
        return rec

    if not ses.get('data_dataset_session_related'):
        return session, pd.DataFrame()
    records = map(_to_record, ses['data_dataset_session_related'])
    index = ['eid', 'id']
    datasets = pd.DataFrame(records).set_index(index).sort_index()
    return session, datasets


def datasets2records(datasets, additional=None) -> pd.DataFrame:
    """Extract datasets DataFrame from one or more Alyx dataset records.

    Parameters
    ----------
    datasets : dict, list
        One or more records from the Alyx 'datasets' endpoint.
    additional : list of str
        A set of optional fields to extract from dataset records.

    Returns
    -------
    pd.DataFrame
        Datasets frame.

    Examples
    --------
    >>> datasets = ONE().alyx.rest('datasets', 'list', subject='foobar')
    >>> df = datasets2records(datasets)
    """
    records = []

    for d in ensure_list(datasets):
        file_record = next((x for x in d['file_records'] if x['data_url'] and x['exists']), None)
        if not file_record:
            continue  # Ignore files that are not accessible
        rec = dict(file_size=d['file_size'], hash=d['hash'], exists=True)
        rec['id'] = d['url'][-36:]
        rec['eid'] = (d['session'] or '')[-36:]
        data_url = urllib.parse.urlsplit(file_record['data_url'], allow_fragments=False)
        file_path = get_alf_path(data_url.path.strip('/'))
        file_path = remove_uuid_string(file_path).as_posix()
        rec['session_path'] = get_session_path(file_path) or ''
        if rec['session_path']:
            rec['session_path'] = rec['session_path'].as_posix()
        rec['rel_path'] = file_path[len(rec['session_path']):].strip('/')
        rec['default_revision'] = d['default_dataset']
        for field in additional or []:
            rec[field] = d.get(field)
        records.append(rec)

    index = ['eid', 'id']
    if not records:
        keys = (*index, 'file_size', 'hash', 'session_path', 'rel_path', 'default_revision')
        return pd.DataFrame(columns=keys).set_index(index)
    return pd.DataFrame(records).set_index(index).sort_index()


def parse_id(method):
    """
    Ensures the input experiment identifier is an experiment UUID string.

    Parameters
    ----------
    method : function
        An ONE method whose second arg is an experiment ID.

    Returns
    -------
    function
        A wrapper function that parses the ID to the expected string.

    Raises
    ------
    ValueError
        Unable to convert input to a valid experiment ID.
    """

    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        eid = self.to_eid(id)
        if eid is None:
            raise ValueError(f'Cannot parse session ID "{id}" (session may not exist)')
        return method(self, eid, *args, **kwargs)

    return wrapper


def refresh(method):
    """
    Refresh cache depending of query_type kwarg.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        mode = kwargs.get('query_type', None)
        if not mode or mode == 'auto':
            mode = self.mode
        self.refresh_cache(mode=mode)
        return method(self, *args, **kwargs)

    return wrapper


def validate_date_range(date_range) -> (pd.Timestamp, pd.Timestamp):
    """
    Validates and arrange date range in a 2 elements list.

    Parameters
    ----------
    date_range : str, datetime.date, datetime.datetime, pd.Timestamp, np.datetime64, list, None
        A single date or tuple/list of two dates.  None represents no bound.

    Returns
    -------
    tuple of pd.Timestamp
        The start and end timestamps.

    Examples
    --------
    >>> validate_date_range('2020-01-01')  # On this day
    >>> validate_date_range(datetime.date(2020, 1, 1))
    >>> validate_date_range(np.array(['2022-01-30', '2022-01-30'], dtype='datetime64[D]'))
    >>> validate_date_range(pd.Timestamp(2020, 1, 1))
    >>> validate_date_range(np.datetime64(2021, 3, 11))
    >>> validate_date_range(['2020-01-01'])  # from date
    >>> validate_date_range(['2020-01-01', None])  # from date
    >>> validate_date_range([None, '2020-01-01'])  # up to date

    Raises
    ------
    ValueError
        Size of date range tuple must be 1 or 2.
    """
    if date_range is None:
        return

    # Ensure we have exactly two values
    if isinstance(date_range, str) or not isinstance(date_range, Iterable):
        # date_range = (date_range, pd.Timestamp(date_range) + pd.Timedelta(days=1))
        dt = pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        date_range = (date_range, pd.Timestamp(date_range) + dt)
    elif len(date_range) == 1:
        date_range = [date_range[0], pd.Timestamp.max]
    elif len(date_range) != 2:
        raise ValueError

    # For comparisons, ensure both values are pd.Timestamp (datetime, date and datetime64
    # objects will be converted)
    start, end = date_range
    start = start or pd.Timestamp.min  # Convert None to lowest possible date
    end = end or pd.Timestamp.max  # Convert None to highest possible date

    # Convert to timestamp
    if not isinstance(start, pd.Timestamp):
        start = pd.Timestamp(start)
    if not isinstance(end, pd.Timestamp):
        end = pd.Timestamp(end)

    return start, end


def _collection_spec(collection=None, revision=None) -> str:
    """
    Return a template string for a collection/revision regular expression.  Because both are
    optional in the ALF spec, None will match any (including absent), while an empty string will
    match absent.

    Parameters
    ----------
    collection : None, str
        An optional collection regular expression.
    revision : None, str
        An optional revision regular expression.

    Returns
    -------
    str
        A string format for matching the collection/revision.
    """
    spec = ''
    for value, default in zip((collection, revision), ('{collection}/', '#{revision}#/')):
        if not value:
            default = f'({default})?' if value is None else ''
        spec += default
    return spec


def _file_spec(**kwargs):
    """
    Return a template string for a ALF dataset regular expression.  Because 'namespace',
    'timescale', and 'extra' are optional None will match any (including absent).  This function
    removes the regex flags from the file spec string that make certain parts optional.

    TODO an empty string should only match absent; this could be achieved by removing parts from
     spec string

    Parameters
    ----------
    namespace : None, str
        If namespace is not None, the namespace section of the returned file spec will not be
        optional.
    timescale : None, str
        If timescale is not None, the namespace section of the returned file spec will not be
        optional.
    extra : None, str
        If extra is not None, the namespace section of the returned file spec will not be
        optional.

    Returns
    -------
    str
        A string format for matching an ALF dataset.
    """
    OPTIONAL = {'namespace': '?', 'timescale': '?', 'extra': '*'}
    filespec = FILE_SPEC
    for k, v in kwargs.items():
        if k in OPTIONAL and v is not None:
            i = filespec.find(k) + len(k)
            i += filespec[i:].find(OPTIONAL[k])
            filespec = filespec[:i] + filespec[i:].replace(OPTIONAL[k], '', 1)
    return filespec


def filter_datasets(all_datasets, filename=None, collection=None, revision=None,
                    revision_last_before=True, assert_unique=True, wildcards=False):
    """
    Filter the datasets cache table by the relative path (dataset name, collection and revision).
    When None is passed, all values will match.  To match on empty parts, use an empty string.
    When revision_last_before is true, None means return latest revision.

    Parameters
    ----------
    all_datasets : pandas.DataFrame
        A datasets cache table.
    filename : str, dict, None
        A filename str or a dict of alf parts.  Regular expressions permitted.
    collection : str, None
        A collection string.  Regular expressions permitted.
    revision : str, None
        A revision string to match.  If revision_last_before is true, regular expressions are
        not permitted.
    revision_last_before : bool
        When true and no exact match exists, the (lexicographically) previous revision is used
        instead.  When false the revision string is matched like collection and filename,
        with regular expressions permitted.
    assert_unique : bool
        When true an error is raised if multiple collections or datasets are found.
    wildcards : bool
        If true, use unix shell style matching instead of regular expressions.

    Returns
    -------
    pd.DataFrame
        A slice of all_datasets that match the filters.

    Examples
    --------
    Filter by dataset name and collection

    >>> datasets = filter_datasets(all_datasets, '.*spikes.times.*', 'alf/probe00')

    Filter datasets not in a collection

    >>> datasets = filter_datasets(all_datasets, collection='')

    Filter by matching revision

    >>> datasets = filter_datasets(all_datasets, 'spikes.times.npy',
    ...                            revision='2020-01-12', revision_last_before=False)

    Filter by filename parts

    >>> datasets = filter_datasets(all_datasets, dict(object='spikes', attribute='times'))

    Notes
    -----
    - It is not possible to match datasets that are in a given collection OR NOT in ANY collection.
      e.g. filter_datasets(dsets, collection=['alf', '']) will not match the latter. For this you
      must use two separate queries.
    """
    # Create a regular expression string to match relative path against
    filename = filename or {}
    regex_args = {'collection': collection}
    spec_str = _collection_spec(collection, None if revision_last_before else revision)

    if isinstance(filename, dict):
        spec_str += _file_spec(**filename)
        regex_args.update(**filename)
    else:
        # Convert to regex is necessary and assert end of string
        filename = [fnmatch.translate(x) if wildcards else x + '$' for x in ensure_list(filename)]
        spec_str += '|'.join(filename)

    # If matching revision name, add to regex string
    if not revision_last_before:
        regex_args.update(revision=revision)

    for k, v in regex_args.items():
        if v is None:
            continue
        if wildcards:
            # Convert to regex, remove \\Z which asserts end of string
            v = (fnmatch.translate(x).replace('\\Z', '') for x in ensure_list(v))
        if not isinstance(v, str):
            regex_args[k] = '|'.join(v)  # logical OR

    # Build regex string
    pattern = alf_regex('^' + spec_str, **regex_args)
    match = all_datasets[all_datasets['rel_path'].str.match(pattern)]
    if len(match) == 0 or not (revision_last_before or assert_unique):
        return match

    revisions = [rel_path_parts(x)[1] or '' for x in match.rel_path.values]
    if assert_unique:
        collections = set(rel_path_parts(x)[0] or '' for x in match.rel_path.values)
        if len(collections) > 1:
            _list = '"' + '", "'.join(collections) + '"'
            raise alferr.ALFMultipleCollectionsFound(_list)
        if not revision_last_before:
            if filename and len(match) > 1:
                _list = '"' + '", "'.join(match['rel_path']) + '"'
                raise alferr.ALFMultipleObjectsFound(_list)
            if len(set(revisions)) > 1:
                _list = '"' + '", "'.join(set(revisions)) + '"'
                raise alferr.ALFMultipleRevisionsFound(_list)
            else:
                return match
        elif filename and len(set(revisions)) != len(revisions):
            _list = '"' + '", "'.join(match['rel_path']) + '"'
            raise alferr.ALFMultipleObjectsFound(_list)

    return filter_revision_last_before(match, revision, assert_unique=assert_unique)


def filter_revision_last_before(datasets, revision=None, assert_unique=True):
    """
    Filter datasets by revision, returning previous revision in ordered list if revision
    doesn't exactly match.

    Parameters
    ----------
    datasets : pandas.DataFrame
        A datasets cache table.
    revision : str
        A revision string to match (regular expressions not permitted).
    assert_unique : bool
        When true an alferr.ALFMultipleRevisionsFound exception is raised when multiple
        default revisions are found; an alferr.ALFError when no default revision is found.

    Returns
    -------
    pd.DataFrame
        A datasets DataFrame with 0 or 1 row per unique dataset.
    """
    def _last_before(df):
        """Takes a DataFrame with only one dataset and multiple revisions, returns matching row"""
        if revision is None and 'default_revision' in df.columns:
            if assert_unique and sum(df.default_revision) > 1:
                revisions = df['revision'][df.default_revision.values]
                rev_list = '"' + '", "'.join(revisions) + '"'
                raise alferr.ALFMultipleRevisionsFound(rev_list)
            if sum(df.default_revision) == 1:
                return df[df.default_revision]
            if len(df) == 1:  # This may be the case when called from load_datasets
                return df  # It's not the default be there's only one available revision
            # default_revision column all False; default isn't copied to remote repository
            dset_name = df['rel_path'].iloc[0]
            if assert_unique:
                raise alferr.ALFError(f'No default revision for dataset {dset_name}')
            else:
                logger.warning(f'No default revision for dataset {dset_name}; using most recent')
        # Compare revisions lexicographically
        if assert_unique and len(df['revision'].unique()) > 1:
            rev_list = '"' + '", "'.join(df['revision'].unique()) + '"'
            raise alferr.ALFMultipleRevisionsFound(rev_list)
        # Square brackets forces 1 row DataFrame returned instead of Series
        idx = index_last_before(df['revision'].tolist(), revision)
        # return df.iloc[slice(0, 0) if idx is None else [idx], :]
        return df.iloc[slice(0, 0) if idx is None else [idx], :]

    with pd.option_context('mode.chained_assignment', None):  # FIXME Explicitly copy?
        datasets['revision'] = [rel_path_parts(x)[1] or '' for x in datasets.rel_path]
    groups = datasets.rel_path.str.replace('#.*#/', '', regex=True).values
    grouped = datasets.groupby(groups, group_keys=False)
    return grouped.apply(_last_before)


def index_last_before(revisions: List[str], revision: Optional[str]) -> Optional[int]:
    """
    Returns the index of string that occurs directly before the provided revision string when
    lexicographic sorted.  If revision is None, the index of the most recent revision is returned.

    Parameters
    ----------
    revisions : list of strings
        A list of revision strings.
    revision : None, str
        The revision string to match on.

    Returns
    -------
    int, None
        Index of revision before matching string in sorted list or None.

    Examples
    --------
    >>> idx = index_last_before([], '2020-08-01')
    """
    if len(revisions) == 0:
        return  # No revisions, just return
    revisions_sorted = sorted(revisions, reverse=True)
    if revision is None:  # Return most recent revision
        return revisions.index(revisions_sorted[0])
    lt = np.array(revisions_sorted) <= revision
    return revisions.index(revisions_sorted[lt.argmax()]) if any(lt) else None


def autocomplete(term, search_terms) -> str:
    """
    Validate search term and return complete name, e.g. autocomplete('subj') == 'subject'.
    """
    term = term.lower()
    # Check if term already complete
    if term in search_terms:
        return term
    full_key = (x for x in search_terms if x.lower().startswith(term))
    key_ = next(full_key, None)
    if not key_:
        raise ValueError(f'Invalid search term "{term}", see `one.search_terms()`')
    elif next(full_key, None):
        raise ValueError(f'Ambiguous search term "{term}"')
    return key_


def ensure_list(value):
    """Ensure input is a list."""
    return [value] if isinstance(value, (str, dict)) or not isinstance(value, Iterable) else value


class LazyId(Mapping):
    """
    Using a paginated response object or list of session records, extracts eid string when required
    """
    def __init__(self, pg, func=None):
        self._pg = pg
        self.func = func or self.ses2eid

    def __getitem__(self, item):
        return self.func(self._pg.__getitem__(item))

    def __len__(self):
        return self._pg.__len__()

    def __iter__(self):
        return map(self.func, self._pg.__iter__())

    @staticmethod
    def ses2eid(ses):
        """Given one or more session dictionaries, extract and return the session UUID.

        Parameters
        ----------
        ses : one.webclient._PaginatedResponse, dict, list
            A collection of Alyx REST sessions endpoint records.

        Returns
        -------
        str, list
            One or more experiment ID strings.
        """
        if isinstance(ses, list):
            return [LazyId.ses2eid(x) for x in ses]
        else:
            return ses.get('id', None) or ses['url'].split('/').pop()


def cache_int2str(table: pd.DataFrame) -> pd.DataFrame:
    """Convert int ids to str ids for cache table.

    Parameters
    ----------
    table : pd.DataFrame
        A cache table (from One._cache).

    """
    # Convert integer uuids to str uuids
    if table.index.nlevels < 2 or not any(x.endswith('_0') for x in table.index.names):
        return table
    table = table.reset_index()
    int_cols = table.filter(regex=r'_\d{1}$').columns.sort_values()
    assert not len(int_cols) % 2, 'expected even number of columns ending in _0 or _1'
    names = sorted(set(c.rsplit('_', 1)[0] for c in int_cols.values))
    for i, name in zip(range(0, len(int_cols), 2), names):
        table[name] = parquet.np2str(table[int_cols[i:i + 2]])
    table = table.drop(int_cols, axis=1).set_index(names)
    return table


def patch_cache(table: pd.DataFrame, min_api_version=None) -> pd.DataFrame:
    """Reformat older cache tables to comply with this version of ONE.

    Currently this function will 1. convert integer UUIDs to string UUIDs; 2. rename the 'project'
    column to 'projects'.

    Parameters
    ----------
    table : pd.DataFrame
        A cache table (from One._cache).
    min_api_version : str
        The minimum API version supported by this cache table.
    """
    min_version = version.parse(min_api_version or '0.0.0')
    table = cache_int2str(table)
    # Rename project column
    if min_version < version.Version('1.13.0') and 'project' in table.columns:
        table.rename(columns={'project': 'projects'}, inplace=True)
    return table
