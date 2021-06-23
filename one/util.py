"""Decorators and small standalone functions for api module"""
from functools import wraps
from typing import Sequence, Union, Iterable, Optional, List

import pandas as pd
from iblutil.io import parquet
import numpy as np

import one.alf.exceptions as alferr
from one.alf.files import rel_path_parts
from one.alf.spec import FILE_SPEC, regex as alf_regex
import one.alf.io as alfio


def Listable(t):
    return Union[t, Sequence[t]]


def ses2records(ses: dict) -> [pd.Series, pd.DataFrame]:
    """Extract session cache record and datasets cache from a remote session data record
    TODO Fix for new tables; use to update caches from remote queries
    :param ses: session dictionary from rest endpoint
    :return: session record, datasets frame
    """
    # Extract session record
    eid = parquet.str2np(ses['url'][-36:])
    session_keys = ('subject', 'start_time', 'lab', 'number', 'task_protocol', 'project')
    session_data = {k: v for k, v in ses.items() if k in session_keys}
    # session_data['id_0'], session_data['id_1'] = eid.flatten().tolist()
    session = (
        (pd.Series(data=session_data, name=tuple(eid.flatten()))
            .rename({'start_time': 'date'}, axis=1))
    )
    session['date'] = session['date'][:10]

    # Extract datasets table
    def _to_record(d):
        rec = dict(file_size=d['file_size'], hash=d['hash'], exists=True)
        rec['id_0'], rec['id_1'] = parquet.str2np(d['id']).flatten().tolist()
        rec['eid_0'], rec['eid_1'] = session.name
        file_path = alfio.get_alf_path(d['data_url'])
        rec['session_path'] = alfio.get_session_path(file_path).as_posix()
        rec['rel_path'] = file_path[len(rec['session_path']):].strip('/')
        return rec

    records = map(_to_record, ses['data_dataset_session_related'])
    datasets = pd.DataFrame(records).set_index(['id_0', 'id_1'])
    return session, datasets


def parse_id(method):
    """
    Ensures the input experiment identifier is an experiment UUID string
    :param method: An ONE method whose second arg is an experiment id
    :return: A wrapper function that parses the id to the expected string
    """

    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        eid = self.to_eid(id)
        if eid is None:
            raise ValueError(f'Cannot parse session ID {id}')
        return method(self, eid, *args, **kwargs)

    return wrapper


def refresh(method):
    """
    Refresh cache depending of query_type kwarg
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        mode = kwargs.get('query_type', None)
        if not mode or mode == 'auto':
            mode = self.mode
        self.refresh_cache(mode=mode)
        return method(self, *args, **kwargs)

    return wrapper


def validate_date_range(date_range):
    """
    Validates and arrange date range in a 2 elements list

    Examples:
        _validate_date_range('2020-01-01')  # On this day
        _validate_date_range(datetime.date(2020, 1, 1))
        _validate_date_range(np.array(['2022-01-30', '2022-01-30'], dtype='datetime64[D]'))
        _validate_date_range(pd.Timestamp(2020, 1, 1))
        _validate_date_range(np.datetime64(2021, 3, 11))
        _validate_date_range(['2020-01-01'])  # from date
        _validate_date_range(['2020-01-01', None])  # from date
        _validate_date_range([None, '2020-01-01'])  # up to date
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


def _collection_spec(collection=None, revision=None):
    """
    Return a template string for a collection/revision regular expression.  Because both are
    optional in the ALF spec, None will match any (including absent), while an empty string will
    match absent.

    :param collection:
    :param revision:
    :return: a string format for matching the collection/revision
    """
    spec = ''
    for value, default in zip((collection, revision), ('{collection}/', '#{revision}#/')):
        if not value:
            default = f'({default})?' if value is None else ''
        spec += default
    return spec


def filter_datasets(all_datasets, filename=None, collection=None, revision=None,
                    revision_last_before=True, assert_unique=True):
    """
    Filter the datasets cache table by the relative path (dataset name, collection and revision).
    When None is passed, all values will match.  To match on empty parts, use an empty string.
    When revision_last_before is true, None means return latest revision.

    Examples:
        # Filter by dataset name and collection
        datasets = filter_datasets(all_datasets, '*.spikes.times.*', 'alf/probe00')
        # Filter datasets not in a collection
        datasets = filter_datasets(all_datasets, collection='')
        # Filter by matching revision
        datasets = filter_datasets(all_datasets, 'spikes.times.npy',
                                   revision='2020-01-12', revision_last_before=False)
        # Filter by filename parts
        datasets = filter_datasets(all_datasets, {object='spikes', attribute='times'})

    :param all_datasets: a datasets cache table
    :param filename: a regex string or a dict of alf parts
    :param collection: a regex string
    :param revision: a regex string
    :param revision_last_before: if true, the datasets are filtered by the last revision before
    the given revision string when ordered lexicographically, otherwise the revision is matched
    like the other strings
    :param assert_unique: raise an error with more than one dataset matches the filters
    :return: a slice of all_datasets that match the filters
    """
    # Create a regular expression string to match relative path against
    filename = filename or {}
    regex_args = {'collection': collection}
    spec_str = _collection_spec(collection, None if revision_last_before else revision)

    if isinstance(filename, dict):
        spec_str += FILE_SPEC
        regex_args.update(**filename)
    else:
        spec_str += filename + '$'  # Assert end of string

    # If matching revision name, add to regex string
    if not revision_last_before:
        regex_args.update(revision=revision)

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
        if filename and len(match) > 1:
            _list = '"' + '", "'.join(match['rel_path']) + '"'
            raise alferr.ALFMultipleObjectsFound(_list)
        if not revision_last_before:
            if len(set(revisions)) > 1:
                _list = '"' + '", "'.join(set(revisions)) + '"'
                raise alferr.ALFMultipleRevisionsFound(_list)
            else:
                return match

    return filter_revision_last_before(match, revision, assert_unique=assert_unique)


def filter_revision_last_before(datasets, revision=None, assert_unique=True):
    def _last_before(df):
        if revision is None and 'default_revision' in df.columns:
            if assert_unique and sum(df.default_revision) > 1:
                revisions = df['revision'][df.default_revision.values]
                rev_list = '"' + '", "'.join(revisions) + '"'
                raise alferr.ALFMultipleRevisionsFound(rev_list)
            return df[df.default_revision]
        else:  # Compare revisions lexicographically
            if assert_unique and len(df['revision'].unique()) > 1:
                rev_list = '"' + '", "'.join(df['revision'].unique()) + '"'
                raise alferr.ALFMultipleRevisionsFound(rev_list)
            # Square brackets forces 1 row DataFrame returned instead of Series
            idx = _index_last_before(df['revision'].tolist(), revision)
            return df.iloc[slice(0, 0) if idx is None else [idx], :]

    with pd.option_context('mode.chained_assignment', None):  # FIXME Explicitly copy?
        datasets['revision'] = [rel_path_parts(x)[1] or '' for x in datasets.rel_path]
    groups = datasets.rel_path.str.replace('#.*#/', '', regex=True).values
    grouped = datasets.groupby(groups)
    return grouped.apply(_last_before)  # .drop('revision', axis=1)


def _index_last_before(revisions: List[str], revision: Optional[str]) -> Optional[int]:
    """
    Returns the index of string that occurs directly before the provided revision string when
    lexicographic sorted.  If revision is None, the index of the most recent revision is returned.

    Example:
        idx = _index_last_before([], '2020-08-01')

    :param revisions: a list of revision strings
    :param revision: revision string to match on
    :return: index of revision before matching string in sorted list or None
    """
    if len(revisions) == 0:
        return  # No revisions, just return
    revisions_sorted = sorted(revisions, reverse=True)
    if revision is None:  # Return most recent revision
        return revisions.index(revisions_sorted[0])
    lt = np.array(revisions_sorted) < revision
    return revisions.index(revisions_sorted[lt.argmax()]) if any(lt) else None