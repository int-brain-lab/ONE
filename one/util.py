"""Decorators and small standalone functions for api module"""
from functools import wraps
from typing import Sequence, Union, Iterable

import pandas as pd
from iblutil.io import parquet

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
    TODO Move to converters.py
    """

    @wraps(method)
    def wrapper(self, id, *args, **kwargs):
        id = self.to_eid(id)
        return method(self, id, *args, **kwargs)

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
