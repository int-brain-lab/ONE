import uuid
import json
import logging

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

from one.lib.brainbox.core import Bunch


_logger = logging.getLogger('ibllib')


# TODO Deprecate
def load(filename):
    """
    Loads parquet file into pandas dataframe
    :param filename:
    :return:
    """
    table = pq.read_table(filename)
    try:
        metadata = json.loads(table.schema.metadata[b'one_metadata'])
    except KeyError:
        _logger.debug("No parquet metadata in %s" % filename)
        metadata = {}
    df = table.to_pandas()
    return df, metadata

# TODO Deprecate
def save(filename, table, metadata=None):
    """
    Save pandas dataframe to parquet
    :param filename:
    :param table:
    :param metadata:
    :return:
    """
    # cf https://towardsdatascience.com/saving-metadata-with-dataframes-71f51f558d8e

    # from dataframe to parquet
    table = pa.Table.from_pandas(table)

    # Add user metadata
    table = table.replace_schema_metadata({
        'one_metadata': json.dumps(metadata or {}).encode(),
        **table.schema.metadata
    })

    # Save to parquet.
    pq.write_table(table, filename)


def uuid2np(eids_uuid):
    return np.asfortranarray(
        np.array([np.frombuffer(eid.bytes, dtype=np.int64) for eid in eids_uuid]))


def str2np(eids_str):
    """
    Converts uuid string or list of uuid strings to int64 numpy array with 2 cols
    Returns [0, 0] for None list entries
    """
    if isinstance(eids_str, str):
        eids_str = [eids_str]
    return uuid2np([uuid.UUID(eid) if eid else uuid.UUID('0' * 32) for eid in eids_str])


def np2uuid(eids_np):
    if isinstance(eids_np, pd.DataFrame) | isinstance(eids_np, pd.Series):
        eids_np = eids_np.to_numpy()
    if eids_np.ndim >= 2:
        return [uuid.UUID(bytes=npu.tobytes()) for npu in eids_np]
    else:
        return uuid.UUID(bytes=eids_np.tobytes())


def np2str(eids_np):
    eids = np2uuid(eids_np)
    eids = str(eids) if isinstance(eids, uuid.UUID) else [str(u) for u in np2uuid(eids_np)]
    return eids


def is_np_id(id):
    """
    The purpose of this is to correctly identify ids even as object arrays
    :param id:
    :return:
    """
    # TODO Document and test
    id = np.asarray(id)
    is_int = id.dtype == int or np.all(isinstance(x, int) for x in id)
    return id.shape[1] == 2 and is_int


def rec2col(rec, join=None, include=None, exclude=None, uuid_fields=None, types=None):
    """
    Change a record list (usually from a REST API endpoint) to a column based dictionary
    (pandas dataframe).
    :param rec (list): list of dictionaries with consistent keys
    :param join (dictionary): dictionary of scalar keys that will be replicated over the full
    array (join operation)
    :param include: list of strings representing dictionary keys: if specified will only include
    the keys specified here
    :param exclude: list of strings representing dictionary keys: if specified will exclude the
    keys specified here
    :param uuid_fields: if the field is an UUID, will split it into 2 distinct int64 columns for
    efficient lookups and intersections
    :param types: for a given key, will force the type; example: types = {'file_size': np.double}
    :return: a Bunch
    """
    if isinstance(rec, dict):
        rec = [rec]
    if len(rec) == 0:
        return Bunch()
    if include is None:
        include = rec[0].keys() if isinstance(rec, list) else rec.keys()
    if exclude is None:
        exclude = []
    if uuid_fields is None:
        uuid_fields = []
    if join is None:
        join = {}

    # first loop over the records and create each columns as a numpy array
    nrecs = len(rec)
    col = {}
    keys = [k for k in rec[0] if k in include and k not in exclude]
    for key in keys:
        if key in uuid_fields:
            npuuid = str2np(np.array([c[key] for c in rec]))
            col[f"{key}_0"] = npuuid[:, 0]
            col[f"{key}_1"] = npuuid[:, 1]
        elif types and key in types:
            col[key] = np.array([c[key] for c in rec]).astype(types[key])
        else:
            col[key] = np.array([c[key] for c in rec])

    # then perform the joins if any
    for key in join:
        if key in uuid_fields:
            npuuid = str2np([join[key]])
            col[f"{key}_0"] = np.tile(npuuid[0, 0], (nrecs,))
            col[f"{key}_1"] = np.tile(npuuid[0, 1], (nrecs,))
        else:
            col[key] = np.tile(np.array(join[key]), (nrecs,))

    return Bunch(col)
