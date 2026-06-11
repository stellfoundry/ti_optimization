import os
import time

import numpy as np
from filelock import FileLock


def copy_params_dict(params_dict):
    """Return a DB-safe copy of a DESC params_dict."""
    return {
        key: np.array(value, copy=True)
        for key, value in params_dict.items()
    }


# def load_DB_safely(database_name):
#     lock = FileLock(database_name + ".lock")
#     with lock:
#         if os.path.exists(database_name):
#             return np.load(database_name, allow_pickle=True)
#         return np.array([], dtype=object)
# 
# 
# def append_DB_safely(database_name, to_save):
#     lock = FileLock(database_name + ".lock")
# 
#     with lock:
#         if os.path.exists(database_name):
#             DB = np.load(database_name, allow_pickle=True)
#         else:
#             DB = np.array([], dtype=object)
# 
#         DB = np.append(DB, to_save)
# 
#         tmp_name = database_name + f".tmp.{os.getpid()}"
#         np.save(tmp_name, DB)
#         os.replace(tmp_name + ".npy", database_name)

def load_DB_safely(database_name):
    if os.path.exists(database_name):
        return np.load(database_name, allow_pickle=True)
    return np.array([], dtype=object)


def append_DB_safely(database_name, to_save):
    if os.path.exists(database_name):
        DB = np.load(database_name, allow_pickle=True)
    else:
        DB = np.array([], dtype=object)

    DB = np.append(DB, to_save)

    tmp_name = database_name + f".tmp.{os.getpid()}"
    np.save(tmp_name, DB)
    os.replace(tmp_name + ".npy", database_name)


def find_existing_entry(database_name, x, *, input_key="x", rtol=1e-12, atol=1e-12):
    DB = load_DB_safely(database_name)
    x = np.asarray(x, dtype=float)

    for entry in DB:
        if input_key in entry and np.allclose(entry[input_key], x, rtol=rtol, atol=atol):
            return entry

    return None


def evaluate_if_needed(
    database_name,
    x,
    expensive_eval,
    *,
    input_key="x",
    rtol=1e-12,
    atol=1e-12,
    metadata_extra=None,
):
    x = np.asarray(x, dtype=float)

    entry = find_existing_entry(
        database_name, x, input_key=input_key, rtol=rtol, atol=atol
    )

    if entry is not None:
        return entry, False

    values, metadata, calc_status = expensive_eval(x)

    if metadata_extra is not None:
        metadata = dict(metadata)
        metadata.update(metadata_extra)

    entry = {
        input_key: x.copy(),
        "values": values,
        "metadata": metadata,
        "calc_status": calc_status,
    }

    append_DB_safely(database_name, entry)
    return entry, True


def flatten_selected_params(eq, param_names):
    pieces = []
    shapes = {}
    sizes = {}

    for name in param_names:
        arr = np.asarray(eq.params_dict[name], dtype=float)
        shapes[name] = arr.shape
        sizes[name] = arr.size
        pieces.append(arr.ravel())

    return np.concatenate(pieces), shapes, sizes


def unflatten_selected_params(x, base_params, param_names, shapes, sizes):
    params = copy_params_dict(base_params)

    offset = 0
    for name in param_names:
        size = sizes[name]
        params[name] = np.asarray(x[offset : offset + size]).reshape(shapes[name])
        offset += size

    return params


def timed_objective_eval(obj, params):
    t0 = time.time()
    f = obj.compute(params)
    elapsed = time.time() - t0
    return float(np.ravel(f)[0]), elapsed
