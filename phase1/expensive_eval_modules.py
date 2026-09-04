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


def fsa_grad_x_from_desc(eq, s):
    """Flux-surface-averaged |grad(rho)| * L_ref at normalized toroidal flux s.

    This is the <grad(x)> normalization factor GX/T3D apply to the raw flux
    that NNITGProxy's training data has divided out, so multiplying
    NNITGProxy's raw output by this factor makes it comparable to GX's
    output. (Formula from R. Michael Churchill.)
    """
    from desc.grid import LinearGrid

    rho = np.sqrt(s)

    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

    data = eq.compute("<|grad(rho)|>", grid=grid)
    grad_rho_fsa = grid.compress(data["<|grad(rho)|>"])[0]  # 1/m

    a = np.asarray(eq.compute("a")["a"]).squeeze()  # m

    return float(a * grad_rho_fsa)


def flux_tube_length(eq, *, psi, alpha, npol, nzgrid=200, sigma_Bxy=-1.0):
    """Physical flux-tube length GX would use for the given npol.

    Reuses DESC's own ``compute_gx_geometry`` (the same code path the ``GX``
    objective calls) so this matches GX's actual geometry exactly, rather
    than re-deriving the arclength integral by hand. ``nzgrid`` here is a
    resolution for this length estimate only, independent of whatever
    ``nzgrid`` the eventual GX run uses.
    """
    from desc.external.gx import compute_gx_geometry

    geo = compute_gx_geometry(
        eq, psi, alpha=alpha, npol=npol, nzgrid=nzgrid, sigma_Bxy=sigma_Bxy
    )
    gradpar_uniform_val = float(np.asarray(geo["gradpar"]).ravel()[0])
    return 2.0 * np.pi / gradpar_uniform_val


def solve_npol_for_target_length(
    eq,
    *,
    psi,
    alpha,
    target_length=75.4,
    npol_bracket=(0.05, 4.0),
    nzgrid=200,
    sigma_Bxy=-1.0,
    xtol=1e-4,
):
    """Solve for the npol that gives GX a flux tube of ``target_length``.

    Flux-tube length grows monotonically with npol, so this brackets and
    bisects (scipy.optimize.brentq) rather than guessing/hardcoding a value
    that was only validated for one particular equilibrium.
    """
    from scipy.optimize import brentq

    def residual(npol):
        return (
            flux_tube_length(
                eq,
                psi=psi,
                alpha=alpha,
                npol=npol,
                nzgrid=nzgrid,
                sigma_Bxy=sigma_Bxy,
            )
            - target_length
        )

    lo, hi = npol_bracket
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"target_length={target_length} not bracketed by npol in "
            f"{npol_bracket}: L({lo})={f_lo + target_length:.4f}, "
            f"L({hi})={f_hi + target_length:.4f}"
        )

    return brentq(residual, lo, hi, xtol=xtol)
