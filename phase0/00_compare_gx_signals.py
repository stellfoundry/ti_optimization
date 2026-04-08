#!/usr/bin/env python
"""Compare GX signals: gx_geometry package vs NNITGProxy internal.

Validates that DESC's internal GX signal computation (used by NNITGProxy)
produces the same 7 geometric coefficients as the gx_geometry reference
package across multiple equilibria. The gx_geometry pipeline is the exact
code used to create the training database.
"""

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from desc.compat import rescale
from desc.examples import get
from desc.objectives import NNITGProxy
from desc.utils import safediv

# ---------------------------------------------------------------------------
# Add gx_geometry to path and import
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "gx_geometry"))
from gx_geometry import (
    add_gx_definitions,
    desc_fieldline_specified_length,
    resample,
    uniform_arclength,
)

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(REPO_ROOT, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

EQUILIBRIA = ["precise_QA", "HELIOTRON", "DSHAPE", "W7-X", "NCSX", "HSX"]
FEATURE_NAMES = [
    "bmag", "gbdrift", "cvdrift", "gbdrift0/shat",
    "gds2", "gds21/shat", "gds22/shat^2",
]


def compute_gx_geometry_reference(eq, rho, alpha, target_flux_tube_length=75.4, nz_cnn=96):
    """Compute 7 GX features using gx_geometry (same pipeline as database)."""
    s = rho ** 2
    L_ref = float(np.asarray(eq.compute("a")["a"]))
    target_length_over_L_ref = target_flux_tube_length / L_ref

    # Same pipeline as create_eik_from_desc_given_length (database used nz=97)
    # then truncate last point to get nz_cnn=96 for the CNN
    nz_gx = nz_cnn + 1  # 97 (and truncate last point)
    nz_big = max(1001, nz_gx * 3)
    fl = desc_fieldline_specified_length(
        eq, s, theta0=alpha, zeta0=0.0, nl=nz_big,
        length=target_length_over_L_ref, verbose=False,
    )
    fl = uniform_arclength(fl, verbose=False)
    add_gx_definitions(fl, sigma_Bxy=-1.0)
    fl = resample(fl, nz_gx)

    # Handle division by zero when shat is zero or very small
    # Using DESC's safediv algorithm (automatically handles numpy->jax conversion)
    gbdrift0_over_shat = np.asarray(safediv(fl.gbdrift0, fl.shat, fill=0, threshold=1e-15))
    gds21_over_shat = np.asarray(safediv(fl.gds21, fl.shat, fill=0, threshold=1e-15))
    gds22_over_shat_squared = np.asarray(safediv(fl.gds22, fl.shat**2, fill=0, threshold=1e-15))

    signals = np.stack([
        fl.bmag,
        fl.gbdrift,
        fl.cvdrift,
        gbdrift0_over_shat,
        fl.gds2,
        gds21_over_shat,
        gds22_over_shat_squared,
    ])
    # Truncate last point: 97 -> 96 (matching CNN input size)
    return fl.z[:nz_cnn], signals[:, :nz_cnn]


def compute_both_signals(eq, rho=0.5, alpha=0.0, nz_cnn=96, target_flux_tube_length=75.4):
    """Compute signals from gx_geometry (reference) and NNITGProxy (internal)."""
    eq = rescale(eq, ("a", 1.0), verbose=False)

    nnitg_obj = NNITGProxy(
        eq=eq, rho=np.array([rho]), alpha=np.array([alpha]),
        ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=1,
        target_flux_tube_length=target_flux_tube_length, a_over_LT=3.0, a_over_Ln=1.0,
    )
    nnitg_obj.build(use_jit=False, verbose=0)

    # Get NNITGProxy signals
    _, signals_info = nnitg_obj.compute(eq.params_dict, return_signals=True)
    z_nn = np.asarray(signals_info["z"])
    signals_nn = np.asarray(signals_info["signals"][0, 0])

    # gx_geometry reference signals (same pipeline as database creation)
    z_ref, signals_ref = compute_gx_geometry_reference(
        eq, rho, alpha, target_flux_tube_length=target_flux_tube_length, nz_cnn=nz_cnn,
    )
    return z_ref, signals_ref, z_nn, signals_nn


def main():
    print("=" * 70)
    print("GX Signal Comparison: gx_geometry vs NNITGProxy")
    print("=" * 70)

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Equilibria: {EQUILIBRIA}")

    data = {}
    for eq_name in EQUILIBRIA:
        print(f"\n  {eq_name}...", end=" ", flush=True)
        try:
            data[eq_name] = compute_both_signals(get(eq_name))
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            data[eq_name] = None

    valid_eq = [name for name in EQUILIBRIA if data[name] is not None]
    print(f"\nComputed: {len(valid_eq)}/{len(EQUILIBRIA)} equilibria")
    if not valid_eq:
        return 1

    # Per-feature max absolute difference and R^2
    print("\n" + "=" * 70)
    print("ACCURACY TABLE")
    print("=" * 70)
    print(f"\n{'Feature':>20s}  {'max|diff|':>10s}  {'R^2':>10s}")
    print("-" * 45)
    for eq_name in valid_eq:
        z_ref, sig_ref, z_nn, sig_nn = data[eq_name]
        print(f"  {eq_name}:")
        for k, name in enumerate(FEATURE_NAMES):
            diff = np.max(np.abs(sig_ref[k] - sig_nn[k]))
            ss_res = np.sum((sig_ref[k] - sig_nn[k]) ** 2)
            ss_tot = np.sum((sig_ref[k] - np.mean(sig_ref[k])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"    {name:>20s}  {diff:10.2e}  {r2:10.6f}")

    # Plot
    fig = plt.figure(figsize=(4 * len(valid_eq), 10))
    n_rows, n_cols = 7, len(valid_eq)
    for j_col, eq_name in enumerate(valid_eq):
        z_ref, sig_ref, z_nn, sig_nn = data[eq_name]
        for j_row in range(n_rows):
            ax = plt.subplot(n_rows, n_cols, j_row * n_cols + j_col + 1)
            ax.plot(z_ref, sig_ref[j_row], "b-", linewidth=1.0)
            ax.plot(z_nn, sig_nn[j_row], "r--", linewidth=1.0)
            ax.set_xlim([z_ref[0], z_ref[-1]])
            if j_col == 0:
                ax.set_ylabel(FEATURE_NAMES[j_row], fontsize=9)
            else:
                ax.set_yticklabels([])
            if j_row < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("z")
            if j_row == 0:
                ax.set_title(eq_name, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=0.87)
    fig.legend(
        [plt.Line2D([0], [0], color="b", linestyle="-", linewidth=1.5),
         plt.Line2D([0], [0], color="r", linestyle="--", linewidth=1.5)],
        ["gx_geometry", "NNITGProxy"], loc="upper right",
        bbox_to_anchor=(0.99, 0.98), fontsize=10,
    )
    path = os.path.join(OUTPUT_DIR, "00_signal_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    exit(main())
