#!/usr/bin/env python
"""Validate ITGProxy and NNITGProxy on a fixed equilibrium.

Checks that both objectives:
1. Produce finite, positive values
2. Support return_signals, return_per_alpha, return_std (NNITGProxy)
3. Are differentiable via JAX (gradient w.r.t. boundary coefficients)
"""

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import os

import numpy as np

from desc.compat import rescale
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import ITGProxy, NNITGProxy, ObjectiveFunction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")


def check(label, condition):
    """Print pass/fail for a check."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main():
    print("=" * 60)
    print("Validate ITGProxy + NNITGProxy Plumbing")
    print("=" * 60)

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return 1

    # Load equilibrium
    print("\nLoading precise_QA...")
    eq = get("precise_QA")
    eq = rescale(eq, ("a", 1.0), verbose=False)

    rho = np.array([np.sqrt(0.5)])
    iota_data = eq.compute(
        ["iota"], grid=LinearGrid(rho=rho, M=eq.M, N=eq.N, NFP=eq.NFP)
    )
    iota = float(iota_data["iota"][0])
    alpha = np.array([0.0, np.pi * iota / eq.NFP])
    print(f"  NFP={eq.NFP}, rho={rho}, alpha={alpha}")

    all_pass = True

    # --- ITGProxy ---
    print("\n--- ITGProxy ---")
    itg = ITGProxy(eq=eq, rho=rho, alpha=alpha, nturns=3, nzetaperturn=100)
    itg.build(use_jit=False, verbose=0)

    f_Q = itg.compute(eq.params_dict)
    print(f"  f_Q = {f_Q}")
    all_pass &= check("f_Q finite", np.isfinite(f_Q).all())
    all_pass &= check("f_Q positive", (f_Q > 0).all())

    obj_fn = ObjectiveFunction([itg])
    obj_fn.build(use_jit=False, verbose=0)
    x0 = obj_fn.x(eq)
    grad_f = obj_fn.grad(x0)
    all_pass &= check("ITGProxy gradient finite", np.isfinite(grad_f).all())
    all_pass &= check("ITGProxy gradient nonzero", np.any(grad_f != 0))
    print(f"  gradient norm = {np.linalg.norm(grad_f):.4e}")

    # --- NNITGProxy ---
    print("\n--- NNITGProxy ---")
    nnitg = NNITGProxy(
        eq=eq, rho=rho, alpha=alpha,
        ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=3,
        target_flux_tube_length=75.4, a_over_LT=3.0, a_over_Ln=1.0,
    )
    nnitg.build(use_jit=False, verbose=0)

    # Basic compute
    Q = nnitg.compute(eq.params_dict)
    print(f"  Q = {Q}")
    all_pass &= check("Q finite", np.isfinite(Q).all())
    all_pass &= check("Q positive", (Q > 0).all())

    # return_signals
    Q_sig, info = nnitg.compute(eq.params_dict, return_signals=True)
    all_pass &= check("return_signals: Q matches", np.allclose(Q, Q_sig))
    all_pass &= check("return_signals: signals finite",
                       np.isfinite(info["signals"]).all())
    all_pass &= check("return_signals: 7 features",
                       len(info["feature_names"]) == 7)
    print(f"  signals shape = {info['signals'].shape}")

    # return_per_alpha
    Q_pa = nnitg.compute(eq.params_dict, return_per_alpha=True)
    all_pass &= check("return_per_alpha: shape (num_rho, num_alpha)",
                       Q_pa.shape == (len(rho), len(alpha)))
    all_pass &= check("return_per_alpha: mean matches Q",
                       np.allclose(np.mean(Q_pa, axis=-1), Q, atol=1e-6))

    # return_std
    Q_std_val, Q_std = nnitg.compute(eq.params_dict, return_std=True)
    all_pass &= check("return_std: Q matches", np.allclose(Q, Q_std_val))
    all_pass &= check("return_std: std finite", np.isfinite(Q_std).all())
    all_pass &= check("return_std: std non-negative", (Q_std >= 0).all())
    print(f"  Q_std = {Q_std}")

    # Gradient
    obj_fn2 = ObjectiveFunction([nnitg])
    obj_fn2.build(use_jit=False, verbose=0)
    x0 = obj_fn2.x(eq)
    grad_f2 = obj_fn2.grad(x0)
    all_pass &= check("NNITGProxy gradient finite", np.isfinite(grad_f2).all())
    all_pass &= check("NNITGProxy gradient nonzero", np.any(grad_f2 != 0))
    print(f"  gradient norm = {np.linalg.norm(grad_f2):.4e}")

    # Summary
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
