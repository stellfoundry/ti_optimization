#!/usr/bin/env python
"""Multi-alpha robustness analysis: compare initial vs optimized equilibrium.

Evaluates NNITGProxy at 8 field lines to check that optimization reduces
turbulence across the full alpha range, not just at alpha=0.
"""

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import os

import matplotlib.pyplot as plt
import numpy as np

from desc.io import load
from desc.objectives import NNITGProxy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def evaluate_multi_alpha(eq, rho, alpha_arr, n_ensemble=3):
    """Evaluate NNITGProxy at multiple field lines."""
    obj = NNITGProxy(
        eq=eq, rho=rho, alpha=alpha_arr,
        ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=n_ensemble,
        target_flux_tube_length=75.4, a_over_LT=3.0, a_over_Ln=1.0,
    )
    obj.build(use_jit=False, verbose=0)
    Q, Q_std = obj.compute(eq.params_dict, return_per_alpha=True, return_std=True)
    return np.asarray(Q), np.asarray(Q_std)


def main():
    print("=" * 60)
    print("Multi-Alpha Robustness Analysis")
    print("=" * 60)

    init_path = os.path.join(OUTPUT_DIR, "03_initial.h5")
    opt_path = os.path.join(OUTPUT_DIR, "03_final.h5")

    if not os.path.exists(init_path) or not os.path.exists(opt_path):
        print("ERROR: Run 03_qs_turbulence_opt.py first to generate equilibria")
        return 1
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return 1

    eq_init = load(init_path)
    eq_opt = load(opt_path)
    print("Loaded initial and optimized equilibria")

    # 8 field lines in [0, pi]
    rho = np.array([np.sqrt(0.5)])
    alpha_arr = np.linspace(0, np.pi, 8)
    n_ensemble = 3

    print(f"\nEvaluating at {len(alpha_arr)} field lines (alpha in [0, pi])...")

    Q_init, Q_std_init = evaluate_multi_alpha(eq_init, rho, alpha_arr, n_ensemble)
    Q_opt, Q_std_opt = evaluate_multi_alpha(eq_opt, rho, alpha_arr, n_ensemble)

    # Q shapes: (1, 8) -> flatten to (8,)
    Q_init, Q_std_init = Q_init[0], Q_std_init[0]
    Q_opt, Q_std_opt = Q_opt[0], Q_std_opt[0]

    # Table
    print(f"\n{'alpha':>8s}  {'Q_init':>10s}  {'Q_opt':>10s}  {'delta':>10s}  {'Q_std_opt':>10s}")
    print("-" * 55)
    for i, a in enumerate(alpha_arr):
        delta = Q_opt[i] - Q_init[i]
        print(f"{a:8.4f}  {Q_init[i]:10.4f}  {Q_opt[i]:10.4f}  {delta:+10.4f}  {Q_std_opt[i]:10.4f}")

    print(f"\nMean Q initial:   {Q_init.mean():.4f}")
    print(f"Mean Q optimized: {Q_opt.mean():.4f}")
    print(f"Mean reduction:   {(Q_init.mean() - Q_opt.mean()) / Q_init.mean() * 100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alpha_arr, Q_init, yerr=Q_std_init, fmt="o-", capsize=4,
                label="Initial", color="C0")
    ax.errorbar(alpha_arr, Q_opt, yerr=Q_std_opt, fmt="s-", capsize=4,
                label="Optimized", color="C1")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Q (ITG heat flux)")
    ax.set_title("Per-Field-Line Q: Initial vs Optimized")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "04_multi_alpha_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
