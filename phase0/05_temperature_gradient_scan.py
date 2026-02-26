#!/usr/bin/env python
"""Temperature gradient scan: Q(a/LT) for initial and optimized equilibria.

Scans the temperature gradient a/LT while holding a/Ln fixed to show
how the ITG heat flux depends on the driving gradient for different
magnetic geometries.
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
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def scan_gradient(eq, rho, alpha, a_over_LT_values, a_over_Ln=1.0, n_ensemble=3):
    """Compute Q for each a/LT value.
    
    Optimized: Build NNITGProxy once, then reuse with modified constants.
    This avoids expensive rebuilds (model loading, geometry computation) for each point.
    """
    print("  Building NNITGProxy (one-time setup)...", end=" ", flush=True)
    obj = NNITGProxy(
        eq=eq, rho=rho, alpha=alpha,
        ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=n_ensemble,
        target_flux_tube_length=75.4, a_over_LT=a_over_LT_values[0], a_over_Ln=a_over_Ln,
    )
    obj.build(use_jit=False, verbose=0)
    print("done")
    
    Q_values = []
    print("  Computing Q for each a/LT...", end=" ", flush=True)
    for a_over_LT in a_over_LT_values:
        modified_constants = obj.constants.copy()
        modified_constants["a_over_LT"] = a_over_LT
        Q_array = obj.compute(eq.params_dict, constants=modified_constants)
        Q = float(Q_array[0])  
        Q_values.append(Q)
    print(f"done ({len(Q_values)} points)")
    return np.array(Q_values)


def main():
    print("=" * 60)
    print("Temperature Gradient Scan: Q(a/LT)")
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

    rho = np.array([np.sqrt(0.5)])
    alpha = np.array([0.0])
    a_over_Ln = 1.0
    a_over_LT_values = np.linspace(0.5, 10.0, 20)

    print(f"\nScanning a/LT from {a_over_LT_values[0]:.1f} to {a_over_LT_values[-1]:.1f} "
          f"({len(a_over_LT_values)} points)")
    print(f"Fixed a/Ln = {a_over_Ln}")

    print("\nInitial equilibrium...")
    Q_init = scan_gradient(eq_init, rho, alpha, a_over_LT_values, a_over_Ln)

    print("Optimized equilibrium...")
    Q_opt = scan_gradient(eq_opt, rho, alpha, a_over_LT_values, a_over_Ln)

    # Table
    print(f"\n{'a/LT':>8s}  {'Q_init':>10s}  {'Q_opt':>10s}  {'ratio':>8s}")
    print("-" * 40)
    for i, lt in enumerate(a_over_LT_values):
        ratio = Q_opt[i] / Q_init[i] if Q_init[i] > 0 else float("nan")
        print(f"{lt:8.2f}  {Q_init[i]:10.4f}  {Q_opt[i]:10.4f}  {ratio:8.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.plot(a_over_LT_values, Q_init, "o-", label="Initial", color="C0")
    ax1.plot(a_over_LT_values, Q_opt, "s-", label="Optimized", color="C1")
    ax1.set_xlabel("a/LT")
    ax1.set_ylabel("Q (ITG heat flux)")
    ax1.set_title("Q vs Temperature Gradient")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.semilogy(a_over_LT_values, Q_init, "o-", label="Initial", color="C0")
    ax2.semilogy(a_over_LT_values, Q_opt, "s-", label="Optimized", color="C1")
    ax2.set_xlabel("a/LT")
    ax2.set_ylabel("Q (log scale)")
    ax2.set_title("Q vs Temperature Gradient (log)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "05_temperature_gradient_scan.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
