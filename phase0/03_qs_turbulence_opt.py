"""Combined QS + turbulence optimization of a quasi-axisymmetric stellarator.

Starts from the precise_QA equilibrium and optimizes QuasisymmetryTwoTerm +
NNITGProxy + AspectRatio simultaneously using proximal-lsq-exact, progressively
freeing more boundary modes. Saves initial and final equilibria as .h5 and
generates a before/after flux surface comparison plot.
"""

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import os

import numpy as np

from desc.compat import rescale
from desc.equilibrium import EquilibriaFamily
from desc.examples import get
from desc.grid import QuadratureGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    NNITGProxy,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer
from desc.plotting import plot_comparison

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "neural_networks")
MODEL_DIR = os.path.join(DATA_DIR, "cyclic_invariant_models")
MODEL_CSV = os.path.join(DATA_DIR, "deephyper_exp", "results.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load and rescale starting equilibrium
eq = get("precise_QA")
eq = rescale(eq, ("a", 1.0), verbose=False)
eq.save(os.path.join(OUTPUT_DIR, "03_initial.h5"))
eqfam = EquilibriaFamily(eq)

# grids and parameters
rho = np.array([np.sqrt(0.5)])
alpha = np.array([0.0])
qs_grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

# optimize in steps
for n in range(2, 3):
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(n))
    print("====================================")
    objective = ObjectiveFunction(
        (
            # QuasisymmetryTwoTerm(
            #     eq=eqfam[-1], helicity=(1, 0), grid=qs_grid,
            #     normalize=False, weight=10.0,
            # ),
            NNITGProxy(
                eq=eqfam[-1], rho=rho, alpha=alpha, weight=0.1,
                ensemble_dir=MODEL_DIR, ensemble_csv=MODEL_CSV, n_ensemble=1,
                target_flux_tube_length=75.4, a_over_LT=3.0, a_over_Ln=1.0,
            ),
            AspectRatio(
                eq=eqfam[-1], target=6.0, weight=1.0 / 6.0,
                grid=qs_grid, normalize=False, normalize_target=False,
            ),
        ),
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eqfam[-1].surface.R_basis.modes[
                np.max(np.abs(eqfam[-1].surface.R_basis.modes), 1) > n, :
            ],
        )
    )
    Z_modes = eqfam[-1].surface.Z_basis.modes[
        np.max(np.abs(eqfam[-1].surface.Z_basis.modes), 1) > n, :
    ]
    constraints = (
        ForceBalance(eq=eqfam[-1]),
        FixBoundaryR(eq=eqfam[-1], modes=R_modes),
        FixBoundaryZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
    )
    optimizer = Optimizer("proximal-lsq-exact")
    eq_new, result = eqfam[-1].optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        copy=True,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        options={
            "initial_trust_radius": 0.5,
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        },
    )
    eqfam.append(eq_new)

# save final equilibrium
eqfam[-1].save(os.path.join(OUTPUT_DIR, "03_final.h5"))

# before/after flux surface comparison
fig, ax = plot_comparison(
    [eqfam[0], eqfam[-1]], labels=["Initial", "Optimized"]
)
fig.savefig(os.path.join(OUTPUT_DIR, "03_flux_surface_comparison.png"), dpi=150)
print("\nSaved comparison plot to {}".format(OUTPUT_DIR))
