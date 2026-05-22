"""Combined QS + turbulence optimization of a quasi-axisymmetric stellarator.

Starts from the precise_QA equilibrium and optimizes QuasisymmetryTwoTerm +
direct GX turbulent heat flux + AspectRatio simultaneously using
proximal-lsq-exact, progressively freeing more boundary modes. Saves initial
and final equilibria as .h5 and generates a before/after flux surface
comparison plot.
"""

import os
from pathlib import Path

# important to run GX on GPU without running out of memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from desc import set_device

try:
    set_device("gpu")
except Exception:
    set_device("cpu")

import numpy as np

from desc.compat import rescale
from desc.equilibrium import EquilibriaFamily
from desc.examples import get
from desc.external.gx import GX
from desc.grid import QuadratureGrid
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    GoodCoordinates,
    ObjectiveFunction,
    PrincipalCurvature,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer
from desc.plotting import plot_comparison

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GX_PATH = Path(os.environ.get("TI_OPT_GX_DIR", REPO_ROOT / "gx")).resolve()
GX_EXEC = os.environ.get("TI_OPT_GX_EXEC", "gx")
GX_INPUT_FILE = Path(
    os.environ.get("TI_OPT_GX_INPUT_FILE", SCRIPT_DIR / "gx.in")
).resolve()
GX_LAUNCH_DEFAULT = SCRIPT_DIR / "gx-launch.sh"
if not GX_LAUNCH_DEFAULT.is_file():
    GX_LAUNCH_DEFAULT = REPO_ROOT / "gx-launch.sh"
GX_LAUNCH = Path(os.environ.get("TI_OPT_GX_LAUNCH", GX_LAUNCH_DEFAULT)).resolve()

# DESC's GX objective shells out via subprocess.run, so this launch wrapper is
# a prefix command for the child process.
LAUNCH_CMD = [str(GX_LAUNCH)] if GX_LAUNCH.is_file() else None

OUTPUT_DIR = SCRIPT_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load and rescale starting equilibrium
eq = get("precise_QA")
eq = rescale(eq, ("a", 1.0), verbose=False)
eq.save(str(OUTPUT_DIR / "03_initial.h5"))
eqfam = EquilibriaFamily(eq)

# grids and parameters
gx_psi = 0.5
gx_alpha = 0.0
qs_grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

# optimize in steps
for n in range(2, 3):
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(n))
    print("====================================")
    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq=eqfam[-1], helicity=(1, 0), grid=qs_grid,
                normalize=False, weight=0.1,
            ),
            GX(
                eqfam[-1],
                weight=0.1,
                abs_step=1e-4,
                rel_step=0,
                processes=1,
                path=str(GX_PATH),
                npol=1,
                nzgrid=48,
                alpha=gx_alpha,
                psi=gx_psi,
                sigma_Bxy=-1.0,
                # Template GX input file. DESC patches geo_option to "eik" and
                # replaces geo_file with the generated geometry path per run.
                gx_input_file=str(GX_INPUT_FILE),
                launch_cmd=LAUNCH_CMD,
                gx_gpu=None,
                timeout=600,
                tmp_dir="tmp_GX",
                save_tmp=False,
                name="GX",
                **{"exec": GX_EXEC}
            ),
            AspectRatio(
                eq=eqfam[-1], target=6.0, weight=1.0 / 6.0,
                grid=qs_grid, normalize=False, normalize_target=False,
            ),
            # New geometric anti-cusp protection
            PrincipalCurvature(
                eq=eq,
                # Using bounds is usually more natural than a target here:
                # penalize only if curvature exceeds this threshold.
                # Units / normalization depend on your setup; tune this.
                bounds=(0.0, 8.0),
                weight=5.0,
                normalize=True,
                name="max principal curvature cap",
            ),
            GoodCoordinates(
                eq=eq,
                weight=1.0,
                normalize=True,
                name="coordinate regularity",
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
        maxiter=1000,
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
eqfam[-1].save(str(OUTPUT_DIR / "03_final.h5"))

# before/after flux surface comparison
fig, ax = plot_comparison(
    [eqfam[0], eqfam[-1]], labels=["Initial", "Optimized"]
)
fig.savefig(OUTPUT_DIR / "03_flux_surface_comparison.png", dpi=150)
print("\nSaved comparison plot to {}".format(OUTPUT_DIR))
