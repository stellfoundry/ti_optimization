#!/usr/bin/env python3
"""Phase 0 workflow for ti_optimization.

This script follows the plan described in README.md:

* start from the DESC "precise QA" example (reactor_QA.py) to obtain a reactor-scale
  stellarator equilibrium with the usual DESC optimization targets and constraints,
* evaluate the AI GX surrogate at fixed flux-tube locations with fixed gradient
  scale lengths (tprim = 3.0, fprim = 0.9 by default), and
* average the predicted ion heat fluxes to form a global optimization target that
  can be tracked across future design iterations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_PARENT = PROJECT_ROOT.parent


def _extend_sys_path(env_var: str, default: Path) -> Path:
    """Ensure external repositories needed for inference are importable."""
    candidate = Path(os.environ.get(env_var, default)).expanduser()
    if candidate.is_dir():
        candidate = candidate.resolve()
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
    return candidate


T3D_ML_ROOT = _extend_sys_path("T3D_ML_ROOT", REPO_PARENT / "t3d_ml")
GX_GEOMETRY_ROOT = _extend_sys_path("GX_GEOMETRY_ROOT", REPO_PARENT / "gx_geometry")

try:
    from scripts.ai_inference_utils import load_engine, run_inference
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Could not import scripts.ai_inference_utils. "
        "Set T3D_ML_ROOT to your t3d_ml checkout."
    ) from exc

try:
    from gx_geometry.driver import create_eik_from_desc
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Could not import gx_geometry. "
        "Set GX_GEOMETRY_ROOT to your gx_geometry checkout."
    ) from exc

from desc import set_device
from desc.compat import rescale
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (
    BootstrapRedlConsistency,
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.profiles import PowerSeriesProfile


@dataclass(frozen=True)
class FluxTubeLocation:
    """Flux-tube location used by the AI surrogate."""

    flux_tube_s: float
    zeta: float
    theta: float = 0.0


@dataclass(frozen=True)
class GradientConfig:
    """Fixed gradient scale lengths for the Phase 0 calculations."""

    tprim: float
    fprim: float


@dataclass
class HeatFluxSample:
    """Single heat-flux prediction at a flux-tube location."""

    location: FluxTubeLocation
    mean_heat_flux: float
    std_heat_flux: float
    geometry_path: Path


DEFAULT_FLUX_SURFACES: Sequence[float] = (0.25, 0.45, 0.65, 0.85)
DEFAULT_ZETA: Sequence[float] = (0.0, math.pi / 6, math.pi / 3)
DEFAULT_GRADIENTS = GradientConfig(tprim=3.0, fprim=0.9)


def build_initial_equilibrium(verbose: int = 3):
    """Replicate the DESC reactor_QA example to obtain an initial equilibrium."""
    eq = get("precise_QA")

    # Apply Landreman-Buller-Drevlak kinetic profiles at ARIES-CS scale.
    a = 1.7  # minor radius [m]
    B = 5.86  # volume averaged |B| [T]
    n0 = 2.38e20  # on-axis density [m^-3]
    T0 = 9.45e3  # on-axis temperature [eV]

    eq.pressure = None
    eq.current = PowerSeriesProfile(np.zeros(eq.L + 1), np.arange(eq.L + 1), sym=False)
    eq.atomic_number = PowerSeriesProfile([1], [0])
    eq.electron_density = PowerSeriesProfile(
        n0 * np.array([1, -1]), np.array([0, 10]), sym=True
    )
    eq.electron_temperature = PowerSeriesProfile(
        T0 * np.array([1, -1]), np.array([0, 2]), sym=True
    )
    eq.ion_temperature = PowerSeriesProfile(
        T0 * np.array([1, -1]), np.array([0, 2]), sym=True
    )

    eq, _ = eq.solve(verbose=verbose, copy=True)
    eq = rescale(eq, L=("a", a), B=("<B>", B), scale_pressure=False, copy=True, verbose=1)
    return eq


def optimize_equilibrium(eq, maxiter: int, gtol: float, verbose: int = 3):
    """Run the DESC optimization used in the reactor QA example."""
    grid = LinearGrid(
        rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid),
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        sym=True,
    )
    objective = ObjectiveFunction(
        BootstrapRedlConsistency(eq=eq, grid=grid, helicity=(1, 0))
    )
    constraints = (
        FixAtomicNumber(eq=eq),
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        FixCurrent(eq=eq, indices=np.array([0, 1])),
        FixElectronDensity(eq=eq),
        FixElectronTemperature(eq=eq),
        FixIonTemperature(eq=eq),
        FixPsi(eq=eq),
        ForceBalance(eq=eq),
    )
    eq_opt, _ = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer="proximal-lsq-exact",
        maxiter=maxiter,
        gtol=gtol,
        verbose=verbose,
        copy=True,
    )
    return eq_opt


def prepare_locations(
    flux_surfaces: Iterable[float], zetas: Iterable[float], theta: float
) -> List[FluxTubeLocation]:
    """Generate a grid of flux-tube locations."""
    locations = []
    for s in flux_surfaces:
        for zeta in zetas:
            locations.append(FluxTubeLocation(float(s), float(zeta), float(theta)))
    return locations


def evaluate_heat_flux(
    eq,
    locations: Sequence[FluxTubeLocation],
    gradients: GradientConfig,
    geometry_dir: Path,
    *,
    engine=None,
    nz: int = 97,
    poloidal_turns: float = 1.0,
    sigma_Bxy: float = -1.0,
) -> List[HeatFluxSample]:
    """Run the AI GX surrogate at the requested flux-tube locations."""
    geometry_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or load_engine()

    samples: List[HeatFluxSample] = []
    for idx, loc in enumerate(locations, start=1):
        geometry_path = geometry_dir / f"phase0_s{loc.flux_tube_s:.3f}_z{loc.zeta:.3f}.eik.nc"
        create_eik_from_desc(
            eq,
            s=loc.flux_tube_s,
            theta0=loc.theta,
            zeta0=loc.zeta,
            nz=nz,
            poloidal_turns=poloidal_turns,
            sigma_Bxy=sigma_Bxy,
            eik_filename=str(geometry_path),
        )
        mean_q, std_q = run_inference(
            geometry=str(geometry_path),
            tprims=[gradients.tprim],
            fprims=[gradients.fprim],
            engine=engine,
        )
        samples.append(
            HeatFluxSample(
                location=loc,
                mean_heat_flux=float(np.atleast_1d(mean_q)[0]),
                std_heat_flux=float(np.atleast_1d(std_q)[0]),
                geometry_path=geometry_path,
            )
        )
        print(
            f"[Phase0] Sample {idx:02d}: s={loc.flux_tube_s:.3f}, "
            f"zeta={loc.zeta:.3f}, q={samples[-1].mean_heat_flux:.4e} (std={samples[-1].std_heat_flux:.2e})"
        )
    return samples


def summarize_samples(
    samples: Sequence[HeatFluxSample],
    gradients: GradientConfig,
    equilibrium_path: Path,
) -> dict:
    """Summarize the averaged heat flux that defines the Phase 0 target."""
    if not samples:
        raise ValueError("No heat-flux samples were generated.")

    values = np.array([sample.mean_heat_flux for sample in samples])
    summary = {
        "equilibrium_path": str(equilibrium_path),
        "num_samples": len(samples),
        "mean_heat_flux": float(values.mean()),
        "std_heat_flux": float(values.std()),
        "tprim": gradients.tprim,
        "fprim": gradients.fprim,
        "samples": [
            {
                "flux_tube_s": sample.location.flux_tube_s,
                "zeta": sample.location.zeta,
                "theta": sample.location.theta,
                "geometry_path": str(sample.geometry_path),
                "mean_heat_flux": sample.mean_heat_flux,
                "std_heat_flux": sample.std_heat_flux,
            }
            for sample in samples
        ],
    }
    summary["global_heat_flux_target"] = summary["mean_heat_flux"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 optimization workflow")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "phase0_results",
        help="Directory used to store equilibria, geometry files, and summaries.",
    )
    parser.add_argument(
        "--desc-device",
        type=str,
        default="gpu",
        help="DESC compute backend (gpu or cpu).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=50, help="Maximum DESC optimization iterations."
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-16,
        help="Gradient tolerance passed to desc Equilibrium.optimize.",
    )
    parser.add_argument(
        "--desc-verbose",
        type=int,
        default=3,
        help="Verbosity level forwarded to DESC solve/optimize calls.",
    )
    parser.add_argument(
        "--flux-surfaces",
        type=float,
        nargs="+",
        default=list(DEFAULT_FLUX_SURFACES),
        help="Flux-surface s values used for flux-tube sampling.",
    )
    parser.add_argument(
        "--zeta",
        type=float,
        nargs="+",
        default=[float(val) for val in DEFAULT_ZETA],
        help="Field-line label(s) in radians.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="Poloidal angle theta0 passed to gx_geometry (radians).",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=97,
        help="Number of parallel grid points when generating GX geometry.",
    )
    parser.add_argument(
        "--poloidal-turns",
        type=float,
        default=1.0,
        help="Number of poloidal turns used when generating GX geometry.",
    )
    parser.add_argument(
        "--tprim",
        type=float,
        default=DEFAULT_GRADIENTS.tprim,
        help="Ion temperature gradient scale length.",
    )
    parser.add_argument(
        "--fprim",
        type=float,
        default=DEFAULT_GRADIENTS.fprim,
        help="Density gradient scale length.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional override for the AI GX checkpoint directory.",
    )
    parser.add_argument(
        "--inference-device",
        type=str,
        default=None,
        help="Torch device used for inference (defaults to CUDA when available).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    geometry_dir = output_dir / "geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_device(args.desc_device)

    eq = build_initial_equilibrium(verbose=args.desc_verbose)
    eq = optimize_equilibrium(eq, maxiter=args.maxiter, gtol=args.gtol, verbose=args.desc_verbose)

    equilibrium_path = output_dir / "phase0_reactor_QA.h5"
    eq.save(str(equilibrium_path))

    engine = load_engine(
        model_dir=args.model_dir if args.model_dir is not None else None,
        device=args.inference_device,
    )

    gradients = GradientConfig(tprim=args.tprim, fprim=args.fprim)
    locations = prepare_locations(args.flux_surfaces, args.zeta, args.theta)
    samples = evaluate_heat_flux(
        eq,
        locations,
        gradients,
        geometry_dir,
        engine=engine,
        nz=args.nz,
        poloidal_turns=args.poloidal_turns,
    )
    summary = summarize_samples(samples, gradients, equilibrium_path)

    summary_path = output_dir / "phase0_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[Phase0] Global optimization target (mean AI heat flux):")
    print(f"         {summary['mean_heat_flux']:.6e}")
    print(f"[Phase0] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
