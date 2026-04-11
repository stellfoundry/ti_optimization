"""General QS + turbulence optimization driver with automatic postprocessing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from desc import set_device
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

from postprocessing import ProxyConfig, TurbulencePostprocessor


def configure_device() -> None:
    """Prefer GPU when available, otherwise fall back to CPU."""
    try:
        set_device("gpu")
    except Exception:
        set_device("cpu")


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout for the phase2 optimization workflow."""

    script_dir: str
    repo_root: str
    data_dir: str
    model_dir: str
    model_csv: str
    output_dir: str
    initial_eq_path: str
    final_eq_path: str

    @classmethod
    def from_script(cls, script_file: str) -> "ProjectPaths":
        script_dir = os.path.dirname(os.path.abspath(script_file))
        repo_root = os.path.dirname(script_dir)
        data_dir = os.path.join(repo_root, "data", "neural_networks")
        model_dir = os.path.join(data_dir, "cyclic_invariant_models")
        model_csv = os.path.join(data_dir, "deephyper_exp", "results.csv")
        output_dir = os.path.join(script_dir, "output")
        return cls(
            script_dir=script_dir,
            repo_root=repo_root,
            data_dir=data_dir,
            model_dir=model_dir,
            model_csv=model_csv,
            output_dir=output_dir,
            initial_eq_path=os.path.join(output_dir, "initial.h5"),
            final_eq_path=os.path.join(output_dir, "final.h5"),
        )


@dataclass
class OptimizationConfig:
    """Configuration for the QS + turbulence optimization and follow-up analysis."""

    example_name: str = "precise_QA"
    rho: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.041, 0.124, 0.206, 0.288, 0.371, 0.453, 0.535, 0.618, 0.700],
            dtype=float,
        )
    )
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=float))
    mode_steps: tuple[int, ...] = (2,)
    qs_weight: float = 0.1
    turbulence_weight: float = 0.1
    aspect_ratio_target: float = 6.0
    aspect_ratio_weight: float = 1.0 / 6.0
    target_flux_tube_length: float = 75.4
    a_over_LT: float = 3.0
    a_over_Ln: float = 1.0
    optimization_n_ensemble: int = 1
    analysis_n_ensemble: int = 3
    optimizer_name: str = "proximal-lsq-exact"
    maxiter: int = 1000
    ftol: float = 1e-4
    xtol: float = 1e-4
    gtol: float = 1e-4
    optimize_options: dict = field(
        default_factory=lambda: {
            "initial_trust_radius": 0.5,
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        }
    )
    multi_alpha_rho: np.ndarray = field(
        default_factory=lambda: np.array([np.sqrt(0.5)], dtype=float)
    )
    multi_alpha_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, np.pi, 8, dtype=float)
    )
    gradient_scan_rho: np.ndarray = field(
        default_factory=lambda: np.array([np.sqrt(0.5)], dtype=float)
    )
    gradient_scan_alpha: np.ndarray = field(
        default_factory=lambda: np.array([0.0], dtype=float)
    )
    gradient_scan_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.5, 10.0, 20, dtype=float)
    )
    rho_scan_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.05, 0.95, 12, dtype=float)
    )
    rho_scan_alpha: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=float))


def ensure_output_dir(paths: ProjectPaths) -> None:
    os.makedirs(paths.output_dir, exist_ok=True)


def build_objective(eq, qs_grid, config: OptimizationConfig, paths: ProjectPaths):
    return ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq=eq,
                helicity=(1, 0),
                grid=qs_grid,
                normalize=False,
                weight=config.qs_weight,
            ),
            NNITGProxy(
                eq=eq,
                rho=config.rho,
                alpha=config.alpha,
                weight=config.turbulence_weight,
                ensemble_dir=paths.model_dir,
                ensemble_csv=paths.model_csv,
                n_ensemble=config.optimization_n_ensemble,
                target_flux_tube_length=config.target_flux_tube_length,
                a_over_LT=config.a_over_LT,
                a_over_Ln=config.a_over_Ln,
            ),
            AspectRatio(
                eq=eq,
                target=config.aspect_ratio_target,
                weight=config.aspect_ratio_weight,
                grid=qs_grid,
                normalize=False,
                normalize_target=False,
            ),
        )
    )


def build_constraints(eq, max_mode: int):
    r_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), axis=1) > max_mode, :],
        )
    )
    z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), axis=1) > max_mode, :]
    return (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=r_modes),
        FixBoundaryZ(eq=eq, modes=z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )


def run_optimization(config: OptimizationConfig, paths: ProjectPaths) -> EquilibriaFamily:
    eq = get(config.example_name)
    eq = rescale(eq, ("a", 1.0), verbose=False)
    eq.save(paths.initial_eq_path)

    eq_family = EquilibriaFamily(eq)
    qs_grid = QuadratureGrid(L=eq.L, M=eq.M, N=eq.N, NFP=eq.NFP)

    for max_mode in config.mode_steps:
        print("\n==================================")
        print(f"Optimizing boundary modes M,N <= {max_mode}")
        print("====================================")

        objective = build_objective(eq_family[-1], qs_grid, config, paths)
        constraints = build_constraints(eq_family[-1], max_mode)
        optimizer = Optimizer(config.optimizer_name)
        eq_new, _ = eq_family[-1].optimize(
            objective=objective,
            constraints=constraints,
            optimizer=optimizer,
            maxiter=config.maxiter,
            verbose=3,
            copy=True,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            options=config.optimize_options,
        )
        eq_family.append(eq_new)

    eq_family[-1].save(paths.final_eq_path)
    return eq_family


def run_postprocessing(
    eq_family: EquilibriaFamily,
    config: OptimizationConfig,
    paths: ProjectPaths,
) -> dict[str, str]:
    proxy_config = ProxyConfig(
        ensemble_dir=paths.model_dir,
        ensemble_csv=paths.model_csv,
        n_ensemble=config.analysis_n_ensemble,
        target_flux_tube_length=config.target_flux_tube_length,
        a_over_LT=config.a_over_LT,
        a_over_Ln=config.a_over_Ln,
    )
    postprocessor = TurbulencePostprocessor(paths.output_dir, proxy_config)
    return postprocessor.run_default_suite(
        eq_family[0],
        eq_family[-1],
        multi_alpha_rho=config.multi_alpha_rho,
        multi_alpha_values=config.multi_alpha_values,
        gradient_scan_rho=config.gradient_scan_rho,
        gradient_scan_alpha=config.gradient_scan_alpha,
        gradient_scan_values=config.gradient_scan_values,
        gradient_scan_a_over_Ln=config.a_over_Ln,
        rho_scan_values=config.rho_scan_values,
        rho_scan_alpha=config.rho_scan_alpha,
        n_ensemble=config.analysis_n_ensemble,
    )


def main() -> int:
    configure_device()
    paths = ProjectPaths.from_script(__file__)
    config = OptimizationConfig()

    ensure_output_dir(paths)
    if not os.path.exists(paths.model_dir):
        raise FileNotFoundError(f"Model directory not found: {paths.model_dir}")

    eq_family = run_optimization(config, paths)
    saved_paths = run_postprocessing(eq_family, config, paths)

    print("\nGenerated outputs:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
