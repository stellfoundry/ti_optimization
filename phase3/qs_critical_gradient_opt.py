"""Phase3 QS + critical-gradient optimization workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from desc import set_device
try:
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        #set if on login node
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"CUDA_VISIBLE_DEVICES set to {os.getenv('CUDA_VISIBLE_DEVICES')}")
    set_device("gpu")
except Exception:
    set_device("cpu")

from desc.compat import rescale
from desc.equilibrium import EquilibriaFamily
from desc.examples import get
from desc.grid import QuadratureGrid
from desc.io import load
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.optimize import Optimizer

from critical_gradient_objective import NNCriticalGradientProxy
from postprocessing import CriticalGradientPostprocessConfig, CriticalGradientPostprocessor



@dataclass(frozen=True)
class ProjectPaths:
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
    critical_gradient_weight: float = 0.1
    aspect_ratio_target: float = 6.0
    aspect_ratio_weight: float = 1.0 / 6.0
    target_flux_tube_length: float = 75.4
    a_over_Ln: float = 0.9
    a_over_LT_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.25, 4.25, 8, dtype=float)
    )
    critical_q_threshold: float = 1.0
    critical_log_width: float = 0.35
    alpha_aggregation: str = "mean"
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
    scan_rho: np.ndarray = field(
        default_factory=lambda: np.array([np.sqrt(0.5)], dtype=float)
    )
    scan_alpha: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=float))
    rho_scan_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.05, 0.95, 12, dtype=float)
    )
    rho_scan_alpha: np.ndarray = field(
        default_factory=lambda: np.array([0.0], dtype=float)
    )


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
            NNCriticalGradientProxy(
                eq=eq,
                rho=config.rho,
                alpha=config.alpha,
                weight=config.critical_gradient_weight,
                ensemble_dir=paths.model_dir,
                ensemble_csv=paths.model_csv,
                n_ensemble=config.optimization_n_ensemble,
                target_flux_tube_length=config.target_flux_tube_length,
                a_over_Ln=config.a_over_Ln,
                a_over_LT_grid=config.a_over_LT_grid,
                critical_q_threshold=config.critical_q_threshold,
                critical_log_width=config.critical_log_width,
                alpha_aggregation=config.alpha_aggregation,
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
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), axis=1) > max_mode, :
            ],
        )
    )
    z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), axis=1) > max_mode, :
    ]
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
):
    post_config = CriticalGradientPostprocessConfig(
        ensemble_dir=paths.model_dir,
        ensemble_csv=paths.model_csv,
        n_ensemble=config.analysis_n_ensemble,
        target_flux_tube_length=config.target_flux_tube_length,
        a_over_Ln=config.a_over_Ln,
        a_over_LT_grid=config.a_over_LT_grid,
        critical_q_threshold=config.critical_q_threshold,
        critical_log_width=config.critical_log_width,
        alpha_aggregation=config.alpha_aggregation,
    )
    postprocessor = CriticalGradientPostprocessor(paths.output_dir, post_config)
    return postprocessor.run_default_suite(
        eq_family[0],
        eq_family[-1],
        scan_rho=config.scan_rho,
        scan_alpha=config.scan_alpha,
        rho_scan_values=config.rho_scan_values,
        rho_scan_alpha=config.rho_scan_alpha,
    )


def summarize_critical_gradient(paths: ProjectPaths, config: OptimizationConfig) -> None:
    post_config = CriticalGradientPostprocessConfig(
        ensemble_dir=paths.model_dir,
        ensemble_csv=paths.model_csv,
        n_ensemble=config.analysis_n_ensemble,
        target_flux_tube_length=config.target_flux_tube_length,
        a_over_Ln=config.a_over_Ln,
        a_over_LT_grid=config.a_over_LT_grid,
        critical_q_threshold=config.critical_q_threshold,
        critical_log_width=config.critical_log_width,
        alpha_aggregation=config.alpha_aggregation,
    )
    postprocessor = CriticalGradientPostprocessor(paths.output_dir, post_config)
    eq_initial = load(paths.initial_eq_path)
    eq_final = load(paths.final_eq_path)
    critical_initial = postprocessor.evaluate_critical_gradient(
        eq_initial, rho=config.rho, alpha=config.alpha
    )
    critical_final = postprocessor.evaluate_critical_gradient(
        eq_final, rho=config.rho, alpha=config.alpha
    )

    summary_path = os.path.join(paths.output_dir, "critical_gradient_summary.txt")
    with open(summary_path, "w", encoding="ascii") as stream:
        stream.write("# Phase3 critical-gradient summary\n")
        stream.write(
            "# Objective value minimized during optimization is a/LT_max - a/LT_crit\n"
        )
        stream.write("# rho  aLTcrit_initial  aLTcrit_final  delta\n")
        for rho_value, crit_i, crit_f in zip(config.rho, critical_initial, critical_final):
            stream.write(
                f"{rho_value: .6f}  {crit_i: .6f}  {crit_f: .6f}  {crit_f - crit_i: .6f}\n"
            )
    print(f"Saved summary: {summary_path}")


def main() -> int:
    paths = ProjectPaths.from_script(__file__)
    config = OptimizationConfig()

    ensure_output_dir(paths)
    if not os.path.exists(paths.model_dir):
        raise FileNotFoundError(f"Model directory not found: {paths.model_dir}")

    eq_family = run_optimization(config, paths)
    saved_paths = run_postprocessing(eq_family, config, paths)
    summarize_critical_gradient(paths, config)

    print("\nGenerated outputs:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
