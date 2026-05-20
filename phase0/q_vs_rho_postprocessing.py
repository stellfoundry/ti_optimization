"""Generate turbulent heat-flux-vs-rho plots for saved equilibrium pairs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

from transport_postprocessing import _log, gpu_preflight, resolve_eq_paths


T3D_ROOT = Path("/global/cfs/projectdirs/m4505/rchurchi/t3d_work/ti_optimization")
DEFAULT_INITIAL_T3D_DIR = T3D_ROOT / "phase0_v1_initial_fixlength"
DEFAULT_FINAL_T3D_DIR = T3D_ROOT / "phase0_v1_final_fixlengthv3"
T3D_ML_ROOT = Path("/global/cfs/cdirs/m4505/rchurchi/t3d_ml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("nnitgproxy", "t3d-ai-gx"),
        default="nnitgproxy",
        help="Flux model backend to use for the q-vs-rho plot.",
    )
    parser.add_argument(
        "--initial-t3d-dir",
        type=Path,
        default=DEFAULT_INITIAL_T3D_DIR,
        help=f"Directory containing the initial T3D run (default: {DEFAULT_INITIAL_T3D_DIR})",
    )
    parser.add_argument(
        "--final-t3d-dir",
        type=Path,
        default=DEFAULT_FINAL_T3D_DIR,
        help=f"Directory containing the final T3D run (default: {DEFAULT_FINAL_T3D_DIR})",
    )
    parser.add_argument(
        "--initial-model-dir",
        type=Path,
        default=None,
        help="Optional override for the initial T3D AI_GX model directory.",
    )
    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=None,
        help="Optional override for the final T3D AI_GX model directory.",
    )
    parser.add_argument(
        "--zeta",
        type=float,
        default=0.0,
        help="Field-line label used by the T3D AI_GX backend.",
    )
    parser.add_argument(
        "--filename",
        default="06_turbulent_heat_flux_vs_rho.png",
        help="Output filename for the q-vs-rho plot.",
    )
    return parser.parse_args()


def resolve_t3d_run_file(folder: Path) -> Path:
    folder = folder.expanduser().resolve()
    path = folder / "ti_optimization.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Expected netCDF file at {path}")
    return path


def resolve_t3d_model_dir(folder: Path, override: Path | None) -> Path:
    if override is not None:
        model_dir = override.expanduser().resolve()
    else:
        model_dir = (folder.expanduser().resolve() / "cyclic_invariant_models.v2")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Expected T3D AI_GX model directory at {model_dir}")
    return model_dir


def load_last_step_gradients(folder: Path) -> dict[str, np.ndarray]:
    path = resolve_t3d_run_file(folder)
    with Dataset(path) as ds:
        grid = ds.groups["grid"]
        species = ds.groups["species"]
        return {
            "midpoints": np.asarray(grid.variables["midpoints"][:], dtype=float),
            "aLT_H": np.asarray(species.variables["aLT_H"][-1, :], dtype=float),
            "aLn_H": np.asarray(species.variables["aLn_H"][-1, :], dtype=float),
        }


def load_eq(eq_path: str):
    from desc.io import load

    eq = load(eq_path)
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    return eq


def compute_grad_rho(eq, rho_values: np.ndarray) -> np.ndarray:
    from desc.grid import LinearGrid

    grid = LinearGrid(rho=np.asarray(rho_values, dtype=float))
    area = np.asarray(eq.compute("S(r)", grid=grid)["S(r)"], dtype=float)
    dVdrho = np.asarray(eq.compute("V_r(r)", grid=grid)["V_r(r)"], dtype=float)
    return area / dVdrho


def evaluate_with_nnitgproxy(
    eq,
    *,
    rho_values,
    a_over_LT,
    a_over_Ln,
    output_dir: str,
):
    from postprocessing import ProxyConfig, TurbulencePostprocessor

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proxy_config = ProxyConfig(
        ensemble_dir=os.path.join(repo_root, "data", "neural_networks", "cyclic_invariant_models"),
        ensemble_csv=os.path.join(repo_root, "data", "neural_networks", "deephyper_exp", "results.csv"),
        n_ensemble=3,
        target_flux_tube_length=75.4,
        a_over_LT=3.0,
        a_over_Ln=1.0,
    )
    post = TurbulencePostprocessor(output_dir, proxy_config)
    return post.evaluate_q_vs_rho(
        eq,
        rho_values=rho_values,
        alpha=0.0,
        a_over_LT=a_over_LT,
        a_over_Ln=a_over_Ln,
        n_ensemble=3,
        return_std=True,
    )


def evaluate_with_t3d_ai_gx(
    eq_path: str,
    *,
    model_dir: Path,
    rho_values,
    a_over_LT,
    a_over_Ln,
    zeta: float,
):
    sys.path.insert(0, str(T3D_ML_ROOT))
    from t3d.ai.inference_engine import AI_GX_InferenceEngine

    eq = load_eq(eq_path)
    grad_rho = compute_grad_rho(eq, rho_values)
    engine = AI_GX_InferenceEngine(model_dir=str(model_dir))

    q_values = []
    q_std_values = []
    for rho_value, a_over_LT_value, a_over_Ln_value, grad_rho_value in zip(
        np.asarray(rho_values, dtype=float),
        np.asarray(a_over_LT, dtype=float),
        np.asarray(a_over_Ln, dtype=float),
        np.asarray(grad_rho, dtype=float),
    ):
        q_local, q_std_local = engine.inference(
            eq=eq_path,
            flux_tube_s=float(rho_value) ** 2,
            zeta=float(zeta),
            tprims=float(a_over_LT_value),
            fprims=float(a_over_Ln_value),
        )
        scale = max(float(grad_rho_value), 1e-12)
        q_values.append(float(np.asarray(q_local, dtype=float).reshape(-1)[0]) / scale)
        q_std_values.append(float(np.asarray(q_std_local, dtype=float).reshape(-1)[0]) / scale)

    return np.asarray(q_values, dtype=float), np.asarray(q_std_values, dtype=float)


def save_plot(
    output_dir: str,
    filename: str,
    rho_values,
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    backend: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o-", "s-", "^-", "d-"]
    for idx, (label, q, q_std) in enumerate(curves):
        ax.errorbar(
            rho_values,
            q,
            yerr=q_std,
            fmt=markers[idx % len(markers)],
            capsize=4,
            label=label,
            color=f"C{idx}",
        )

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Q (ITG heat flux)")
    ax.set_title(f"Turbulent Heat Flux vs rho ({backend})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(f"Saved: {path}")
    return path


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    sys.path.insert(1, os.path.join(repo_root, "phase2"))

    gpu_preflight()

    output_dir = os.path.join(repo_root, "phase0", "output_v1")
    init_path, final_path = resolve_eq_paths(output_dir)

    _log(f"Using {init_path} and {final_path}")
    _log(f"Loading T3D gradients from {args.initial_t3d_dir}")
    initial_t3d = load_last_step_gradients(args.initial_t3d_dir)
    _log(f"Loading T3D gradients from {args.final_t3d_dir}")
    final_t3d = load_last_step_gradients(args.final_t3d_dir)

    if not np.allclose(
        initial_t3d["midpoints"],
        final_t3d["midpoints"],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Initial and final T3D midpoint grids do not match")

    rho_values = initial_t3d["midpoints"]
    _log(
        f"Starting q-vs-rho postprocessing for {output_dir} using backend={args.backend} "
        f"on {rho_values.size} rho surfaces from T3D"
    )

    if args.backend == "nnitgproxy":
        eq_init = load_eq(init_path)
        eq_final = load_eq(final_path)
        curves = [
            (
                "Initial",
                *evaluate_with_nnitgproxy(
                    eq_init,
                    rho_values=rho_values,
                    a_over_LT=initial_t3d["aLT_H"],
                    a_over_Ln=initial_t3d["aLn_H"],
                    output_dir=output_dir,
                ),
            ),
            (
                "Final",
                *evaluate_with_nnitgproxy(
                    eq_final,
                    rho_values=rho_values,
                    a_over_LT=final_t3d["aLT_H"],
                    a_over_Ln=final_t3d["aLn_H"],
                    output_dir=output_dir,
                ),
            ),
        ]
    else:
        initial_model_dir = resolve_t3d_model_dir(args.initial_t3d_dir, args.initial_model_dir)
        final_model_dir = resolve_t3d_model_dir(args.final_t3d_dir, args.final_model_dir)
        _log(f"Using initial T3D AI_GX model dir {initial_model_dir}")
        _log(f"Using final T3D AI_GX model dir {final_model_dir}")
        curves = [
            (
                "Initial",
                *evaluate_with_t3d_ai_gx(
                    init_path,
                    model_dir=initial_model_dir,
                    rho_values=rho_values,
                    a_over_LT=initial_t3d["aLT_H"],
                    a_over_Ln=initial_t3d["aLn_H"],
                    zeta=args.zeta,
                ),
            ),
            (
                "Final",
                *evaluate_with_t3d_ai_gx(
                    final_path,
                    model_dir=final_model_dir,
                    rho_values=rho_values,
                    a_over_LT=final_t3d["aLT_H"],
                    a_over_Ln=final_t3d["aLn_H"],
                    zeta=args.zeta,
                ),
            ),
        ]

    path = save_plot(
        output_dir,
        args.filename,
        rho_values,
        curves,
        backend=args.backend,
    )
    _log(f"Saved q-vs-rho postprocessing: {path}")


if __name__ == "__main__":
    main()
