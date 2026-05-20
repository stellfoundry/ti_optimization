"""Run the phase0 turbulence postprocessing suite on phase3 equilibria."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PLOT_CHOICES = ("03", "04", "05", "06", "07")
DEFAULT_PLOTS = ("04", "05", "06", "07")


def _log(*parts) -> None:
    """Emit a UTC-timestamped log line."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}]", *parts, flush=True)


def resolve_eq_paths(output_dir: Path) -> tuple[Path, Path]:
    """Resolve the initial/final equilibrium pair within an output directory."""
    candidates = (
        ("initial.h5", "final.h5"),
        ("03_initial.h5", "03_final.h5"),
    )
    for init_name, final_name in candidates:
        init_path = output_dir / init_name
        final_path = output_dir / final_name
        if init_path.is_file() and final_path.is_file():
            return init_path, final_path
    raise FileNotFoundError(f"No equilibrium pair found in {output_dir}")


def normalize_plots(raw_plots: list[str]) -> tuple[str, ...]:
    """Expand the user plot selection into an ordered plot list."""
    if "all" in raw_plots:
        return PLOT_CHOICES
    return tuple(plot for plot in PLOT_CHOICES if plot in raw_plots)


def configure_device(device: str) -> None:
    """Select the requested DESC/JAX device, with optional GPU fallback."""
    from desc import set_device
    import jax

    _log("python executable:", sys.executable)
    _log("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    _log("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
    _log("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS"))

    if device in {"auto", "gpu"}:
        try:
            import torch

            _log("torch.cuda.is_available():", torch.cuda.is_available())
            _log("torch.cuda.device_count():", torch.cuda.device_count())
        except Exception as exc:
            _log(f"torch CUDA probe failed: {exc}")

        try:
            set_device("gpu")
            platforms = [dev.platform for dev in jax.devices()]
            _log("jax.default_backend():", jax.default_backend())
            _log("jax platforms:", platforms)
            if "gpu" in platforms:
                return
            raise RuntimeError(f"GPU backend unavailable; platforms={platforms}")
        except Exception as exc:
            if device == "gpu":
                raise SystemExit(f"Failed to initialize GPU backend: {exc}") from exc
            _log(f"GPU unavailable ({exc}); falling back to CPU")

    set_device("cpu")
    _log("Using CPU backend")


def load_phase0_postprocessing_module(repo_root: Path):
    """Load phase0/postprocessing.py without conflicting with phase3/postprocessing.py."""
    phase2_dir = str(repo_root / "phase2")
    if phase2_dir not in sys.path:
        sys.path.insert(0, phase2_dir)

    module_path = repo_root / "phase0" / "postprocessing.py"
    spec = importlib.util.spec_from_file_location("phase0_postprocessing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_equilibrium(eq_path: Path):
    """Load one equilibrium object from an HDF5 file."""
    from desc.io import load

    eq = load(str(eq_path))
    if hasattr(eq, "__len__"):
        eq = eq[-1]
    return eq


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "phase3" / "output",
        help="Directory containing the saved initial/final equilibria.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all",) + PLOT_CHOICES,
        default=list(DEFAULT_PLOTS),
        help="Plots to generate. Default: 04 05 06 07",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Execution device for DESC/JAX work.",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=3,
        help="Number of surrogate models to average in postprocessing.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("Initial", "Final"),
        metavar=("INITIAL_LABEL", "FINAL_LABEL"),
        help="Legend labels for the two equilibria.",
    )
    parser.add_argument(
        "--gradient-scan-a-over-Ln",
        type=float,
        default=None,
        help="Optional fixed a/Ln value for the temperature-gradient scan.",
    )
    parser.add_argument(
        "--transport-verbose",
        action="store_true",
        help="Enable verbose transport-solver logging for plot 07.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = args.output_dir.expanduser().resolve()
    plots = normalize_plots(list(args.plots))

    os.chdir(repo_root)
    configure_device(args.device)

    phase0_post = load_phase0_postprocessing_module(repo_root)
    init_path, final_path = resolve_eq_paths(output_dir)
    _log(f"Using {init_path} and {final_path}")

    proxy_config = phase0_post.ProxyConfig(
        ensemble_dir=str(repo_root / "data" / "neural_networks" / "cyclic_invariant_models"),
        ensemble_csv=str(repo_root / "data" / "neural_networks" / "deephyper_exp" / "results.csv"),
        n_ensemble=args.n_ensemble,
        target_flux_tube_length=75.4,
        a_over_LT=3.0,
        a_over_Ln=1.0,
    )

    _log(f"Loading equilibria from {output_dir}")
    initial_eq = load_equilibrium(init_path)
    final_eq = load_equilibrium(final_path)
    post = phase0_post.TurbulencePostprocessor(str(output_dir), proxy_config)

    labels = list(args.labels)
    saved_paths: dict[str, str] = {}

    if "03" in plots:
        _log("Running plot 03: flux-surface comparison")
        saved_paths["03"] = post.plot_flux_surface_comparison(
            [initial_eq, final_eq],
            labels,
        )
    if "04" in plots:
        _log("Running plot 04: multi-alpha comparison")
        saved_paths["04"] = post.plot_multi_alpha_comparison(
            [initial_eq, final_eq],
            labels,
            rho=np.sqrt(0.5),
            alpha_values=np.linspace(0.0, np.pi, 8),
            n_ensemble=args.n_ensemble,
        )
    if "05" in plots:
        _log("Running plot 05: temperature-gradient scan")
        saved_paths["05"] = post.plot_temperature_gradient_scan(
            [initial_eq, final_eq],
            labels,
            rho=np.sqrt(0.5),
            alpha=0.0,
            a_over_LT_values=np.linspace(0.5, 10.0, 20),
            a_over_Ln=args.gradient_scan_a_over_Ln,
            n_ensemble=args.n_ensemble,
        )
    if "06" in plots:
        _log("Running plot 06: turbulent heat flux vs rho")
        saved_paths["06"] = post.plot_q_vs_rho(
            [initial_eq, final_eq],
            labels,
            rho_values=np.linspace(0.05, 0.95, 12),
            alpha=0.0,
            n_ensemble=args.n_ensemble,
        )
    if "07" in plots:
        _log("Running plot 07: transport temperature estimate")
        transport_config = phase0_post.TransportEstimateConfig(verbose=args.transport_verbose)
        saved_paths["07"] = post.plot_transport_temperature_estimate(
            [initial_eq, final_eq],
            labels,
            transport_config=transport_config,
            n_ensemble=args.n_ensemble,
        )

    _log("Saved paths:")
    for key, path in saved_paths.items():
        _log(f"  {key}: {path}")


if __name__ == "__main__":
    main()
