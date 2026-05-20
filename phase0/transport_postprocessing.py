"""Run transport postprocessing for saved equilibria."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone


def _log(*parts) -> None:
    """Emit a UTC-timestamped log line."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}]", *parts, flush=True)


def resolve_eq_paths(output_dir: str) -> tuple[str, str]:
    """Resolve the initial/final equilibrium pair within an output directory."""
    candidates = [
        ("initial.h5", "final.h5"),
        ("03_initial.h5", "03_final.h5"),
    ]
    for init_name, final_name in candidates:
        init_path = os.path.join(output_dir, init_name)
        final_path = os.path.join(output_dir, final_name)
        if os.path.exists(init_path) and os.path.exists(final_path):
            return init_path, final_path
    raise FileNotFoundError(f"No equilibrium pair found in {output_dir}")


def gpu_preflight() -> None:
    """Fail fast if the batch job does not expose a usable GPU to Torch and JAX."""
    import torch

    _log("python executable:", sys.executable)
    _log("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    _log("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
    _log("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS"))

    _log("torch.cuda.is_available():", torch.cuda.is_available())
    _log("torch.cuda.device_count():", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False in GPU batch job")
    _log("torch device 0:", torch.cuda.get_device_name(0))

    from desc import set_device
    import jax

    set_device("gpu")
    devices = jax.devices()
    platforms = [device.platform for device in devices]
    _log("jax.default_backend():", jax.default_backend())
    _log("jax.devices():", devices)
    _log("jax platforms:", platforms)
    if "gpu" not in platforms:
        raise SystemExit("desc.set_device('gpu') did not expose a GPU-backed JAX device")


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)

    gpu_preflight()

    from desc.io import load
    from postprocessing import ProxyConfig, TransportEstimateConfig, TurbulencePostprocessor

    model_dir = os.path.join(repo_root, "data", "neural_networks", "cyclic_invariant_models")
    model_csv = os.path.join(repo_root, "data", "neural_networks", "deephyper_exp", "results.csv")

    proxy_config = ProxyConfig(
        ensemble_dir=model_dir,
        ensemble_csv=model_csv,
        n_ensemble=3,
        target_flux_tube_length=75.4,
        a_over_LT=3.0,
        a_over_Ln=1.0,
    )
    transport_config = TransportEstimateConfig(verbose=True)

    targets = [
        os.path.join(repo_root, "phase0", "output"),
        os.path.join(repo_root, "phase2", "output"),
    ]

    for output_dir in targets:
        init_path, final_path = resolve_eq_paths(output_dir)
        _log(f"Using {init_path} and {final_path}")
        post = TurbulencePostprocessor(output_dir, proxy_config)
        _log(f"Loading equilibria from {output_dir}")
        eq_init = load(init_path)
        eq_final = load(final_path)
        _log(f"Starting transport postprocessing for {output_dir}")
        path = post.plot_transport_temperature_estimate(
            [eq_init, eq_final],
            ["Initial", "Optimized"],
            transport_config=transport_config,
            n_ensemble=3,
            filename="07_transport_temperature_estimate.png",
        )
        _log(f"Saved transport postprocessing: {path}")


if __name__ == "__main__":
    main()
