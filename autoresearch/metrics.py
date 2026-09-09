"""Metric extraction from phase2 and phase3 trial artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float(value: Any) -> float | None:
    try:
        result = float(np.asarray(value, dtype=float).reshape(-1)[0])
    except Exception:
        return None
    if not np.isfinite(result):
        return None
    return result


def _array_stats(prefix: str, value: Any) -> dict[str, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _phase2_metrics(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    final_diag = run_dir / "07_transport_temperature_estimate_optimized_diagnostics.npz"
    if not final_diag.exists():
        final_diag = run_dir / "07_transport_temperature_estimate_final_diagnostics.npz"
    if final_diag.exists():
        payload = _load_npz(final_diag)
        for key in ("nn_heat_flux_samples", "Q_faces", "q_nodes", "Ti", "residual"):
            if key in payload:
                metrics.update(_array_stats(key, payload[key]))
        if "nn_heat_flux_samples" in payload:
            metrics["mean_heat_flux"] = metrics.get("nn_heat_flux_samples_mean")
            metrics["turbulence"] = metrics.get("nn_heat_flux_samples_mean")
        elif "Q_faces" in payload:
            metrics["mean_heat_flux"] = metrics.get("Q_faces_mean")
            metrics["turbulence"] = metrics.get("Q_faces_mean")
        if "success" in payload:
            success = _as_float(payload["success"])
            metrics["transport_success"] = bool(success) if success is not None else None
        metrics["transport_diagnostics"] = str(final_diag)
    return metrics


def _phase3_metrics(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    summary = run_dir / "critical_gradient_summary.txt"
    rows: list[tuple[float, float, float, float]] = []
    if summary.exists():
        for line in summary.read_text(encoding="ascii").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    rows.append(tuple(float(part) for part in parts[:4]))
                except ValueError:
                    continue
    if rows:
        data = np.asarray(rows, dtype=float)
        metrics["critical_gradient_initial"] = float(np.mean(data[:, 1]))
        metrics["critical_gradient"] = float(np.mean(data[:, 2]))
        metrics["critical_gradient_delta"] = float(np.mean(data[:, 3]))
        target = float(config.get("critical_gradient_target", 3.0))
        metrics["critical_gradient_deficit"] = float(
            max(target - metrics["critical_gradient"], 0.0)
        )
        metrics["critical_gradient_summary"] = str(summary)
    return metrics


def artifact_inventory(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(run_dir.iterdir()):
        if path.suffix.lower() in {".h5", ".png", ".npz", ".txt", ".json"}:
            artifacts[path.stem] = str(path)
    return artifacts


def extract_metrics(run_dir: Path, family: str, config: dict[str, Any]) -> dict[str, Any]:
    metrics = _phase2_metrics(run_dir) if family == "phase2" else _phase3_metrics(run_dir, config)
    continuation_path = run_dir / "continuation_stages.json"
    if continuation_path.exists():
        try:
            stages = json.loads(continuation_path.read_text(encoding="ascii"))
        except json.JSONDecodeError:
            stages = []
        completed = [stage for stage in stages if stage.get("status") == "completed"]
        metrics["completed_stages"] = len(completed)
        metrics["stage_elapsed_seconds"] = [
            float(stage.get("elapsed_seconds", 0.0)) for stage in stages
        ]
        metrics["continuation_stage_count"] = len(stages)
    else:
        metrics["completed_stages"] = len(list(run_dir.glob("stage_*_mode_*.h5")))
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return metrics
