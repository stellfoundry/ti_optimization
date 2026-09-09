"""Run one materialized autoresearch trial."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import repo_root
from .metrics import artifact_inventory, extract_metrics
from .scoring import score_result
from .state import utc_now


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _import_driver(family: str):
    phase_dir = repo_root() / family
    if family == "phase2":
        module_name = "qs_turbulence_opt_multi"
    elif family == "phase3":
        module_name = "qs_critical_gradient_opt"
    else:
        raise ValueError(f"Unsupported family: {family}")

    sys.path.insert(0, str(phase_dir))
    for name in ("postprocessing", "transport", "critical_gradient_objective", module_name):
        sys.modules.pop(name, None)
    return importlib.import_module(module_name)


def _coerce_for_field(value: Any, current: Any) -> Any:
    if isinstance(current, np.ndarray):
        return np.asarray(value, dtype=float)
    if isinstance(current, tuple):
        return tuple(value)
    return value


def _build_config(driver, overrides: dict[str, Any]):
    config = driver.OptimizationConfig()
    valid_fields = {field.name for field in fields(config)}
    for key, value in overrides.items():
        if key not in valid_fields:
            continue
        setattr(config, key, _coerce_for_field(value, getattr(config, key)))
    return config


def _build_paths(driver, run_dir: Path):
    base = driver.ProjectPaths.from_script(str(Path(driver.__file__).resolve()))
    return replace(
        base,
        output_dir=str(run_dir),
        initial_eq_path=str(run_dir / "initial.h5"),
        final_eq_path=str(run_dir / "final.h5"),
    )


def _save_stage_artifacts(eq_family, mode_steps: list[int], run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for index, mode in enumerate(mode_steps, start=1):
        if index >= len(eq_family):
            break
        path = run_dir / f"stage_{index:02d}_mode_{int(mode)}.h5"
        eq_family[index].save(str(path))
        artifacts[f"stage_{index:02d}_mode_{int(mode)}"] = str(path)
    return artifacts


def _geometry_metrics(eq) -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        data = eq.compute("R0/a")
        metrics["aspect_ratio"] = float(np.asarray(data["R0/a"], dtype=float).reshape(-1)[0])
    except Exception:
        pass
    return metrics


def run_trial(record: dict[str, Any], score_config: dict[str, Any] | None = None) -> dict[str, Any]:
    run_dir = Path(str(record["run_dir"])).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = record.get("gpu")
    if gpu is not None and os.getenv("AUTORESEARCH_SKIP_CUDA_VISIBLE_DEVICES") != "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(
        "[autoresearch] trial startup:"
        f" run_id={record.get('run_id')}"
        f" family={record.get('family')}"
        f" gpu_label={gpu}"
        f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}"
        f" SLURM_STEP_GPUS={os.getenv('SLURM_STEP_GPUS')}",
        flush=True,
    )

    start = time.monotonic()
    result = dict(record)
    result.update(
        {
            "status": "running",
            "start_time": result.get("start_time") or utc_now(),
            "end_time": None,
            "failure_reason": None,
        }
    )

    family = str(result["family"])
    trial_config = dict(result.get("config") or {})
    artifacts: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    status = "completed"
    failure_reason = None

    try:
        try:
            from desc import set_device

            set_device("gpu")
        except Exception:
            pass
        driver = _import_driver(family)
        if hasattr(driver, "configure_device"):
            driver.configure_device()
        paths = _build_paths(driver, run_dir)
        driver.ensure_output_dir(paths)
        if not os.path.exists(paths.model_dir):
            raise FileNotFoundError(f"Model directory not found: {paths.model_dir}")
        config = _build_config(driver, trial_config)
        eq_family = driver.run_optimization(config, paths)
        artifacts.update(_save_stage_artifacts(eq_family, list(trial_config.get("mode_steps") or []), run_dir))
        saved_paths = driver.run_postprocessing(eq_family, config, paths) or {}
        artifacts.update({str(name): str(path) for name, path in saved_paths.items()})
        if family == "phase3" and hasattr(driver, "summarize_critical_gradient"):
            driver.summarize_critical_gradient(paths, config)
        metrics = extract_metrics(run_dir, family, trial_config)
        metrics.update(_geometry_metrics(eq_family[-1]))
        artifacts.update(artifact_inventory(run_dir))
    except Exception as exc:
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        artifacts.update(artifact_inventory(run_dir))

    elapsed = time.monotonic() - start
    result.update(
        {
            "status": status,
            "end_time": utc_now(),
            "elapsed_seconds": elapsed,
            "metrics": _jsonify(metrics),
            "artifacts": _jsonify(artifacts),
            "failure_reason": failure_reason,
        }
    )
    result["score"] = score_result(
        family=family,
        status=status,
        metrics=metrics,
        config=trial_config,
        elapsed_seconds=elapsed,
        score_config=score_config,
    )
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-json", required=True, help="Path to a materialized trial JSON file.")
    parser.add_argument("--score-json", help="Optional path to campaign score config JSON.")
    args = parser.parse_args(argv)

    trial_path = Path(args.trial_json)
    record = json.loads(trial_path.read_text(encoding="utf-8"))
    score_config = None
    if args.score_json:
        score_config = json.loads(Path(args.score_json).read_text(encoding="utf-8"))
    result = run_trial(record, score_config=score_config)
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
