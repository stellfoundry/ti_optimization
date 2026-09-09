"""Plot autoresearch critical-gradient metrics with ensemble uncertainty."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import repo_root
from .state import load_latest_records, score_key


REPO_ROOT = repo_root()
PHASE3_DIR = REPO_ROOT / "phase3"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _resolve_path(raw_path: Any, campaign_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return campaign_dir / path


def _run_dir(record: dict[str, Any], campaign_dir: Path) -> Path:
    raw = record.get("run_dir")
    if raw:
        return _resolve_path(raw, campaign_dir)
    generation = int(record["generation"])
    run_id = str(record["run_id"])
    return campaign_dir / f"generation_{generation:02d}" / f"run_{run_id}"


def _artifact_path(record: dict[str, Any], name: str, campaign_dir: Path) -> Path:
    raw = (record.get("artifacts") or {}).get(name)
    if raw:
        return _resolve_path(raw, campaign_dir)
    return _run_dir(record, campaign_dir) / f"{name}.h5"


def _completed_phase3_records(
    records: list[dict[str, Any]], campaign_dir: Path
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if record.get("status") != "completed":
            continue
        if record.get("family") != "phase3":
            continue
        if not _artifact_path(record, "final", campaign_dir).exists():
            continue
        selected.append(record)
    return selected


def _select_records(
    records: list[dict[str, Any]], selection: str
) -> list[dict[str, Any]]:
    if selection == "all-completed":
        return sorted(
            records,
            key=lambda item: (
                int(item.get("generation", -1)),
                str(item.get("run_id", "")),
            ),
        )

    by_generation: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_generation.setdefault(int(record.get("generation", -1)), []).append(record)

    selected: list[dict[str, Any]] = []
    for generation in sorted(by_generation):
        candidates = by_generation[generation]
        if selection == "best-critical-gradient":
            selected.append(
                max(
                    candidates,
                    key=lambda item: _finite_float(
                        (item.get("metrics") or {}).get("critical_gradient"),
                        default=float("-inf"),
                    ),
                )
            )
        elif selection == "best-score":
            selected.append(min(candidates, key=score_key))
        else:
            raise ValueError(f"Unsupported selection: {selection}")
    return selected


def _phase3_runtime():
    if str(PHASE3_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE3_DIR))
    from desc.io import load
    from postprocessing import CriticalGradientPostprocessConfig, CriticalGradientPostprocessor

    return load, CriticalGradientPostprocessConfig, CriticalGradientPostprocessor


def _postprocess_config(record: dict[str, Any], n_ensemble: int, config_cls):
    config = record.get("config") or {}
    data_dir = REPO_ROOT / "data" / "neural_networks"
    return config_cls(
        ensemble_dir=str(data_dir / "cyclic_invariant_models"),
        ensemble_csv=str(data_dir / "deephyper_exp" / "results.csv"),
        n_ensemble=n_ensemble,
        target_flux_tube_length=float(config.get("target_flux_tube_length", 75.4)),
        a_over_Ln=float(config.get("a_over_Ln", 0.9)),
        a_over_LT_grid=np.asarray(
            config.get("a_over_LT_grid", np.linspace(0.5, 10.0, 20)),
            dtype=float,
        ),
        critical_q_threshold=float(config.get("critical_q_threshold", 1.0)),
        critical_log_width=float(config.get("critical_log_width", 0.35)),
        alpha_aggregation=str(config.get("alpha_aggregation", "mean")),
    )


def _metric_uncertainty(
    record: dict[str, Any],
    campaign_dir: Path,
    *,
    n_ensemble_override: int | None,
    force: bool,
) -> dict[str, Any]:
    config = record.get("config") or {}
    n_ensemble = int(n_ensemble_override or config.get("analysis_n_ensemble", 3))
    run_dir = _run_dir(record, campaign_dir)
    cache_path = run_dir / f"critical_gradient_uncertainty_n{n_ensemble:03d}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    load, config_cls, postprocessor_cls = _phase3_runtime()
    final_eq_path = _artifact_path(record, "final", campaign_dir)
    post_config = _postprocess_config(record, n_ensemble, config_cls)
    postprocessor = postprocessor_cls(str(run_dir), post_config)
    eq = load(str(final_eq_path))
    rho = np.asarray(config.get("rho", []), dtype=float)
    alpha = np.asarray(config.get("alpha", [0.0]), dtype=float)
    if rho.size == 0:
        raise ValueError(f"{record.get('run_id')} does not have a rho grid in its config")

    uncertainty = postprocessor.evaluate_critical_gradient_metric_uncertainty(
        eq,
        rho=rho,
        alpha=alpha,
        alpha_aggregation=post_config.alpha_aggregation,
    )
    metrics = record.get("metrics") or {}
    payload = {
        "run_id": record.get("run_id"),
        "generation": record.get("generation"),
        "score": record.get("score"),
        "n_ensemble": n_ensemble,
        "selection_metrics": {
            "critical_gradient_existing": metrics.get("critical_gradient"),
            "critical_gradient_delta_existing": metrics.get("critical_gradient_delta"),
            "aspect_ratio": metrics.get("aspect_ratio"),
        },
        "config": {
            "rho": rho.tolist(),
            "alpha": alpha.tolist(),
            "alpha_aggregation": post_config.alpha_aggregation,
            "a_over_Ln": post_config.a_over_Ln,
            "a_over_LT_grid": post_config.a_over_LT_grid.tolist(),
            "critical_q_threshold": post_config.critical_q_threshold,
            "critical_log_width": post_config.critical_log_width,
        },
        "final_eq_path": str(final_eq_path),
        **uncertainty,
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _configure_device(device: str) -> None:
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from desc import set_device

    if device == "cpu":
        set_device("cpu")
        return
    if device == "gpu":
        set_device("gpu")
        return
    try:
        set_device("gpu")
    except Exception:
        set_device("cpu")


def _plot_results(
    results: list[dict[str, Any]],
    output_path: Path,
    *,
    selection: str,
    yerr_key: str,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generations = np.asarray([int(item["generation"]) for item in results], dtype=float)
    means = np.asarray(
        [float(item["critical_gradient"]) for item in results], dtype=float
    )
    errors = np.asarray([float(item[yerr_key]) for item in results], dtype=float)
    scores = np.asarray([_finite_float(item.get("score")) for item in results], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if selection == "all-completed":
        ax.errorbar(
            generations,
            means,
            yerr=errors,
            fmt="o",
            ms=4,
            alpha=0.55,
            capsize=2,
            linewidth=0.8,
            color="C0",
            ecolor="0.55",
            label="Completed runs",
        )
    else:
        ax.errorbar(
            generations,
            means,
            yerr=errors,
            fmt="o-",
            ms=5,
            capsize=3,
            linewidth=1.5,
            color="C0",
            ecolor="0.4",
            label=selection.replace("-", " "),
        )

    if scores.size and np.any(np.isfinite(scores)):
        best_idx = int(np.nanargmin(scores))
        ax.scatter(
            [generations[best_idx]],
            [means[best_idx]],
            s=90,
            facecolors="none",
            edgecolors="C3",
            linewidths=1.8,
            label=f"Best score: {results[best_idx]['run_id']}",
            zorder=5,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Final $a/L_{T,crit}$")
    ax.set_title("Autoresearch critical gradient by generation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(
        0.01,
        0.01,
        "Centerline: same ensemble-mean critical-gradient metric used by index.html. "
        f"Error bars: {yerr_key.replace('critical_gradient_ensemble_', '')} "
        "of per-model propagated metrics.",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        default=str(REPO_ROOT / "autoresearch" / "output" / "latest"),
        help="Autoresearch campaign directory containing runs.jsonl.",
    )
    parser.add_argument(
        "--selection",
        choices=("best-score", "best-critical-gradient", "all-completed"),
        default="best-score",
        help="Which completed phase3 runs to evaluate.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        help="Evaluate a specific run id. May be supplied more than once.",
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        help="Override per-run analysis_n_ensemble. Omit to use each run config.",
    )
    parser.add_argument(
        "--error",
        choices=("std", "sem"),
        default="std",
        help="Use ensemble standard deviation or standard error as the plotted error bar.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="DESC/JAX device to request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute per-run uncertainty caches.",
    )
    parser.add_argument(
        "--output",
        help="PNG output path. Defaults inside the campaign directory.",
    )
    parser.add_argument(
        "--json-output",
        help="JSON output path. Defaults next to the PNG output.",
    )
    args = parser.parse_args(argv)

    if args.n_ensemble is not None and args.n_ensemble < 1:
        raise ValueError("--n-ensemble must be positive")

    _configure_device(args.device)

    campaign_dir = Path(args.campaign_dir).resolve()
    records = _completed_phase3_records(load_latest_records(campaign_dir), campaign_dir)
    if args.run_id:
        requested = set(args.run_id)
        selected = [record for record in records if str(record.get("run_id")) in requested]
        missing = requested - {str(record.get("run_id")) for record in selected}
        if missing:
            raise ValueError(f"Requested run ids were not completed phase3 runs: {sorted(missing)}")
    else:
        selected = _select_records(records, args.selection)
    if not selected:
        raise ValueError(f"No completed phase3 records found in {campaign_dir}")

    results: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {record.get('run_id')} "
            f"generation={record.get('generation')}",
            flush=True,
        )
        results.append(
            _metric_uncertainty(
                record,
                campaign_dir,
                n_ensemble_override=args.n_ensemble,
                force=args.force,
            )
        )

    output_path = (
        Path(args.output)
        if args.output
        else campaign_dir / f"critical_gradient_by_generation_{args.selection}_uncertainty.png"
    )
    json_path = (
        Path(args.json_output)
        if args.json_output
        else output_path.with_suffix(".json")
    )
    yerr_key = (
        "critical_gradient_ensemble_std"
        if args.error == "std"
        else "critical_gradient_ensemble_sem"
    )
    _plot_results(results, output_path, selection=args.selection, yerr_key=yerr_key)
    json_path.write_text(
        json.dumps(
            {
                "campaign_dir": str(campaign_dir),
                "selection": args.selection,
                "error": args.error,
                "results": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
