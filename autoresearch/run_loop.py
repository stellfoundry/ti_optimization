"""Run a restartable generation-based autoresearch loop on local GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import campaign_dir, configured_gpu_ids, load_config, repo_root
from .render_dashboard import render_dashboard
from .sampling import planned_trials_for_generation, write_trial_json
from .state import (
    FINAL_STATUSES,
    append_record,
    latest_by_run,
    load_jsonl,
    load_latest_records,
    mark_stale_running_records,
    utc_now,
    write_state,
)


def _load_result_or_failure(record: dict[str, Any], returncode: int) -> dict[str, Any]:
    run_dir = Path(str(record["run_dir"]))
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["stdout_path"] = record.get("stdout_path")
        result["stderr_path"] = record.get("stderr_path")
        result["gpu"] = record.get("gpu")
        if returncode != 0 and result.get("status") == "completed":
            result["status"] = "failed"
            result["failure_reason"] = f"Trial process returned {returncode} after writing completed result."
        return result
    result = dict(record)
    result.update(
        {
            "status": "failed",
            "end_time": utc_now(),
            "score": 1_000_000.0,
            "failure_reason": f"Trial process returned {returncode} without writing result.json.",
        }
    )
    return result


def _materialize_generation(
    config: dict[str, Any],
    generation: int,
    state_dir: Path,
    previous_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = latest_by_run(load_jsonl(state_dir / "runs.jsonl"))
    planned = planned_trials_for_generation(config, generation, previous_records)
    for record in planned:
        existing = latest.get(str(record["run_id"]))
        if existing and existing.get("status") in FINAL_STATUSES:
            continue
        if existing and existing.get("status") == "pending":
            write_trial_json(existing)
            continue
        write_trial_json(record)
        append_record(state_dir, record)
    return planned


def _generation_records(state_dir: Path, generation: int) -> list[dict[str, Any]]:
    records = load_latest_records(state_dir)
    return [record for record in records if int(record.get("generation", -1)) == generation]


def _generation_finished(state_dir: Path, generation: int, population_size: int) -> bool:
    records = _generation_records(state_dir, generation)
    if len(records) < population_size:
        return False
    return all(str(record.get("status")) in FINAL_STATUSES for record in records)


def _pending_records(state_dir: Path, generation: int) -> list[dict[str, Any]]:
    return [
        record
        for record in _generation_records(state_dir, generation)
        if record.get("status") == "pending"
    ]


def _run_generation(
    config: dict[str, Any],
    state_dir: Path,
    generation: int,
    gpu_ids: list[str],
    concurrency: int,
) -> None:
    score_path = state_dir / "score_config.json"
    score_path.write_text(json.dumps(config.get("score") or {}, indent=2) + "\n", encoding="utf-8")
    use_srun = bool(os.getenv("SLURM_JOB_ID"))
    cpus_per_trial = max(
        1,
        int(os.getenv("SLURM_CPUS_ON_NODE", str(max(1, concurrency)))) // max(1, concurrency),
    )

    inflight: dict[str, tuple[subprocess.Popen, dict[str, Any], Any, Any]] = {}
    while True:
        pending = _pending_records(state_dir, generation)
        if not pending and not inflight:
            break

        busy_gpus = set(inflight)
        free_gpus = [gpu for gpu in gpu_ids if gpu not in busy_gpus]
        while pending and free_gpus and len(inflight) < concurrency:
            gpu = free_gpus.pop(0)
            record = dict(pending.pop(0))
            record["gpu"] = gpu
            record["start_time"] = utc_now()
            record["status"] = "running"
            write_trial_json(record)

            run_dir = Path(str(record["run_dir"]))
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout = open(record["stdout_path"], "ab")
            stderr = open(record["stderr_path"], "ab")
            env = os.environ.copy()
            if use_srun:
                env["AUTORESEARCH_SKIP_CUDA_VISIBLE_DEVICES"] = "1"
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(repo_root()) + os.pathsep + env.get("PYTHONPATH", "")
            trial_cmd = [
                sys.executable,
                "-m",
                "autoresearch.run_trial",
                "--trial-json",
                str(record["trial_json"]),
                "--score-json",
                str(score_path),
            ]
            if use_srun:
                cmd = [
                    "srun",
                    "--exclusive",
                    "--nodes=1",
                    "--ntasks=1",
                    "--gpus=1",
                    "--gpu-bind=single:1",
                    f"--cpus-per-task={cpus_per_trial}",
                    *trial_cmd,
                ]
            else:
                cmd = trial_cmd
            append_record(state_dir, record)
            write_state(state_dir, config)
            render_dashboard(state_dir)
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root()),
                stdout=stdout,
                stderr=stderr,
                env=env,
            )
            inflight[gpu] = (proc, record, stdout, stderr)
            print(f"Launched {record['run_id']} on GPU {gpu}", flush=True)

        completed_gpus: list[str] = []
        for gpu, (proc, record, stdout, stderr) in list(inflight.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            stdout.close()
            stderr.close()
            result = _load_result_or_failure(record, int(returncode))
            append_record(state_dir, result)
            write_state(state_dir, config)
            render_dashboard(state_dir)
            completed_gpus.append(gpu)
            print(
                f"Finished {record['run_id']} on GPU {gpu}: "
                f"{result.get('status')} score={result.get('score')}",
                flush=True,
            )
        for gpu in completed_gpus:
            inflight.pop(gpu, None)

        if inflight:
            time.sleep(5.0)


def run_loop(config: dict[str, Any], gpu_ids: list[str]) -> Path:
    state_dir = campaign_dir(config)
    state_dir.mkdir(parents=True, exist_ok=True)
    mark_stale_running_records(state_dir)
    max_generations = int(config.get("max_generations", 1))
    population_size = int(config.get("population_size", 1))
    concurrency = min(int(config.get("concurrency", len(gpu_ids))), len(gpu_ids))

    for generation in range(max_generations):
        previous_records = load_latest_records(state_dir)
        _materialize_generation(config, generation, state_dir, previous_records)
        write_state(state_dir, config)
        render_dashboard(state_dir)
        if _generation_finished(state_dir, generation, population_size):
            print(f"Generation {generation} already finished; skipping.", flush=True)
            continue
        _run_generation(config, state_dir, generation, gpu_ids, concurrency)
        write_state(state_dir, config)
        render_dashboard(state_dir)
    return state_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to campaign config.")
    parser.add_argument("--gpus", help="Comma-separated GPU ids, for example 0,1,2,3.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    gpu_ids = configured_gpu_ids(config, args.gpus)
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")
    state_dir = run_loop(config, gpu_ids)
    print(f"Campaign state: {state_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
