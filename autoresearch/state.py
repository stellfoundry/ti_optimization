"""Append-friendly run logging and compact campaign state."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STATUSES = {"completed", "failed", "skipped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_by_run(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record.get("run_id", "unknown"))] = record
    return latest


def append_record(state_dir: Path, record: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "runs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def score_key(record: dict[str, Any]) -> float:
    try:
        score = record.get("score")
        if score is None:
            return float("inf")
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def ranked_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(record) for record in records]
    completed = sorted(
        (record for record in ranked if record.get("status") == "completed"),
        key=score_key,
    )
    ranks = {str(record.get("run_id")): rank for rank, record in enumerate(completed, start=1)}
    for record in ranked:
        record["rank"] = ranks.get(str(record.get("run_id")))
    return ranked


def load_latest_records(state_dir: Path) -> list[dict[str, Any]]:
    return list(latest_by_run(load_jsonl(state_dir / "runs.jsonl")).values())


def write_state(state_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    records = ranked_records(load_latest_records(state_dir))
    records.sort(key=lambda item: (int(item.get("generation", -1)), str(item.get("run_id", ""))))
    counts = Counter(str(record.get("status", "unknown")) for record in records)
    by_generation: dict[str, dict[str, int]] = {}
    for record in records:
        generation = str(record.get("generation", -1))
        by_generation.setdefault(generation, {})
        status = str(record.get("status", "unknown"))
        by_generation[generation][status] = by_generation[generation].get(status, 0) + 1
    completed = [record for record in records if record.get("status") == "completed"]
    best = min(completed, key=score_key, default=None)
    state = {
        "campaign_id": config.get("campaign_id"),
        "updated_at": utc_now(),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "generation_status_counts": by_generation,
        "best_run_id": None if best is None else best.get("run_id"),
        "best_score": None if best is None else best.get("score"),
        "runs": records,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return state


def mark_stale_running_records(state_dir: Path) -> list[dict[str, Any]]:
    """Turn interrupted running records into failed records on restart."""

    updates: list[dict[str, Any]] = []
    latest = latest_by_run(load_jsonl(state_dir / "runs.jsonl"))
    for record in latest.values():
        if record.get("status") != "running":
            continue
        updated = dict(record)
        updated.update(
            {
                "status": "failed",
                "end_time": utc_now(),
                "failure_reason": "Marked failed on restart because the run was still running.",
            }
        )
        append_record(state_dir, updated)
        updates.append(updated)
    return updates
