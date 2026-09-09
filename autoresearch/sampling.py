"""Trial generation, mutation, and crossover."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

from .config import campaign_dir
from .state import score_key


def _sample_spec(rng: random.Random, spec: Any) -> Any:
    if isinstance(spec, list):
        return copy.deepcopy(rng.choice(spec))
    if not isinstance(spec, dict) or "type" not in spec:
        return copy.deepcopy(spec)
    kind = spec["type"]
    if kind == "choice":
        return copy.deepcopy(rng.choice(spec["values"]))
    if kind == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if kind == "loguniform":
        low = math.log(float(spec["low"]))
        high = math.log(float(spec["high"]))
        return math.exp(rng.uniform(low, high))
    raise ValueError(f"Unsupported search-space spec: {spec}")


def _grid_from_spec(spec: dict[str, Any]) -> list[float]:
    start = float(spec["start"])
    stop = float(spec["stop"])
    num = int(spec["num"])
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + step * idx for idx in range(num)]


def _normalize_sampled_config(raw: dict[str, Any], family: str) -> dict[str, Any]:
    config = dict(raw)
    radius = config.pop("initial_trust_radius", None)
    optimize_options = dict(config.get("optimize_options") or {})
    if radius is not None:
        optimize_options["initial_trust_radius"] = radius
    optimize_options.setdefault("perturb_options", {"verbose": 0})
    optimize_options.setdefault("solve_options", {"verbose": 0})
    config["optimize_options"] = optimize_options
    config["mode_steps"] = [int(item) for item in config.get("mode_steps", [2])]
    config["rho"] = [float(item) for item in config.get("rho", [0.124, 0.288, 0.453, 0.618])]
    config["alpha"] = [float(item) for item in config.get("alpha", [0.0])]
    config.setdefault("optimization_n_ensemble", 1)
    config.setdefault("analysis_n_ensemble", 3)

    if family == "phase2":
        config.setdefault("multi_alpha_rho", [math.sqrt(0.5)])
        config.setdefault("multi_alpha_values", [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0])
        config.setdefault("gradient_scan_rho", [math.sqrt(0.5)])
        config.setdefault("gradient_scan_alpha", [config["alpha"][0]])
        config.setdefault("gradient_scan_values", [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        config.setdefault("rho_scan_values", config["rho"])
        config.setdefault("rho_scan_alpha", [config["alpha"][0]])
    else:
        grid = config.get("a_over_LT_grid")
        if isinstance(grid, dict):
            config["a_over_LT_grid"] = _grid_from_spec(grid)
        config.setdefault("scan_rho", [math.sqrt(0.5)])
        config.setdefault("scan_alpha", config["alpha"])
        config.setdefault("rho_scan_values", config["rho"])
        config.setdefault("rho_scan_alpha", config["alpha"])
    return config


def sample_config(config: dict[str, Any], family: str, rng: random.Random) -> dict[str, Any]:
    search = config.get("search_space") or {}
    raw: dict[str, Any] = {}
    for key, spec in (search.get("shared") or {}).items():
        raw[key] = _sample_spec(rng, spec)
    for key, spec in (search.get(family) or {}).items():
        raw[key] = _sample_spec(rng, spec)
    return _normalize_sampled_config(raw, family)


def seed_configs(config: dict[str, Any], rng: random.Random) -> list[tuple[str, dict[str, Any], list[str]]]:
    families = list(config.get("families") or ["phase2", "phase3"])
    seeds: list[tuple[str, dict[str, Any], list[str]]] = []
    for family in families:
        base = sample_config(config, family, rng)
        base["mode_steps"] = [1, 2]
        base["maxiter"] = min(int(base.get("maxiter", 500)), 500)
        seeds.append((family, base, []))

        deeper = sample_config(config, family, rng)
        deeper["mode_steps"] = [1, 2, 3]
        seeds.append((family, deeper, []))
    return seeds


def mutate_config(
    config: dict[str, Any],
    family: str,
    parent_config: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    candidate = copy.deepcopy(parent_config)
    sampled = sample_config(config, family, rng)
    keys = list(sampled)
    if not keys:
        return candidate
    n_changes = max(1, min(4, len(keys) // 3))
    for key in rng.sample(keys, n_changes):
        candidate[key] = copy.deepcopy(sampled[key])
    return _normalize_sampled_config(candidate, family)


def crossover_config(
    config: dict[str, Any],
    family: str,
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    child = copy.deepcopy(parent_a)
    for key, value in parent_b.items():
        if rng.random() < 0.5:
            child[key] = copy.deepcopy(value)
    return mutate_config(config, family, child, rng)


def planned_trials_for_generation(
    config: dict[str, Any],
    generation: int,
    previous_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(config.get("random_seed", 7)) + generation * 7919)
    population_size = int(config.get("population_size", 16))
    state_dir = campaign_dir(config)
    campaign_id = str(config.get("campaign_id", "latest"))

    planned: list[tuple[str, dict[str, Any], list[str]]] = []
    if generation == 0:
        planned.extend(seed_configs(config, rng))
        while len(planned) < population_size:
            family = rng.choice(list(config.get("families") or ["phase2", "phase3"]))
            planned.append((family, sample_config(config, family, rng), []))
    else:
        completed = sorted(
            [record for record in (previous_records or []) if record.get("status") == "completed"],
            key=score_key,
        )
        elite_count = max(1, int(population_size * float(config.get("elite_fraction", 0.25))))
        elites = completed[:elite_count]
        confirm_n = int(config.get("confirm_elites_with_analysis_n_ensemble", 3))
        for elite in elites:
            elite_config = copy.deepcopy(elite.get("config") or {})
            elite_config["analysis_n_ensemble"] = confirm_n
            if "optimization_n_ensemble" in elite_config:
                elite_config["optimization_n_ensemble"] = max(1, int(elite_config["optimization_n_ensemble"]))
            planned.append((str(elite.get("family")), elite_config, [str(elite.get("run_id"))]))
        while len(planned) < population_size and elites:
            parent = rng.choice(elites)
            family = str(parent.get("family"))
            if len(elites) > 1 and rng.random() < 0.35:
                other = rng.choice(elites)
                child = crossover_config(
                    config,
                    family,
                    parent.get("config") or {},
                    other.get("config") or {},
                    rng,
                )
                parents = [str(parent.get("run_id")), str(other.get("run_id"))]
            else:
                child = mutate_config(config, family, parent.get("config") or {}, rng)
                parents = [str(parent.get("run_id"))]
            planned.append((family, child, parents))
        while len(planned) < population_size:
            family = rng.choice(list(config.get("families") or ["phase2", "phase3"]))
            planned.append((family, sample_config(config, family, rng), []))

    records: list[dict[str, Any]] = []
    for index, (family, trial_config, parent_run_ids) in enumerate(planned[:population_size]):
        run_id = f"g{generation:03d}_r{index:03d}"
        run_dir = state_dir / f"generation_{generation:02d}" / f"run_{run_id}"
        records.append(
            {
                "campaign_id": campaign_id,
                "generation": generation,
                "run_id": run_id,
                "family": family,
                "status": "pending",
                "score": None,
                "rank": None,
                "gpu": None,
                "start_time": None,
                "end_time": None,
                "elapsed_seconds": None,
                "config": trial_config,
                "metrics": {},
                "artifacts": {},
                "stdout_path": str(run_dir / "stdout.log"),
                "stderr_path": str(run_dir / "stderr.log"),
                "failure_reason": None,
                "parent_run_ids": parent_run_ids,
                "run_dir": str(run_dir),
                "trial_json": str(run_dir / "trial.json"),
            }
        )
    return records


def write_trial_json(record: dict[str, Any]) -> Path:
    path = Path(str(record["trial_json"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
