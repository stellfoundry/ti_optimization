# Codex `/goal`: Autoresearch loop for turbulence-AI-GX optimization

Implement an **autoresearch loop** for `ti_optimization` that repeatedly tries different optimization targets, constraints, and Fourier-continuation schedules for the phase2 and phase3 turbulence-AI-GX surrogate workflows. The loop must run efficiently on a node with **4 GPUs**, where each optimization uses **1 GPU** and usually takes **5--10 minutes**.

The implementation should be production-oriented but simple enough to iterate on. Prefer standard-library Python plus dependencies that are already in the repo environment. Do not patch external DESC code.

## Repository context

Current useful entry points:

- `phase2/qs_turbulence_opt_multi.py`: fixed-gradient, multi-rho/alpha NNITGProxy optimization.
- `phase3/qs_critical_gradient_opt.py`: critical-gradient optimization using `NNCriticalGradientProxy`.
- `phase2/postprocessing.py`, `phase3/postprocessing.py`: generate analysis plots/summaries after an optimization.
- Existing optimization configs are dataclasses named `OptimizationConfig`.
- Existing Fourier-mode freezing is controlled by `build_constraints(eq, max_mode)` and `mode_steps`; optimize low modes first, then relax higher modes.

## Desired deliverable

Create a new `autoresearch/` package with scripts, configs, and docs that can be run from the repo root. The user workflow should be roughly:

```bash
# dry-run the planned first generation
python -m autoresearch.plan --config autoresearch/configs/default.yaml --dry-run

# run the loop on a 4-GPU node
python -m autoresearch.run_loop --config autoresearch/configs/default.yaml --gpus 0,1,2,3

# rebuild dashboard only
python -m autoresearch.render_dashboard --state-dir autoresearch/output/latest
```

Also add an example Slurm script for a 4-GPU node:

```bash
sbatch autoresearch/run_autoresearch_4gpu.sbatch
```

## Research loop design

Use a generation-based loop:

1. **Generation 0** starts from a curated seed set of phase2 and phase3 configurations.
2. Run up to 4 trials concurrently, one per GPU, setting `CUDA_VISIBLE_DEVICES` per process.
3. Each trial writes all outputs into an isolated directory:
   `autoresearch/output/<campaign_id>/generation_<NN>/run_<run_id>/`.
4. After every trial, parse scalar metrics from its postprocessing outputs and write a machine-readable result record.
5. After all trials in a generation finish, rank trials, keep elites, mutate/crossover configs, and launch the next generation.
6. Continuously regenerate `autoresearch/output/<campaign_id>/index.html` so progress can be watched while the loop runs.

The loop should be restartable. If it is interrupted, rerunning with the same campaign directory should skip completed runs, mark incomplete/stale runs as failed or pending according to a clear rule, and continue.

## What to vary

Support both `phase2` and `phase3` trial families.

### Shared parameters

Vary at least:

- `mode_steps`: Fourier continuation schedules such as:
  - `[1]`
  - `[2]`
  - `[1, 2]`
  - `[1, 2, 3]`
  - `[2, 3, 4]`
  - `[1, 2, 3, 4]`
- `qs_weight`
- `aspect_ratio_target`
- `aspect_ratio_weight`
- optimizer tolerances: `maxiter`, `ftol`, `xtol`, `gtol`
- optimizer trust radius: `optimize_options.initial_trust_radius`
- radial grids `rho`
- alpha grids `alpha`
- postprocessing grids for rho, alpha, and gradient scans
- desc.objectives imported and applied (can apply or not apply what is currently defined, or pull in new)

### Phase2 fixed-gradient parameters

Vary at least:

- `turbulence_weight`
- `a_over_LT`
- `a_over_Ln`
- `optimization_n_ensemble` initially `1`; optionally test `2` for elite confirmation only
- target aggregation choices if implemented; otherwise use existing NNITGProxy behavior

### Phase3 critical-gradient parameters

Vary at least:

- `critical_gradient_weight`
- `a_over_Ln`
- `a_over_LT_grid`, including coarse and wider grids
- `critical_q_threshold`
- `critical_log_width`
- `alpha_aggregation`: `mean`, `worst`, and `separate` when compatible with objective shape
- `optimization_n_ensemble` initially `1`; optionally test `2` for elite confirmation only

## Fourier continuation requirement

Make Fourier continuation explicit in the autoresearch config. A run with `mode_steps: [1, 2, 3]` means:

1. optimize only boundary modes with `M,N <= 1` unfrozen;
2. use that result as the initial equilibrium;
3. unfreeze `M,N <= 2` and optimize again;
4. use that result as the initial equilibrium;
5. unfreeze `M,N <= 3` and optimize again.

Save per-stage artifacts when possible:

- final equilibrium for each continuation stage;
- objective summary for each stage;
- elapsed time per stage;
- failure reason if any stage fails.

If changing phase2/phase3 drivers, keep backwards-compatible defaults so existing `python phase2/qs_turbulence_opt_multi.py` and `python phase3/qs_critical_gradient_opt.py` still work.

## Ranking / scoring

Implement a configurable scalar score that combines:

- lower turbulence proxy heat flux for phase2;
- higher critical gradient / lower critical-gradient deficit for phase3;
- lower quasisymmetry residual;
- aspect ratio close to target;
- successful completion of all Fourier-continuation stages;
- penalties for crashes, NaN metrics, very long runtime, or invalid geometry.

The score does not need to be perfect on the first pass. Implement a clear default and store all raw metrics so I can change the scoring later.

Minimum record fields for every run:

```json
{
  "campaign_id": "...",
  "generation": 0,
  "run_id": "g000_r000",
  "family": "phase2|phase3",
  "status": "pending|running|completed|failed|skipped",
  "score": 1.23,
  "rank": 1,
  "gpu": "0",
  "start_time": "...",
  "end_time": "...",
  "elapsed_seconds": 612.3,
  "config": {},
  "metrics": {},
  "artifacts": {},
  "stdout_path": "...",
  "stderr_path": "...",
  "failure_reason": null,
  "parent_run_ids": []
}
```

Use JSONL for append-friendly logging and also maintain a compact `state.json` summary.

## Config format

Add `autoresearch/configs/default.yaml`. If PyYAML is not already available, either add a tiny JSON-compatible YAML reader for the simple config shape or use `.json` instead. Prefer a readable config like:

```yaml
campaign_id: latest
max_generations: 8
population_size: 16
concurrency: 4
random_seed: 7
output_root: autoresearch/output
families: [phase2, phase3]
elite_fraction: 0.25
mutation_fraction: 0.65
random_fraction: 0.10
confirm_elites_with_analysis_n_ensemble: 3

gpu:
  ids: [0, 1, 2, 3]
  one_process_per_gpu: true

score:
  weights:
    turbulence: 1.0
    critical_gradient: 1.0
    quasisymmetry: 0.25
    aspect_ratio: 0.25
    runtime: 0.05
    failure: 1000000.0

search_space:
  shared:
    mode_steps:
      - [1]
      - [2]
      - [1, 2]
      - [1, 2, 3]
      - [2, 3, 4]
      - [1, 2, 3, 4]
    qs_weight: {type: loguniform, low: 0.01, high: 1.0}
    aspect_ratio_target: {type: choice, values: [5.0, 5.5, 6.0, 6.5, 7.0]}
    aspect_ratio_weight: {type: loguniform, low: 0.02, high: 1.0}
    initial_trust_radius: {type: loguniform, low: 0.05, high: 1.0}
    maxiter: {type: choice, values: [250, 500, 750, 1000]}
    ftol: {type: loguniform, low: 1.0e-5, high: 1.0e-3}
    xtol: {type: loguniform, low: 1.0e-5, high: 1.0e-3}
    gtol: {type: loguniform, low: 1.0e-5, high: 1.0e-3}
    rho:
      - [0.124, 0.288, 0.453, 0.618]
      - [0.041, 0.124, 0.206, 0.288, 0.371, 0.453, 0.535, 0.618, 0.700]
      - [0.25, 0.5, 0.75]
    alpha:
      - [0.0]
      - [0.0, 1.5707963267948966]
      - [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345]
  phase2:
    turbulence_weight: {type: loguniform, low: 0.01, high: 1.0}
    a_over_LT: {type: uniform, low: 1.5, high: 5.0}
    a_over_Ln: {type: uniform, low: 0.3, high: 1.5}
    target_flux_tube_length: {type: choice, values: [50.0, 75.4, 100.0]}
  phase3:
    critical_gradient_weight: {type: loguniform, low: 0.01, high: 1.0}
    a_over_Ln: {type: uniform, low: 0.3, high: 1.5}
    a_over_LT_grid:
      - {start: 0.25, stop: 4.25, num: 8}
      - {start: 0.5, stop: 6.0, num: 12}
      - {start: 0.5, stop: 10.0, num: 20}
    critical_q_threshold: {type: loguniform, low: 0.3, high: 3.0}
    critical_log_width: {type: uniform, low: 0.15, high: 0.75}
    alpha_aggregation: {type: choice, values: [mean, worst]}
```

## Implementation details

### Trial execution

Add a trial runner that can receive a fully materialized trial config as JSON:

```bash
python -m autoresearch.run_trial --trial-json path/to/trial.json
```

The runner should:

1. set `CUDA_VISIBLE_DEVICES` from the trial record;
2. import the correct phase driver;
3. construct `OptimizationConfig` overrides from the trial config;
4. redirect output paths to the isolated run directory;
5. run optimization and postprocessing;
6. parse and write `result.json`;
7. return nonzero on failure after writing a useful failure record.

If the existing phase2/phase3 drivers are not easy to override, refactor them minimally so `run_optimization(config, paths)` and `run_postprocessing(...)` can be reused with custom `ProjectPaths`/`OptimizationConfig` values.

### GPU scheduler

Implement a simple local scheduler using `subprocess.Popen`:

- maintain a queue of pending trial JSON files;
- start at most one child process per GPU id;
- pass `CUDA_VISIBLE_DEVICES=<gpu_id>` to each child environment;
- write `stdout.log` and `stderr.log` in the run directory;
- poll until children finish;
- update JSONL/state/dashboard after each completion.

### HTML dashboard

Add `autoresearch/render_dashboard.py`. It should read JSONL/state files and generate a static `index.html` with:

- campaign metadata and last updated timestamp;
- progress by generation;
- status counts by generation;
- top overall run with score, key metrics, config, and artifact links;
- top runs per generation;
- sortable-looking tables for all generations;
- collapsible details for every run showing config, metrics, artifacts, stdout/stderr links, and failure reason;
- lightweight CSS only, no external CDN requirement.

Use this exact implementation as the starting point if helpful, adapting field names as needed:

```python
"""Render a static HTML dashboard for autoresearch campaigns."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "run_id": f"jsonl_error_{line_number}",
                        "generation": -1,
                        "status": "failed",
                        "score": None,
                        "failure_reason": f"Could not parse {path}:{line_number}: {exc}",
                    }
                )
    return records


def _latest_by_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        run_id = str(record.get("run_id", "unknown"))
        latest[run_id] = record
    return list(latest.values())


def _safe_score(record: dict[str, Any]) -> float:
    score = record.get("score")
    if score is None:
        return float("inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def _json_block(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2, sort_keys=True, default=str))


def _artifact_links(record: dict[str, Any], state_dir: Path) -> str:
    artifacts = record.get("artifacts") or {}
    links: list[str] = []
    for name, raw_path in sorted(artifacts.items()):
        if raw_path is None:
            continue
        path = Path(str(raw_path))
        try:
            href = path.resolve().relative_to(state_dir.resolve()).as_posix()
        except ValueError:
            href = path.as_posix()
        links.append(f'<a href="{html.escape(href)}">{html.escape(str(name))}</a>')
    for log_key in ("stdout_path", "stderr_path"):
        raw_path = record.get(log_key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        try:
            href = path.resolve().relative_to(state_dir.resolve()).as_posix()
        except ValueError:
            href = path.as_posix()
        links.append(f'<a href="{html.escape(href)}">{html.escape(log_key)}</a>')
    return "<br>".join(links) if links else "&mdash;"


def _status_badge(status: str) -> str:
    status_class = html.escape(status.lower().replace("_", "-"))
    return f'<span class="badge {status_class}">{html.escape(status)}</span>'


def _metric_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "&mdash;"
    preferred = [
        "objective",
        "turbulence",
        "mean_heat_flux",
        "critical_gradient",
        "critical_gradient_deficit",
        "quasisymmetry",
        "aspect_ratio",
        "runtime_seconds",
    ]
    parts: list[str] = []
    for key in preferred:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                value_text = f"{value:.6g}"
            else:
                value_text = str(value)
            parts.append(f"<b>{html.escape(key)}</b>: {html.escape(value_text)}")
    if not parts:
        for key, value in list(sorted(metrics.items()))[:6]:
            parts.append(f"<b>{html.escape(str(key))}</b>: {html.escape(str(value))}")
    return "<br>".join(parts)


def render_dashboard(state_dir: Path, output_path: Path | None = None) -> Path:
    state_dir = state_dir.resolve()
    output_path = output_path or state_dir / "index.html"
    records = _latest_by_run(_load_jsonl(state_dir / "runs.jsonl"))
    records.sort(key=lambda record: (int(record.get("generation", -1)), str(record.get("run_id", ""))))

    completed = [record for record in records if record.get("status") == "completed"]
    ranked = sorted(completed, key=_safe_score)
    best = ranked[0] if ranked else None

    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_generation[int(record.get("generation", -1))].append(record)

    generation_cards: list[str] = []
    for generation in sorted(by_generation):
        generation_records = by_generation[generation]
        counts = Counter(str(record.get("status", "unknown")) for record in generation_records)
        best_generation = min(
            (record for record in generation_records if record.get("status") == "completed"),
            key=_safe_score,
            default=None,
        )
        count_text = " ".join(
            f'{_status_badge(status)} <span class="count">{count}</span>'
            for status, count in sorted(counts.items())
        )
        best_text = "No completed runs yet."
        if best_generation is not None:
            best_text = (
                f'Best: <a href="#run-{html.escape(str(best_generation.get("run_id")))}">'
                f'{html.escape(str(best_generation.get("run_id")))}</a> '
                f'score={_safe_score(best_generation):.6g}'
            )
        generation_cards.append(
            f"""
            <section class="card generation-card">
              <h2>Generation {generation}</h2>
              <p>{count_text}</p>
              <p>{best_text}</p>
            </section>
            """
        )

    top_rows: list[str] = []
    for rank, record in enumerate(ranked[:10], start=1):
        top_rows.append(
            f"""
            <tr>
              <td>{rank}</td>
              <td><a href="#run-{html.escape(str(record.get('run_id')))}">{html.escape(str(record.get('run_id')))}</a></td>
              <td>{html.escape(str(record.get('family', '&mdash;')))}</td>
              <td>{html.escape(str(record.get('generation', '&mdash;')))}</td>
              <td>{_safe_score(record):.6g}</td>
              <td>{_metric_summary(record.get('metrics') or {})}</td>
            </tr>
            """
        )

    detail_rows: list[str] = []
    for record in sorted(records, key=lambda item: (_safe_score(item), int(item.get("generation", -1)), str(item.get("run_id", "")))):
        run_id = html.escape(str(record.get("run_id", "unknown")))
        score = _safe_score(record)
        score_text = "&mdash;" if score == float("inf") else f"{score:.6g}"
        failure = record.get("failure_reason")
        detail_rows.append(
            f"""
            <details class="run-detail" id="run-{run_id}">
              <summary>
                <span class="run-id">{run_id}</span>
                {_status_badge(str(record.get('status', 'unknown')))}
                <span>family={html.escape(str(record.get('family', '&mdash;')))}</span>
                <span>generation={html.escape(str(record.get('generation', '&mdash;')))}</span>
                <span>score={score_text}</span>
                <span>elapsed={html.escape(str(record.get('elapsed_seconds', '&mdash;')))}s</span>
              </summary>
              <div class="detail-grid">
                <section>
                  <h3>Metrics</h3>
                  <pre>{_json_block(record.get('metrics') or {})}</pre>
                </section>
                <section>
                  <h3>Config</h3>
                  <pre>{_json_block(record.get('config') or {})}</pre>
                </section>
                <section>
                  <h3>Artifacts</h3>
                  <p>{_artifact_links(record, state_dir)}</p>
                  <h3>Failure reason</h3>
                  <pre>{html.escape(str(failure)) if failure else '&mdash;'}</pre>
                </section>
                <section>
                  <h3>Raw record</h3>
                  <pre>{_json_block(record)}</pre>
                </section>
              </div>
            </details>
            """
        )

    best_panel = "<p>No completed runs yet.</p>"
    if best is not None:
        best_panel = f"""
        <p><b>Run:</b> <a href="#run-{html.escape(str(best.get('run_id')))}">{html.escape(str(best.get('run_id')))}</a></p>
        <p><b>Score:</b> {_safe_score(best):.6g}</p>
        <p><b>Family:</b> {html.escape(str(best.get('family', '&mdash;')))}</p>
        <p><b>Generation:</b> {html.escape(str(best.get('generation', '&mdash;')))}</p>
        <h3>Key metrics</h3>
        <p>{_metric_summary(best.get('metrics') or {})}</p>
        <h3>Artifacts</h3>
        <p>{_artifact_links(best, state_dir)}</p>
        """

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ti_optimization autoresearch dashboard</title>
  <style>
    :root {{ color-scheme: light dark; --bg: #0f172a; --panel: #111827; --text: #e5e7eb; --muted: #9ca3af; --line: #374151; --accent: #38bdf8; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 2rem; background: linear-gradient(135deg, #0f172a, #1e3a8a); border-bottom: 1px solid var(--line); }}
    header h1 {{ margin: 0 0 0.5rem; }}
    main {{ padding: 1.5rem; max-width: 1400px; margin: 0 auto; }}
    a {{ color: var(--accent); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    .card, .run-detail {{ background: color-mix(in srgb, var(--panel), white 3%); border: 1px solid var(--line); border-radius: 14px; padding: 1rem; box-shadow: 0 10px 30px rgb(0 0 0 / 0.18); }}
    .generation-card h2 {{ margin-top: 0; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.8rem; margin-right: 0.25rem; background: #475569; color: white; }}
    .badge.completed {{ background: #15803d; }} .badge.running {{ background: #0369a1; }} .badge.pending {{ background: #a16207; }} .badge.failed {{ background: #b91c1c; }} .badge.skipped {{ background: #64748b; }}
    .count {{ margin-right: 0.75rem; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 0.6rem; vertical-align: top; text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    details {{ margin: 0.75rem 0; }}
    summary {{ cursor: pointer; display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
    .run-id {{ font-weight: 700; color: var(--accent); }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    pre {{ max-height: 30rem; overflow: auto; background: #020617; border: 1px solid var(--line); padding: 0.75rem; border-radius: 10px; }}
  </style>
</head>
<body>
  <header>
    <h1>ti_optimization autoresearch dashboard</h1>
    <p>State directory: {html.escape(str(state_dir))}</p>
    <p>Last updated: {html.escape(now)}</p>
  </header>
  <main>
    <section class="card">
      <h2>Top overall run</h2>
      {best_panel}
    </section>

    <h2>Progress by generation</h2>
    <div class="grid">
      {''.join(generation_cards) if generation_cards else '<p>No runs recorded yet.</p>'}
    </div>

    <section class="card">
      <h2>Top completed runs</h2>
      <table>
        <thead><tr><th>Rank</th><th>Run</th><th>Family</th><th>Generation</th><th>Score</th><th>Metrics</th></tr></thead>
        <tbody>{''.join(top_rows) if top_rows else '<tr><td colspan="6">No completed runs yet.</td></tr>'}</tbody>
      </table>
    </section>

    <h2>All run details</h2>
    {''.join(detail_rows) if detail_rows else '<p>No runs recorded yet.</p>'}
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = render_dashboard(args.state_dir, args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Acceptance criteria

- `python -m autoresearch.plan --config autoresearch/configs/default.yaml --dry-run` prints the planned initial generation without running DESC.
- `python -m autoresearch.render_dashboard --state-dir <campaign_dir>` works with an empty campaign directory and with sample completed/failed records.
- `python -m autoresearch.run_trial --trial-json <trial.json>` can run one phase2 or phase3 trial on one GPU and writes `result.json`.
- `python -m autoresearch.run_loop --config autoresearch/configs/default.yaml --gpus 0,1,2,3` launches up to 4 concurrent trials and updates `runs.jsonl`, `state.json`, and `index.html`.
- Existing phase2 and phase3 scripts still run with their current defaults.
- Add a short `autoresearch/README.md` explaining how to start, restart, inspect results, and modify the search space.
- Keep generated campaign outputs out of git, except for tiny sample fixtures if tests need them.

## Suggested tests / checks

Run at least:

```bash
python -m autoresearch.plan --config autoresearch/configs/default.yaml --dry-run
python -m autoresearch.render_dashboard --state-dir autoresearch/tests/fixtures/sample_campaign --output /tmp/ti_autoresearch_dashboard.html
python -m compileall autoresearch
```

If the environment has the DESC/GX surrogate data and a GPU available, also run one tiny smoke trial with low `maxiter`, one rho value, one alpha value, and `mode_steps: [1]`.
