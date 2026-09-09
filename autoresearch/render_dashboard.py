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
        latest[str(record.get("run_id", "unknown"))] = record
    return list(latest.values())


def _safe_score(record: dict[str, Any]) -> float:
    try:
        score = record.get("score")
        if score is None:
            return float("inf")
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def _json_block(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2, sort_keys=True, default=str))


def _script_json(value: Any) -> str:
    """Serialize JSON safely for embedding inside an inline script tag."""

    return json.dumps(value, sort_keys=True, default=str).replace("<", "\\u003c")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _relative_artifacts(record: dict[str, Any], state_dir: Path) -> dict[str, str]:
    links: dict[str, str] = {}
    artifacts = record.get("artifacts") or {}
    for name, raw_path in sorted(artifacts.items()):
        if raw_path is None:
            continue
        path = Path(str(raw_path))
        try:
            href = path.resolve().relative_to(state_dir.resolve()).as_posix()
        except ValueError:
            href = path.as_posix()
        links[str(name)] = href
    for log_key in ("stdout_path", "stderr_path"):
        raw_path = record.get(log_key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        try:
            href = path.resolve().relative_to(state_dir.resolve()).as_posix()
        except ValueError:
            href = path.as_posix()
        links[str(log_key)] = href
    return links


def _artifact_links(record: dict[str, Any], state_dir: Path) -> str:
    links: list[str] = []
    for name, href in _relative_artifacts(record, state_dir).items():
        links.append(f'<a href="{html.escape(href)}">{html.escape(str(name))}</a>')
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
            value_text = f"{value:.6g}" if isinstance(value, float) else str(value)
            parts.append(f"<b>{html.escape(key)}</b>: {html.escape(value_text)}")
    if not parts:
        for key, value in list(sorted(metrics.items()))[:6]:
            parts.append(f"<b>{html.escape(str(key))}</b>: {html.escape(str(value))}")
    return "<br>".join(parts)


def render_dashboard(state_dir: Path, output_path: Path | None = None) -> Path:
    state_dir = state_dir.resolve()
    output_path = output_path or state_dir / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _latest_by_run(_load_jsonl(state_dir / "runs.jsonl"))
    records.sort(key=lambda record: (int(record.get("generation", -1)), str(record.get("run_id", ""))))

    completed = [record for record in records if record.get("status") == "completed"]
    ranked = sorted(completed, key=_safe_score)
    best = ranked[0] if ranked else None

    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_generation[int(record.get("generation", -1))].append(record)

    chart_records: list[dict[str, Any]] = []
    for record in records:
        metrics = record.get("metrics") or {}
        chart_records.append(
            {
                "run_id": str(record.get("run_id", "unknown")),
                "family": record.get("family"),
                "generation": record.get("generation"),
                "status": record.get("status"),
                "score": _finite_float(record.get("score")),
                "elapsed_seconds": _finite_float(record.get("elapsed_seconds")),
                "parent_run_ids": record.get("parent_run_ids") or [],
                "metrics": {
                    "critical_gradient": _finite_float(metrics.get("critical_gradient")),
                    "critical_gradient_deficit": _finite_float(metrics.get("critical_gradient_deficit")),
                    "critical_gradient_delta": _finite_float(metrics.get("critical_gradient_delta")),
                    "mean_heat_flux": _finite_float(metrics.get("mean_heat_flux")),
                    "aspect_ratio": _finite_float(metrics.get("aspect_ratio")),
                    "completed_stages": _finite_float(metrics.get("completed_stages")),
                },
                "config": record.get("config") or {},
                "failure_reason": record.get("failure_reason"),
                "artifacts": _relative_artifacts(record, state_dir),
                "detail_anchor": f"run-{record.get('run_id', 'unknown')}",
            }
        )

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
                <section><h3>Metrics</h3><pre>{_json_block(record.get('metrics') or {})}</pre></section>
                <section><h3>Config</h3><pre>{_json_block(record.get('config') or {})}</pre></section>
                <section>
                  <h3>Artifacts</h3><p>{_artifact_links(record, state_dir)}</p>
                  <h3>Failure reason</h3><pre>{html.escape(str(failure)) if failure else '&mdash;'}</pre>
                </section>
                <section><h3>Raw record</h3><pre>{_json_block(record)}</pre></section>
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
        <h3>Key metrics</h3><p>{_metric_summary(best.get('metrics') or {})}</p>
        <h3>Artifacts</h3><p>{_artifact_links(best, state_dir)}</p>
        """

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    chart_json = _script_json(chart_records)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ti_optimization autoresearch dashboard</title>
  <style>
    :root {{ color-scheme: light dark; --bg: #0f172a; --panel: #111827; --text: #e5e7eb; --muted: #9ca3af; --line: #374151; --accent: #38bdf8; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 2rem; background: #111827; border-bottom: 1px solid var(--line); }}
    header h1 {{ margin: 0 0 0.5rem; }}
    main {{ padding: 1.5rem; max-width: 1400px; margin: 0 auto; }}
    a {{ color: var(--accent); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }}
    .card, .run-detail {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1rem; }}
    .chart-card {{ margin-top: 1rem; }}
    .chart-toolbar {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem; }}
    .chart-toolbar label {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
    .chart-toolbar select {{ background: #020617; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 0.45rem 0.6rem; }}
    .chart-wrap {{ min-height: 430px; border: 1px solid var(--line); border-radius: 8px; background: #020617; overflow: hidden; }}
    #progress-chart {{ display: block; width: 100%; height: 430px; }}
    .chart-point {{ cursor: pointer; stroke: #020617; stroke-width: 1.5; }}
    .chart-point:hover, .chart-point.selected {{ stroke: #f8fafc; stroke-width: 3; }}
    .chart-axis, .chart-grid {{ stroke: #334155; stroke-width: 1; }}
    .chart-grid {{ opacity: 0.8; }}
    .chart-label {{ fill: var(--muted); font-size: 12px; }}
    .chart-title {{ fill: var(--text); font-size: 13px; font-weight: 700; }}
    .chart-line {{ fill: none; stroke: #fbbf24; stroke-width: 2.5; }}
    .chart-empty {{ color: var(--muted); padding: 1rem; }}
    .chart-detail {{ margin-top: 0.75rem; display: grid; grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.5fr); gap: 1rem; }}
    .chart-detail dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 0.75rem; margin: 0; }}
    .chart-detail dt {{ color: var(--muted); }}
    .chart-detail dd {{ margin: 0; }}
    .artifact-list {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .artifact-list a {{ border: 1px solid var(--line); border-radius: 999px; padding: 0.2rem 0.5rem; text-decoration: none; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.45rem; border-radius: 999px; background: #334155; color: #f8fafc; font-size: 0.8rem; }}
    .completed {{ background: #047857; }} .failed {{ background: #b91c1c; }} .running {{ background: #1d4ed8; }} .pending {{ background: #6b7280; }} .skipped {{ background: #92400e; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 0.65rem; text-align: left; vertical-align: top; }}
    th {{ color: #f8fafc; cursor: default; }}
    pre {{ overflow-x: auto; background: #020617; padding: 0.75rem; border-radius: 6px; }}
    summary {{ cursor: pointer; display: flex; flex-wrap: wrap; gap: 0.7rem; align-items: center; }}
    .run-id {{ font-weight: 700; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-top: 1rem; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>ti_optimization autoresearch dashboard</h1>
    <p class="muted">State directory: {html.escape(str(state_dir))} · Updated {html.escape(now)}</p>
  </header>
  <main>
    <section class="grid">
      <section class="card"><h2>Best run</h2>{best_panel}</section>
      {''.join(generation_cards)}
    </section>
    <section class="card chart-card">
      <h2>Generation Progress</h2>
      <div class="chart-toolbar">
        <label for="metric-select">Metric</label>
        <select id="metric-select">
          <option value="score">Score (lower is better)</option>
          <option value="critical_gradient">Critical gradient</option>
          <option value="critical_gradient_delta">Critical-gradient delta</option>
          <option value="critical_gradient_deficit">Critical-gradient deficit</option>
          <option value="aspect_ratio">Aspect ratio</option>
          <option value="elapsed_seconds">Runtime seconds</option>
        </select>
        <label><input type="checkbox" id="focus-best" checked>Within 10x of best</label>
        <label><input type="checkbox" id="log-scale" checked>Log y-axis when possible</label>
        <span class="muted">Dots are completed runs. Yellow line is best-so-far for the selected metric. Click a dot for run details.</span>
      </div>
      <div class="chart-wrap"><svg id="progress-chart" role="img" aria-label="Autoresearch metric progress by generation"></svg></div>
      <div id="chart-detail" class="chart-detail card">
        <section><h3>Selected run</h3><p class="muted">Click a point in the chart.</p></section>
      </div>
    </section>
    <section class="card">
      <h2>Top completed runs</h2>
      <table>
        <thead><tr><th>Rank</th><th>Run</th><th>Family</th><th>Generation</th><th>Score</th><th>Metrics</th></tr></thead>
        <tbody>{''.join(top_rows) if top_rows else '<tr><td colspan="6">No completed runs yet.</td></tr>'}</tbody>
      </table>
    </section>
    <section>
      <h2>All runs</h2>
      {''.join(detail_rows) if detail_rows else '<p>No run records yet.</p>'}
    </section>
  </main>
  <script>
    const runData = {chart_json};
    const metricInfo = {{
      score: {{ label: "Score", direction: "min", getter: run => run.score }},
      critical_gradient: {{ label: "Critical gradient", direction: "max", getter: run => run.metrics.critical_gradient }},
      critical_gradient_delta: {{ label: "Critical-gradient delta", direction: "max", getter: run => run.metrics.critical_gradient_delta }},
      critical_gradient_deficit: {{ label: "Critical-gradient deficit", direction: "min", getter: run => run.metrics.critical_gradient_deficit }},
      aspect_ratio: {{ label: "Aspect ratio", direction: "none", getter: run => run.metrics.aspect_ratio }},
      elapsed_seconds: {{ label: "Runtime seconds", direction: "min", getter: run => run.elapsed_seconds }},
    }};
    const colors = {{
      phase2: "#fb7185",
      phase3: "#38bdf8",
      unknown: "#a78bfa",
    }};

    function isFiniteNumber(value) {{
      return typeof value === "number" && Number.isFinite(value);
    }}

    function fmt(value) {{
      if (!isFiniteNumber(value)) return "n/a";
      const abs = Math.abs(value);
      if ((abs >= 10000 || abs < 0.001) && abs !== 0) return value.toExponential(4);
      return value.toPrecision(7).replace(/\\.0+$/, "");
    }}

    function escapeText(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }}[ch]));
    }}

    function metricValue(run, metricName) {{
      const info = metricInfo[metricName] || metricInfo.score;
      return info.getter(run);
    }}

    function completedMetricRuns(metricName) {{
      return runData
        .filter(run => run.status === "completed")
        .map(run => ({{ ...run, value: metricValue(run, metricName) }}))
        .filter(run => isFiniteNumber(run.value) && Number.isFinite(Number(run.generation)))
        .sort((a, b) => Number(a.generation) - Number(b.generation) || a.run_id.localeCompare(b.run_id));
    }}

    function bestPointFor(points, metricName) {{
      const direction = metricInfo[metricName].direction;
      if (direction === "max") {{
        return points.reduce((best, point) => point.value > best.value ? point : best, points[0]);
      }}
      return points.reduce((best, point) => point.value < best.value ? point : best, points[0]);
    }}

    function focusPoints(points, metricName, enabled) {{
      const direction = metricInfo[metricName].direction;
      if (!enabled || direction === "none" || !points.length) {{
        return {{ points, hiddenCount: 0, cutoffText: "" }};
      }}
      const best = bestPointFor(points, metricName).value;
      const baseline = Math.max(Math.abs(best), 1e-12);
      let visible = points;
      let cutoffText = "";
      if (direction === "min") {{
        const cutoff = best <= 0 ? baseline * 10 : best * 10;
        visible = points.filter(point => point.value <= cutoff);
        cutoffText = `showing values <= ${{fmt(cutoff)}}`;
      }} else if (best > 0) {{
        const cutoff = best / 10;
        visible = points.filter(point => point.value >= cutoff);
        cutoffText = `showing values >= ${{fmt(cutoff)}}`;
      }} else {{
        const cutoff = best - baseline * 10;
        visible = points.filter(point => point.value >= cutoff);
        cutoffText = `showing values >= ${{fmt(cutoff)}}`;
      }}
      return {{ points: visible, hiddenCount: points.length - visible.length, cutoffText }};
    }}

    function generationBestLine(points, metricName) {{
      const direction = metricInfo[metricName].direction;
      if (direction === "none") return [];
      const bestByGeneration = new Map();
      for (const point of points) {{
        const generation = Number(point.generation);
        const existing = bestByGeneration.get(generation);
        if (!existing || (direction === "min" ? point.value < existing.value : point.value > existing.value)) {{
          bestByGeneration.set(generation, point);
        }}
      }}
      let current = null;
      const line = [];
      for (const generation of [...bestByGeneration.keys()].sort((a, b) => a - b)) {{
        const best = bestByGeneration.get(generation);
        if (!current || (direction === "min" ? best.value < current.value : best.value > current.value)) {{
          current = best;
        }}
        line.push({{ generation, value: current.value, run_id: current.run_id }});
      }}
      return line;
    }}

    function showDetails(run, metricName) {{
      const detail = document.getElementById("chart-detail");
      const metrics = run.metrics || {{}};
      const config = run.config || {{}};
      const artifacts = run.artifacts || {{}};
      const artifactHtml = Object.entries(artifacts).length
        ? `<div class="artifact-list">${{Object.entries(artifacts).map(([name, href]) => `<a href="${{escapeText(href)}}">${{escapeText(name)}}</a>`).join("")}}</div>`
        : "<p class=\\"muted\\">No artifacts listed.</p>";
      detail.innerHTML = `
        <section>
          <h3>${{escapeText(run.run_id)}}</h3>
          <dl>
            <dt>Status</dt><dd>${{escapeText(run.status)}}</dd>
            <dt>Family</dt><dd>${{escapeText(run.family || "n/a")}}</dd>
            <dt>Generation</dt><dd>${{escapeText(run.generation)}}</dd>
            <dt>${{escapeText(metricInfo[metricName].label)}}</dt><dd>${{fmt(metricValue(run, metricName))}}</dd>
            <dt>Score</dt><dd>${{fmt(run.score)}}</dd>
            <dt>Critical gradient</dt><dd>${{fmt(metrics.critical_gradient)}}</dd>
            <dt>Delta</dt><dd>${{fmt(metrics.critical_gradient_delta)}}</dd>
            <dt>Aspect ratio</dt><dd>${{fmt(metrics.aspect_ratio)}}</dd>
            <dt>Stages</dt><dd>${{fmt(metrics.completed_stages)}}</dd>
            <dt>Parents</dt><dd>${{(run.parent_run_ids || []).map(escapeText).join(", ") || "none"}}</dd>
          </dl>
          <p><a href="#${{escapeText(run.detail_anchor)}}">Open full run record</a></p>
          <h3>Artifacts</h3>${{artifactHtml}}
        </section>
        <section>
          <h3>Config</h3>
          <pre>${{escapeText(JSON.stringify(config, null, 2))}}</pre>
        </section>
      `;
    }}

    function renderChart(metricName) {{
      const svg = document.getElementById("progress-chart");
      const detail = document.getElementById("chart-detail");
      const allPoints = completedMetricRuns(metricName);
      const focused = focusPoints(allPoints, metricName, document.getElementById("focus-best").checked);
      const points = focused.points;
      svg.innerHTML = "";
      if (!points.length) {{
        svg.innerHTML = `<foreignObject width="100%" height="100%"><div class="chart-empty">No completed runs with this metric yet.</div></foreignObject>`;
        detail.innerHTML = `<section><h3>Selected run</h3><p class="muted">No points available for this metric.</p></section>`;
        return;
      }}

      const width = svg.clientWidth || 1100;
      const height = svg.clientHeight || 430;
      const margin = {{ top: 32, right: 28, bottom: 54, left: 78 }};
      const generations = [...new Set(points.map(point => Number(point.generation)))].sort((a, b) => a - b);
      const minGeneration = generations[0];
      const maxGeneration = generations[generations.length - 1];
      const useLogScale = document.getElementById("log-scale").checked
        && points.every(point => point.value > 0)
        && Math.max(...points.map(point => point.value)) / Math.min(...points.map(point => point.value)) >= 10;
      const plotValue = value => useLogScale ? Math.log10(value) : value;
      const axisValue = value => useLogScale ? 10 ** value : value;
      let minValue = Math.min(...points.map(point => point.value));
      let maxValue = Math.max(...points.map(point => point.value));
      let minPlotValue = Math.min(...points.map(point => plotValue(point.value)));
      let maxPlotValue = Math.max(...points.map(point => plotValue(point.value)));
      if (minPlotValue === maxPlotValue) {{
        minPlotValue -= Math.abs(minPlotValue || 1) * 0.1;
        maxPlotValue += Math.abs(maxPlotValue || 1) * 0.1;
      }}
      const pad = (maxPlotValue - minPlotValue) * 0.08;
      minPlotValue -= pad;
      maxPlotValue += pad;
      minValue = axisValue(minPlotValue);
      maxValue = axisValue(maxPlotValue);

      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const xScale = generation => {{
        if (maxGeneration === minGeneration) return margin.left + innerWidth / 2;
        return margin.left + ((generation - minGeneration) / (maxGeneration - minGeneration)) * innerWidth;
      }};
      const yScale = value => margin.top + (1 - (plotValue(value) - minPlotValue) / (maxPlotValue - minPlotValue)) * innerHeight;

      function line(x1, y1, x2, y2, cls) {{
        const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
        el.setAttribute("x1", x1); el.setAttribute("y1", y1);
        el.setAttribute("x2", x2); el.setAttribute("y2", y2);
        el.setAttribute("class", cls);
        svg.appendChild(el);
      }}
      function text(x, y, value, cls, anchor = "middle") {{
        const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
        el.setAttribute("x", x); el.setAttribute("y", y);
        el.setAttribute("class", cls);
        el.setAttribute("text-anchor", anchor);
        el.textContent = value;
        svg.appendChild(el);
      }}

      line(margin.left, margin.top, margin.left, margin.top + innerHeight, "chart-axis");
      line(margin.left, margin.top + innerHeight, margin.left + innerWidth, margin.top + innerHeight, "chart-axis");
      for (let i = 0; i <= 5; i++) {{
        const value = axisValue(minPlotValue + ((maxPlotValue - minPlotValue) * i / 5));
        const y = yScale(value);
        line(margin.left, y, margin.left + innerWidth, y, "chart-grid");
        text(margin.left - 10, y + 4, fmt(value), "chart-label", "end");
      }}
      for (const generation of generations) {{
        const x = xScale(generation);
        line(x, margin.top, x, margin.top + innerHeight, "chart-grid");
        text(x, margin.top + innerHeight + 24, `Gen ${{generation}}`, "chart-label");
      }}
      const hiddenText = focused.hiddenCount ? `; hidden ${{focused.hiddenCount}} outlier${{focused.hiddenCount === 1 ? "" : "s"}} (${{focused.cutoffText}})` : "";
      text(width / 2, 20, `${{metricInfo[metricName].label}} by generation${{useLogScale ? " (log y)" : ""}}${{hiddenText}}`, "chart-title");

      const bestLine = generationBestLine(points, metricName);
      if (bestLine.length > 1) {{
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("class", "chart-line");
        path.setAttribute("d", bestLine.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{xScale(point.generation)}} ${{yScale(point.value)}}`).join(" "));
        svg.appendChild(path);
      }}

      const generationCounts = new Map();
      for (const point of points) {{
        const generation = Number(point.generation);
        const index = generationCounts.get(generation) || 0;
        generationCounts.set(generation, index + 1);
        const x = xScale(generation) + ((index % 9) - 4) * 5;
        const y = yScale(point.value);
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", x);
        circle.setAttribute("cy", y);
        circle.setAttribute("r", point.run_id === points[0].run_id ? 6 : 5);
        circle.setAttribute("fill", colors[point.family] || colors.unknown);
        circle.setAttribute("class", "chart-point");
        circle.dataset.runId = point.run_id;
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${{point.run_id}}: ${{fmt(point.value)}}`;
        circle.appendChild(title);
        circle.addEventListener("click", () => {{
          svg.querySelectorAll(".chart-point.selected").forEach(item => item.classList.remove("selected"));
          circle.classList.add("selected");
          showDetails(point, metricName);
        }});
        svg.appendChild(circle);
      }}
      const bestPoint = bestPointFor(points, metricName);
      showDetails(bestPoint, metricName);
      const selected = [...svg.querySelectorAll(".chart-point")].find(point => point.dataset.runId === bestPoint.run_id);
      if (selected) selected.classList.add("selected");
    }}

    document.getElementById("metric-select").addEventListener("change", event => renderChart(event.target.value));
    document.getElementById("focus-best").addEventListener("change", () => renderChart(document.getElementById("metric-select").value));
    document.getElementById("log-scale").addEventListener("change", () => renderChart(document.getElementById("metric-select").value));
    window.addEventListener("resize", () => renderChart(document.getElementById("metric-select").value));
    renderChart(document.getElementById("metric-select").value);
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, help="Campaign output directory.")
    parser.add_argument("--output", help="Optional output HTML path.")
    args = parser.parse_args(argv)
    output = render_dashboard(Path(args.state_dir), Path(args.output) if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
