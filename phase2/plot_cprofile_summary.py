#!/usr/bin/env python
"""Create a compact plot from a cProfile .cprof file.

The plot uses only the function/method name for labels, not absolute paths.
"""

from __future__ import annotations

import argparse
import pstats
from pathlib import Path
import os

import matplotlib.pyplot as plt


PLUMBING_NAMES = {
    "wrapper",
    "cache_miss",
    "_python_pjit_helper",
    "reraise_with_filtered_traceback",
    "_true_bind",
    "bind",
    "bind_with_trace",
    "call_wrapped",
    "memoized_fun",
    "_infer_params",
    "_infer_params_internal",
    "flatten_fun",
    "_argnums_partial",
}

PLUMBING_PATH_PARTS = (
    "/jax/_src/profiler.py",
    "/jax/_src/traceback_util.py",
)


def _collect_rows(
    stats: pstats.Stats,
    top_n: int,
    turbulence_only: bool = False,
    strip_plumbing: bool = False,
):
    rows = []
    for func, stat in stats.stats.items():
        cc, nc, tt, ct, callers = stat
        filename, line, name = func
        if name in {"<module>", "<built-in method exec>", "<built-in method builtins.exec>"}:
            continue
        if strip_plumbing and (
            name in PLUMBING_NAMES
            or any(part in filename for part in PLUMBING_PATH_PARTS)
        ):
            continue
        if turbulence_only and not (
            "turbulence" in filename or "NNITGProxy" in name or "_turbulence" in filename
        ):
            continue
        rows.append(
            {
                "filename": filename,
                "line": line,
                "name": name,
                "ncalls": nc,
                "tottime": tt,
                "cumtime": ct,
            }
        )
    rows.sort(key=lambda row: row["cumtime"], reverse=True)
    return rows[:top_n]


def _label(row):
    filename = row["filename"]
    name = row["name"]
    if filename.startswith("<") and filename.endswith(">"):
        return name
    return f"{os.path.basename(filename)}:{name}"


def _draw_panel(ax, rows, title: str):
    if not rows:
        ax.text(0.5, 0.5, "No matching functions", ha="center", va="center")
        ax.set_axis_off()
        return

    labels = [_label(row) for row in rows][::-1]
    values = [row["cumtime"] for row in rows][::-1]
    bars = ax.barh(range(len(rows)), values, color="#2f5d8c")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cumulative Time (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    xmax = max(values)
    pad = xmax * 0.015 if xmax > 0 else 0.1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}s",
            va="center",
            fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path, help="Path to .cprof file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of rows per panel",
    )
    args = parser.parse_args()

    profile = args.profile
    output = args.output or profile.with_suffix(".png")

    stats = pstats.Stats(str(profile))
    top_all = _collect_rows(stats, args.top_n, turbulence_only=False, strip_plumbing=False)
    top_stripped = _collect_rows(
        stats, args.top_n, turbulence_only=False, strip_plumbing=True
    )
    top_turb = _collect_rows(stats, args.top_n, turbulence_only=True, strip_plumbing=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), constrained_layout=True)
    _draw_panel(axes[0], top_all, f"Top {len(top_all)} Overall Methods")
    _draw_panel(axes[1], top_stripped, f"Top {len(top_stripped)} Sans Wrapper/Plumbing")
    _draw_panel(axes[2], top_turb, f"Top {len(top_turb)} Turbulence Methods")

    fig.suptitle(
        f"cProfile Summary: {profile.name}\n"
        "Labels show basename:function, not absolute paths",
        fontsize=14,
    )

    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
