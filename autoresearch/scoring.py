"""Default scalar scoring for autoresearch trials."""

from __future__ import annotations

import math
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def score_result(
    *,
    family: str,
    status: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    elapsed_seconds: float | None,
    score_config: dict[str, Any] | None,
) -> float:
    weights = ((score_config or {}).get("weights") or {})
    failure_weight = _finite(weights.get("failure"), 1_000_000.0)
    if status != "completed":
        return failure_weight

    score = 0.0
    if family == "phase2":
        if metrics.get("transport_success") is False:
            return failure_weight * 0.25
        if metrics.get("mean_heat_flux", metrics.get("turbulence")) is None:
            return failure_weight * 0.5
        turbulence = _finite(
            metrics.get("mean_heat_flux", metrics.get("turbulence")),
            default=0.0,
        )
        score += _finite(weights.get("turbulence"), 1.0) * math.log1p(max(turbulence, 0.0))
    elif family == "phase3":
        if metrics.get("critical_gradient") is None:
            return failure_weight * 0.5
        critical_gradient = _finite(metrics.get("critical_gradient"), default=0.0)
        score -= _finite(weights.get("critical_gradient"), 1.0) * critical_gradient
        score += _finite(
            metrics.get("critical_gradient_deficit"),
            default=0.0,
        )

    score += _finite(weights.get("quasisymmetry"), 0.25) * abs(
        _finite(metrics.get("quasisymmetry"), default=0.0)
    )
    target = _finite(config.get("aspect_ratio_target"), default=0.0)
    aspect = metrics.get("aspect_ratio")
    if aspect is not None:
        score += _finite(weights.get("aspect_ratio"), 0.25) * abs(_finite(aspect) - target)
    if elapsed_seconds is not None:
        score += _finite(weights.get("runtime"), 0.05) * max(float(elapsed_seconds), 0.0) / 3600.0

    expected_stages = len(config.get("mode_steps") or [])
    completed_stages = int(_finite(metrics.get("completed_stages"), default=expected_stages))
    if expected_stages and completed_stages < expected_stages:
        score += failure_weight * 0.1
    return float(score)
