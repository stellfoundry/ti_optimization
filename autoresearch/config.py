"""Configuration loading and path helpers for autoresearch campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML config with only the standard library.

    The repository runtime used for this project does not include PyYAML, so the
    default ``.yaml`` file is intentionally written as JSON-compatible YAML.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must be JSON-compatible YAML because PyYAML is not "
            f"available in this environment: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping at the top level")
    return config


def campaign_dir(config: dict[str, Any]) -> Path:
    output_root = Path(str(config.get("output_root", "autoresearch/output")))
    if not output_root.is_absolute():
        output_root = repo_root() / output_root
    return output_root / str(config.get("campaign_id", "latest"))


def configured_gpu_ids(config: dict[str, Any], override: str | None = None) -> list[str]:
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    gpu_config = config.get("gpu") or {}
    ids = gpu_config.get("ids", [0])
    return [str(item) for item in ids]
