"""Configuration helpers for YAML-driven experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when experiment configuration is invalid."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a config file with optional inheritance."""

    path = Path(config_path).resolve()
    cfg = _load_yaml(path)
    parent_path = cfg.pop("inherits", None)
    if parent_path:
        parent_candidate = Path(parent_path)
        if parent_candidate.is_absolute():
            resolved_parent = parent_candidate
        else:
            local_parent = (path.parent / parent_candidate).resolve()
            cwd_parent = parent_candidate.resolve()
            resolved_parent = local_parent if local_parent.exists() else cwd_parent
        parent = load_config(resolved_parent)
        cfg = _deep_merge(parent, cfg)
    cfg.setdefault("experiment_name", path.stem)
    cfg["config_path"] = str(path)
    return cfg


def dump_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Persist an effective config snapshot."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
