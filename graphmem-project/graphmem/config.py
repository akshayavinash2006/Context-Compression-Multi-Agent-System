"""Loads config.yaml into an attribute-access dict (cfg.retrieval.ppr_alpha)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class Cfg(dict):
    """dict with attribute access; nested dicts become Cfg too."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Cfg()
            node = node[part]
        node[parts[-1]] = Cfg(value) if isinstance(value, dict) else value

    def copy(self) -> "Cfg":  # deep copy, so overrides never leak between runs
        return copy.deepcopy(self)


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_config(path: str | Path | None = None, overrides: dict | list[str] | None = None) -> Cfg:
    """Load config.yaml. `overrides` is {"a.b": v} or ["a.b=v", ...]."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    if isinstance(overrides, list):
        overrides = dict(o.split("=", 1) for o in overrides)
        overrides = {k: _parse_value(v) for k, v in overrides.items()}
    for k, v in (overrides or {}).items():
        cfg.set_path(k, v)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Paths in config are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
