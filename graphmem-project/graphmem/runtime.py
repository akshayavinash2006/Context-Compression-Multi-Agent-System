"""Decides whether heavy models are used (runtime.use_heavy_models: auto | true | false)."""
from __future__ import annotations

import importlib.util

_MODE = "auto"


def set_heavy_mode(mode) -> None:
    global _MODE
    _MODE = {True: "true", False: "false"}.get(mode, str(mode).lower())


def heavy_available(package: str) -> bool:
    if _MODE == "false":
        return False
    found = importlib.util.find_spec(package) is not None
    if _MODE == "true" and not found:
        raise ImportError(f"use_heavy_models is true but '{package}' is not installed "
                          f"(pip install -r requirements-models.txt)")
    return found
