"""Plugin discovery helpers."""
from __future__ import annotations
from typing import Any
from .loader import PluginLoader


def discover_models(namespace: str = "sicm.models") -> dict[str, Any]:
    return PluginLoader().load(namespace)
