"""Plugin discovery helpers."""
from __future__ import annotations
from .loader import PluginLoader
def discover_models(namespace: str = "sicm.models"):
    return PluginLoader().load(namespace)
