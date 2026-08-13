"""Python entry-point plugin loader."""
from __future__ import annotations
from typing import Any
from importlib.metadata import entry_points


class PluginLoader:
    def discover(self, namespace: str) -> tuple[Any, ...]:
        return tuple(entry_points(group=namespace))

    def load(self, namespace: str) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for entry in self.discover(namespace):
            loaded[entry.name] = entry.load()
        return loaded
