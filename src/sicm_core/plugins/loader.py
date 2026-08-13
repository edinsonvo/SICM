"""Python entry-point plugin loader."""
from __future__ import annotations
from importlib.metadata import entry_points
class PluginLoader:
    def discover(self, namespace: str):
        return tuple(entry_points(group=namespace))
    def load(self, namespace: str):
        loaded = {}
        for entry in self.discover(namespace):
            loaded[entry.name] = entry.load()
        return loaded
