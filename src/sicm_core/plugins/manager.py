"""Plugin manager."""
from __future__ import annotations
from sicm_core.core.registry import Registry
from .loader import PluginLoader
class PluginManager:
    def __init__(self, registry: Registry, namespace: str = "sicm.models", loader: PluginLoader | None = None) -> None:
        self.registry, self.namespace, self.loader = registry, namespace, loader or PluginLoader()
    def load(self) -> tuple[str, ...]:
        loaded = self.loader.load(self.namespace)
        for model in loaded.values():
            self.registry.register(model)
        return tuple(loaded)
