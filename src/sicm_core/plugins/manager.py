from .loader import PluginLoader

class PluginManager:
    def __init__(self, registry, namespace="sicm.models", loader=None):
        self.registry = registry
        self.namespace = namespace
        self.loader = loader or PluginLoader()

    def load(self):
        loaded = self.loader.load(self.namespace)
        for model in loaded.values():
            self.registry.register(model)
        return tuple(loaded)
