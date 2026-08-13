from .registry import Registry

class Dispatcher:
    def __init__(self, registry):
        self.registry = registry

    def dispatch(self, model_name):
        return self.registry.get(model_name)
