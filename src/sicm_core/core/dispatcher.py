from sicm_core.core.registry import Registry


class Dispatcher:

    def __init__(

        self,

        registry: Registry

    ):

        self.registry = registry

    def dispatch(self, model_name: str):

        return self.registry.get(model_name)
