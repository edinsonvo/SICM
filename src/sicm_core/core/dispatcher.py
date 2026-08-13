from typing import Type, Any
from sicm_core.core.registry import Registry


class Dispatcher:

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def dispatch(self, model_name: str) -> Type[Any]:
        return self.registry.get(model_name)
