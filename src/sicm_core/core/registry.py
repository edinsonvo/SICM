from __future__ import annotations

from typing import Type, Any
from sicm_core.exceptions import ModelNotFoundError

class Registry:

    def __init__(self) -> None:
        self._models: dict[str, Type[Any]] = {}

    def register(self, model: Type[Any]) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> Type[Any]:
        try:
            return self._models[name]
        except KeyError:
            raise ModelNotFoundError(name)
