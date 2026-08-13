from __future__ import annotations

from sicm_core.exceptions import ModelNotFoundError


class Registry:

    def __init__(self):

        self._models: dict[str, type] = {}

    def register(self, model: type) -> None:

        self._models[model.name] = model

    def get(self, name: str):

        try:

            return self._models[name]

        except KeyError:

            raise ModelNotFoundError(name)
