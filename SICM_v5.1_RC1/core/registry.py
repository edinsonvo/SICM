from typing import Dict, Type

from models.base_model import BaseModel


class Registry:

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):

            cls._models[name] = model_cls

            return model_cls

        return decorator

    @classmethod
    def create(cls, name, config):

        if name not in cls._models:

            raise ValueError(
                f"Modelo '{name}' no registrado."
            )

        return cls._models[name](config)

    @classmethod
    def available_models(cls):

        return sorted(cls._models.keys())
