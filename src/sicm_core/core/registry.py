from sicm_core.exceptions import ModelNotFoundError

class Registry:
    def __init__(self):
        self._models = {}

    def register(self, model):
        name = getattr(model, "name", None)
        if not name:
            raise ValueError("Registered model must define 'name'.")
        self._models[name] = model

    def get(self, name):
        try:
            return self._models[name]
        except KeyError as exc:
            raise ModelNotFoundError(f"Unknown model: {name}") from exc

    def names(self):
        return tuple(sorted(self._models))
