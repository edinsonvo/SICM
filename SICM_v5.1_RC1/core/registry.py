class Registry:

    _models = {}

    @classmethod
    def register(

        cls,

        name,

        model

    ):

        cls._models[name] = model

    def create(

        self,

        name,

        config

    ):

        if name not in self._models:

            raise ValueError(

                f"Modelo {name} no registrado."

            )

        return self._models[name](config)
