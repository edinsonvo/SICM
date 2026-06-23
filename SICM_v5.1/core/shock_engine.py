from copy import deepcopy


class ShockEngine:

    def apply_shock(
        self,
        config,
        shock
    ):

        new_config = deepcopy(config)

        shock.apply(new_config)

        return new_config

    def apply_multiple(
        self,
        config,
        shocks
    ):

        result = deepcopy(config)

        for shock in shocks:
            shock.apply(result)

        return result
