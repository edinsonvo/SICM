from core.model_factory import (
    ModelFactory
)


class SICMEngine:

    def run(
        self,
        model_name,
        config
    ):

        solver = (
            ModelFactory
            .get_solver(model_name)
        )

        result = (
            solver.solve(config)
        )

        return result
