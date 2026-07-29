from core.registry import Registry


class Engine:

    def __init__(self):

        self.registry = Registry()

    def run(

        self,

        model_name,

        config

    ):

        model = self.registry.create(

            model_name,

            config

        )

        return model.solve()
