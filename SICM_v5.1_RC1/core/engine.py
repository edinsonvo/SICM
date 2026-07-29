from core.registry import Registry


class Engine:

    def run(

        self,

        model,

        config

    ):

        instance = Registry.create(

            model,

            config

        )

        return instance.solve()
