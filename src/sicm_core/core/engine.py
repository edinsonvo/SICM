from sicm_core.core.context import ExecutionContext


class Engine:

    def __init__(

        self,

        pipeline

    ):

        self.pipeline = pipeline

    def run(self, experiment):

        context = ExecutionContext(

            experiment=experiment
        )

        return self.pipeline.execute(context)
