from .context import ExecutionContext

class Engine:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, experiment):
        return self.pipeline.execute(ExecutionContext(experiment))
