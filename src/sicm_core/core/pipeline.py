from .context import ExecutionContext

class Pipeline:
    def __init__(self, validator, dispatcher):
        self.validator = validator
        self.dispatcher = dispatcher

    def execute(self, context: ExecutionContext):
        self.validator.validate(context.experiment)
        model_cls = self.dispatcher.dispatch(context.experiment.scenario.model.value)
        model = model_cls()
        result = model.solve(context.experiment)
        return result, model.interpret(result)
