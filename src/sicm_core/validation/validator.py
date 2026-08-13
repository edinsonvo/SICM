from sicm_core.domain.experiment import Experiment
from sicm_core.exceptions import ValidationError

class Validator:
    def validate(self, experiment: Experiment) -> None:
        if not experiment.name.strip():
            raise ValidationError("Experiment name cannot be empty.")
        if experiment.scenario.parameters.P <= 0:
            raise ValidationError("Price level P must be positive.")
        if not 0 <= experiment.scenario.parameters.c < 1:
            raise ValidationError("Marginal propensity to consume c must be in [0, 1).")
