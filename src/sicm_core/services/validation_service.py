"""Validation application service."""
from sicm_core.domain.experiment import Experiment
from sicm_core.validation.validator import Validator
class ValidationService:
    def __init__(self, validator: Validator) -> None:
        self.validator = validator
    def validate(self, experiment: Experiment) -> None:
        self.validator.validate(experiment)
