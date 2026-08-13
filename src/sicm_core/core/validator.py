from sicm_core.domain.experiment import Experiment
from sicm_core.exceptions import ValidationError


class Validator:

    def validate(

        self,

        experiment: Experiment

    ) -> None:

        if experiment.name == "":

            raise ValidationError(
                "Experiment name cannot be empty."
            )
