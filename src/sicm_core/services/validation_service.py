class ValidationService:
    def __init__(self, validator):
        self.validator = validator

    def validate(self, experiment):
        self.validator.validate(experiment)
