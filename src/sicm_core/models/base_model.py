from abc import abstractmethod
from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment
from .model_contract import ModelContract

class BaseModel(ModelContract):
    @abstractmethod
    def solve(self, experiment: Experiment) -> ExecutionResult: ...

    @abstractmethod
    def interpret(self, result: ExecutionResult) -> str: ...
