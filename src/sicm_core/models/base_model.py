from abc import ABC, abstractmethod
from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment

class BaseModel(ABC):
    name = ""
    family = ""
    version = "1.0"

    @abstractmethod
    def solve(self, experiment: Experiment) -> ExecutionResult: ...

    @abstractmethod
    def interpret(self, result: ExecutionResult) -> str: ...
