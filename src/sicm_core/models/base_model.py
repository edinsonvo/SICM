from abc import ABC
from abc import abstractmethod

from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment


class BaseModel(ABC):

    name: str = ""

    family: str = ""

    version: str = "1.0"

    @abstractmethod
    def solve(

        self,

        experiment: Experiment

    ) -> ExecutionResult:

        ...

    @abstractmethod
    def interpret(

        self,

        result: ExecutionResult

    ) -> str:

        ...
