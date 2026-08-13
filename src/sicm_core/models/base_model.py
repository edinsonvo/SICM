from abc import abstractmethod
from .model_contract import ModelContract
from .result import ModelResult

class BaseModel(ModelContract):
    @abstractmethod
    def solve(self, scenario) -> ModelResult: ...

    @abstractmethod
    def interpret(self, result: ModelResult) -> str: ...
