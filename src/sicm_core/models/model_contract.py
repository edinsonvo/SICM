from abc import ABC, abstractmethod
from typing import Mapping
from .equation import Equation
from .parameter import Parameter
from .variable import Variable

class ModelContract(ABC):
    name: str = ""
    family: str = ""
    version: str = "1.0"

    @abstractmethod
    def variables(self) -> tuple[Variable, ...]: ...

    @abstractmethod
    def parameters(self) -> tuple[Parameter, ...]: ...

    @abstractmethod
    def equations(self) -> tuple[Equation, ...]: ...

    def variable_map(self) -> Mapping[str, Variable]:
        return {v.symbol: v for v in self.variables()}

    def parameter_map(self) -> Mapping[str, Parameter]:
        return {p.symbol: p for p in self.parameters()}

    def equation_map(self) -> Mapping[str, Equation]:
        return {e.equation_id: e for e in self.equations()}

    def validate_contract(self) -> None:
        if not self.name.strip() or not self.family.strip():
            raise ValueError("Model name and family are required.")
        variables, parameters = self.variable_map(), self.parameter_map()
        if len(variables) != len(self.variables()):
            raise ValueError("Variable symbols must be unique.")
        if len(parameters) != len(self.parameters()):
            raise ValueError("Parameter symbols must be unique.")
        for p in self.parameters():
            p.validate()
        for e in self.equations():
            unknown_v = set(e.variables) - set(variables)
            unknown_p = set(e.parameters) - set(parameters)
            if unknown_v:
                raise ValueError(f"Unknown variables: {sorted(unknown_v)}")
            if unknown_p:
                raise ValueError(f"Unknown parameters: {sorted(unknown_p)}")
