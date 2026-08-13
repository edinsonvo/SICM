from dataclasses import dataclass
from enum import Enum
from typing import Any

class EquationType(str, Enum):
    IDENTITY = "identity"
    BEHAVIORAL = "behavioral"
    EQUILIBRIUM = "equilibrium"
    DEFINITION = "definition"
    CONSTRAINT = "constraint"

@dataclass(frozen=True, slots=True)
class Equation:
    equation_id: str
    name: str
    expression: str
    variables: tuple[str, ...]
    parameters: tuple[str, ...] = ()
    equation_type: EquationType = EquationType.BEHAVIORAL
    description: str = ""

    def __post_init__(self) -> None:
        if not self.equation_id.strip() or not self.name.strip() or not self.expression.strip():
            raise ValueError("Equation id, name and expression are required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id, "name": self.name,
            "expression": self.expression, "variables": self.variables,
            "parameters": self.parameters, "equation_type": self.equation_type.value,
            "description": self.description,
        }
