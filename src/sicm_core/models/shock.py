from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class ShockDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class ShockTarget(str, Enum):
    PARAMETER = "parameter"
    VARIABLE = "variable"
    EQUATION = "equation"
    EXTERNAL = "external"

@dataclass(frozen=True, slots=True)
class ModelShock:
    shock_id: str
    shock_type: str
    target: ShockTarget
    magnitude: float
    direction: ShockDirection
    variable: str | None = None
    parameter: str | None = None
    description: str = ""
    metadata: Mapping[str, str] = None

    def __post_init__(self) -> None:
        if not self.shock_id.strip():
            raise ValueError("shock_id is required.")
        if not self.shock_type.strip():
            raise ValueError("shock_type is required.")
        if self.magnitude < 0:
            raise ValueError("magnitude must be non-negative.")
        if self.target is ShockTarget.PARAMETER and not self.parameter:
            raise ValueError("A parameter shock requires parameter.")
        if self.target is ShockTarget.VARIABLE and not self.variable:
            raise ValueError("A variable shock requires variable.")

    @property
    def signed_magnitude(self) -> float:
        return self.magnitude if self.direction is ShockDirection.POSITIVE else -self.magnitude
