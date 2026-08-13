from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    symbol: str
    unit: str
    description: str
    initial_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None

    def validate_value(self, value: float) -> bool:
        return not (
            (self.lower_bound is not None and value < self.lower_bound)
            or (self.upper_bound is not None and value > self.upper_bound)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "symbol": self.symbol, "unit": self.unit,
            "description": self.description, "initial_value": self.initial_value,
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
        }
