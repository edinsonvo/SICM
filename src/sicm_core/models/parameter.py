from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class Parameter:
    name: str
    symbol: str
    value: float
    unit: str
    description: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibratable: bool = True

    def validate(self) -> None:
        if self.lower_bound is not None and self.value < self.lower_bound:
            raise ValueError(f"{self.symbol} is below its lower bound.")
        if self.upper_bound is not None and self.value > self.upper_bound:
            raise ValueError(f"{self.symbol} is above its upper bound.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "symbol": self.symbol, "value": self.value,
            "unit": self.unit, "description": self.description,
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "calibratable": self.calibratable,
        }
