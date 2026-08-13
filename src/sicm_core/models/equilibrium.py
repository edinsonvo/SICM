from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True, slots=True)
class Equilibrium:
    values: Mapping[str, float]
    residuals: Mapping[str, float] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0
    tolerance: float | None = None

    def value(self, symbol: str) -> float | None:
        return self.values.get(symbol)

    def max_residual(self) -> float:
        return max((abs(v) for v in self.residuals.values()), default=0.0)

    def to_dict(self) -> dict:
        return {"values": dict(self.values), "residuals": dict(self.residuals),
                "converged": self.converged, "iterations": self.iterations,
                "tolerance": self.tolerance}
