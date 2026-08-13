from dataclasses import dataclass
from .curve_set import CurveSet
from .equilibrium import Equilibrium
from .shock import ModelShock
from .transmission import TransmissionMechanism

@dataclass(frozen=True, slots=True)
class ShockResult:
    shock: ModelShock
    initial_equilibrium: Equilibrium
    final_equilibrium: Equilibrium
    transmission: TransmissionMechanism
    curves_before: CurveSet = CurveSet()
    curves_after: CurveSet = CurveSet()
    warnings: tuple[str, ...] = ()

    def delta(self, symbol: str) -> float | None:
        before = self.initial_equilibrium.value(symbol)
        after = self.final_equilibrium.value(symbol)
        if before is None or after is None:
            return None
        return after - before

    def to_dict(self) -> dict:
        return {
            "shock": {
                "shock_id": self.shock.shock_id,
                "shock_type": self.shock.shock_type,
                "target": self.shock.target.value,
                "magnitude": self.shock.magnitude,
                "direction": self.shock.direction.value,
                "variable": self.shock.variable,
                "parameter": self.shock.parameter,
                "description": self.shock.description,
            },
            "initial_equilibrium": self.initial_equilibrium.to_dict(),
            "final_equilibrium": self.final_equilibrium.to_dict(),
            "transmission": {
                "shock_id": self.transmission.shock_id,
                "summary": self.transmission.summary,
                "steps": [s.__dict__ for s in self.transmission.ordered_steps()],
            },
            "warnings": list(self.warnings),
        }
