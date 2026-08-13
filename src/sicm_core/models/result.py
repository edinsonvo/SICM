from dataclasses import dataclass, field
from typing import Mapping
from .curve_set import CurveSet
from .equilibrium import Equilibrium
from .shock import ModelShock
from .transmission import TransmissionMechanism

@dataclass(frozen=True, slots=True)
class ModelResult:
    model_name: str
    model_family: str
    initial_equilibrium: Equilibrium
    final_equilibrium: Equilibrium
    curves_before: CurveSet = CurveSet()
    curves_after: CurveSet = CurveSet()
    shock: ModelShock | None = None
    transmission: TransmissionMechanism | None = None
    interpretation: str = ""
    diagnostics: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def delta(self, symbol: str) -> float | None:
        before = self.initial_equilibrium.value(symbol)
        after = self.final_equilibrium.value(symbol)
        if before is None or after is None:
            return None
        return after - before

    def deltas(self) -> dict[str, float]:
        symbols = set(self.initial_equilibrium.values) | set(self.final_equilibrium.values)
        return {s: self.delta(s) for s in symbols if self.delta(s) is not None}

    @property
    def shocked(self) -> bool:
        return self.shock is not None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "initial_equilibrium": self.initial_equilibrium.to_dict(),
            "final_equilibrium": self.final_equilibrium.to_dict(),
            "curves_before": [c.to_dict() for c in self.curves_before.curves],
            "curves_after": [c.to_dict() for c in self.curves_after.curves],
            "shock": None if self.shock is None else {
                "shock_id": self.shock.shock_id,
                "shock_type": self.shock.shock_type,
                "target": self.shock.target.value,
                "magnitude": self.shock.magnitude,
                "direction": self.shock.direction.value,
            },
            "transmission": None if self.transmission is None else {
                "shock_id": self.transmission.shock_id,
                "summary": self.transmission.summary,
                "steps": [
                    {
                        "order": s.order, "source": s.source, "channel": s.channel,
                        "affected_variable": s.affected_variable,
                        "effect": s.effect, "magnitude": s.magnitude,
                        "description": s.description,
                    }
                    for s in self.transmission.ordered_steps()
                ],
            },
            "interpretation": self.interpretation,
            "diagnostics": list(self.diagnostics),
            "metadata": dict(self.metadata),
        }
