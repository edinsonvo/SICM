from dataclasses import dataclass
from .equilibrium import Equilibrium
from .curve_set import CurveSet

@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    equilibrium: Equilibrium
    curves: CurveSet = CurveSet()
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"equilibrium": self.equilibrium.to_dict(),
                "curves": [c.to_dict() for c in self.curves.curves],
                "diagnostics": list(self.diagnostics)}
