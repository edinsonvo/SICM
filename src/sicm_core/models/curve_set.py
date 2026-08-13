from dataclasses import dataclass
from typing import Iterable
from .curve import Curve

@dataclass(frozen=True, slots=True)
class CurveSet:
    curves: tuple[Curve, ...] = ()

    def __post_init__(self) -> None:
        ids = [c.curve_id for c in self.curves]
        if len(ids) != len(set(ids)):
            raise ValueError("Curve IDs must be unique.")

    def get(self, curve_id: str) -> Curve | None:
        return next((c for c in self.curves if c.curve_id == curve_id), None)

    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.curves)

    @classmethod
    def from_iterable(cls, curves: Iterable[Curve]) -> "CurveSet":
        return cls(tuple(curves))
