from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True, slots=True)
class EconomyParameters:

    C0: float = 50.0
    c: float = 0.80

    I0: float = 100.0
    b: float = 5.0

    G: float = 150.0
    T: float = 120.0

    M: float = 500.0
    P: float = 1.0

    k: float = 0.5
    h: float = 10.0

    NX0: float = 20.0

    A: float = 1.0
    K: float = 100.0
    alpha: float = 0.33

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
