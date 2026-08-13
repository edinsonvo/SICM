from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from sicm_core.config.enums import ShockType

@dataclass(frozen=True, slots=True)
class Shock:
    shock_type: ShockType
    magnitude: float
    variable: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
