from dataclasses import dataclass
from sicm_core.config.enums import ShockType

@dataclass(frozen=True, slots=True)
class Shock:
    shock_type: ShockType
    magnitude: float
    variable: str
    description: str = ""
