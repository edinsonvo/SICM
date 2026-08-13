from dataclasses import dataclass, field
from sicm_core.config.enums import ModelType
from .parameters import EconomyParameters
from .shock import Shock

@dataclass(frozen=True, slots=True)
class Scenario:
    model: ModelType
    parameters: EconomyParameters
    shocks: tuple[Shock, ...] = field(default_factory=tuple)
    name: str = ""
    notes: str = ""
