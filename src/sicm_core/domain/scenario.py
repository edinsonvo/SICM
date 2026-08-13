from __future__ import annotations

from dataclasses import dataclass, field

from .parameters import EconomyParameters
from .shock import Shock

from sicm_core.config.enums import ModelType


@dataclass(frozen=True, slots=True)
class Scenario:

    model: ModelType

    parameters: EconomyParameters

    shocks: tuple[Shock, ...] = field(default_factory=tuple)

    name: str = ""

    notes: str = ""
