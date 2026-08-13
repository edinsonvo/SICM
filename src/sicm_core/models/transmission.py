from dataclasses import dataclass
from typing import Mapping, Sequence

@dataclass(frozen=True, slots=True)
class TransmissionStep:
    order: int
    source: str
    channel: str
    affected_variable: str
    effect: str
    magnitude: float | None = None
    description: str = ""

@dataclass(frozen=True, slots=True)
class TransmissionMechanism:
    shock_id: str
    steps: Sequence[TransmissionStep]
    summary: str = ""
    assumptions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        orders = [s.order for s in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError("Transmission step orders must be unique.")

    def ordered_steps(self) -> tuple[TransmissionStep, ...]:
        return tuple(sorted(self.steps, key=lambda s: s.order))

    def channels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.channel for s in self.ordered_steps()))
