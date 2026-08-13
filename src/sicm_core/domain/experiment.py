from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from .metadata import Metadata
from .scenario import Scenario


@dataclass(frozen=True, slots=True)
class Experiment:

    name: str

    scenario: Scenario

    metadata: Metadata

    id: UUID = field(default_factory=uuid4)
