"""Domain events emitted by SICM Core."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

@dataclass(frozen=True, slots=True)
class Event:
    name: str
    execution_id: UUID | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

class ExperimentStarted(Event):
    def __init__(self, execution_id: UUID) -> None:
        super().__init__(name="experiment.started", execution_id=execution_id)

class ExperimentFinished(Event):
    def __init__(self, execution_id: UUID) -> None:
        super().__init__(name="experiment.finished", execution_id=execution_id)

class ExperimentFailed(Event):
    def __init__(self, execution_id: UUID, error: str) -> None:
        super().__init__(name="experiment.failed", execution_id=execution_id, payload={"error": error})
