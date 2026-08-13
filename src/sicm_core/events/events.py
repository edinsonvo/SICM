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
