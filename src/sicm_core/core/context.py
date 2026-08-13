from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4
from sicm_core.domain.experiment import Experiment

@dataclass(slots=True)
class ExecutionContext:
    experiment: Experiment
    execution_id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
