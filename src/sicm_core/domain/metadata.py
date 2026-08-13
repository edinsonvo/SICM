from dataclasses import dataclass, field
from datetime import UTC, datetime

@dataclass(frozen=True, slots=True)
class Metadata:
    author: str
    institution: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    description: str = ""
    tags: tuple[str, ...] = ()
