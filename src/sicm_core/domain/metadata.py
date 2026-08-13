from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC


@dataclass(frozen=True, slots=True)
class Metadata:
    author: str
    institution: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
