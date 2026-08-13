"""Execution metrics service."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
@dataclass(slots=True)
class Timer:
    started: float
    @classmethod
    def start(cls) -> "Timer": return cls(perf_counter())
    def elapsed(self) -> float: return perf_counter() - self.started
class MetricsService:
    def start_timer(self) -> Timer: return Timer.start()
