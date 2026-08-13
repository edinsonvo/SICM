from dataclasses import dataclass
from time import perf_counter

@dataclass(slots=True)
class Timer:
    started: float

    @classmethod
    def start(cls):
        return cls(perf_counter())

    def elapsed(self):
        return perf_counter() - self.started

class MetricsService:
    def start_timer(self):
        return Timer.start()
