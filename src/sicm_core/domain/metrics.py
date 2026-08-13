from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metrics:

    execution_time: float

    iterations: int

    converged: bool
