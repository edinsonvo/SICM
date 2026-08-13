from dataclasses import dataclass, field
from .experiment import Experiment
from .execution_result import ExecutionResult

@dataclass(frozen=True, slots=True)
class ExecutionReport:
    experiment: Experiment
    result: ExecutionResult
    interpretation: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
