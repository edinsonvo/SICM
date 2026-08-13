from dataclasses import dataclass, field

from .execution_result import ExecutionResult
from .experiment import Experiment


@dataclass(frozen=True, slots=True)
class ExecutionReport:

    experiment: Experiment

    result: ExecutionResult

    interpretation: str

    warnings: tuple[str, ...] = field(default_factory=tuple)
