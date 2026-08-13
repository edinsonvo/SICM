from dataclasses import dataclass

from .equilibrium import Equilibrium
from .metrics import Metrics


@dataclass(frozen=True, slots=True)
class ExecutionResult:

    equilibrium: Equilibrium

    metrics: Metrics
