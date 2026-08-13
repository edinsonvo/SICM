from __future__ import annotations

from typing import Protocol

from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment


class EconomicModel(Protocol):
    """Contrato mínimo que debe cumplir cualquier modelo económico."""

    name: str
    family: str

    def solve(self, experiment: Experiment) -> ExecutionResult:
        ...
