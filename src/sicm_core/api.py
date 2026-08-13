"""
Public API for SICM Core.
"""

from __future__ import annotations

from sicm_core.core.engine import Engine
from sicm_core.domain.equilibrium import Equilibrium
from sicm_core.domain.execution_report import ExecutionReport
from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment
from sicm_core.domain.metrics import Metrics
from sicm_core.domain.parameters import EconomyParameters
from sicm_core.domain.scenario import Scenario
from sicm_core.domain.shock import Shock
from sicm_core.config.enums import EconomyType, ModelType, ShockType
from sicm_core.core.bootstrap import bootstrap


def create_engine() -> Engine:
    """
    Creates a fully configured SICM engine.

    Returns:
        A ready-to-use Engine instance.
    """
    return bootstrap().resolve("engine")


__all__ = [
    "EconomyParameters",
    "EconomyType",
    "Engine",
    "Equilibrium",
    "ExecutionReport",
    "ExecutionResult",
    "Experiment",
    "Metrics",
    "ModelType",
    "Scenario",
    "Shock",
    "ShockType",
    "create_engine",
]
