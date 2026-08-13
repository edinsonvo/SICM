"""Execution report service."""
from sicm_core.domain.execution_report import ExecutionReport
from sicm_core.domain.execution_result import ExecutionResult
from sicm_core.domain.experiment import Experiment
class ReportService:
    def build(self, experiment: Experiment, result: ExecutionResult, interpretation: str, warnings: tuple[str, ...] = ()) -> ExecutionReport:
        return ExecutionReport(experiment=experiment, result=result, interpretation=interpretation, warnings=warnings)
