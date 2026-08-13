from sicm_core.domain.execution_report import ExecutionReport

class ReportService:
    def build(self, experiment, result, interpretation, warnings=()):
        return ExecutionReport(experiment, result, interpretation, warnings)
