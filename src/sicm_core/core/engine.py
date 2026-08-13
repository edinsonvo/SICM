from typing import Any, Tuple
from sicm_core.core.context import ExecutionContext


class Engine:

    def __init__(

        self,

        pipeline: Any

    ) -> None:

        self.pipeline = pipeline

    def run(self, experiment: Any) -> Tuple[Any, Any]:

        context = ExecutionContext(

            experiment=experiment
        )

        return self.pipeline.execute(context)
