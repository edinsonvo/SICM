from enum import Enum


class ExecutionState(str, Enum):

    CREATED = "created"

    VALIDATING = "validating"

    RUNNING = "running"

    FINISHED = "finished"

    FAILED = "failed"
