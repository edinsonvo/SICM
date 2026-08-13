"""Built-in event subscribers."""
from __future__ import annotations
import logging
from typing import Callable, Any
from .events import Event


def logging_subscriber(logger: logging.Logger) -> Callable[[Event], None]:
    def handle(event: Event) -> None:
        logger.info("Event %s execution=%s payload=%s", event.name, event.execution_id, event.payload)
    return handle
