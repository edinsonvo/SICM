"""Built-in event subscribers."""
from __future__ import annotations
import logging
from .events import Event

def logging_subscriber(logger: logging.Logger):
    def handle(event: Event) -> None:
        logger.info("Event %s execution=%s payload=%s", event.name, event.execution_id, event.payload)
    return handle
