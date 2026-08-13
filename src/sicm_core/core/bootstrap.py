"""
Application bootstrap.
"""

from __future__ import annotations

from sicm_core.config.manager import ConfigManager
from sicm_core.core.dispatcher import Dispatcher
from sicm_core.core.engine import Engine
from sicm_core.core.pipeline import Pipeline
from sicm_core.core.registry import Registry
from sicm_core.events.event_bus import EventBus
from sicm_core.services.container import ServiceContainer
from sicm_core.validation.validator import Validator


def bootstrap() -> ServiceContainer:
    """
    Build and configure the SICM dependency container.

    Returns:
        Configured service container.
    """

    container = ServiceContainer()

    config = ConfigManager()
    registry = Registry()
    validator = Validator()
    event_bus = EventBus()

    dispatcher = Dispatcher(registry)

    pipeline = Pipeline(
        validator=validator,
        dispatcher=dispatcher,
    )

    engine = Engine(
        pipeline=pipeline,
    )

    container.register("config", config)
    container.register("registry", registry)
    container.register("validator", validator)
    container.register("event_bus", event_bus)
    container.register("dispatcher", dispatcher)
    container.register("pipeline", pipeline)
    container.register("engine", engine)

    return container
