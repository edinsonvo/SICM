from sicm_core.config.manager import ConfigManager
from sicm_core.config.logging_config import configure_logging
from sicm_core.core.dispatcher import Dispatcher
from sicm_core.core.engine import Engine
from sicm_core.core.pipeline import Pipeline
from sicm_core.core.registry import Registry
from sicm_core.events.event_bus import EventBus
from sicm_core.plugins.manager import PluginManager
from sicm_core.services.container import ServiceContainer
from sicm_core.services.metrics_service import MetricsService
from sicm_core.services.report_service import ReportService
from sicm_core.services.validation_service import ValidationService
from sicm_core.validation.validator import Validator

def bootstrap(load_plugins=True):
    container = ServiceContainer()
    config = ConfigManager()
    logger = configure_logging(config.settings.log_level, config.settings.log_file)
    registry = Registry()
    validator = Validator()
    event_bus = EventBus()
    dispatcher = Dispatcher(registry)
    pipeline = Pipeline(validator, dispatcher)
    plugin_manager = PluginManager(registry, config.settings.plugin_namespace)
    if load_plugins:
        plugin_manager.load()
    services = {
        "config": config, "logger": logger, "registry": registry,
        "validator": validator, "event_bus": event_bus,
        "dispatcher": dispatcher, "pipeline": pipeline,
        "report_service": ReportService(), "metrics_service": MetricsService(),
        "validation_service": ValidationService(validator),
        "plugin_manager": plugin_manager, "engine": Engine(pipeline),
    }
    for name, service in services.items():
        container.register(name, service)
    return container
