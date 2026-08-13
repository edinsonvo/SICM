from sicm_core import create_engine, ModelType
from sicm_core.config.settings import Settings
from sicm_core.events.event_bus import EventBus
from sicm_core.events.events import Event
from sicm_core.services.container import ServiceContainer

def test_public_engine():
    assert create_engine() is not None

def test_settings():
    assert Settings().app_name == "SICM Core"

def test_events():
    bus, received = EventBus(), []
    bus.subscribe("test", received.append)
    bus.publish(Event("test"))
    assert len(received) == 1

def test_container():
    container, value = ServiceContainer(), object()
    container.register("value", value)
    assert container.resolve("value") is value

def test_model_enum():
    assert ModelType.ISLM.value == "islm"
