from sicm_core.config.settings import Settings
from sicm_core.events.event_bus import EventBus
from sicm_core.events.events import Event
from sicm_core.services.container import ServiceContainer

def test_settings_defaults():
    assert Settings().app_name == "SICM Core"

def test_event_bus():
    bus, received = EventBus(), []
    bus.subscribe("test", received.append)
    bus.publish(Event(name="test"))
    assert len(received) == 1

def test_container():
    c = ServiceContainer(); value = object(); c.register("x", value)
    assert c.resolve("x") is value
