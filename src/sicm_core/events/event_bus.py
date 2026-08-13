"""Synchronous event bus."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
from .events import Event

EventHandler = Callable[[Event], None]

class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        if handler not in self._handlers[event_name]:
            self._handlers[event_name].append(handler)
    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        if handler in self._handlers[event_name]:
            self._handlers[event_name].remove(handler)
    def publish(self, event: Event) -> None:
        for handler in tuple(self._handlers.get(event.name, ())):
            handler(event)
