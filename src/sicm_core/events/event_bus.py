from collections import defaultdict

class EventBus:
    def __init__(self):
        self._handlers = defaultdict(list)

    def subscribe(self, event_name, handler):
        if handler not in self._handlers[event_name]:
            self._handlers[event_name].append(handler)

    def publish(self, event):
        for handler in tuple(self._handlers.get(event.name, ())):
            handler(event)
