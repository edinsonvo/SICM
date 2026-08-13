class ServiceContainer:
    def __init__(self):
        self._services = {}

    def register(self, name, service):
        self._services[name] = service

    def resolve(self, name):
        if name not in self._services:
            raise KeyError(f"Service not registered: {name}")
        return self._services[name]
