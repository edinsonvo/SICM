from importlib.metadata import entry_points

class PluginLoader:
    def discover(self, namespace):
        return tuple(entry_points(group=namespace))

    def load(self, namespace):
        return {entry.name: entry.load() for entry in self.discover(namespace)}
