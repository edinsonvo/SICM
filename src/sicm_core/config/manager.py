from .settings import Settings

class ConfigManager:
    def __init__(self, settings=None):
        self._settings = settings or Settings.from_env()

    @property
    def settings(self):
        return self._settings
