"""Configuration manager."""
from __future__ import annotations
from .settings import Settings

class ConfigManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
    @property
    def settings(self) -> Settings:
        return self._settings
