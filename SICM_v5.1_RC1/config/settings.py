from dataclasses import dataclass


@dataclass(slots=True)
class Settings:

    debug: bool = False

    enable_cache: bool = True

    enable_logging: bool = True

    enable_animation: bool = False

    language: str = "es"
