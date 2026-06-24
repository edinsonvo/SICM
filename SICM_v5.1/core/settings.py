from dataclasses import dataclass


@dataclass
class SICMSettings:

    APP_NAME = "SICM v5.1 Research Lab"

    VERSION = "5.1-alpha"

    AUTHOR = (
        "edvalenciao@unal.edu.co"
    )

    DEFAULT_INFLATION_TARGET = 2.0

    DEFAULT_NATURAL_UNEMPLOYMENT = 5.0

    ENABLE_LOGGING = True

    ENABLE_CACHE = True
