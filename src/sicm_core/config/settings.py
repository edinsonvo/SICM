"""Application settings."""
from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "SICM Core"
    version: str = "1.0.0a1"
    debug: bool = False
    log_level: str = "INFO"
    default_model: str = "islm"
    plugin_namespace: str = "sicm.models"
    log_file: str = "logs/sicm.log"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            debug=os.getenv("SICM_DEBUG", "false").lower() in {"1","true","yes"},
            log_level=os.getenv("SICM_LOG_LEVEL", "INFO").upper(),
            default_model=os.getenv("SICM_DEFAULT_MODEL", "islm"),
            plugin_namespace=os.getenv("SICM_PLUGIN_NAMESPACE", "sicm.models"),
            log_file=os.getenv("SICM_LOG_FILE", "logs/sicm.log"),
        )
