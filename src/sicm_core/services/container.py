"""Minimal dependency injection container."""
from __future__ import annotations
from typing import Any, TypeVar
T = TypeVar("T")
class ServiceContainer:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
    def register(self, name: str, service: Any) -> None:
        self._services[name] = service
    def resolve(self, name: str) -> Any:
        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(f"Service not registered: {name}") from exc
    def has(self, name: str) -> bool:
        return name in self._services
