"""Plugin registration and discovery utilities."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Type

from .base import BaseSubsystem, ConfigurationError


class PluginRegistry:
    """Simple registry that allows subsystems to self-register as plugins."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[dict], BaseSubsystem]] = {}

    def register(self, name: str, factory: Callable[[dict], BaseSubsystem]) -> None:
        """Register a factory callable for the provided plugin name."""
        if name in self._registry:
            raise ConfigurationError(f"Plugin '{name}' is already registered")
        self._registry[name] = factory

    def create(self, name: str, config: dict) -> BaseSubsystem:
        """Instantiate a plugin by name using the provided configuration."""
        try:
            factory = self._registry[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ConfigurationError(f"Unknown plugin '{name}'") from exc
        return factory(config)

    def available(self) -> Iterable[str]:
        """Return names of all registered plugins."""
        return sorted(self._registry.keys())


plugin_registry = PluginRegistry()
"""Global plugin registry used across the package."""
