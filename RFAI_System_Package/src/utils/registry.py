"""Plugin registry utilities for the RFAI system."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Iterable


class PluginRegistry:
    """Registry that tracks subsystem plugins.

    The registry keeps factories grouped by component type. Components are created
    on demand which avoids constructing unnecessary subsystems when they are
    disabled in the configuration.
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def register(self, component_type: str, name: str, factory: Callable[..., Any]) -> None:
        """Register a factory for a component type.

        Args:
            component_type: The logical subsystem name such as ``"fractal_engine"``.
            name: Identifier of the implementation variant. ``"default"`` is
                typically used.
            factory: Callable that returns an initialized component instance.

        Raises:
            ValueError: If an implementation with the same name is already
                registered for the component type.
        """

        component_factories = self._factories.setdefault(component_type, {})
        if name in component_factories:
            raise ValueError(
                f"Factory for {component_type!r} with name {name!r} already registered."
            )
        component_factories[name] = factory

    def create(self, component_type: str, *, name: str = "default", **kwargs: Any) -> Any:
        """Instantiate a registered component.

        Args:
            component_type: The subsystem type.
            name: The registered implementation name.
            **kwargs: Configuration passed to the factory.

        Returns:
            Any: The instantiated subsystem.

        Raises:
            KeyError: If the component type or name has not been registered.
        """

        try:
            factory = self._factories[component_type][name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(
                f"No factory registered for type={component_type!r}, name={name!r}."
            ) from exc
        return factory(**kwargs)

    def load_plugins(self, modules: Iterable[str]) -> None:
        """Import plugin modules and register their factories."""

        for module_name in modules:
            module = import_module(module_name)
            register = getattr(module, "register_plugin", None)
            if register is None:
                raise ImportError(
                    f"Module {module_name!r} does not define register_plugin(registry)."
                )
            register(self)


__all__ = ["PluginRegistry"]
