"""Base classes and common interfaces for RFAI subsystems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSubsystem(ABC):
    """Abstract base class for every subsystem managed by the orchestrator."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the subsystem with validated configuration values."""
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one processing step for the provided task."""

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the subsystem state."""

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the subsystem from a previously saved state."""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return lightweight operational status information."""

    def shutdown(self) -> None:
        """Allow subsystems to release resources when the orchestrator stops."""


class OrchestratorError(RuntimeError):
    """Base exception raised for orchestrator level failures."""


class ConfigurationError(OrchestratorError):
    """Exception raised when configuration loading or validation fails."""


class StatePersistenceError(OrchestratorError):
    """Exception raised when persistence or restoration of state fails."""
