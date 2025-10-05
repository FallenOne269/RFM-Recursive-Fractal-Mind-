"""Semantic goal definitions for the RFIM core package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping


@dataclass
class SemanticGoal:
    """Represents an optimisation goal for a semantic node."""

    name: str
    description: str = ""
    success_criteria: Mapping[str, Callable[[float], bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        for key, criterion in self.success_criteria.items():
            if not callable(criterion):
                raise TypeError(f"Criterion for '{key}' must be callable")

    def is_satisfied(self, metrics: Mapping[str, float]) -> bool:
        """Return ``True`` when all success criteria hold for ``metrics``."""

        for key, criterion in self.success_criteria.items():
            value = metrics.get(key)
            if value is None or not criterion(value):
                return False
        return True

    def describe(self) -> Dict[str, str]:  # pragma: no cover - trivial
        return {"name": self.name, "description": self.description}
