"""Meta-learning subsystem handling long-term adaptation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.base import BaseSubsystem
from ..core.registry import plugin_registry


class MetaLearner(BaseSubsystem):
    """Coordinate system-wide adaptation signals."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.base_learning_rate: float = float(self.config.get("base_learning_rate", 0.01))
        self.adaptation_factor: float = float(self.config.get("adaptation_factor", 1.05))
        self.performance_threshold: float = float(self.config.get("performance_threshold", 0.75))
        self.max_learning_rate: float = float(self.config.get("max_learning_rate", 0.2))
        self.min_learning_rate: float = float(self.config.get("min_learning_rate", 1e-4))
        self.learning_rate: float = self.base_learning_rate
        self.performance_history: List[float] = []
        self.adaptation_history: List[Dict[str, float | str]] = []

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update learning parameters based on current task signals."""
        hint = float(task.get("performance_hint", 0.5))
        history = list(task.get("performance_history", []))
        combined = history + [hint]
        trend = float(np.mean(combined)) if combined else hint

        if trend < self.performance_threshold:
            self.learning_rate = float(
                np.clip(
                    self.learning_rate * self.adaptation_factor,
                    self.min_learning_rate,
                    self.max_learning_rate,
                )
            )
            action = "increase"
        else:
            self.learning_rate = float(
                np.clip(
                    self.learning_rate / self.adaptation_factor,
                    self.min_learning_rate,
                    self.max_learning_rate,
                )
            )
            action = "decrease"

        record: Dict[str, float | str] = {
            "learning_rate": self.learning_rate,
            "trend": trend,
            "action": action,
        }
        self.performance_history.append(hint)
        self.adaptation_history.append(record)

        return {
            "learning_rate": self.learning_rate,
            "trend": trend,
            "action": action,
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialise meta-learner state."""
        return {
            "config": {
                "base_learning_rate": self.base_learning_rate,
                "adaptation_factor": self.adaptation_factor,
                "performance_threshold": self.performance_threshold,
                "max_learning_rate": self.max_learning_rate,
                "min_learning_rate": self.min_learning_rate,
            },
            "learning_rate": self.learning_rate,
            "performance_history": list(self.performance_history[-50:]),
            "adaptation_history": list(self.adaptation_history[-20:]),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the meta-learner from persisted state."""
        config = state.get("config", {})
        self.base_learning_rate = float(config.get("base_learning_rate", self.base_learning_rate))
        self.adaptation_factor = float(config.get("adaptation_factor", self.adaptation_factor))
        self.performance_threshold = float(
            config.get("performance_threshold", self.performance_threshold)
        )
        self.max_learning_rate = float(config.get("max_learning_rate", self.max_learning_rate))
        self.min_learning_rate = float(config.get("min_learning_rate", self.min_learning_rate))
        self.learning_rate = float(state.get("learning_rate", self.base_learning_rate))
        self.performance_history = [float(x) for x in state.get("performance_history", [])]
        self.adaptation_history = [
            {"learning_rate": float(entry.get("learning_rate", self.learning_rate)), "trend": float(entry.get("trend", 0.0)), "action": str(entry.get("action", "stable"))}
            for entry in state.get("adaptation_history", [])
        ]

    def get_status(self) -> Dict[str, Any]:
        """Return current learning parameters."""
        recent = self.adaptation_history[-1] if self.adaptation_history else {"action": "stable"}
        return {
            "learning_rate": self.learning_rate,
            "last_action": recent.get("action", "stable"),
            "history_length": len(self.performance_history),
        }


def _factory(config: Dict[str, Any]) -> MetaLearner:
    """Factory used for plugin registration."""
    return MetaLearner(config)


plugin_registry.register("meta_learner", _factory)
