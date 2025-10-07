"""Meta-learning subsystem."""

from __future__ import annotations

from dataclasses import dataclass

from utils import MetaLearnerConfig


@dataclass(frozen=True)
class MetaLearnerOutput:
    """Meta-learner update summary."""

    updated_learning_rate: float
    adaptation_applied: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "updated_learning_rate": self.updated_learning_rate,
            "adaptation_applied": self.adaptation_applied,
            "status": self.status,
        }


class MetaLearner:
    """Adaptive meta-learner adjusting system parameters."""

    def __init__(self, config: MetaLearnerConfig) -> None:
        self._config = config
        self._current_lr = config.base_learning_rate

    def step(self, performance_score: float) -> MetaLearnerOutput:
        if performance_score < self._config.performance_threshold:
            self._current_lr *= self._config.adaptation_factor
            status = "accelerated"
            adaptation = self._config.adaptation_factor
        else:
            self._current_lr /= self._config.adaptation_factor
            status = "stabilized"
            adaptation = 1 / self._config.adaptation_factor
        return MetaLearnerOutput(
            updated_learning_rate=self._current_lr,
            adaptation_applied=adaptation,
            status=status,
        )


def register_plugin(registry: "PluginRegistry") -> None:
    from utils import PluginRegistry

    if not isinstance(registry, PluginRegistry):  # pragma: no cover - defensive
        raise TypeError("registry must be a PluginRegistry instance")

    registry.register(
        "meta_learner",
        "default",
        lambda config: MetaLearner(config),
    )


__all__ = ["MetaLearner", "MetaLearnerOutput", "register_plugin"]
