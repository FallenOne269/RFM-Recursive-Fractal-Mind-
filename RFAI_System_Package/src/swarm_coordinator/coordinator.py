"""Swarm coordination subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils import SwarmCoordinatorConfig


@dataclass(frozen=True)
class SwarmCoordinatorOutput:
    """Result of coordinating agents for a task."""

    participating_agents: int
    overall_success_rate: float
    swarm_efficiency: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "participating_agents": self.participating_agents,
            "overall_success_rate": self.overall_success_rate,
            "swarm_efficiency": self.swarm_efficiency,
        }


class SwarmCoordinator:
    """Simple stochastic swarm coordinator."""

    def __init__(self, config: SwarmCoordinatorConfig) -> None:
        self._config = config

    def coordinate(self, complexity: float, agent_load: Sequence[float]) -> SwarmCoordinatorOutput:
        """Coordinate agents for a task complexity."""

        participating = int(max(1, round(self._config.swarm_size * complexity)))
        load = np.asarray(agent_load, dtype=float)
        if load.size == 0:
            load = np.zeros(participating)
        else:
            load = load[:participating]
        efficiency = float(np.clip(1.0 - load.mean(), 0.0, 1.0))
        success_rate = float(np.clip(0.5 * (efficiency + (1.0 - complexity / 2)), 0.0, 1.0))
        return SwarmCoordinatorOutput(
            participating_agents=participating,
            overall_success_rate=success_rate,
            swarm_efficiency=efficiency,
        )


def register_plugin(registry: "PluginRegistry") -> None:
    from utils import PluginRegistry

    if not isinstance(registry, PluginRegistry):  # pragma: no cover - defensive
        raise TypeError("registry must be a PluginRegistry instance")

    registry.register(
        "swarm_coordinator",
        "default",
        lambda config: SwarmCoordinator(config),
    )


__all__ = ["SwarmCoordinator", "SwarmCoordinatorOutput", "register_plugin"]
