"""Unit tests for the swarm coordinator."""

from __future__ import annotations

from swarm_coordinator import SwarmCoordinator
from utils import SwarmCoordinatorConfig


def test_swarm_coordination_metrics() -> None:
    coordinator = SwarmCoordinator(SwarmCoordinatorConfig(swarm_size=10, max_parallel_tasks=3))
    output = coordinator.coordinate(0.6, [0.1] * 10)
    assert 0 <= output.overall_success_rate <= 1
    assert 0 <= output.swarm_efficiency <= 1
    assert 1 <= output.participating_agents <= 10
