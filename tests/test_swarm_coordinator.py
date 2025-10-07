"""Tests for the swarm coordinator subsystem."""

import pytest

from src.swarm_coordinator import SwarmCoordinator


def test_swarm_coordinator_builds_consensus() -> None:
    coordinator = SwarmCoordinator({"agent_count": 3, "consensus_threshold": 0.8})
    result = coordinator.process({"final": [1.0, 2.0, 3.0]})
    assert result["consensus"] == pytest.approx(2.0)
    assert len(result["agent_votes"]) == 3
    assert 0 <= result["confidence"] <= 1


def test_swarm_coordinator_handles_unexpected_payload() -> None:
    coordinator = SwarmCoordinator()
    result = coordinator.process("not-a-dict")
    assert result["consensus"] is None
    assert result["confidence"] == 0.0


def test_swarm_coordinator_invalid_config() -> None:
    with pytest.raises(ValueError):
        SwarmCoordinator({"agent_count": 0})
