"""High level smoke tests for Recursive Fractal Mind."""

from __future__ import annotations

from rfai_system import RecursiveFractalMind


def test_run_cycle_returns_all_outputs(config_path) -> None:
    orchestrator = RecursiveFractalMind(str(config_path))
    result = orchestrator.run_cycle(
        {
            "id": "smoke-001",
            "type": "pattern",
            "complexity": 0.5,
            "payload": [0.1, 0.2, 0.3],
            "metadata": {"scope": "test"},
        }
    )
    assert result["fractal_output"] is not None
    assert result["swarm_output"] is not None
    assert "meta_output" in result
    assert "performance_score" in result


def test_state_round_trip(config_path) -> None:
    orchestrator = RecursiveFractalMind(str(config_path))
    orchestrator.run_cycle(
        {
            "id": "state-001",
            "type": "analysis",
            "complexity": 0.4,
            "payload": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"stage": "initial"},
        }
    )
    saved_path = orchestrator.save_state()
    restored = RecursiveFractalMind(str(config_path))
    restored.load_state(saved_path)
    assert restored.get_status()["cycles"] == 1
