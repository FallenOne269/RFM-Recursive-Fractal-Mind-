"""Tests for the meta learner subsystem."""

import pytest

from src.meta_learner import MetaLearner


def test_meta_learner_combines_component_signals() -> None:
    learner = MetaLearner({"learning_rate": 0.2, "momentum": 0.8})
    payload = {
        "fractal_output": {"final": [0.2, 0.4, 0.6]},
        "swarm_output": {"confidence": 0.7},
        "quantum_output": {"stability": 0.6},
    }
    result = learner.process(payload)
    assert 0.0 <= result["score"] <= 1.0
    assert result["recommendations"]


def test_meta_learner_handles_missing_inputs() -> None:
    learner = MetaLearner()
    result = learner.process({})
    assert result["score"] == 0.0
    assert "Collect more data" in result["recommendations"]


def test_meta_learner_invalid_config() -> None:
    with pytest.raises(ValueError):
        MetaLearner({"learning_rate": 0})
