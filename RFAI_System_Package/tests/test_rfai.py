"""Comprehensive pytest suite for the modular RFAI system."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfai_system import Orchestrator  # noqa: E402
from rfai_system.api import server as api_server  # noqa: E402
from rfai_system.fractal_engine import FractalEngine  # noqa: E402
from rfai_system.meta_learner import MetaLearner  # noqa: E402
from rfai_system.quantum_processor import QuantumProcessor  # noqa: E402
from rfai_system.swarm_coordinator import SwarmCoordinator  # noqa: E402
from rfai_system.utils.validation import (  # noqa: E402
    ConfigurationError,
    load_and_validate_config,
    validate_config,
    validate_task,
)


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    """Load the packaged default configuration once per test session."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


@pytest.fixture()
def orchestrator(config: Dict[str, Any]) -> Orchestrator:
    """Instantiate an orchestrator for a test."""
    return Orchestrator(config=config)


def test_config_schema_validation(config: Dict[str, Any]) -> None:
    """The default configuration satisfies the validation schema."""
    validate_config(config)


def test_invalid_config_rejected() -> None:
    """Missing module definitions trigger configuration errors."""
    bad_config = {
        "system": {
            "max_fractal_depth": 2,
            "base_dimensions": 8,
            "swarm_size": 2,
            "quantum_enabled": False,
            "recursion_limit": 2,
        }
    }
    with pytest.raises(ConfigurationError):
        load_and_validate_config(bad_config)


def test_task_validation_bounds() -> None:
    """Task validation enforces complexity boundaries."""
    with pytest.raises(ConfigurationError):
        validate_task({"id": "bad", "type": "x", "complexity": 1.5})


def test_fractal_engine_processing(config: Dict[str, Any]) -> None:
    """Fractal engine returns embeddings of the configured dimension."""
    engine = FractalEngine(config=config["modules"]["fractal_engine"]["settings"])
    task = {"id": "fractal", "type": "pattern_recognition", "complexity": 0.5, "data": np.ones(64)}
    output = engine.process(task)
    assert (
        len(output["representation"])
        == config["modules"]["fractal_engine"]["settings"]["base_dimensions"]
    )
    assert 0.0 <= output["activation_energy"]


@pytest.mark.performance
def test_fractal_engine_high_depth_performance() -> None:
    """High recursion depths remain performant within bounds."""
    engine = FractalEngine(
        {
            "max_depth": 6,
            "base_dimensions": 48,
            "branching_factor": 2,
            "recursion_limit": 6,
        }
    )
    task = {"id": "perf", "type": "pattern_recognition", "complexity": 0.7, "data": np.zeros(48)}
    start = time.perf_counter()
    engine.process(task)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"Fractal engine too slow: {elapsed}"


def test_swarm_coordinator_synergy_bounds(config: Dict[str, Any]) -> None:
    """Swarm synergy values stay within [0, 1]."""
    coordinator = SwarmCoordinator(config=config["modules"]["swarm_coordinator"]["settings"])
    task = {"id": "swarm", "type": "pattern_recognition", "complexity": 0.9}
    output = coordinator.process(task)
    assert 0.0 <= output["synergy"] <= 1.0
    assert output["selected_agents"], "At least one agent participates"


def test_quantum_processor_disabled() -> None:
    """When disabled the quantum processor returns an inactive state."""
    processor = QuantumProcessor({"enabled": False, "qubits": 4})
    result = processor.process({"id": "q", "type": "t", "complexity": 0.2})
    assert result["active"] is False
    assert result["entanglement_entropy"] == 0.0


def test_meta_learner_adjusts_learning_rate() -> None:
    """Meta learner increases learning rate when performance dips."""
    learner = MetaLearner(
        {"base_learning_rate": 0.01, "adaptation_factor": 2.0, "performance_threshold": 0.9}
    )
    initial = learner.learning_rate
    result = learner.process({"performance_hint": 0.2})
    assert result["learning_rate"] > initial


def test_orchestrator_process_task(orchestrator: Orchestrator) -> None:
    """Orchestrator returns all component outputs for a task."""
    task = {
        "id": "orchestrator",
        "type": "pattern_recognition",
        "complexity": 0.6,
        "data": np.ones(64),
    }
    result = orchestrator.process_task(task)
    for key in ["fractal_output", "swarm_output", "quantum_output", "meta_output"]:
        assert key in result
    assert 0.0 <= result["performance_score"] <= 1.0


def test_state_persistence_round_trip(orchestrator: Orchestrator, tmp_path: Path) -> None:
    """Persisted state can be restored with full fidelity."""
    for index in range(3):
        orchestrator.process_task(
            {"id": f"state-{index}", "type": "pattern_recognition", "complexity": 0.4 + index * 0.1}
        )
    state_file = tmp_path / "rfai_state.json"
    orchestrator.save_state(str(state_file))

    restored = Orchestrator(config=orchestrator.config)
    restored.load_state(str(state_file))

    assert orchestrator.history == restored.history
    assert orchestrator.metrics == restored.metrics
    for name in orchestrator.modules:
        assert orchestrator.modules[name].get_state() == restored.modules[name].get_state()


def test_fastapi_endpoints(orchestrator: Orchestrator, monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI endpoints respond with expected schemas."""
    api_server.get_orchestrator.cache_clear()
    monkeypatch.setattr(api_server, "get_orchestrator", lambda: orchestrator)
    client = TestClient(api_server.app)

    health = client.get("/health").json()
    expected = {
        "status",
        "fractal_output",
        "swarm_output",
        "quantum_output",
        "meta_output",
        "tasks_processed",
    }
    assert expected.issubset(health)

    status = client.get("/status").json()
    assert status["fractal_output"] is not None

    payload = {"id": "api", "type": "pattern_recognition", "complexity": 0.5}
    response = client.post("/process_task", json=payload)
    if response.status_code != 200:
        pytest.fail(f"API error: {response.status_code} {response.text}")
    body = response.json()
    for key in ["fractal_output", "swarm_output", "quantum_output", "meta_output"]:
        assert key in body


def test_integration_recursive_loop(orchestrator: Orchestrator) -> None:
    """Integration test covering a recursive processing loop."""
    tasks = [
        {"id": f"integration-{idx}", "type": "pattern_recognition", "complexity": 0.3 + idx * 0.1}
        for idx in range(5)
    ]
    scores = [orchestrator.process_task(task)["performance_score"] for task in tasks]
    assert len(scores) == len(tasks)
    assert orchestrator.metrics["tasks_processed"] >= len(tasks)


def test_plugin_registration_allows_extension(
    monkeypatch: pytest.MonkeyPatch, config: Dict[str, Any]
) -> None:
    """New plugins can self-register and be instantiated."""
    from rfai_system.core.registry import plugin_registry
    from rfai_system.core.base import BaseSubsystem

    class DummySubsystem(BaseSubsystem):
        def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
            return {"dummy": True}

        def get_state(self) -> Dict[str, Any]:
            return {"dummy": True}

        def set_state(self, state: Dict[str, Any]) -> None:
            return None

        def get_status(self) -> Dict[str, Any]:
            return {"dummy": True}

    monkeypatch.setattr(
        plugin_registry, "_registry", dict(plugin_registry._registry), raising=False
    )
    plugin_registry.register("dummy", lambda cfg: DummySubsystem(cfg))
    config = json.loads(json.dumps(config))
    config["modules"]["dummy"] = {"plugin": "dummy", "settings": {}}
    orchestrator = Orchestrator(config=config)
    assert "dummy" in orchestrator.modules


def test_quantum_processor_entropy_progression() -> None:
    """Entropy history records are updated per task."""
    processor = QuantumProcessor({"enabled": True, "qubits": 3})
    base_len = len(processor.entropy_history)
    processor.process({"id": "q1", "type": "pattern_recognition", "complexity": 0.5})
    assert len(processor.entropy_history) == base_len + 1


def test_meta_learner_state_serialisation() -> None:
    """Meta learner state serialises and restores correctly."""
    learner = MetaLearner({"base_learning_rate": 0.01})
    learner.process({"performance_hint": 0.2})
    state = learner.get_state()
    restored = MetaLearner({})
    restored.set_state(state)
    assert restored.get_state() == state
