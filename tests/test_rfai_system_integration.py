"""Integration tests for the full RFAI system."""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rfai_system import RFAISystem
from src.utils.state_manager import load_state, save_state


def test_rfai_system_cycle_with_disabled_component() -> None:
    system = RFAISystem({"quantum_processor": {"enabled": False}})
    result = system.run_cycle([0.1, 0.2, 0.3])
    assert set(result) >= {
        "fractal_output",
        "swarm_output",
        "quantum_output",
        "meta_output",
        "errors",
    }
    assert result["quantum_output"] is None
    assert result["errors"] == []


def test_state_persistence_round_trip(tmp_path: Path) -> None:
    state = {"cycles": 5, "status": {"ok": True}}
    path = save_state(tmp_path / "state.json", state)
    loaded = load_state(path)
    assert loaded == state


def test_api_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "secret")
    api_module = importlib.import_module("src.api")
    importlib.reload(api_module)
    client = TestClient(api_module.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/process_task",
        json={"values": [1, 2, 3]},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data) >= {
        "fractal_output",
        "swarm_output",
        "quantum_output",
        "meta_output",
        "errors",
    }

    status_response = client.get("/status", headers={"X-API-Key": "secret"})
    assert status_response.status_code == 200
    assert "components" in status_response.json()
