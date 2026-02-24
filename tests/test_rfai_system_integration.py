"""Integration tests for the full RFAI system."""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rfai_system import RFAISystem
from src.utils.state_manager import load_state, save_state


# ---------------------------------------------------------------------------
# Orchestrator unit tests
# ---------------------------------------------------------------------------


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


def test_rfai_system_disabled_component_tracked_in_status() -> None:
    system = RFAISystem({"swarm_coordinator": {"enabled": False}})
    status = system.get_status()
    assert "swarm_coordinator" in status["disabled_components"]
    assert not status["components"]["swarm_coordinator"]


def test_rfai_system_invalid_input_returns_error() -> None:
    system = RFAISystem()
    result = system.run_cycle("not-valid")
    assert result["errors"]
    assert result["fractal_output"] is None


def test_rfai_system_cycles_increment() -> None:
    system = RFAISystem()
    assert system.get_status()["cycles"] == 0
    system.run_cycle([1.0])
    assert system.get_status()["cycles"] == 1
    system.run_cycle([2.0])
    assert system.get_status()["cycles"] == 2


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------


def test_state_persistence_round_trip(tmp_path: Path) -> None:
    state = {"cycles": 5, "status": {"ok": True}}
    path = save_state(tmp_path / "state.json", state)
    loaded = load_state(path)
    assert loaded == state


def test_state_load_detects_tampering(tmp_path: Path) -> None:
    import json

    state = {"cycles": 3}
    path = save_state(tmp_path / "state.json", state)
    # Corrupt the stored state without updating the checksum.
    payload = json.loads(path.read_text())
    payload["state"]["cycles"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_state(path)


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


def _fresh_client(monkeypatch: pytest.MonkeyPatch, api_key: str = "secret") -> TestClient:
    """Reload the api module with the given API_KEY and return a TestClient."""
    monkeypatch.setenv("API_KEY", api_key)
    api_module = importlib.import_module("src.api")
    importlib.reload(api_module)
    return TestClient(api_module.app)


def test_api_health_no_auth_required() -> None:
    """Health endpoint must be reachable without credentials."""
    api_module = importlib.import_module("src.api")
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fresh_client(monkeypatch)

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
    status_data = status_response.json()
    assert "components" in status_data
    assert "disabled_components" in status_data
    assert "failed_components" in status_data


def test_api_rejects_wrong_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fresh_client(monkeypatch)
    response = client.get("/status", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_api_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fresh_client(monkeypatch)
    response = client.get("/status")
    assert response.status_code == 401


def test_api_rejects_empty_values_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty values list must fail Pydantic validation (422)."""
    client = _fresh_client(monkeypatch)
    response = client.post(
        "/process_task",
        json={"values": []},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 422


def test_api_rejects_missing_values_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body without the required `values` field must return 422."""
    client = _fresh_client(monkeypatch)
    response = client.post(
        "/process_task",
        json={"data": [1, 2, 3]},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 422


def test_api_response_contains_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every response must carry a unique X-Request-ID header."""
    client = _fresh_client(monkeypatch)
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert "x-request-id" in r1.headers
    assert "x-request-id" in r2.headers
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_api_process_task_result_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the shape of a successful process_task response."""
    client = _fresh_client(monkeypatch)
    response = client.post(
        "/process_task",
        json={"values": [0.5, 1.0, 1.5]},
        headers={"X-API-Key": "secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["errors"] == []
    fractal = data["fractal_output"]
    assert fractal is not None
    assert "initial" in fractal
    assert "final" in fractal
    assert "iterations" in fractal
