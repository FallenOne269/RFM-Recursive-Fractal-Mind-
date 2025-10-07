"""Integration tests for the FastAPI application."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from api import create_app
from rfai_system import RecursiveFractalMind



def test_api_endpoints_with_api_key(monkeypatch, config_path) -> None:
    monkeypatch.setenv("RFAI_API_KEY", "test-key")

    def orchestrator_factory() -> RecursiveFractalMind:
        return RecursiveFractalMind(str(config_path))

    app = create_app(orchestrator_factory)
    client = TestClient(app)

    headers = {"X-API-Key": "test-key"}
    assert client.get("/health", headers=headers).status_code == 200
    status_response = client.get("/status", headers=headers)
    assert status_response.status_code == 200

    response = client.post(
        "/process_task",
        json={
            "id": "api-001",
            "type": "analysis",
            "complexity": 0.4,
            "payload": [0.1, 0.2, 0.3],
            "metadata": {"origin": "test"},
        },
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fractal_output"] is not None
    assert body["meta_output"] is not None


def test_api_rejects_invalid_key(monkeypatch, config_path) -> None:
    monkeypatch.setenv("RFAI_API_KEY", "expected")

    def orchestrator_factory() -> RecursiveFractalMind:
        return RecursiveFractalMind(str(config_path))

    app = create_app(orchestrator_factory)
    client = TestClient(app)
    response = client.get("/health", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
