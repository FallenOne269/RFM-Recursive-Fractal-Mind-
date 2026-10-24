"""ethical_dna_api.py imports, serves its endpoint, and fails closed on auth."""

import importlib

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token-0123456789"

REQUEST = {
    "action_proposal": {
        "action_id": "a1",
        "action_type": "update",
        "description": "Rotate the cache",
        "target_system": "cache",
        "parameters": {"ttl": 60},
        "urgency_level": 2,
        "reversibility": True,
    },
    "context_data": {},
    "stakeholder_impact": [
        {
            "stakeholder_id": "s1",
            "stakeholder_type": "human_individual",
            "impact_description": "brief slowdown",
            "impact_magnitude": -0.2,
            "confidence": 0.9,
            "mitigation_possible": True,
        }
    ],
    "value_system": {
        "core_values": ["safety", "autonomy"],
        "value_weights": {"safety": 1.0},
        "cultural_context": "test",
        "adaptation_allowed": False,
    },
}


@pytest.fixture
def client():
    module = importlib.import_module("ethical_dna_api")
    return TestClient(module.app)


def _post(client, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/api/v1/ethical/dna", json=REQUEST, headers=headers)


def test_refuses_every_request_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.delenv("ETHICAL_DNA_API_TOKEN", raising=False)
    assert _post(client, "anything-at-all-long-enough").status_code == 503


def test_requires_a_bearer_token(client, monkeypatch):
    monkeypatch.setenv("ETHICAL_DNA_API_TOKEN", TOKEN)
    assert _post(client).status_code in (401, 403)
    assert _post(client, "wrong-token-0123456789").status_code == 401


def test_scores_all_five_principles(client, monkeypatch):
    monkeypatch.setenv("ETHICAL_DNA_API_TOKEN", TOKEN)
    resp = _post(client, TOKEN)
    assert resp.status_code == 200, resp.text
    scores = resp.json()["ethical_assessment"]
    assert set(scores) == {
        "harm_prevention",
        "autonomy_respect",
        "fairness",
        "transparency",
        "accountability",
    }
    assert scores["fairness"] == 1.0  # one stakeholder: no spread
    assert scores["transparency"] == pytest.approx(0.9)
    assert scores["accountability"] == pytest.approx(0.8)
