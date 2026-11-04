import json

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server import gemini

pytestmark = pytest.mark.anyio
FAKE_KEY = "test-key-not-real"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server():
    mcp = FastMCP("t")
    gemini.register_gemini_tools(mcp)
    return mcp


@pytest.fixture
def google(monkeypatch):
    calls = []
    responses = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status, body = responses.get(request.url.path, (404, {}))
        return httpx.Response(status, json=body)

    monkeypatch.setattr(gemini, "TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    return calls, responses


async def call(server, name, args):
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(name, args)
    text = result.content[0].text
    return result.isError, (text if result.isError else json.loads(text))


async def test_requires_api_key(server, google):
    calls, _ = google
    is_error, text = await call(server, "ask_gemini", {"prompt": "hi"})
    assert is_error and "GEMINI_API_KEY" in text
    assert calls == []


async def test_ask_gemini_sends_key_in_header_and_labels_source(
    server, google, monkeypatch
):
    calls, responses = google
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    responses["/v1beta/models/gemini-3.8-flash:generateContent"] = (
        200,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": "It depends on the timeline model."},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.8-flash-001",
            "usageMetadata": {"totalTokenCount": 42},
        },
    )
    is_error, data = await call(server, "ask_gemini", {"prompt": "  paradox?  "})
    assert not is_error
    assert data["answer"] == "It depends on the timeline model."
    assert data["source"] == "google-gemini" and data["rfm_engines_used"] is False
    assert data["model_version"] == "gemini-3.8-flash-001"
    assert data["prompt"] == "paradox?"
    request = calls[0]
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in str(request.url)
    sent = json.loads(request.content)
    assert sent["contents"][0]["parts"][0]["text"] == "paradox?"


async def test_key_file_and_model_override(server, google, monkeypatch, tmp_path):
    calls, responses = google
    key_file = tmp_path / "key"
    key_file.write_text(FAKE_KEY + "\n")
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    responses["/v1beta/models/gemini-2.5-flash:generateContent"] = (
        200,
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
    )
    is_error, data = await call(server, "ask_gemini", {"prompt": "hi"})
    assert not is_error and data["requested_model"] == "gemini-2.5-flash"
    assert calls[0].headers["x-goog-api-key"] == FAKE_KEY


async def test_rate_limit_and_errors_do_not_fall_back(server, google, monkeypatch):
    calls, responses = google
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    path = "/v1beta/models/gemini-3.8-flash:generateContent"
    responses[path] = (429, {"error": {"message": "quota"}})
    is_error, text = await call(server, "ask_gemini", {"prompt": "hi"})
    assert is_error and "429" in text and "No paid fallback" in text
    assert len(calls) == 1
    responses[path] = (403, {"error": {"message": "API key not valid"}})
    is_error, text = await call(server, "ask_gemini", {"prompt": "hi"})
    assert is_error and "403" in text and FAKE_KEY not in text


@pytest.mark.parametrize(
    "args",
    [
        {"prompt": ""},
        {"prompt": "x" * (gemini.MAX_PROMPT_CHARS + 1)},
        {"prompt": "hi", "model": "../../etc"},
        {"prompt": "hi", "max_output_tokens": 0},
    ],
)
async def test_rejects_bad_input(server, google, monkeypatch, args):
    calls, _ = google
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    is_error, _ = await call(server, "ask_gemini", args)
    assert is_error and calls == []


async def test_list_models_filters_generate_content(server, google, monkeypatch):
    _, responses = google
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    responses["/v1beta/models"] = (
        200,
        {
            "models": [
                {
                    "name": "models/gemini-3.8-flash",
                    "displayName": "Gemini 3.8 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/text-embedding",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        },
    )
    is_error, data = await call(server, "list_gemini_models", {})
    assert not is_error
    assert [m["id"] for m in data["models"]] == ["gemini-3.8-flash"]
    assert data["default_model"] == "gemini-3.8-flash"
