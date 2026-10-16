import base64
import hashlib
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from mcp_server.auth import OAuthSettings
from mcp_server.server import DEFAULT_REPO_ROOT, Settings, create_server

PASSWORD = "correct-horse-battery"
pytestmark = pytest.mark.anyio
REDIRECT = "http://localhost:3030/callback"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    settings = Settings(
        repo_root=DEFAULT_REPO_ROOT,
        port=port,
        oauth=OAuthSettings(public_url=url, password=PASSWORD),
    )
    app = create_server(settings).streamable_http_app()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    yield url
    server.should_exit = True
    thread.join(5)


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.decode().rstrip("=")


async def _register(http, base_url):
    resp = await http.post(
        f"{base_url}/register",
        json={
            "redirect_uris": [REDIRECT],
            "client_name": "Sesame test",
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _authorize(http, base_url, client, challenge, state="xyz"):
    resp = await http.get(
        f"{base_url}/authorize",
        params={
            "response_type": "code",
            "client_id": client["client_id"],
            "redirect_uri": REDIRECT,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "resource": f"{base_url}/mcp",
        },
    )
    assert resp.status_code == 302, resp.text
    login_url = resp.headers["location"]
    assert login_url.startswith(f"{base_url}/oauth/login?txn=")
    return parse_qs(urlparse(login_url).query)["txn"][0]


async def test_metadata_and_unauthenticated_challenge(base_url):
    async with httpx.AsyncClient() as http:
        prm = (
            await http.get(f"{base_url}/.well-known/oauth-protected-resource/mcp")
        ).json()
        assert prm["resource"] == f"{base_url}/mcp"
        assert prm["authorization_servers"] == [f"{base_url}/"]
        assert prm["scopes_supported"] == ["rfm"]

        asm = (
            await http.get(f"{base_url}/.well-known/oauth-authorization-server")
        ).json()
        assert asm["registration_endpoint"] == f"{base_url}/register"
        assert asm["revocation_endpoint"] == f"{base_url}/revoke"
        assert asm["code_challenge_methods_supported"] == ["S256"]

        resp = await http.post(f"{base_url}/mcp", json={})
        assert resp.status_code == 401
        assert "resource_metadata=" in resp.headers["www-authenticate"]

        assert (await http.get(f"{base_url}/healthz")).json() == {
            "status": "ok",
            "oauth": True,
        }


async def _mcp(http, base_url, body, headers):
    return (await http.post(f"{base_url}/mcp", json=body, headers=headers)).status_code


async def test_full_flow_refresh_and_revoke(base_url):
    async with httpx.AsyncClient() as http:
        client = await _register(http, base_url)
        verifier, challenge = _pkce()
        txn = await _authorize(http, base_url, client, challenge)

        page = await http.get(f"{base_url}/oauth/login", params={"txn": txn})
        assert page.status_code == 200
        assert "Sesame test" in page.text
        assert page.headers["x-frame-options"] == "DENY"

        bad = await http.post(
            f"{base_url}/oauth/login", data={"txn": txn, "password": "nope"}
        )
        assert bad.status_code == 401

        ok = await http.post(
            f"{base_url}/oauth/login", data={"txn": txn, "password": PASSWORD}
        )
        assert ok.status_code == 302
        query = parse_qs(urlparse(ok.headers["location"]).query)
        assert query["state"] == ["xyz"]

        token_req = {
            "grant_type": "authorization_code",
            "code": query["code"][0],
            "redirect_uri": REDIRECT,
            "code_verifier": verifier,
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "resource": f"{base_url}/mcp",
        }
        tokens = (await http.post(f"{base_url}/token", data=token_req)).json()
        assert tokens["token_type"] == "Bearer" and tokens["scope"] == "rfm"

        replay = await http.post(f"{base_url}/token", data=token_req)
        assert replay.status_code == 400

        refreshed = (
            await http.post(
                f"{base_url}/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id": client["client_id"],
                    "client_secret": client["client_secret"],
                },
            )
        ).json()
        assert refreshed["access_token"] != tokens["access_token"]

        reused = await http.post(
            f"{base_url}/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": client["client_id"],
                "client_secret": client["client_secret"],
            },
        )
        assert reused.status_code == 400

        mcp_headers = {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Accept": "application/json, text/event-stream",
        }
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        }
        assert (await _mcp(http, base_url, init, mcp_headers)) == 401

        mcp_headers["Authorization"] = f"Bearer {refreshed['access_token']}"
        assert (await _mcp(http, base_url, init, mcp_headers)) == 200

        revoke = await http.post(
            f"{base_url}/revoke",
            data={
                "token": refreshed["access_token"],
                "client_id": client["client_id"],
                "client_secret": client["client_secret"],
            },
        )
        assert revoke.status_code == 200
        assert (await _mcp(http, base_url, init, mcp_headers)) == 401


async def test_deny_and_lockout(base_url):
    async with httpx.AsyncClient() as http:
        client = await _register(http, base_url)
        _, challenge = _pkce()

        txn = await _authorize(http, base_url, client, challenge, state="s1")
        denied = await http.post(
            f"{base_url}/oauth/login", data={"txn": txn, "action": "deny"}
        )
        assert denied.status_code == 302
        assert parse_qs(urlparse(denied.headers["location"]).query)["error"] == [
            "access_denied"
        ]

        txn = await _authorize(http, base_url, client, challenge)
        statuses = [
            (
                await http.post(
                    f"{base_url}/oauth/login", data={"txn": txn, "password": "x"}
                )
            ).status_code
            for _ in range(5)
        ]
        assert statuses == [401, 401, 401, 401, 403]
        gone = await http.post(
            f"{base_url}/oauth/login", data={"txn": txn, "password": PASSWORD}
        )
        assert gone.status_code == 400


async def test_wrong_resource_is_rejected(base_url):
    async with httpx.AsyncClient() as http:
        client = await _register(http, base_url)
        _, challenge = _pkce()
        resp = await http.get(
            f"{base_url}/authorize",
            params={
                "response_type": "code",
                "client_id": client["client_id"],
                "redirect_uri": REDIRECT,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": "https://evil.example/mcp",
            },
        )
        assert resp.status_code == 302
        assert "error=invalid_request" in resp.headers["location"]


class MemoryStorage:
    def __init__(self):
        self.tokens = self.client = None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client

    async def set_client_info(self, client_info: OAuthClientInformationFull):
        self.client = client_info


async def test_sdk_oauth_client_end_to_end(base_url):
    """A spec-compliant MCP client discovers, registers, authorizes and calls tools."""
    result = {}

    async def redirect_handler(authorization_url: str) -> None:
        async with httpx.AsyncClient() as http:
            login = (await http.get(authorization_url)).headers["location"]
            txn = parse_qs(urlparse(login).query)["txn"][0]
            done = await http.post(
                f"{base_url}/oauth/login", data={"txn": txn, "password": PASSWORD}
            )
            query = parse_qs(urlparse(done.headers["location"]).query)
            result["code"], result["state"] = query["code"][0], query["state"][0]

    async def callback_handler():
        return result["code"], result["state"]

    auth = OAuthClientProvider(
        server_url=f"{base_url}/mcp",
        client_metadata=OAuthClientMetadata(
            client_name="SDK e2e",
            redirect_uris=[REDIRECT],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=MemoryStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    async with (
        httpx.AsyncClient(auth=auth, timeout=30) as http,
        streamable_http_client(f"{base_url}/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {t.name for t in (await session.list_tools()).tools}
        assert {"project_overview", "run_tests"} <= tools
        res = await session.call_tool("describe_architecture", {})
        assert not res.isError
        assert "Resonant" in res.content[0].text or "rfc" in res.content[0].text


def test_oauth_settings_require_password(monkeypatch):
    monkeypatch.setenv("RFM_MCP_OAUTH_PASSWORD", "short")
    with pytest.raises(ValueError, match="RFM_MCP_OAUTH_PASSWORD"):
        OAuthSettings.from_env("https://rfm.example.com")
    with pytest.raises(ValueError, match="RFM_MCP_PUBLIC_URL"):
        OAuthSettings.from_env(None)
