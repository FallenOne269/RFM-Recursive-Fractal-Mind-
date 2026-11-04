"""Regression tests for the OAuth token lifecycle: expiry, refresh, rotation and
revocation.

The provider tests drive ``RFMOAuthProvider`` directly with a fake clock. The HTTP
test runs the real token endpoint and bearer middleware from the MCP SDK against a
server whose access tokens live for one second.
"""

import base64
import hashlib
import secrets
import socket
import threading
import time
import types
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import uvicorn
from mcp.server.auth.provider import AuthorizationCode, TokenError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

import mcp_server.auth as auth_mod
from mcp_server.auth import OAuthSettings, RFMOAuthProvider
from mcp_server.server import DEFAULT_REPO_ROOT, Settings, create_server

pytestmark = pytest.mark.anyio

PASSWORD = "correct-horse-battery"
PUBLIC_URL = "https://rfm.example.com"
RESOURCE = f"{PUBLIC_URL}/mcp"
ACCESS_TTL = 60
REFRESH_TTL = 3600


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    # auth.py only calls time.time(); replace the module it sees, not the global.
    monkeypatch.setattr(auth_mod, "time", types.SimpleNamespace(time=fake.time))
    return fake


def _settings(**kwargs):
    return OAuthSettings(
        public_url=PUBLIC_URL,
        password=PASSWORD,
        access_token_ttl=ACCESS_TTL,
        refresh_token_ttl=REFRESH_TTL,
        **kwargs,
    )


def _client(client_id="client-1"):
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="s3cret",
        redirect_uris=[AnyUrl("https://app.example/callback")],
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="rfm",
    )


async def _login(provider, client, clock):
    """Issue a token pair through the authorization-code exchange."""
    code = AuthorizationCode(
        code=secrets.token_urlsafe(16),
        scopes=["rfm"],
        expires_at=clock.now + 300,
        client_id=client.client_id,
        code_challenge="x" * 43,
        redirect_uri=AnyUrl("https://app.example/callback"),
        redirect_uri_provided_explicitly=True,
        resource=RESOURCE,
    )
    provider.codes[code.code] = code
    return await provider.exchange_authorization_code(client, code)


async def _refresh(provider, client, refresh_token):
    loaded = await provider.load_refresh_token(client, refresh_token)
    assert loaded is not None, "refresh token should still be valid"
    return await provider.exchange_refresh_token(client, loaded, [])


async def test_expired_access_token_keeps_paired_refresh_token(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)

    clock.advance(ACCESS_TTL + 1)
    assert await provider.load_access_token(tokens.access_token) is None
    # The expired access token was presented first; its refresh token must survive.
    refreshed = await _refresh(provider, client, tokens.refresh_token)

    assert refreshed.access_token != tokens.access_token
    assert refreshed.refresh_token != tokens.refresh_token
    assert await provider.load_access_token(refreshed.access_token) is not None


async def test_expired_access_token_is_removed_from_the_store(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    clock.advance(ACCESS_TTL + 1)

    assert await provider.load_access_token(tokens.access_token) is None
    assert auth_mod._hash(tokens.access_token) not in provider.grants
    assert auth_mod._hash(tokens.refresh_token) in provider.grants


async def test_refresh_rotation_invalidates_the_old_pair(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    refreshed = await _refresh(provider, client, tokens.refresh_token)

    assert await provider.load_refresh_token(client, tokens.refresh_token) is None
    assert await provider.load_access_token(tokens.access_token) is None
    assert await provider.load_access_token(refreshed.access_token) is not None
    assert await provider.load_refresh_token(client, refreshed.refresh_token)


async def test_rotation_after_access_expiry_drops_old_refresh_token(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    clock.advance(ACCESS_TTL + 1)
    await provider.load_access_token(tokens.access_token)  # expires + removes it

    await _refresh(provider, client, tokens.refresh_token)
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None


async def test_refresh_token_cannot_be_exchanged_twice(clock):
    """Two concurrent refresh requests can both pass load_refresh_token; only the
    first exchange may succeed."""
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    first = await provider.load_refresh_token(client, tokens.refresh_token)
    second = await provider.load_refresh_token(client, tokens.refresh_token)

    await provider.exchange_refresh_token(client, first, [])
    with pytest.raises(TokenError) as err:
        await provider.exchange_refresh_token(client, second, [])
    assert err.value.error == "invalid_grant"


async def test_expired_refresh_token_is_rejected(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    loaded = await provider.load_refresh_token(client, tokens.refresh_token)

    clock.advance(REFRESH_TTL + 1)
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None
    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(client, loaded, [])


async def test_refresh_token_is_bound_to_its_client(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    other = _client("client-2")
    tokens = await _login(provider, client, clock)

    assert await provider.load_refresh_token(other, tokens.refresh_token) is None
    loaded = await provider.load_refresh_token(client, tokens.refresh_token)
    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(other, loaded, [])
    # The failed cross-client attempt must not burn the legitimate client's token.
    assert await provider.load_refresh_token(client, tokens.refresh_token)


async def test_revoking_access_token_revokes_its_refresh_token(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    access = await provider.load_access_token(tokens.access_token)

    await provider.revoke_token(access)
    assert await provider.load_access_token(tokens.access_token) is None
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None


async def test_revoking_refresh_token_revokes_its_access_token(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    refresh = await provider.load_refresh_token(client, tokens.refresh_token)

    await provider.revoke_token(refresh)
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None
    assert await provider.load_access_token(tokens.access_token) is None


async def test_revoking_refresh_token_after_access_expiry(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    tokens = await _login(provider, client, clock)
    clock.advance(ACCESS_TTL + 1)
    assert await provider.load_access_token(tokens.access_token) is None

    refresh = await provider.load_refresh_token(client, tokens.refresh_token)
    await provider.revoke_token(refresh)
    assert await provider.load_refresh_token(client, tokens.refresh_token) is None


async def test_lifecycle_survives_a_restart(clock, tmp_path):
    state = tmp_path / "oauth-state.json"
    provider, client = RFMOAuthProvider(_settings(state_file=state)), _client()
    provider.clients[client.client_id] = client
    tokens = await _login(provider, client, clock)
    clock.advance(ACCESS_TTL + 1)
    assert await provider.load_access_token(tokens.access_token) is None

    restarted = RFMOAuthProvider(_settings(state_file=state))
    assert await restarted.load_access_token(tokens.access_token) is None
    refreshed = await _refresh(restarted, client, tokens.refresh_token)

    again = RFMOAuthProvider(_settings(state_file=state))
    assert await again.load_refresh_token(client, tokens.refresh_token) is None
    assert await again.load_access_token(refreshed.access_token) is not None


async def test_expired_grants_are_pruned_when_new_tokens_are_issued(clock):
    provider, client = RFMOAuthProvider(_settings()), _client()
    stale = await _login(provider, client, clock)
    clock.advance(REFRESH_TTL + 1)
    await _login(provider, client, clock)

    assert auth_mod._hash(stale.access_token) not in provider.grants
    assert auth_mod._hash(stale.refresh_token) not in provider.grants
    assert len(provider.grants) == 2


# -- end to end through the MCP SDK token endpoint and bearer middleware ------

REDIRECT = "http://localhost:3030/callback"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def short_ttl_url():
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    settings = Settings(
        repo_root=DEFAULT_REPO_ROOT,
        port=port,
        oauth=OAuthSettings(public_url=url, password=PASSWORD, access_token_ttl=1),
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


INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}


async def _mcp_status(http, url, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json, text/event-stream",
    }
    return (await http.post(f"{url}/mcp", json=INIT, headers=headers)).status_code


async def test_http_refresh_after_access_token_expiry(short_ttl_url):
    url = short_ttl_url
    async with httpx.AsyncClient() as http:
        client = (
            await http.post(
                f"{url}/register",
                json={
                    "redirect_uris": [REDIRECT],
                    "client_name": "expiry test",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "token_endpoint_auth_method": "client_secret_post",
                },
            )
        ).json()
        verifier = secrets.token_urlsafe(48)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        auth = await http.get(
            f"{url}/authorize",
            params={
                "response_type": "code",
                "client_id": client["client_id"],
                "redirect_uri": REDIRECT,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "st",
                "resource": f"{url}/mcp",
            },
        )
        txn = parse_qs(urlparse(auth.headers["location"]).query)["txn"][0]
        login = await http.post(
            f"{url}/oauth/login", data={"txn": txn, "password": PASSWORD}
        )
        code = parse_qs(urlparse(login.headers["location"]).query)["code"][0]
        creds = {
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
        }
        tokens = (
            await http.post(
                f"{url}/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": REDIRECT,
                    "code_verifier": verifier,
                    "resource": f"{url}/mcp",
                    **creds,
                },
            )
        ).json()
        assert await _mcp_status(http, url, tokens["access_token"]) == 200

        time.sleep(2.2)  # let the one-second access token expire
        assert await _mcp_status(http, url, tokens["access_token"]) == 401

        refresh_req = {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            **creds,
        }
        resp = await http.post(f"{url}/token", data=refresh_req)
        assert resp.status_code == 200, resp.text
        refreshed = resp.json()
        assert await _mcp_status(http, url, refreshed["access_token"]) == 200

        replay = await http.post(f"{url}/token", data=refresh_req)
        assert replay.status_code == 400
