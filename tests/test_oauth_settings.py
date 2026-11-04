"""OAuthSettings must refuse plain-HTTP public URLs except on loopback."""

import pytest
from mcp.server.auth.routes import validate_issuer_url
from pydantic import AnyHttpUrl

from mcp_server.auth import OAuthSettings
from mcp_server.server import Settings, main

PASSWORD = "correct-horse-battery"

ACCEPTED = [
    "https://rfm.example.com",
    "https://rfm.example.com/",
    "https://rfm.example.com:8443/base",
    "HTTPS://RFM.EXAMPLE.COM",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://LOCALHOST:8000",
]

REJECTED = [
    ("http://rfm.example.com", "https"),
    ("http://0.0.0.0:8000", "https"),
    ("http://192.168.1.10:8000", "https"),
    ("http://10.0.0.5", "https"),
    ("http://127.0.0.2:8000", "https"),
    ("http://[::1]:8000", "https"),
    ("http://localhost.evil.example", "https"),
    ("http://127.0.0.1.nip.io", "https"),
    ("http://evil.example#@localhost", "fragment"),
    ("ftp://rfm.example.com", "https"),
    ("rfm.example.com", "https"),
    ("https://", "host"),
    ("https://user:pw@rfm.example.com", "credentials"),
    ("https://rfm.example.com/?next=x", "query"),
    ("https://rfm.example.com/#top", "fragment"),
    (" https://rfm.example.com", "whitespace"),
]


@pytest.fixture
def good_password(monkeypatch):
    monkeypatch.setenv("RFM_MCP_OAUTH_PASSWORD", PASSWORD)


@pytest.mark.parametrize("url", ACCEPTED)
def test_from_env_accepts_https_and_loopback_http(good_password, url):
    settings = OAuthSettings.from_env(url)
    assert settings.public_url == url


@pytest.mark.parametrize("url,reason", REJECTED)
def test_from_env_rejects_insecure_or_malformed_urls(good_password, url, reason):
    with pytest.raises(ValueError, match="RFM_MCP_PUBLIC_URL") as err:
        OAuthSettings.from_env(url)
    assert reason in str(err.value)


@pytest.mark.parametrize("url", ACCEPTED)
def test_accepted_urls_also_pass_the_sdk_issuer_check(good_password, url):
    """Anything from_env accepts must not be refused later by the MCP SDK."""
    validate_issuer_url(AnyHttpUrl(OAuthSettings.from_env(url).issuer_url))


def test_direct_construction_is_validated_too():
    with pytest.raises(ValueError, match="https"):
        OAuthSettings(public_url="http://rfm.example.com", password=PASSWORD)


def test_settings_from_env_rejects_public_http(monkeypatch, good_password):
    monkeypatch.setenv("RFM_MCP_OAUTH", "1")
    monkeypatch.setenv("RFM_MCP_PUBLIC_URL", "http://rfm.example.com")
    with pytest.raises(ValueError, match="https"):
        Settings.from_env()


def test_settings_from_env_accepts_public_https(monkeypatch, good_password):
    monkeypatch.setenv("RFM_MCP_OAUTH", "1")
    monkeypatch.setenv("RFM_MCP_PUBLIC_URL", "https://rfm.example.com")
    settings = Settings.from_env()
    assert settings.oauth is not None
    assert settings.oauth.resource_url == "https://rfm.example.com/mcp"


def test_cli_refuses_to_start_with_public_http(monkeypatch, good_password, capsys):
    monkeypatch.setenv("RFM_MCP_OAUTH", "1")
    monkeypatch.setenv("RFM_MCP_PUBLIC_URL", "http://rfm.example.com")
    with pytest.raises(SystemExit) as exc:
        main(["--transport", "streamable-http"])
    assert exc.value.code == 2
    assert "https" in capsys.readouterr().err


def test_password_and_url_still_required(monkeypatch):
    monkeypatch.setenv("RFM_MCP_OAUTH_PASSWORD", "short")
    with pytest.raises(ValueError, match="RFM_MCP_OAUTH_PASSWORD"):
        OAuthSettings.from_env("https://rfm.example.com")
    with pytest.raises(ValueError, match="RFM_MCP_PUBLIC_URL"):
        OAuthSettings.from_env(None)
