"""Self-contained OAuth 2.1 authorization server for the RFM MCP server.

Implements the MCP authorization spec so the server can be added as a remote/custom app
by OAuth-capable MCP clients (e.g. Sesame): RFC 8414 metadata, RFC 7591 dynamic client
registration, PKCE authorization-code + refresh-token grants, RFC 7009 revocation and
RFC 9728 protected-resource metadata (the last four are provided by the MCP SDK routes).

The resource owner is the server operator: the authorization page asks for the
``RFM_MCP_OAUTH_PASSWORD`` before an authorization code is issued.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

logger = logging.getLogger(__name__)

SCOPE = "rfm"
LOGIN_PATH = "/oauth/login"
AUTH_CODE_TTL = 300
PENDING_TTL = 600
MAX_LOGIN_ATTEMPTS = 5
SUBJECT = "rfm-operator"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class OAuthSettings:
    public_url: str
    password: str
    access_token_ttl: int = 3600
    refresh_token_ttl: int = 30 * 24 * 3600
    state_file: Optional[Path] = None
    static_client_id: Optional[str] = None
    static_client_secret: Optional[str] = None
    static_redirect_uris: list[str] = field(default_factory=list)

    @property
    def issuer_url(self) -> str:
        return self.public_url.rstrip("/")

    @property
    def resource_url(self) -> str:
        return f"{self.issuer_url}/mcp"

    @classmethod
    def from_env(cls, public_url: Optional[str]) -> "OAuthSettings":
        if not public_url:
            raise ValueError("RFM_MCP_PUBLIC_URL is required when OAuth is enabled")
        password = os.environ.get("RFM_MCP_OAUTH_PASSWORD", "")
        if len(password) < 12:
            raise ValueError(
                "RFM_MCP_OAUTH_PASSWORD must be set (at least 12 characters)"
            )
        state_file = os.environ.get("RFM_MCP_OAUTH_STATE_FILE")
        redirect_uris = os.environ.get("RFM_MCP_OAUTH_REDIRECT_URIS", "")
        return cls(
            public_url=public_url,
            password=password,
            access_token_ttl=int(os.environ.get("RFM_MCP_ACCESS_TOKEN_TTL", "3600")),
            refresh_token_ttl=int(
                os.environ.get("RFM_MCP_REFRESH_TOKEN_TTL", str(30 * 86400))
            ),
            state_file=Path(state_file) if state_file else None,
            static_client_id=os.environ.get("RFM_MCP_OAUTH_CLIENT_ID") or None,
            static_client_secret=os.environ.get("RFM_MCP_OAUTH_CLIENT_SECRET") or None,
            static_redirect_uris=[
                u.strip() for u in redirect_uris.split(",") if u.strip()
            ],
        )


@dataclass
class _Grant:
    """A stored access or refresh token, keyed by the token's SHA-256."""

    client_id: str
    scopes: list[str]
    expires_at: Optional[int]
    resource: Optional[str]
    kind: str  # "access" | "refresh"
    pair: str  # hash of the sibling token issued alongside this one


@dataclass
class _Pending:
    client_id: str
    params: AuthorizationParams
    created_at: float
    attempts: int = 0


class RFMOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """In-process OAuth authorization server with optional JSON persistence."""

    def __init__(self, settings: OAuthSettings) -> None:
        self.settings = settings
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.grants: dict[str, _Grant] = {}
        self.codes: dict[str, AuthorizationCode] = {}
        self.pending: dict[str, _Pending] = {}
        self._load_state()
        if settings.static_client_id:
            self._register_static_client()

    # -- persistence -------------------------------------------------------

    def _load_state(self) -> None:
        path = self.settings.state_file
        if not path or not path.is_file():
            return
        data = json.loads(path.read_text())
        self.clients = {
            cid: OAuthClientInformationFull.model_validate(info)
            for cid, info in data.get("clients", {}).items()
        }
        now = time.time()
        self.grants = {
            h: _Grant(**g)
            for h, g in data.get("grants", {}).items()
            if not g["expires_at"] or g["expires_at"] > now
        }

    def _save_state(self) -> None:
        path = self.settings.state_file
        if not path:
            return
        data = {
            "clients": {
                cid: c.model_dump(mode="json")
                for cid, c in self.clients.items()
                if cid != self.settings.static_client_id
            },
            "grants": {h: asdict(g) for h, g in self.grants.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def _register_static_client(self) -> None:
        s = self.settings
        if not s.static_redirect_uris:
            raise ValueError(
                "RFM_MCP_OAUTH_REDIRECT_URIS is required with RFM_MCP_OAUTH_CLIENT_ID"
            )
        client_id = s.static_client_id
        assert client_id is not None
        self.clients[client_id] = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=s.static_client_secret,
            client_id_issued_at=int(time.time()),
            redirect_uris=[AnyUrl(uri) for uri in s.static_redirect_uris],
            token_endpoint_auth_method=(
                "client_secret_post" if s.static_client_secret else "none"
            ),
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope=SCOPE,
            client_name="Pre-registered client",
        )

    # -- clients -----------------------------------------------------------

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        assert client_info.client_id
        self.clients[client_info.client_id] = client_info
        logger.info(
            "registered OAuth client %s (%s)",
            client_info.client_id,
            client_info.client_name,
        )
        self._save_state()

    # -- authorization -----------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.resource and not _same_url(
            params.resource, self.settings.resource_url
        ):
            raise AuthorizeError(
                error="invalid_request",
                error_description=f"unknown resource {params.resource!r}",
            )
        self._expire()
        txn = secrets.token_urlsafe(32)
        assert client.client_id
        self.pending[txn] = _Pending(client.client_id, params, time.time())
        return f"{self.settings.issuer_url}{LOGIN_PATH}?{urlencode({'txn': txn})}"

    def _expire(self) -> None:
        now = time.time()
        self.pending = {
            k: v for k, v in self.pending.items() if now - v.created_at < PENDING_TTL
        }
        self.codes = {k: v for k, v in self.codes.items() if v.expires_at > now}

    async def handle_login(self, request: Request) -> Response:
        """GET renders the consent form; POST checks the operator password."""
        form = await request.form() if request.method == "POST" else {}
        txn = str(form.get("txn") or request.query_params.get("txn") or "")
        self._expire()
        pending = self.pending.get(txn)
        if pending is None:
            return _page(
                "Authorization request expired",
                "<p>Start the connection again from your app.</p>",
                status=400,
            )
        client = self.clients.get(pending.client_id)
        client_name = (client.client_name if client else None) or pending.client_id

        if request.method == "GET":
            return _login_form(txn, client_name, pending.params)

        if form.get("action") == "deny":
            self.pending.pop(txn, None)
            return _redirect(
                construct_redirect_uri(
                    str(pending.params.redirect_uri),
                    error="access_denied",
                    state=pending.params.state,
                )
            )

        password = str(form.get("password") or "")
        if not hmac.compare_digest(password.encode(), self.settings.password.encode()):
            pending.attempts += 1
            await asyncio.sleep(0.5)
            if pending.attempts >= MAX_LOGIN_ATTEMPTS:
                self.pending.pop(txn, None)
                logger.warning(
                    "OAuth login locked out for client %s", pending.client_id
                )
                return _page(
                    "Too many attempts",
                    "<p>Start the connection again.</p>",
                    status=403,
                )
            return _login_form(
                txn,
                client_name,
                pending.params,
                error="Incorrect password.",
                status=401,
            )

        self.pending.pop(txn, None)
        code = secrets.token_urlsafe(32)
        params = pending.params
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [SCOPE],
            expires_at=time.time() + AUTH_CODE_TTL,
            client_id=pending.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource or self.settings.resource_url,
            subject=SUBJECT,
        )
        logger.info("issued authorization code to client %s", pending.client_id)
        return _redirect(
            construct_redirect_uri(
                str(params.redirect_uri), code=code, state=params.state
            )
        )

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        code = self.codes.get(authorization_code)
        if code and code.client_id == client.client_id:
            return code
        return None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        if self.codes.pop(authorization_code.code, None) is None:
            raise TokenError(
                error="invalid_grant", error_description="code already used"
            )
        return self._issue(
            authorization_code.client_id,
            authorization_code.scopes,
            authorization_code.resource,
        )

    # -- tokens ------------------------------------------------------------

    def _issue(
        self, client_id: str, scopes: list[str], resource: Optional[str]
    ) -> OAuthToken:
        now = int(time.time())
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        ah, rh = _hash(access), _hash(refresh)
        self.grants[ah] = _Grant(
            client_id,
            scopes,
            now + self.settings.access_token_ttl,
            resource,
            "access",
            rh,
        )
        self.grants[rh] = _Grant(
            client_id,
            scopes,
            now + self.settings.refresh_token_ttl,
            resource,
            "refresh",
            ah,
        )
        self._save_state()
        return OAuthToken(
            access_token=access,
            expires_in=self.settings.access_token_ttl,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    def _grant(self, token: str, kind: str) -> Optional[_Grant]:
        grant = self.grants.get(_hash(token))
        if grant is None or grant.kind != kind:
            return None
        if grant.expires_at and grant.expires_at < time.time():
            self._drop(_hash(token))
            return None
        return grant

    def _drop(self, token_hash: str) -> None:
        grant = self.grants.pop(token_hash, None)
        if grant:
            self.grants.pop(grant.pair, None)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        grant = self._grant(refresh_token, "refresh")
        if grant is None or grant.client_id != client.client_id:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=grant.client_id,
            scopes=grant.scopes,
            expires_at=grant.expires_at,
            resource=grant.resource,
            subject=SUBJECT,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        self._drop(
            _hash(refresh_token.token)
        )  # rotate: old refresh + access become invalid
        return self._issue(
            refresh_token.client_id,
            scopes or refresh_token.scopes,
            refresh_token.resource,
        )

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        grant = self._grant(token, "access")
        if grant is None:
            return None
        return AccessToken(
            token=token,
            client_id=grant.client_id,
            scopes=grant.scopes,
            expires_at=grant.expires_at,
            resource=grant.resource,
            subject=SUBJECT,
        )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._drop(_hash(token.token))
        self._save_state()


def _same_url(a: str, b: str) -> bool:
    return a.rstrip("/").lower() == b.rstrip("/").lower()


SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
    "form-action 'self' https: http:; frame-ancestors 'none'",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}


def _redirect(url: str) -> Response:
    return RedirectResponse(url, status_code=302, headers={"Cache-Control": "no-store"})


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:420px;margin:10vh auto;
padding:0 1rem}}
input[type=password]{{width:100%;padding:.6rem;margin:.5rem 0 1rem;
box-sizing:border-box}}
button{{padding:.6rem 1.2rem;margin-right:.5rem}}.err{{color:#b00020}}</style></head>
<body><h2>{html.escape(title)}</h2>{body}</body></html>"""
    return HTMLResponse(doc, status_code=status, headers=SECURITY_HEADERS)


def _login_form(
    txn: str,
    client_name: str,
    params: AuthorizationParams,
    error: str = "",
    status: int = 200,
) -> HTMLResponse:
    scopes = ", ".join(params.scopes or [SCOPE])
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    body = f"""<p><b>{html.escape(client_name)}</b> wants to access the RFM MCP server
(scopes: {html.escape(scopes)}).
It will be able to read the repository and run its tests.</p>
<p>Redirects to: <code>{html.escape(str(params.redirect_uri))}</code></p>{err}
<form method="post" action="{LOGIN_PATH}">
<input type="hidden" name="txn" value="{html.escape(txn)}">
<label>Server password<input type="password" name="password"
autocomplete="current-password"
autofocus></label>
<button type="submit" name="action" value="allow">Allow</button>
<button type="submit" name="action" value="deny" formnovalidate>Deny</button>
</form>"""
    return _page("Connect to RFM", body, status=status)
