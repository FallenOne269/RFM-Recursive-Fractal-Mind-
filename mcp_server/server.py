"""MCP server that lets an assistant query Recursive Fractal Mind and run its tests."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from mcp.server.auth.settings import (
    AuthSettings,
    ClientRegistrationOptions,
    RevocationOptions,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_server.auth import LOGIN_PATH, SCOPE, OAuthSettings, RFMOAuthProvider
from mcp_server.domain import register_domain_tools
from mcp_server.gemini import register_gemini_tools
from mcp_server.runs import PytestRunner, RunError

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".idea",
    ".vscode",
}
SECRET_FILE_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "id_rsa*")
# read_file never holds more than about MAX_READ_BYTES + READ_CHUNK_BYTES of a file
# in memory and never returns more than MAX_READ_LINES lines / MAX_READ_BYTES bytes.
MAX_READ_BYTES = 256 * 1024
MAX_READ_LINES = 2000
READ_CHUNK_BYTES = 64 * 1024
# total_lines is counted (streaming) only for files up to this size.
MAX_LINE_COUNT_BYTES = 16 * 1024 * 1024
MAX_SEARCH_FILE_BYTES = 1024 * 1024

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

INSTRUCTIONS = """\
Tools for the RFM / Recursive Fractal Mind repository: the RFAI reference stack
(`rfai/`: AMIFS, DFE, RSA, fractal graph, simulations), Resonant Fractal Cognition
(`rfc/`), and the component orchestrator (`src/`: fractal engine, swarm, quantum
processor, meta learner).

- Start with `project_overview` or `describe_architecture`.
- Use `list_files`, `read_file`, and `search_code` to inspect source and docs.
- Use `list_tests` to discover tests and `run_tests` to execute them. Long runs can be
  started with `wait=false` and polled via `get_test_run`; `list_test_runs` shows
  history.
- `run_rfai_scenario`, `run_rfc_episode` and `run_orchestrator_cycle` execute the
  deterministic simulations / engines in-process and return their metrics.
"""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class LineRangeError(ValueError):
    """The requested line range is invalid for the file."""


@dataclass
class LineWindow:
    lines: list[str]
    start_line: int
    end_line: int  # last line returned; start_line - 1 when nothing was returned
    total_lines: Optional[int]  # None when the file was too large to count
    truncated: bool  # the requested range was cut short by a limit
    line_truncated: bool  # the last returned line was cut at the byte limit


def read_line_window(
    path: Path,
    start_line: int = 1,
    end_line: Optional[int] = None,
    *,
    max_lines: int = MAX_READ_LINES,
    max_bytes: int = MAX_READ_BYTES,
    count_limit: int = MAX_LINE_COUNT_BYTES,
) -> LineWindow:
    """Stream lines ``start_line..end_line`` (1-based, inclusive) out of *path*.

    The file is read in bounded pieces, so memory stays near ``max_bytes`` no
    matter how large the file or how long a single line is. At most ``max_lines``
    lines and ``max_bytes`` bytes are returned; ``truncated`` says the range was
    cut short. Lines are split on ``\\n`` with a trailing ``\\r`` removed.
    """
    if start_line < 1:
        raise LineRangeError(f"start_line must be >= 1 (got {start_line})")
    if end_line is not None and end_line < start_line:
        raise LineRangeError(
            f"end_line ({end_line}) must be >= start_line ({start_line})"
        )
    last_wanted = start_line + max_lines - 1
    if end_line is not None:
        last_wanted = min(last_wanted, end_line)

    out: list[bytes] = []
    used = 0  # bytes already placed in `out`
    current = bytearray()  # the in-range line being read
    lineno = 0  # number of the line currently being read
    at_line_start = True
    collecting = True
    truncated = line_truncated = False
    reached_eof = False

    def finish_line() -> None:
        nonlocal used, current
        out.append(bytes(current))
        used += len(current)
        current = bytearray()

    file_size = path.stat().st_size
    with path.open("rb") as fh:
        while True:
            if not collecting and file_size > count_limit:
                break  # do not scan the rest of a huge file only to count lines
            piece = fh.readline(READ_CHUNK_BYTES)
            if not piece:
                reached_eof = True
                break
            if at_line_start:
                lineno += 1
                if collecting and lineno > last_wanted:
                    collecting = False
                    # More lines exist beyond the ones returned.
                    truncated = end_line is None or end_line > last_wanted
            ends_line = piece.endswith(b"\n")
            if collecting and lineno >= start_line:
                room = max_bytes - used - len(current)
                if len(piece) > room:
                    if out:
                        current = bytearray()  # return whole lines only
                    else:
                        current += piece[:room]  # one line longer than the cap
                        finish_line()
                        line_truncated = True
                    collecting = False
                    truncated = True
                else:
                    current += piece
                    if ends_line:
                        finish_line()
            at_line_start = ends_line

    if collecting and current:
        finish_line()  # last line of a file without a trailing newline

    if reached_eof and start_line > max(lineno, 1):
        raise LineRangeError(
            f"start_line {start_line} is past the end of the file ({lineno} lines)"
        )

    lines = [line.rstrip(b"\n").removesuffix(b"\r") for line in out]
    return LineWindow(
        lines=[line.decode(errors="replace") for line in lines],
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        total_lines=lineno if reached_eof else None,
        truncated=truncated,
        line_truncated=line_truncated,
    )


@dataclass
class Settings:
    repo_root: Path = DEFAULT_REPO_ROOT
    test_timeout: int = 600
    python: str = sys.executable
    host: str = "127.0.0.1"
    port: int = 8000
    oauth: Optional[OAuthSettings] = None

    @classmethod
    def from_env(
        cls, host: Optional[str] = None, port: Optional[int] = None
    ) -> "Settings":
        public_url = os.environ.get("RFM_MCP_PUBLIC_URL")
        oauth = (
            OAuthSettings.from_env(public_url) if _env_flag("RFM_MCP_OAUTH") else None
        )
        return cls(
            repo_root=Path(
                os.environ.get("RFM_REPO_ROOT", DEFAULT_REPO_ROOT)
            ).resolve(),
            test_timeout=int(os.environ.get("RFM_MCP_TEST_TIMEOUT", "600")),
            python=os.environ.get("RFM_MCP_PYTHON", sys.executable),
            host=host or os.environ.get("RFM_MCP_HOST") or "127.0.0.1",
            port=port or int(os.environ.get("RFM_MCP_PORT", "8000")),
            oauth=oauth,
        )


class RepoBrowser:
    """Read-only, path-sandboxed access to the repository."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError(f"path {rel_path!r} is outside the repository")
        rel_parts = candidate.relative_to(self.root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            raise ToolError(f"path {rel_path!r} is in an excluded directory")
        if candidate.is_file() and self._is_secret(candidate.name):
            raise ToolError(f"refusing to expose potential secret file {rel_path!r}")
        return candidate

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _is_secret(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in SECRET_FILE_PATTERNS)

    def walk(self, start: Path):
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
            for name in sorted(filenames):
                if not self._is_secret(name):
                    yield Path(dirpath) / name

    def git(self, *args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None


def create_server(settings: Optional[Settings] = None) -> FastMCP:
    settings = settings or Settings.from_env()
    repo = RepoBrowser(settings.repo_root)
    runner = PytestRunner(
        settings.repo_root,
        python=settings.python,
        default_timeout=settings.test_timeout,
    )
    mcp = FastMCP(
        "rfm",
        instructions=INSTRUCTIONS,
        host=settings.host,
        port=settings.port,
        **_auth_kwargs(settings),
    )

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok", "oauth": settings.oauth is not None})

    if settings.oauth is not None:
        provider = mcp._auth_server_provider
        assert isinstance(provider, RFMOAuthProvider)
        mcp.custom_route(LOGIN_PATH, methods=["GET", "POST"])(provider.handle_login)

    # ------------------------------------------------------------------
    # Project query tools
    # ------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    def project_overview() -> dict[str, Any]:
        """Summarise the RFM repository: README, layout, docs, tests and git state."""
        files = [repo.rel(p) for p in repo.walk(repo.root)]
        readme = repo.root / "README.md"
        log = repo.git("log", "-5", "--pretty=format:%h %ad %an: %s", "--date=short")
        return {
            "name": repo.root.name,
            "readme": readme.read_text(errors="replace") if readme.is_file() else None,
            "top_level": sorted(
                p.name + ("/" if p.is_dir() else "")
                for p in repo.root.iterdir()
                if p.name not in EXCLUDED_DIRS and not repo._is_secret(p.name)
            ),
            "file_count": len(files),
            "python_modules": [f for f in files if f.endswith(".py")],
            "docs": [f for f in files if f.startswith("docs/")],
            "test_files": [
                f for f in files if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", f)
            ],
            "git": {
                "branch": repo.git("rev-parse", "--abbrev-ref", "HEAD"),
                "head": repo.git("rev-parse", "--short", "HEAD"),
                "recent_commits": log.splitlines() if log else [],
            },
        }

    @mcp.tool(annotations=READ_ONLY)
    def list_files(
        path: str = ".", pattern: str = "*", max_results: int = 500
    ) -> dict[str, Any]:
        """List repository files under `path` whose relative path matches the glob
        `pattern` (e.g. "*.py", "docs/*"). Ignores .git, virtualenvs, caches and secret
        files."""
        start = repo.resolve(path)
        if not start.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        matches = [
            repo.rel(p)
            for p in repo.walk(start)
            if fnmatch.fnmatch(repo.rel(p), pattern)
        ]
        return {
            "files": matches[:max_results],
            "total": len(matches),
            "truncated": len(matches) > max_results,
        }

    @mcp.tool(annotations=READ_ONLY)
    def read_file(
        path: str, start_line: int = 1, end_line: Optional[int] = None
    ) -> dict[str, Any]:
        """Read a text file from the repository, optionally limited to a 1-based
        inclusive line range. Returns at most 2000 lines / 256 KiB per call; when
        `truncated` is true, call again with `start_line=next_start_line`."""
        target = repo.resolve(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file")
        with target.open("rb") as fh:
            if b"\0" in fh.read(8192):
                raise ToolError(f"{path!r} looks like a binary file")
        try:
            window = read_line_window(target, start_line, end_line)
        except LineRangeError as exc:
            raise ToolError(f"{path!r}: {exc}") from None
        return {
            "path": repo.rel(target),
            "start_line": window.start_line,
            "end_line": window.end_line,
            "total_lines": window.total_lines,
            "content": "\n".join(window.lines),
            "truncated": window.truncated,
            "line_truncated": window.line_truncated,
            "next_start_line": window.end_line + 1 if window.truncated else None,
        }

    @mcp.tool(annotations=READ_ONLY)
    def search_code(
        query: str,
        path: str = ".",
        file_pattern: str = "*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """Search repository text files for `query` (plain text or a Python regex)
        and return matching lines as {path, line, text}."""
        start = repo.resolve(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query if regex else re.escape(query), flags)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        files = [start] if start.is_file() else repo.walk(start)
        results: list[dict[str, Any]] = []
        truncated = False
        for file in files:
            rel = repo.rel(file)
            if not fnmatch.fnmatch(rel, file_pattern):
                continue
            if file.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            raw = file.read_bytes()
            if b"\0" in raw[:8192]:
                continue
            for lineno, line in enumerate(raw.decode(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    if len(results) >= max_results:
                        truncated = True
                        break
                    results.append(
                        {"path": rel, "line": lineno, "text": line.strip()[:500]}
                    )
            if truncated:
                break
        return {"matches": results, "count": len(results), "truncated": truncated}

    @mcp.tool(annotations=READ_ONLY)
    async def list_tests(target: Optional[str] = None) -> dict[str, Any]:
        """List pytest node ids, optionally limited to a test file/directory/node id."""
        try:
            tests = await runner.collect(target)
        except RunError as exc:
            raise ToolError(str(exc)) from exc
        return {"tests": tests, "count": len(tests)}

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        )
    )
    async def run_tests(
        target: Optional[str] = None,
        keyword: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        wait: bool = True,
        include_output: bool = False,
    ) -> dict[str, Any]:
        """Trigger a pytest run.

        Args:
            target: Optional test path or node id
                (e.g. "tests/test_rfc_engine.py::test_x").
            keyword: Optional pytest `-k` expression.
            timeout_seconds: Kill the run after this many seconds (server default
                if unset).
            wait: If false, return immediately with a run_id to poll via `get_test_run`.
            include_output: Include the tail of pytest's console output.
        """
        try:
            if wait:
                record = await runner.run(target, keyword, timeout_seconds)
            else:
                record = runner.start(target, keyword, timeout_seconds)
        except RunError as exc:
            raise ToolError(str(exc)) from exc
        return record.to_dict(include_output=include_output)

    @mcp.tool(annotations=READ_ONLY)
    async def get_test_run(
        run_id: str, wait: bool = False, include_output: bool = True
    ) -> dict[str, Any]:
        """Get the status and results of a test run. Set `wait` to block until it
        finishes."""
        try:
            record = await runner.wait(run_id) if wait else runner.get(run_id)
        except RunError as exc:
            raise ToolError(str(exc)) from exc
        return record.to_dict(include_output=include_output)

    @mcp.tool(annotations=READ_ONLY)
    def list_test_runs(limit: int = 20) -> dict[str, Any]:
        """List recent test runs (newest first) from this server session."""
        return {"runs": [r.to_dict() for r in runner.list(limit)]}

    register_domain_tools(mcp, settings.repo_root)
    register_gemini_tools(mcp)

    return mcp


def _auth_kwargs(settings: Settings) -> dict[str, Any]:
    """FastMCP constructor arguments that enable OAuth and allow the public hostname."""
    oauth = settings.oauth
    if oauth is None:
        return {}
    public = urlparse(oauth.issuer_url)
    local_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    return {
        "auth_server_provider": RFMOAuthProvider(oauth),
        "auth": AuthSettings(
            issuer_url=AnyHttpUrl(oauth.issuer_url),
            resource_server_url=AnyHttpUrl(oauth.resource_url),
            required_scopes=[SCOPE],
            validate_token_resource=True,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=[SCOPE],
                default_scopes=[SCOPE],
            ),
            revocation_options=RevocationOptions(enabled=True),
        ),
        "transport_security": TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[public.netloc, *local_hosts],
            allowed_origins=[
                f"{public.scheme}://{public.netloc}",
                "http://127.0.0.1:*",
                "http://localhost:*",
            ],
        ),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mcp_server", description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.environ.get("RFM_MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=None, help="bind port (default 8000)"
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env(host=args.host, port=args.port)
    except ValueError as exc:
        parser.error(str(exc))
    if settings.oauth is not None and args.transport == "stdio":
        parser.error(
            "OAuth requires an HTTP transport (use --transport streamable-http)"
        )
    if (
        args.transport != "stdio"
        and settings.oauth is None
        and settings.host not in LOCAL_HOSTS
    ):
        print(
            f"WARNING: serving without authentication on {settings.host}; "
            "set RFM_MCP_OAUTH=1",
            file=sys.stderr,
        )
    if settings.oauth is not None:
        print(
            "OAuth enabled; add this URL as the custom app: "
            f"{settings.oauth.resource_url}",
            file=sys.stderr,
        )
    create_server(settings).run(transport=args.transport)


if __name__ == "__main__":
    main()
