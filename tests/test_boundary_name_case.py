"""Repository Boundary Gate v1.1: secret and excluded names match in any case.

On a case-sensitive filesystem ``.ENV`` is a different file from ``.env``, and
v1 matched names case-sensitively, so a real ``.ENV``, ``KEY.PEM`` or
``.GIT/config`` was readable and searchable. The policy now ignores case on
every platform instead of relying on the filesystem.
"""

import json
import sys
import uuid

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from mcp_server.boundary import is_excluded_name, is_secret_name  # noqa: E402
from mcp_server.server import Settings, create_server  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def call(session, name, **args):
    result = await session.call_tool(name, args)
    text = result.content[0].text if result.content else ""
    if result.isError:
        return {"error": text}
    return json.loads(text)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".ENV",
        ".Env.local",
        ".env.production",
        "id_rsa",
        "ID_RSA_backup",
        "private.pem",
        "KEY.PEM",
        "private.Key",
    ],
)
def test_secret_names_match_in_any_case(name):
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ["notes.txt", "environment.md", "keys.py"])
def test_ordinary_names_are_not_secret(name):
    assert not is_secret_name(name)


@pytest.mark.parametrize("name", [".git", ".GIT", ".Venv", "NODE_MODULES"])
def test_excluded_names_match_in_any_case(name):
    assert is_excluded_name(name)


async def test_upper_case_secret_and_excluded_files_are_not_exposed(tmp_path):
    token = f"BOUNDARY_{uuid.uuid4().hex}"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.txt").write_text(f"inside {token}\n")
    (repo / ".ENV").write_text(f"SECRET {token}\n")
    (repo / "KEY.PEM").write_text(f"PEM {token}\n")
    (repo / ".GIT").mkdir()
    (repo / ".GIT" / "config").write_text(f"GIT {token}\n")
    srv = create_server(Settings(repo_root=repo, python=sys.executable))
    async with create_connected_server_and_client_session(srv) as s:
        reads = [
            await call(s, "read_file", path=p)
            for p in (".ENV", "KEY.PEM", ".GIT/config")
        ]
        found = await call(s, "search_code", query=token)
        listed = await call(s, "list_files")
        overview = await call(s, "project_overview")
    for read in reads:
        assert "error" in read, read
    assert [m["path"] for m in found["matches"]] == ["inside.txt"]
    assert listed["files"] == ["inside.txt"]
    assert overview["top_level"] == ["inside.txt"]
    for leaked in (f"SECRET {token}", f"PEM {token}", f"GIT {token}"):
        assert leaked not in json.dumps([reads, found, listed, overview])
