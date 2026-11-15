"""Repository Boundary Gate v1.1: hard links cannot carry a file past the gate.

A hard link is a second name for the same file, so resolving its path never
leaves the repository: a link inside the repository to a file outside it, or a
link named ``notes.txt`` to the repository's ``.env``, passed every v1 check.
"""

import json
import os
import sys
import uuid

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

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


def hardlink(target, path):
    try:
        os.link(target, path)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are not supported here")


@pytest.fixture
def token():
    return f"BOUNDARY_{uuid.uuid4().hex}"


@pytest.fixture
def repo(tmp_path, token):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "inside.txt").write_text(f"inside {token}\n")
    return root


def server_for(repo):
    return create_server(Settings(repo_root=repo, python=sys.executable))


async def test_hardlink_to_file_outside_repository_is_refused(tmp_path, repo, token):
    outside = tmp_path / "outside.txt"
    outside.write_text(f"outside {token}\n")
    hardlink(outside, repo / "notes.txt")
    async with create_connected_server_and_client_session(server_for(repo)) as s:
        read = await call(s, "read_file", path="notes.txt")
        found = await call(s, "search_code", query=token)
        listed = await call(s, "list_files")
    assert "hard-linked" in read.get("error", ""), read
    assert f"outside {token}" not in json.dumps(found)
    assert [m["path"] for m in found["matches"]] == ["inside.txt"]
    assert listed["files"] == ["inside.txt"]


async def test_hardlink_to_secret_file_is_refused(repo, token):
    (repo / ".env").write_text(f"SECRET={token}\n")
    hardlink(repo / ".env", repo / "notes.txt")
    async with create_connected_server_and_client_session(server_for(repo)) as s:
        read = await call(s, "read_file", path="notes.txt")
        found = await call(s, "search_code", query="SECRET")
    assert "error" in read, read
    assert f"SECRET={token}" not in json.dumps(read)
    assert f"SECRET={token}" not in json.dumps(found)


async def test_ordinary_file_still_reads(repo, token):
    """Control: a file with a single link is unaffected."""
    async with create_connected_server_and_client_session(server_for(repo)) as s:
        read = await call(s, "read_file", path="inside.txt")
    assert token in read["content"]
