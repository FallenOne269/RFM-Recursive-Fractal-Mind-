"""search_code must not follow a symlink to a file outside the repository.

Regression test for the Repository Boundary Gate: read_file already refuses a
symlink whose target is outside the repository root, but search_code walks the
tree and reads every file it finds, so it followed the same link.
"""

import json
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


async def test_search_code_does_not_follow_symlink_outside_repository(tmp_path):
    token = f"BOUNDARY_{uuid.uuid4().hex}"

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inside.txt").write_text(f"inside {token}\n")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "notes.txt"
    outside_file.write_text(f"outside {token}\n")

    link = repo / "notes_link.txt"
    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")

    try:
        server = create_server(Settings(repo_root=repo, python=sys.executable))
        async with create_connected_server_and_client_session(server) as s:
            # read_file already refuses the link: search_code must match it.
            refused = await call(s, "read_file", path="notes_link.txt")
            assert "outside the repository" in refused.get("error", "")

            found = await call(s, "search_code", query=token)

        # Control: search still finds the file that really is in the repository.
        assert {"path": "inside.txt", "line": 1, "text": f"inside {token}"} in found[
            "matches"
        ]
        # The outside file's content must not come back through the link.
        assert f"outside {token}" not in json.dumps(found)
        assert all(m["path"] != "notes_link.txt" for m in found["matches"])
    finally:
        link.unlink(missing_ok=True)
        outside_file.unlink(missing_ok=True)
