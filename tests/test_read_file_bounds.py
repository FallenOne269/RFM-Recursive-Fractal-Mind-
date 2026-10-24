"""read_file must stream bounded line windows instead of loading whole files."""

import json
import tracemalloc

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server.server import (
    MAX_READ_BYTES,
    MAX_READ_LINES,
    LineRangeError,
    Settings,
    create_server,
    read_line_window,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def big_file(tmp_path):
    """About 7 MB: 200,000 numbered lines."""
    path = tmp_path / "big.txt"
    with path.open("w") as fh:
        for i in range(1, 200_001):
            fh.write(f"line {i:06d} " + "x" * 24 + "\n")
    return path


def _peak_alloc(fn):
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


# -- read_line_window ---------------------------------------------------------


def test_reads_a_range_deep_in_a_large_file_with_bounded_memory(big_file):
    window, peak = _peak_alloc(lambda: read_line_window(big_file, 150_000, 150_002))

    assert [line[:11] for line in window.lines] == [
        "line 150000",
        "line 150001",
        "line 150002",
    ]
    assert (window.start_line, window.end_line) == (150_000, 150_002)
    assert window.total_lines == 200_000
    assert not window.truncated
    assert big_file.stat().st_size > 6_000_000
    assert peak < 1_000_000, f"peak allocation {peak} bytes"


def test_open_ended_read_of_a_large_file_returns_the_first_window(big_file):
    window, peak = _peak_alloc(lambda: read_line_window(big_file))

    assert window.start_line == 1
    assert window.end_line == MAX_READ_LINES
    assert window.truncated
    assert len(window.lines) == MAX_READ_LINES
    assert peak < MAX_READ_BYTES * 4


def test_line_count_is_skipped_for_files_over_the_count_limit(big_file):
    window = read_line_window(big_file, 1, 5, count_limit=1024)
    assert window.total_lines is None
    assert len(window.lines) == 5


def test_range_longer_than_the_line_cap_is_truncated(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("".join(f"{i}\n" for i in range(1, 51)))
    window = read_line_window(path, 10, 40, max_lines=5)

    assert window.lines == ["10", "11", "12", "13", "14"]
    assert window.end_line == 14
    assert window.truncated


def test_byte_cap_returns_whole_lines_only(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a" * 40 + "\n" + "b" * 40 + "\n" + "c" * 40 + "\n")
    window = read_line_window(path, 1, None, max_bytes=100)

    assert window.lines == ["a" * 40, "b" * 40]
    assert window.truncated and not window.line_truncated


def test_single_huge_line_is_cut_at_the_byte_cap(tmp_path):
    path = tmp_path / "minified.js"
    path.write_bytes(b"y" * (5 * 1024 * 1024))  # one 5 MiB line, no newline
    window, peak = _peak_alloc(lambda: read_line_window(path))

    assert len(window.lines) == 1
    assert len(window.lines[0]) == MAX_READ_BYTES
    assert window.truncated and window.line_truncated
    assert window.total_lines == 1
    assert peak < MAX_READ_BYTES * 4, f"peak allocation {peak} bytes"


def test_end_line_past_eof_is_clamped(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("one\ntwo\nthree\n")
    window = read_line_window(path, 2, 99)

    assert window.lines == ["two", "three"]
    assert (window.end_line, window.total_lines) == (3, 3)
    assert not window.truncated


def test_crlf_and_missing_final_newline(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"one\r\ntwo\r\nthree")
    window = read_line_window(path)
    assert window.lines == ["one", "two", "three"]
    assert window.total_lines == 3


def test_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    window = read_line_window(path)
    assert window.lines == [] and window.total_lines == 0 and window.end_line == 0


@pytest.mark.parametrize(
    "start,end,message",
    [
        (0, None, "start_line must be >= 1"),
        (-5, 3, "start_line must be >= 1"),
        (5, 4, "end_line (4) must be >= start_line (5)"),
        (4, None, "past the end of the file (3 lines)"),
        (10, 20, "past the end of the file (3 lines)"),
    ],
)
def test_invalid_ranges_are_rejected(tmp_path, start, end, message):
    path = tmp_path / "f.txt"
    path.write_text("one\ntwo\nthree\n")
    with pytest.raises(LineRangeError) as err:
        read_line_window(path, start, end)
    assert message in str(err.value)


# -- through the MCP tool -----------------------------------------------------


async def _call(repo, **arguments):
    server = create_server(Settings(repo_root=repo))
    async with create_connected_server_and_client_session(server) as session:
        return await session.call_tool("read_file", arguments)


async def test_tool_pages_through_a_large_file(big_file):
    first = await _call(big_file.parent, path="big.txt")
    data = json.loads(first.content[0].text)
    assert not first.isError
    assert data["truncated"] is True
    assert data["end_line"] == MAX_READ_LINES
    assert data["next_start_line"] == MAX_READ_LINES + 1
    assert len(data["content"].encode()) <= MAX_READ_BYTES

    second = await _call(
        big_file.parent, path="big.txt", start_line=data["next_start_line"]
    )
    data2 = json.loads(second.content[0].text)
    assert data2["content"].startswith(f"line {MAX_READ_LINES + 1:06d}")


async def test_tool_reports_invalid_ranges_as_errors(tmp_path):
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    for args in (
        {"start_line": 0},
        {"start_line": 3, "end_line": 2},
        {"start_line": 9},
    ):
        result = await _call(tmp_path, path="f.txt", **args)
        assert result.isError, args


async def test_tool_small_file_has_no_continuation(tmp_path):
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    result = await _call(tmp_path, path="f.txt")
    data = json.loads(result.content[0].text)
    assert data["content"] == "one\ntwo"
    assert data["truncated"] is False and data["next_start_line"] is None
    assert data["total_lines"] == 2


async def test_tool_still_rejects_binary_files(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    result = await _call(tmp_path, path="blob.bin")
    assert result.isError
