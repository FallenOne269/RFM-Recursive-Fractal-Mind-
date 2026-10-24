"""Pytest configuration for the RFAI test suite."""

import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-import the local src package so later path adjustments do not shadow it.
importlib.import_module("src")

# The MCP server needs the `mcp` SDK, which requirements.txt installs only on
# Python 3.10+. Skip its tests where the SDK is not installed (the 3.9 CI job).
MCP_TEST_FILES = [
    "test_gemini.py",
    "test_junit_parsing.py",
    "test_mcp_server.py",
    "test_oauth.py",
    "test_oauth_lifecycle.py",
    "test_oauth_settings.py",
    "test_read_file_bounds.py",
]
collect_ignore = [] if importlib.util.find_spec("mcp") else MCP_TEST_FILES
