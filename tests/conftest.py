"""Pytest configuration for the RFAI test suite."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-import the local src package so later path adjustments do not shadow it.
import importlib

importlib.import_module("src")
