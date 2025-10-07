"""State persistence helpers for the RFAI system."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def save_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Persist orchestration state to disk with an integrity checksum."""

    serialized = json.dumps(state, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    payload = {"checksum": checksum, "state": state}

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_state(path: str | Path) -> Mapping[str, Any]:
    """Load state from disk and verify its checksum."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    checksum = payload.get("checksum")
    state = payload.get("state")
    if checksum is None or state is None:
        raise ValueError("State payload is malformed")

    serialized = json.dumps(state, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if checksum != expected:
        raise ValueError("Checksum mismatch detected while loading state")

    return state


__all__ = ["save_state", "load_state"]
