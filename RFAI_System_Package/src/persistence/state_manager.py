"""Persistence utilities for saving and restoring orchestrator state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from utils import PersistenceConfig, sanitize_path


@dataclass(frozen=True)
class PersistedState:
    """Container describing a persisted orchestrator state."""

    version: str
    timestamp: str
    checksum: str
    payload: Dict[str, Any]


class StateManager:
    """Serialize and restore the orchestrator state with integrity checks."""

    def __init__(self, config: PersistenceConfig) -> None:
        self._config = config
        config.state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Mapping[str, Any], *, path: Path | None = None) -> Path:
        """Persist the given state to disk."""

        payload = dict(state)
        timestamp = datetime.now(timezone.utc).isoformat()
        serialized_payload = json.dumps(payload, sort_keys=True, default=_default_serializer)
        checksum = hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()
        target_path = self._resolve_path(path)
        document = {
            "version": self._config.version,
            "timestamp": timestamp,
            "checksum": checksum,
            "payload": payload,
        }
        target_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        return target_path

    def load(self, path: Path | None = None) -> PersistedState:
        """Load a previously persisted orchestrator state."""

        source_path = self._resolve_path(path)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        payload_serialized = json.dumps(data["payload"], sort_keys=True)
        checksum = hashlib.sha256(payload_serialized.encode("utf-8")).hexdigest()
        if checksum != data["checksum"]:
            raise ValueError("State checksum mismatch. File may be corrupted.")
        return PersistedState(
            version=data["version"],
            timestamp=data["timestamp"],
            checksum=data["checksum"],
            payload=data["payload"],
        )

    def _resolve_path(self, path: Path | None) -> Path:
        if path is None:
            path = self._config.state_dir / "rfai_state.json"
        return sanitize_path(path, base_dir=self._config.state_dir)


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {type(obj)!r} is not JSON serializable.")


__all__ = ["StateManager", "PersistedState"]
