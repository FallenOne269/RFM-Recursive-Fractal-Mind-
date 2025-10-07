"""FastAPI wrapper for the Recursive Fractal Autonomous Intelligence system."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PACKAGE_ROOT = Path(__file__).resolve().parent / "RFAI_System_Package"
SRC_ROOT = PACKAGE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:
    from rfai_system import RecursiveFractalAutonomousIntelligence
except ImportError as exc:  # pragma: no cover - fail fast during startup
    raise RuntimeError("Unable to import the RFAI core implementation.") from exc


def _load_config() -> Dict[str, Any]:
    """Load the default configuration shipped with the package."""
    config_path = PACKAGE_ROOT / "config" / "default_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    # Provide a sensible fallback when the file is missing.
    return {
        "max_fractal_depth": 4,
        "base_dimensions": 64,
        "swarm_size": 12,
        "quantum_enabled": True,
        "system_name": "Prometheus 2.0 API",
    }


def _initialise_core(config: Dict[str, Any]) -> RecursiveFractalAutonomousIntelligence:
    try:
        return RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=config.get("max_fractal_depth", 4),
            base_dimensions=config.get("base_dimensions", 64),
            swarm_size=config.get("swarm_size", 12),
            quantum_enabled=config.get("quantum_enabled", True),
        )
    except Exception as exc:  # pragma: no cover - fail fast during startup
        raise RuntimeError(f"Failed to initialise the RFAI core: {exc}") from exc

APP_CONFIG = _load_config()
RFAI_CORE = _initialise_core(APP_CONFIG)
app = FastAPI(title=APP_CONFIG.get("system_name", "RFAI System"), version="2.0.0")


class TaskInput(BaseModel):
    """Incoming task payload for the RFAI processing pipeline."""

    id: str = Field(..., description="Unique identifier for the task")
    type: str = Field(..., description="Task archetype or intent descriptor")
    complexity: float = Field(..., ge=0, description="Relative difficulty of the task")
    data: Optional[List[float]] = Field(
        None, description="Optional numeric payload consumed by the fractal processors"
    )
    priority: float = Field(0.5, ge=0, le=1, description="Scheduling priority in the swarm")
    requirements: List[str] = Field(
        default_factory=list, description="Human-readable constraints or acceptance criteria"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Arbitrary metadata propagated to the swarm layer"
    )


def _serialise_value(value: Any) -> Any:
    """Recursively convert numpy objects into JSON serialisable types."""

    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _serialise_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialise_value(item) for item in value)
    return value


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Lightweight liveness probe used by orchestration platforms."""

    status_payload = RFAI_CORE.get_system_status()
    return {"status": "ok", "system_status": status_payload.get("status", "INITIALISED")}


@app.get("/status")
def get_rfai_status() -> Dict[str, Any]:
    """Expose the detailed runtime status of the RFAI core."""

    return _serialise_value(RFAI_CORE.get_system_status())


@app.post("/process_task")
async def process_task(task: TaskInput) -> Dict[str, Any]:
    """Submit a task to the RFAI core and return the processed result."""

    core_task = task.model_dump()
    if core_task.get("data") is not None:
        core_task["data"] = np.array(core_task["data"], dtype=float)

    try:
        result = RFAI_CORE.process_task(core_task)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RFAI processing error: {exc}") from exc

    serialised_result = _serialise_value(result)
    serialised_result["system_state"] = _serialise_value(RFAI_CORE.system_state)
    return serialised_result


__all__ = ["app"]
