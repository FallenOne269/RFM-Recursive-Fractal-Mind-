"""FastAPI interface exposing core orchestrator capabilities."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..core.orchestrator import Orchestrator
from ..utils.validation import ConfigurationError


class TaskPayload(BaseModel):
    """Pydantic model describing incoming task requests."""

    id: str = Field(..., description="Unique identifier for the task")
    type: str = Field(..., description="Task category")
    complexity: float = Field(..., ge=0.0, le=1.0, description="Task complexity between 0 and 1")
    data: Optional[Any] = Field(default=None, description="Optional input data")
    priority: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    requirements: Optional[list[str]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("requirements", mode="before")
    def _ensure_list(cls, value: Any) -> Optional[list[str]]:  # noqa: D401
        """Convert missing requirement entries to None."""
        if value is None:
            return None
        return list(value)


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """Create a singleton orchestrator for the API process."""
    config_path = os.getenv("RFAI_CONFIG_PATH")
    if config_path:
        return Orchestrator(config_path=config_path)
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "default_config.json"
    )
    return Orchestrator(config_path=os.path.abspath(default_path))


app = FastAPI(title="Recursive Fractal Autonomous Intelligence API")


def _ensure_all_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure payload contains expected output keys with None defaults."""
    template = {
        "fractal_output": None,
        "swarm_output": None,
        "quantum_output": None,
        "meta_output": None,
    }
    return template | payload


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health probe providing readiness information."""
    orchestrator = get_orchestrator()
    return {
        "status": "healthy",
        **_ensure_all_keys({}),
        "tasks_processed": orchestrator.metrics.get("tasks_processed", 0),
    }


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Return orchestrator status including subsystem summaries."""
    orchestrator = get_orchestrator()
    snapshot = orchestrator.get_status()
    return _ensure_all_keys(
        {
            "fractal_output": snapshot.get("fractal_output"),
            "swarm_output": snapshot.get("swarm_output"),
            "quantum_output": snapshot.get("quantum_output"),
            "meta_output": snapshot.get("meta_output"),
            "system": snapshot.get("system"),
        }
    )


@app.post("/process_task")
async def process_task(payload: TaskPayload) -> Dict[str, Any]:
    """Process a task via the orchestrator and return structured output."""
    orchestrator = get_orchestrator()
    try:
        result = orchestrator.process_task(payload.model_dump(exclude_none=True))
    except ConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Internal processing error") from exc
    return _ensure_all_keys(result)


__all__ = ["app", "get_orchestrator", "TaskPayload"]
