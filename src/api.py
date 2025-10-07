"""FastAPI entrypoint for the RFAI orchestrator."""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader

from .rfai_system import RFAISystem

app = FastAPI(title="Recursive Fractal Mind", version="1.0.0")
_system = RFAISystem()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _expected_api_key() -> str:
    return os.getenv("API_KEY", "")


def get_system() -> RFAISystem:
    """Expose the orchestrator instance for dependency injection and tests."""

    return _system


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    expected = _expected_api_key()
    if not expected:
        return api_key or ""
    if api_key == expected:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Service liveness probe."""

    return {"status": "ok"}


@app.get("/status")
async def status_endpoint(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return orchestrator status information."""

    return get_system().get_status()


@app.post("/process_task")
async def process_task(
    payload: Dict[str, Any],
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Trigger a full orchestration cycle."""

    system = get_system()
    return system.run_cycle(payload)


__all__ = ["app", "get_system"]
