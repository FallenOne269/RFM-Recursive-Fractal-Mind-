"""FastAPI interface for the RFAI system."""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator, ConfigDict

from rfai_system import RecursiveFractalMind
from utils import validate_task

API_KEY_ENV = "RFAI_API_KEY"


class TaskRequest(BaseModel):
    """Schema for task processing requests."""

    id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., alias="type", description="Task category")
    complexity: float = Field(..., ge=0.0, le=1.0)
    payload: list[float] = Field(default_factory=list, description="Input signal")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Arbitrary metadata")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("payload", mode="before")
    @classmethod
    def _ensure_payload(cls, value: Optional[list[float]]) -> list[float]:
        return list(value or [])

    def as_validated_dict(self) -> Dict[str, object]:
        task_dict = {
            "id": self.id,
            "type": self.task_type,
            "complexity": self.complexity,
            "payload": self.payload,
            "metadata": self.metadata,
        }
        validate_task(task_dict)
        return task_dict


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv(API_KEY_ENV)
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def create_app(orchestrator_factory: Optional[Callable[[], RecursiveFractalMind]] = None) -> FastAPI:
    factory = orchestrator_factory or (lambda: RecursiveFractalMind())
    orchestrator = factory()
    app = FastAPI(title="Recursive Fractal Mind API", version="1.0.0")

    dependency = require_api_key

    @app.get("/health")
    async def health(_: None = Depends(dependency)) -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status_endpoint(_: None = Depends(dependency)) -> Dict[str, object]:
        return orchestrator.get_status()

    @app.post("/process_task")
    async def process_task(request: TaskRequest, _: None = Depends(dependency)) -> Dict[str, object]:
        task = request.as_validated_dict()
        result = orchestrator.run_cycle(task)
        return result

    return app


__all__ = ["create_app", "TaskRequest", "require_api_key"]
