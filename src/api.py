"""FastAPI entrypoint for the RFAI orchestrator."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import get_settings
from .rfai_system import RFAISystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    """Input payload for a single orchestration cycle."""

    values: List[float] = Field(
        ...,
        min_length=1,
        description="Numeric values to feed into the fractal pipeline.",
        examples=[[1.0, 2.0, 3.0]],
    )


class TaskResponse(BaseModel):
    """Full orchestration cycle result."""

    fractal_output: Optional[Dict[str, Any]] = None
    swarm_output: Optional[Dict[str, Any]] = None
    quantum_output: Optional[Dict[str, Any]] = None
    meta_output: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service liveness payload."""

    status: str


class StatusResponse(BaseModel):
    """Orchestrator status snapshot."""

    components: Dict[str, bool]
    disabled_components: List[str] = Field(default_factory=list)
    failed_components: List[str] = Field(default_factory=list)
    cycles: int
    last_run: Optional[str] = None
    last_errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level orchestrator (initialised once at import time)
# ---------------------------------------------------------------------------

_system = RFAISystem()


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup and emit lifecycle log messages."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("RFAI system starting up")
    yield
    logger.info("RFAI system shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Recursive Fractal Mind",
    version="1.0.0",
    description=(
        "Modular orchestration framework for recursive fractal intelligence. "
        "Components: FractalEngine → SwarmCoordinator → QuantumProcessor → MetaLearner."
    ),
    lifespan=lifespan,
)

# CORS — origins are configurable via the CORS_ORIGINS environment variable.
_cors_origins = get_settings().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    """Attach a unique request identifier to every response."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Validate the ``X-API-Key`` header against the ``API_KEY`` env variable.

    Authentication is skipped when ``API_KEY`` is not set, which simplifies
    local development and testing.
    """
    expected = get_settings().api_key
    if not expected:
        return api_key or ""
    if api_key == expected:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_system() -> RFAISystem:
    """Return the module-level orchestrator instance.

    Exposed for dependency injection and test overrides.
    """
    return _system


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Service liveness probe — no authentication required."""
    return HealthResponse(status="ok")


@app.get(
    "/status",
    response_model=StatusResponse,
    tags=["ops"],
    dependencies=[Depends(verify_api_key)],
)
async def status_endpoint(
    system: RFAISystem = Depends(get_system),
) -> StatusResponse:
    """Return a snapshot of orchestrator status."""
    return StatusResponse(**system.get_status())


@app.post(
    "/process_task",
    response_model=TaskResponse,
    tags=["tasks"],
    dependencies=[Depends(verify_api_key)],
)
async def process_task(
    body: TaskRequest,
    system: RFAISystem = Depends(get_system),
) -> TaskResponse:
    """Trigger a full orchestration cycle and return all component outputs."""
    result = system.run_cycle(body.values)
    return TaskResponse(**result)


__all__ = ["app", "get_system"]
