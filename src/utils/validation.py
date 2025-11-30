"""Validation utilities for the RFAI system."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError


class FractalEngineConfig(BaseModel):
    """Configuration schema for :class:`FractalEngine` components."""

    max_depth: int = Field(default=3, ge=1, le=32)
    scale_factor: float = Field(default=0.5, gt=0, le=10)
    recursion_limit: int = Field(default=8, ge=1, le=64)


class SwarmCoordinatorConfig(BaseModel):
    """Configuration schema for :class:`SwarmCoordinator` components."""

    agent_count: int = Field(default=8, ge=1, le=256)
    consensus_threshold: float = Field(default=0.75, gt=0, le=1)


class QuantumProcessorConfig(BaseModel):
    """Configuration schema for :class:`QuantumProcessor` components."""

    shots: int = Field(default=128, ge=1, le=4096)
    decoherence: float = Field(default=0.02, ge=0, le=1)


class MetaLearnerConfig(BaseModel):
    """Configuration schema for :class:`MetaLearner` components."""

    learning_rate: float = Field(default=0.1, gt=0, le=1)
    momentum: float = Field(default=0.9, ge=0, le=1)


CONFIG_MODELS: Mapping[str, type[BaseModel]] = {
    "fractal_engine": FractalEngineConfig,
    "swarm_coordinator": SwarmCoordinatorConfig,
    "quantum_processor": QuantumProcessorConfig,
    "meta_learner": MetaLearnerConfig,
}


def validate_component_config(
    name: str, config: Optional[MutableMapping[str, Any]]
) -> Dict[str, Any]:
    """Validate and normalise a component configuration mapping."""

    model = CONFIG_MODELS.get(name)
    if model is None:
        return dict(config or {})
    payload: Dict[str, Any] = dict(config or {})
    validated = model(**payload)
    return validated.model_dump()


def _coerce_sequence(values: Any) -> Sequence[float]:
    if isinstance(values, Mapping):
        for key in ("values", "data", "input"):
            if key in values:
                return _coerce_sequence(values[key])
        raise ValueError("Mapping input must contain a 'values' or 'data' key")
    if isinstance(values, (bytes, bytearray)):
        raise ValueError("Binary payloads are not supported")
    if isinstance(values, str):
        raise ValueError("String payloads are not supported; provide a numeric sequence")
    if isinstance(values, (int, float)):
        return (float(values),)
    if isinstance(values, Sequence):
        coerced = []
        for item in values:
            if isinstance(item, (int, float)):
                coerced.append(float(item))
            else:
                raise ValueError("Sequence values must be numeric")
        return tuple(coerced)
    raise ValueError("Unsupported input payload type")


def validate_input_payload(input_data: Any) -> list[float]:
    """Validate task input payloads, returning a list of floats."""

    values = list(_coerce_sequence(input_data))
    if not values:
        raise ValueError("Input payload must contain at least one numeric value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Numeric payload contains non-finite values")
    return values


__all__ = [
    "FractalEngineConfig",
    "SwarmCoordinatorConfig",
    "QuantumProcessorConfig",
    "MetaLearnerConfig",
    "validate_component_config",
    "validate_input_payload",
    "ValidationError",
]
