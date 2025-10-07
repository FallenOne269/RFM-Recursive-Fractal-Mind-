"""Configuration and validation helpers for the RFAI system."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from jsonschema import Draft202012Validator

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
SCHEMA_DIR = BASE_DIR / "schemas"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"


@dataclass(frozen=True)
class FractalEngineConfig:
    """Configuration for the fractal engine."""

    max_depth: int
    base_dimensions: int
    noise_scale: float


@dataclass(frozen=True)
class SwarmCoordinatorConfig:
    """Configuration for the swarm coordinator."""

    swarm_size: int
    max_parallel_tasks: int


@dataclass(frozen=True)
class QuantumProcessorConfig:
    """Configuration for the quantum processor."""

    enabled: bool
    qubit_count: int
    sampling_runs: int


@dataclass(frozen=True)
class MetaLearnerConfig:
    """Configuration for the meta learner."""

    base_learning_rate: float
    adaptation_factor: float
    performance_threshold: float


@dataclass(frozen=True)
class PersistenceConfig:
    """Persistence settings."""

    state_dir: Path
    version: str


@dataclass(frozen=True)
class SystemConfig:
    """Aggregated configuration for the orchestrator."""

    fractal: FractalEngineConfig
    swarm: SwarmCoordinatorConfig
    quantum: QuantumProcessorConfig
    meta: MetaLearnerConfig
    persistence: PersistenceConfig


def _load_json_schema(name: str) -> Dict[str, Any]:
    schema_path = SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_path(path_str: os.PathLike[str] | str, *, base_dir: Path = BASE_DIR, allow_external: bool = False) -> Path:
    """Sanitize file-system paths to avoid directory traversal."""

    base = base_dir.resolve()
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not allow_external:
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"Path {candidate} is outside the allowed base directory {base}."
            ) from exc
    return candidate


def _validate(data: Mapping[str, Any], schema_name: str) -> None:
    schema = _load_json_schema(schema_name)
    validator = Draft202012Validator(schema)
    validator.validate(dict(data))


def load_config(config_path: Optional[os.PathLike[str] | str] = None) -> SystemConfig:
    """Load and validate the system configuration."""

    if config_path is None:
        path = sanitize_path(DEFAULT_CONFIG_PATH, base_dir=BASE_DIR)
    else:
        path = Path(config_path).expanduser().resolve()
    with Path(path).open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    _validate(config_data, "config.schema.json")

    persistence_dir = sanitize_path(
        config_data["persistence"]["state_dir"], base_dir=BASE_DIR, allow_external=True
    )
    return SystemConfig(
        fractal=FractalEngineConfig(
            max_depth=config_data["fractal_engine"]["max_depth"],
            base_dimensions=config_data["fractal_engine"]["base_dimensions"],
            noise_scale=config_data["fractal_engine"]["noise_scale"],
        ),
        swarm=SwarmCoordinatorConfig(
            swarm_size=config_data["swarm_coordinator"]["swarm_size"],
            max_parallel_tasks=config_data["swarm_coordinator"]["max_parallel_tasks"],
        ),
        quantum=QuantumProcessorConfig(
            enabled=config_data["quantum_processor"]["enabled"],
            qubit_count=config_data["quantum_processor"]["qubit_count"],
            sampling_runs=config_data["quantum_processor"]["sampling_runs"],
        ),
        meta=MetaLearnerConfig(
            base_learning_rate=config_data["meta_learner"]["base_learning_rate"],
            adaptation_factor=config_data["meta_learner"]["adaptation_factor"],
            performance_threshold=config_data["meta_learner"]["performance_threshold"],
        ),
        persistence=PersistenceConfig(
            state_dir=persistence_dir,
            version=config_data["persistence"]["version"],
        ),
    )


def validate_task(task: Mapping[str, Any]) -> None:
    """Validate task payloads before processing."""

    _validate(task, "task.schema.json")


__all__ = [
    "FractalEngineConfig",
    "SwarmCoordinatorConfig",
    "QuantumProcessorConfig",
    "MetaLearnerConfig",
    "PersistenceConfig",
    "SystemConfig",
    "load_config",
    "sanitize_path",
    "validate_task",
]
