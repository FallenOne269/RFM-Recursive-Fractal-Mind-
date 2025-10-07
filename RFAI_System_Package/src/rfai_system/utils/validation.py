"""Validation utilities for configuration, task payloads, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator

from ..core.base import ConfigurationError


CONFIG_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "system": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "max_fractal_depth": {"type": "integer", "minimum": 1, "maximum": 12},
                "base_dimensions": {"type": "integer", "minimum": 4, "maximum": 4096},
                "swarm_size": {"type": "integer", "minimum": 1, "maximum": 512},
                "quantum_enabled": {"type": "boolean"},
                "recursion_limit": {"type": "integer", "minimum": 1, "maximum": 64},
            },
            "required": [
                "max_fractal_depth",
                "base_dimensions",
                "swarm_size",
                "quantum_enabled",
                "recursion_limit",
            ],
        },
        "modules": {
            "type": "object",
            "properties": {
                "fractal_engine": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                        "settings": {"type": "object"},
                    },
                    "required": ["plugin"],
                },
                "swarm_coordinator": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                        "settings": {"type": "object"},
                    },
                    "required": ["plugin"],
                },
                "quantum_processor": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                        "settings": {"type": "object"},
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["plugin", "enabled"],
                },
                "meta_learner": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                        "settings": {"type": "object"},
                    },
                    "required": ["plugin"],
                },
            },
            "required": [
                "fractal_engine",
                "swarm_coordinator",
                "meta_learner",
            ],
        },
        "persistence": {
            "type": "object",
            "properties": {
                "state_path": {"type": "string"},
                "autosave": {"type": "boolean"},
            },
        },
    },
    "required": ["system", "modules"],
}


TASK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "complexity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "data": {"type": ["array", "object", "number", "string", "null"]},
        "priority": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "requirements": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
    "required": ["id", "type", "complexity"],
}


def sanitize_path(path: str, base: Path | None = None) -> Path:
    """Resolve a filesystem path safely within an optional base directory."""
    candidate = Path(path).expanduser()
    base_dir = base or Path.cwd()
    resolved = (base_dir / candidate).resolve()
    if base and base not in resolved.parents and base != resolved:
        raise ConfigurationError(f"Path '{path}' is outside the allowed directory")
    return resolved


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load JSON content from disk with consistent error handling."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ConfigurationError(f"Configuration file '{path}' not found") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Malformed JSON in configuration '{path}'") from exc


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration against the canonical schema."""
    validator = Draft202012Validator(CONFIG_SCHEMA)
    if errors := sorted(
        validator.iter_errors(config), key=lambda err: err.path
    ):
        messages = ", ".join(error.message for error in errors)
        raise ConfigurationError(f"Invalid configuration: {messages}")
    return config


def load_and_validate_config(
    config: Dict[str, Any] | None = None,
    *,
    config_path: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load configuration from disk and validate it."""
    data: Dict[str, Any]
    if config_path:
        path = sanitize_path(config_path)
        data = load_json_file(path)
    else:
        data = config.copy() if config else {}
    if overrides:
        data = {**data, **overrides}
    if not data:
        raise ConfigurationError("No configuration data provided")
    return validate_config(data)


def validate_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a task payload against the schema."""
    sanitized = dict(task)
    data = sanitized.get("data")
    if data is not None and hasattr(data, "tolist"):
        sanitized["data"] = data.tolist()
    validator = Draft202012Validator(TASK_SCHEMA)
    errors = sorted(validator.iter_errors(sanitized), key=lambda err: err.path)
    if errors:
        messages = ", ".join(error.message for error in errors)
        raise ConfigurationError(f"Invalid task payload: {messages}")
    return sanitized
