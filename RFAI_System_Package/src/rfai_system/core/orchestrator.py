"""Central orchestrator coordinating all RFAI subsystems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import BaseSubsystem, StatePersistenceError
from .registry import plugin_registry
from ..utils.validation import load_and_validate_config, sanitize_path, validate_task


def _ensure_builtin_plugins() -> None:
    """Import builtin plugins to trigger registration."""
    from .. import fractal_engine, meta_learner, quantum_processor, swarm_coordinator  # noqa: F401


class Orchestrator:
    """High level orchestrator that wires fractal, swarm, quantum, and meta subsystems."""

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        *,
        config_path: str | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> None:
        _ensure_builtin_plugins()
        self.config = load_and_validate_config(config, config_path=config_path, overrides=overrides)
        self.system_config = self.config.get("system", {})
        self.persistence_config = self.config.get("persistence", {})
        self.modules: Dict[str, BaseSubsystem] = {}
        self.history: list[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {"tasks_processed": 0}
        self._initialise_modules()

    def _initialise_modules(self) -> None:
        """Instantiate subsystems according to configuration."""
        modules_cfg = self.config.get("modules", {})
        for key, module_cfg in modules_cfg.items():
            plugin_name = module_cfg.get("plugin", key)
            settings = module_cfg.get("settings", {})
            if key == "quantum_processor":
                settings = {"enabled": module_cfg.get("enabled", True), **settings}
            subsystem = plugin_registry.create(plugin_name, settings)
            self.modules[key] = subsystem

    @property
    def fractal_engine(self) -> BaseSubsystem:
        """Return the fractal engine instance."""
        return self.modules["fractal_engine"]

    @property
    def swarm_coordinator(self) -> BaseSubsystem:
        """Return the swarm coordinator instance."""
        return self.modules["swarm_coordinator"]

    @property
    def quantum_processor(self) -> BaseSubsystem | None:
        """Return the quantum processor instance if configured."""
        return self.modules.get("quantum_processor")

    @property
    def meta_learner(self) -> BaseSubsystem:
        """Return the meta learner instance."""
        return self.modules["meta_learner"]

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a full recursive processing cycle for the task."""
        validated_task = validate_task(task)
        fractal_output = self.fractal_engine.process(validated_task)
        swarm_payload = {**validated_task, "fractal_output": fractal_output}
        swarm_output = self.swarm_coordinator.process(swarm_payload)

        quantum_default = {
            "active": False,
            "state_projection": [],
            "entanglement_entropy": 0.0,
        }
        quantum_output: Dict[str, Any]
        if self.quantum_processor is not None:
            quantum_output = self.quantum_processor.process(validated_task)
        else:
            quantum_output = quantum_default

        performance_history = [record["performance_score"] for record in self.history[-10:]]
        meta_payload = {
            **validated_task,
            "performance_hint": float(swarm_output.get("swarm_success", 0.0)),
            "performance_history": performance_history,
            "fractal_output": fractal_output,
            "swarm_output": swarm_output,
            "quantum_output": quantum_output,
        }
        meta_output = self.meta_learner.process(meta_payload)

        performance_components = np.array(
            [
                swarm_output.get("swarm_success", 0.0),
                1.0 / (1.0 + fractal_output.get("activation_energy", 0.0)),
                (
                    quantum_output.get("entanglement_entropy", 0.0)
                    if quantum_output.get("active")
                    else 0.0
                ),
                meta_output.get("learning_rate", 0.0),
            ]
        )
        weights = np.array([0.35, 0.25, 0.15, 0.25])
        performance_score = float(np.clip(np.dot(performance_components, weights), 0.0, 1.0))

        record = {
            "task_id": validated_task["id"],
            "performance_score": performance_score,
            "fractal_output": fractal_output,
            "swarm_output": swarm_output,
            "quantum_output": quantum_output,
            "meta_output": meta_output,
        }
        self.history.append(record)
        self.metrics["tasks_processed"] = self.metrics.get("tasks_processed", 0) + 1
        self.metrics["last_task_id"] = validated_task["id"]

        if self.persistence_config.get("autosave"):
            default_path = self.persistence_config.get("state_path", "rfai_state.json")
            self.save_state(default_path)

        return {
            "fractal_output": fractal_output,
            "swarm_output": swarm_output,
            "quantum_output": quantum_output,
            "meta_output": meta_output,
            "performance_score": performance_score,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return an aggregated status snapshot."""
        return {
            "system": {
                "name": self.system_config.get("name", "RFAI"),
                "tasks_processed": self.metrics.get("tasks_processed", 0),
                "modules": list(self.modules.keys()),
            },
            "fractal_output": self.fractal_engine.get_status(),
            "swarm_output": self.swarm_coordinator.get_status(),
            "quantum_output": (
                self.quantum_processor.get_status() if self.quantum_processor else None
            ),
            "meta_output": self.meta_learner.get_status(),
        }

    def save_state(self, path: str) -> Path:
        """Persist orchestrator and subsystem state to the provided path."""
        destination = sanitize_path(path)
        payload = {
            "config": self.config,
            "history": self.history,
            "metrics": self.metrics,
            "modules": {name: module.get_state() for name, module in self.modules.items()},
        }
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:  # pragma: no cover - defensive guard
            raise StatePersistenceError(f"Failed to persist state to '{destination}'") from exc
        return destination

    def load_state(self, path: str) -> None:
        """Load orchestrator state from disk."""
        source = sanitize_path(path)
        try:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise StatePersistenceError(f"State file '{source}' was not found") from exc
        except json.JSONDecodeError as exc:
            raise StatePersistenceError(f"Malformed state file '{source}'") from exc

        config = payload.get("config")
        if config:
            self.config = load_and_validate_config(config)
            self.system_config = self.config.get("system", {})
            self.persistence_config = self.config.get("persistence", {})
        self.history = [dict(entry) for entry in payload.get("history", [])]
        self.metrics = dict(payload.get("metrics", {}))

        modules_state = payload.get("modules", {})
        for name, state in modules_state.items():
            if name not in self.modules:
                raise StatePersistenceError(f"State contains unknown module '{name}'")
            self.modules[name].set_state(state)

    def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        for module in self.modules.values():
            module.shutdown()


__all__ = ["Orchestrator"]
