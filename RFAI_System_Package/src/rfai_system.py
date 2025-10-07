"""Recursive Fractal Mind orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from fractal_engine import FractalEngine, FractalEngineOutput
from meta_learner import MetaLearner
from persistence import StateManager
from quantum_processor import QuantumProcessor, QuantumProcessorOutput
from swarm_coordinator import SwarmCoordinator, SwarmCoordinatorOutput
from utils import (
    load_config,
    validate_task,
    PluginRegistry,
)

PLUGINS = [
    "fractal_engine",
    "swarm_coordinator",
    "quantum_processor",
    "meta_learner",
]
SYSTEM_VERSION = "2.0.0"


@dataclass
class OrchestratorState:
    """Track orchestrator execution history."""

    cycle_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"cycle_count": self.cycle_count, "history": self.history}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OrchestratorState":
        return cls(cycle_count=payload.get("cycle_count", 0), history=list(payload.get("history", [])))


class RecursiveFractalMind:
    """High level orchestrator for the Recursive Fractal Mind system."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config = load_config(config_path)
        self._registry = PluginRegistry()
        self._registry.load_plugins(PLUGINS)
        self._fractal_engine: FractalEngine = self._registry.create(
            "fractal_engine", config=self._config.fractal
        )
        self._swarm: SwarmCoordinator = self._registry.create(
            "swarm_coordinator", config=self._config.swarm
        )
        self._quantum: QuantumProcessor = self._registry.create(
            "quantum_processor", config=self._config.quantum
        )
        self._meta: MetaLearner = self._registry.create("meta_learner", config=self._config.meta)
        self._state_manager = StateManager(self._config.persistence)
        self._state = OrchestratorState()

    def run_cycle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        validate_task(task)
        payload = task.get("payload", [])
        fractal_output = self._fractal_engine.run(payload)
        swarm_output = self._swarm.coordinate(task["complexity"], task.get("agent_load", []))
        quantum_output = self._quantum.run(fractal_output.aggregated_signal, task["complexity"])
        performance_score = self._compute_performance(fractal_output, swarm_output, quantum_output)
        meta_output = self._meta.step(performance_score)

        result = {
            "task_id": task["id"],
            "fractal_output": fractal_output.to_dict(),
            "swarm_output": swarm_output.to_dict(),
            "quantum_output": quantum_output.to_dict() if quantum_output.active else None,
            "meta_output": meta_output.to_dict(),
            "performance_score": performance_score,
            "system_version": SYSTEM_VERSION,
        }

        self._state.cycle_count += 1
        self._state.history.append({
            "task_id": task["id"],
            "performance_score": performance_score,
        })
        return result

    def _compute_performance(
        self,
        fractal_output: FractalEngineOutput,
        swarm_output: SwarmCoordinatorOutput,
        quantum_output: QuantumProcessorOutput,
    ) -> float:
        score = np.mean(fractal_output.aggregated_signal)
        score += swarm_output.overall_success_rate
        if quantum_output.active:
            score += quantum_output.confidence
        return float(np.clip(score / 3.0, 0.0, 1.0))

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": SYSTEM_VERSION,
            "cycles": self._state.cycle_count,
            "config": {
                "fractal": self._config.fractal,
                "swarm": self._config.swarm,
                "quantum": self._config.quantum,
                "meta": self._config.meta,
            },
            "history_length": len(self._state.history),
        }

    def save_state(self, path: Optional[str] = None) -> str:
        target = self._state_manager.save(self._state.to_dict(), path=Path(path) if path else None)
        return str(target)

    def load_state(self, path: Optional[str] = None) -> None:
        restored = self._state_manager.load(Path(path) if path else None)
        self._state = OrchestratorState.from_dict(restored.payload)


__all__ = ["RecursiveFractalMind", "OrchestratorState"]
