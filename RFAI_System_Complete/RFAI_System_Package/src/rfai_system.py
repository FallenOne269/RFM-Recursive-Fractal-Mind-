"""Recursive Fractal Autonomous Intelligence (RFAI) core implementation."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:
    from .components import MetaLearner, QuantumProcessor, SwarmCoordinator, FractalEngine
except ImportError:  # pragma: no cover - allow running as a script without package context
    from components import MetaLearner, QuantumProcessor, SwarmCoordinator, FractalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RecursiveFractalAutonomousIntelligence:
    """Main RFAI system implementing the integrated architecture."""

    def __init__(
        self,
        max_fractal_depth: int = 5,
        base_dimensions: int = 128,
        swarm_size: int = 50,
        quantum_enabled: bool = True,
        config_path: Optional[str] = None,
    ) -> None:
        logger.info("Initializing Recursive Fractal Autonomous Intelligence System...")

        self.max_depth = max_fractal_depth
        self.base_dims = base_dimensions
        self.swarm_size = swarm_size
        self.quantum_enabled = quantum_enabled

        if config_path:
            self._apply_configuration(config_path)

        self.fractal_engine = FractalEngine(self.max_depth, self.base_dims)
        self.fractal_hierarchy = self.fractal_engine.hierarchy

        self.swarm_coordinator = SwarmCoordinator(self.swarm_size)
        self.agent_swarm = self.swarm_coordinator.agent_swarm

        self.meta_learner = MetaLearner()
        self.architecture_search = self.meta_learner.architecture_search

        self.quantum_processor = QuantumProcessor(self.base_dims, self.quantum_enabled)

        self.system_state: Dict[str, Any] = {
            "performance_history": [],
            "adaptation_events": [],
            "emergent_behaviors": [],
            "optimization_trajectory": [],
        }

        logger.info("RFAI System initialized successfully!")
        logger.info("- Fractal depth: %s levels", self.max_depth)
        logger.info("- Base dimensions: %s", self.base_dims)
        logger.info("- Swarm size: %s agents", len(self.agent_swarm))
        logger.info("- Quantum processing: %s", self.quantum_enabled)

    def _apply_configuration(self, config_path: str) -> None:
        """Load configuration overrides from the provided path."""

        try:
            if not os.path.exists(config_path):
                logger.warning("Configuration file %s not found; using defaults.", config_path)
                return

            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load configuration from %s: %s", config_path, exc)
            return

        self.max_depth = config.get("max_fractal_depth", self.max_depth)
        self.base_dims = config.get("base_dimensions", self.base_dims)
        self.swarm_size = config.get("swarm_size", self.swarm_size)
        self.quantum_enabled = config.get("quantum_enabled", self.quantum_enabled)

    def fractal_processing(self, input_data: np.ndarray, level: int = 0) -> np.ndarray:
        """Recursive fractal processing with self-similar patterns."""

        prepared = input_data
        if level == 0:
            prepared = self.fractal_engine.prepare_input(input_data, strict=True)

        return self.fractal_engine.process(prepared, level=level)

    def swarm_coordination(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate swarm agents for collaborative problem solving."""

        return self.swarm_coordinator.coordinate(task)

    def quantum_classical_hybrid_processing(self, data: np.ndarray) -> np.ndarray:
        """Quantum-classical hybrid processing."""

        prepared = self.fractal_engine.prepare_input(data)
        if not self.quantum_enabled:
            return self.fractal_engine.process(prepared)

        quantum_result = self.quantum_processor.simulate(prepared)
        refined = self.fractal_engine.prepare_input(quantum_result.astype(float))
        return self.fractal_engine.process(refined)

    def meta_learning_optimization(self) -> Dict[str, Any]:
        """Meta-learning system for continuous self-improvement."""

        return self.meta_learner.optimise()

    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score."""

        scores = []

        if "fractal_output" in results:
            fractal_score = min(1.0, float(np.mean(np.abs(results["fractal_output"]))))
            scores.append(fractal_score)

        if "swarm_output" in results:
            swarm_score = float(results["swarm_output"].get("overall_success_rate", 0.5))
            scores.append(swarm_score)

        if "quantum_output" in results:
            quantum_score = min(1.0, float(np.mean(np.abs(results["quantum_output"]))))
            scores.append(quantum_score)

        return float(np.mean(scores)) if scores else 0.5

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Main task processing pipeline."""

        task_id = task.get("id", "unknown")
        logger.info("Processing task: %s", task_id)
        start_time = datetime.now()

        raw_data = np.real(np.array(task.get("data", np.random.randn(self.base_dims)), dtype=float))
        try:
            prepared_data = self.fractal_engine.prepare_input(raw_data)
        except ValueError as exc:
            raise ValueError(f"Invalid task data for processing: {exc}") from exc

        results: Dict[str, Any] = {
            "fractal_output": self.fractal_processing(prepared_data),
            "swarm_output": self.swarm_coordination(task),
        }

        if self.quantum_enabled:
            results["quantum_output"] = self.quantum_classical_hybrid_processing(prepared_data)

        meta_result = self.meta_learning_optimization()
        results["meta_optimization"] = meta_result

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        performance_score = self._calculate_performance_score(results)
        self.system_state.setdefault("performance_history", []).append(performance_score)

        final_result = {
            "task_id": task_id,
            "processing_time": processing_time,
            "performance_score": performance_score,
            "component_results": results,
            "system_adaptations": meta_result,
            "timestamp": end_time.isoformat(),
        }

        logger.info("Task completed - Performance: %.3f", performance_score)
        return final_result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""

        total_params = self.fractal_engine.total_parameters()
        active_agents = sum(
            1
            for agent in self.agent_swarm
            if agent.performance_metrics["task_completion_rate"] > 0.5
        )
        performance_history = self.system_state.get("performance_history", [])
        tasks_processed = len(performance_history)
        avg_performance = float(np.mean(performance_history)) if performance_history else 0.0
        learning_trend = 0.0
        if len(performance_history) > 1:
            x_axis = np.arange(len(performance_history))
            learning_trend = float(np.polyfit(x_axis, performance_history, 1)[0])

        return {
            "system_id": "RFAI_v1.0",
            "status": "OPERATIONAL",
            "fractal_hierarchy": {
                "levels": self.max_depth,
                "total_modules": sum(len(modules) for modules in self.fractal_hierarchy.values()),
                "total_parameters": total_params,
            },
            "agent_swarm": {
                "total_agents": len(self.agent_swarm),
                "active_agents": active_agents,
                "specializations": [*{agent.specialization for agent in self.agent_swarm}],
            },
            "quantum_processor": {
                "enabled": self.quantum_enabled,
                "qubits": self.quantum_processor.configuration.get("qubit_allocation", 0)
                if self.quantum_enabled
                else 0,
            },
            "performance": {
                "tasks_processed": tasks_processed,
                "avg_performance": avg_performance,
                "learning_trend": learning_trend,
            },
        }

    def save_state(self, filepath: str) -> None:
        """Save system state to file."""

        state = {
            "config": {
                "max_depth": self.max_depth,
                "base_dims": self.base_dims,
                "swarm_size": self.swarm_size,
                "quantum_enabled": self.quantum_enabled,
            },
            "system_state": self.system_state,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as file_handle:
                json.dump(state, file_handle, indent=2, default=str)
        except OSError as exc:
            logger.error("Failed to save state to %s: %s", filepath, exc)
            raise

        logger.info("System state saved to %s", filepath)

    def load_state(self, filepath: str) -> Dict[str, Any]:
        """Load system state from file."""

        try:
            with open(filepath, "r", encoding="utf-8") as file_handle:
                state = json.load(file_handle)
        except FileNotFoundError as exc:
            logger.error("State file not found: %s", filepath)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from state file %s: %s", filepath, exc)
            raise
        except OSError as exc:
            logger.error("Error reading state file %s: %s", filepath, exc)
            raise

        self.system_state = state.get("system_state", self.system_state)
        logger.info("System state loaded from %s", filepath)
        return state


if __name__ == "__main__":
    rfai = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=4,
        base_dimensions=64,
        swarm_size=12,
        quantum_enabled=True,
    )

    sample_task = {
        "id": "test_001",
        "type": "pattern_recognition",
        "complexity": 0.7,
        "data": np.random.randn(64),
        "requirements": ["accuracy", "speed"],
        "priority": 0.8,
    }

    result = rfai.process_task(sample_task)
    print(f"Task processed - Performance: {result['performance_score']:.3f}")
    print(f"System status: {rfai.get_system_status()}")
