"""Basic example demonstrating the refactored RFAI orchestrator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rfai_system import Orchestrator


def load_default_config() -> Dict[str, Any]:
    """Load the packaged default configuration file."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Run the basic demonstration pipeline."""
    print("RFAI System - Basic Usage Example")
    print("=" * 40)

    config = load_default_config()
    orchestrator = Orchestrator(config=config)

    status = orchestrator.get_status()
    print(f"System Name: {status['system']['name']}")
    print(f"Modules: {', '.join(status['system']['modules'])}")
    print(f"Fractal depth: {status['fractal_output']['depth']}")
    print(f"Swarm size: {status['swarm_output']['swarm_size']}")
    if status["quantum_output"]:
        print(f"Quantum qubits: {status['quantum_output']['qubits']}")
    print()

    tasks = [
        {
            "id": "pattern_001",
            "type": "pattern_recognition",
            "complexity": 0.6,
            "data": np.sin(np.linspace(0, 4 * np.pi, 64)) + np.random.randn(64) * 0.1,
            "requirements": ["accuracy", "speed"],
            "priority": 0.8,
        },
        {
            "id": "optimization_001",
            "type": "optimization",
            "complexity": 0.8,
            "data": np.random.exponential(1.0, 64),
            "requirements": ["global_optimum", "convergence"],
            "priority": 0.9,
        },
        {
            "id": "memory_001",
            "type": "memory_management",
            "complexity": 0.4,
            "data": np.random.beta(2, 5, 64),
            "requirements": ["efficiency", "compression"],
            "priority": 0.7,
        },
    ]

    print("Processing tasks...")
    for task in tasks:
        print(f"Task {task['id']} (complexity: {task['complexity']:.2f})")
        result = orchestrator.process_task(task)
        print(f"  Performance: {result['performance_score']:.3f}")
        print(f"  Swarm synergy: {result['swarm_output']['synergy']:.3f}")
        if result["quantum_output"]["active"]:
            print(f"  Entanglement entropy: {result['quantum_output']['entanglement_entropy']:.3f}")
        print()

    print("Performance history:")
    for index, record in enumerate(orchestrator.history, 1):
        print(f"  {index}. {record['task_id']} -> {record['performance_score']:.3f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
