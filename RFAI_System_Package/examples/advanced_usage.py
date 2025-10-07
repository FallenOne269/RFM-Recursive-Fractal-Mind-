"""Advanced usage of the modular RFAI orchestrator."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rfai_system import Orchestrator


def build_custom_config() -> Dict[str, Any]:
    """Return a tweaked configuration emphasising deeper recursion."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["system"]["max_fractal_depth"] = 5
    config["system"]["recursion_limit"] = 5
    config["modules"]["fractal_engine"]["settings"]["max_depth"] = 5
    config["modules"]["fractal_engine"]["settings"]["recursion_limit"] = 5
    config["modules"]["swarm_coordinator"]["settings"]["swarm_size"] = 16
    config["modules"]["meta_learner"]["settings"]["performance_threshold"] = 0.8
    config["persistence"]["state_path"] = "state/advanced_rfai_state.json"
    config["persistence"]["autosave"] = False
    return config


def generate_task(seed: int, complexity: float) -> Dict[str, Any]:
    """Generate a synthetic benchmark task."""
    rng = np.random.default_rng(seed)
    dimension = 64
    data = rng.normal(0.0, 1.0, size=dimension)
    return {
        "id": f"advanced_{seed:03d}",
        "type": rng.choice(
            ["pattern_recognition", "optimization", "memory_management"]
        ),
        "complexity": complexity,
        "data": data,
        "priority": float(rng.uniform(0.5, 1.0)),
    }


def run_pipeline(orchestrator: Orchestrator, iterations: int = 10) -> List[float]:
    """Execute a pipeline of synthetic tasks and return performance scores."""
    scores: List[float] = []
    for iteration in range(iterations):
        task = generate_task(iteration, complexity=0.3 + iteration / (iterations + 2))
        result = orchestrator.process_task(task)
        scores.append(result["performance_score"])
        if iteration % 3 == 0:
            print(
                f"Iteration {iteration + 1:02d}: performance={result['performance_score']:.3f} "
                f"synergy={result['swarm_output']['synergy']:.3f}"
            )
    return scores


def main() -> None:
    """Showcase custom configuration, persistence, and benchmarking."""
    print("RFAI System - Advanced Usage Example")
    print("=" * 45)
    config = build_custom_config()

    orchestrator = Orchestrator(config=config)
    baseline_status = orchestrator.get_status()
    print("Initial status:")
    print(f"  Depth: {baseline_status['fractal_output']['depth']}")
    print(f"  Swarm size: {baseline_status['swarm_output']['swarm_size']}")
    print()

    print("Executing synthetic pipeline...")
    scores = run_pipeline(orchestrator, iterations=12)
    mean_score = float(np.mean(scores))
    print(f"Average performance: {mean_score:.3f}")

    state_path = orchestrator.save_state(config["persistence"]["state_path"])
    print(f"State saved to {state_path}")

    reloaded = Orchestrator(config=config)
    reloaded.load_state(str(state_path))
    print("Reloaded orchestrator state verified.")

    benchmark_task = generate_task(999, complexity=0.95)
    start = time.perf_counter()
    benchmark_result = reloaded.process_task(benchmark_task)
    duration = time.perf_counter() - start
    print("Benchmark run:")
    print(f"  Performance score: {benchmark_result['performance_score']:.3f}")
    print(f"  Duration: {duration:.4f}s")
    print(f"  Learning rate: {benchmark_result['meta_output']['learning_rate']:.5f}")

    print("\nAdvanced example complete.")


if __name__ == "__main__":
    main()
