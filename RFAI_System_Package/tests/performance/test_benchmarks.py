"""Lightweight performance benchmarks."""

from __future__ import annotations

import time
import tracemalloc

from rfai_system import RecursiveFractalMind


def test_recursion_cycle_performance(config_path) -> None:
    orchestrator = RecursiveFractalMind(str(config_path))
    tracemalloc.start()
    start = time.perf_counter()
    for index in range(3):
        orchestrator.run_cycle(
            {
                "id": f"bench-{index}",
                "type": "benchmark",
                "complexity": 0.5,
                "payload": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"iteration": str(index)},
            }
        )
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 1.0
    assert peak < 10_000_000  # 10 MB
