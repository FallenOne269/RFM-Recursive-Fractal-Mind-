"""Integration tests for the orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from rfai_system import RecursiveFractalMind


def _build_task(index: int) -> dict[str, object]:
    return {
        "id": f"cycle-{index}",
        "type": "analysis",
        "complexity": 0.2 + 0.2 * index,
        "payload": [float(index), float(index + 1), float(index + 2)],
        "metadata": {"iteration": str(index)},
    }


def test_three_cycle_run_and_state(config_path: Path) -> None:
    orchestrator = RecursiveFractalMind(str(config_path))
    outputs = [orchestrator.run_cycle(_build_task(i)) for i in range(3)]
    assert len(outputs) == 3
    status = orchestrator.get_status()
    assert status["cycles"] == 3
    assert status["history_length"] == 3

    saved_path = Path(orchestrator.save_state())
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["payload"]["cycle_count"] == 3

    restored = RecursiveFractalMind(str(config_path))
    restored.load_state(saved_path)
    assert restored.get_status()["cycles"] == status["cycles"]
    assert restored.get_status()["history_length"] == status["history_length"]
