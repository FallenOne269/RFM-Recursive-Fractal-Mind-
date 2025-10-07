"""Advanced orchestration example demonstrating persistence and multiple cycles."""

from __future__ import annotations

import sys
from pathlib import Path

SYS_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_ROOT))

from pathlib import Path
from pprint import pprint

from rfai_system import RecursiveFractalMind


def main() -> None:
    orchestrator = RecursiveFractalMind()
    tasks = [
        {
            "id": f"advanced-{index:03d}",
            "type": "optimization",
            "complexity": 0.3 + index * 0.2,
            "payload": [float(index), float(index + 1), float(index + 2)],
            "metadata": {"scenario": "advanced"},
        }
        for index in range(3)
    ]
    outputs = [orchestrator.run_cycle(task) for task in tasks]
    for output in outputs:
        pprint(output)

    state_path = Path("advanced_state.json")
    orchestrator.save_state(state_path)
    orchestrator.load_state(state_path)


if __name__ == "__main__":
    main()
