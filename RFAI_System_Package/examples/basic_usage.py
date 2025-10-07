"""Basic usage example for the Recursive Fractal Mind system."""

from __future__ import annotations

import sys
from pathlib import Path

SYS_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SYS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYS_ROOT))

from pprint import pprint

from rfai_system import RecursiveFractalMind


def main() -> None:
    orchestrator = RecursiveFractalMind()
    task = {
        "id": "demo-001",
        "type": "pattern_recognition",
        "complexity": 0.5,
        "payload": [0.1, 0.2, 0.3, 0.4],
        "metadata": {"source": "example"},
    }
    result = orchestrator.run_cycle(task)
    pprint(result)


if __name__ == "__main__":
    main()
