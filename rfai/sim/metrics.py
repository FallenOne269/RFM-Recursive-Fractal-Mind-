from __future__ import annotations

from typing import Dict, List
import numpy as np


def task_completion_rate(progress: int, total: int) -> float:
    return float(progress) / max(1, total)


def time_efficiency(steps: int, progress: int) -> float:
    return float(progress) / max(1, steps)


def autonomous_recovery(failures: int, recoveries: int) -> float:
    return recoveries / max(1, failures + recoveries)


def coordination_effectiveness(signal_count: int, node_count: int) -> float:
    return signal_count / max(1, node_count)


def oscillation_score(collisions: List[bool]) -> float:
    if not collisions:
        return 0.0
    return sum(collisions) / len(collisions)


def recursion_depth_used(depths: List[int]) -> int:
    return max(depths) if depths else 0


def summary(metrics: Dict[str, float]) -> Dict[str, float]:
    return metrics

