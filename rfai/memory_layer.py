from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple
import time


@dataclass
class MemoryRecord:
    timestamp: float
    complexity: float
    delta: float
    notes: str


class MemoryLayer:
    """Stores bounded adaptation history for a node."""

    def __init__(self, maxlen: int = 50):
        self.records: Deque[MemoryRecord] = deque(maxlen=maxlen)

    def record(self, complexity: float, delta: float, notes: str = "") -> None:
        self.records.append(
            MemoryRecord(timestamp=time.time(), complexity=complexity, delta=delta, notes=notes)
        )

    def summarize(self) -> Dict[str, float]:
        if not self.records:
            return {"mean_delta": 0.0, "recent_trend": 0.0, "count": 0}
        deltas = [r.delta for r in self.records]
        mean_delta = sum(deltas) / len(deltas)
        recent = deltas[-5:]
        trend = (recent[-1] - recent[0]) / max(1, len(recent) - 1) if len(recent) > 1 else 0.0
        return {"mean_delta": mean_delta, "recent_trend": trend, "count": len(deltas)}

