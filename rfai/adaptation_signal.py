from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import time


@dataclass
class AdaptationSignal:
    delta: float
    reason: str
    timestamp: float = field(default_factory=lambda: time.time())
    payload: Dict = field(default_factory=dict)

