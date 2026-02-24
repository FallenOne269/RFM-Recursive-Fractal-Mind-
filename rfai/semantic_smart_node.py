from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .semantic_goal import SemanticGoal
from .smart_fractal_node import SmartFractalNode
from .adaptation_signal import AdaptationSignal


@dataclass
class SemanticSmartNode(SmartFractalNode):
    goal: SemanticGoal | None = None

    def apply_adaptation(self, signal: AdaptationSignal) -> None:
        goal_boost = 0.0
        if self.goal is not None and "semantic_vector" in signal.payload:
            goal_boost = self.goal.similarity(np.asarray(signal.payload["semantic_vector"])) * 0.1
        adjusted_signal = AdaptationSignal(
            delta=signal.delta + goal_boost,
            reason=signal.reason,
            timestamp=signal.timestamp,
            payload=signal.payload,
        )
        super().apply_adaptation(adjusted_signal)

