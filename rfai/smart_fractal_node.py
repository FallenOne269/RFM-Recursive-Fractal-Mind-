from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .memory_layer import MemoryLayer
from .adaptation_signal import AdaptationSignal
from .dfe import DynamicFractalEncoder


@dataclass
class SmartFractalNode:
    node_id: str
    fim_id: str
    complexity: float = 1.0
    memory: MemoryLayer = field(default_factory=MemoryLayer)

    def apply_adaptation(self, signal: AdaptationSignal) -> None:
        self.complexity = max(0.1, self.complexity + signal.delta)
        self.memory.record(self.complexity, signal.delta, notes=signal.reason)

    def should_bifurcate(self, signal: AdaptationSignal) -> bool:
        return signal.delta > 0.5 and self.memory.summarize()["recent_trend"] > 0

    def bifurcate(self, dfe: DynamicFractalEncoder, semantic_vector: np.ndarray) -> List["SmartFractalNode"]:
        children: List[SmartFractalNode] = []
        rng = np.random.default_rng(0)
        for i in range(2):
            perturb = semantic_vector + rng.normal(0, 0.05, size=semantic_vector.shape)
            fim = dfe.encode({"child": i, "parent": self.node_id}, perturb, context={})
            child_node = SmartFractalNode(node_id=f"{self.node_id}.{i}", fim_id=fim.fim_id, complexity=self.complexity * 0.8)
            children.append(child_node)
        return children

    def should_consolidate(self) -> bool:
        summary = self.memory.summarize()
        return summary["count"] >= 3 and abs(summary["recent_trend"]) < 0.05 and abs(summary["mean_delta"]) < 0.1

