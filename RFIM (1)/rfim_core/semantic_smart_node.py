"""Semantic smart node implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .adaptation_signal import AdaptationSignal
from .dfe import DynamicFractalEncoder
from .memory_layer import MemoryLayer
from .rsa import AdaptationResult, RecursiveStructuralAdapter
from .semantic_goal import SemanticGoal


@dataclass
class SemanticSmartNode:
    """Combines encoding, adaptation and semantic goals in a single node."""

    goal: SemanticGoal
    encoder: DynamicFractalEncoder
    adapter: RecursiveStructuralAdapter
    memory_capacity: int = 128
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory = MemoryLayer(self.memory_capacity)
        self._last_adaptation: Optional[AdaptationResult] = None
        self._counter = 0

    def process(self, data, extractor_name: str) -> AdaptationSignal:
        """Encode ``data`` and adapt the internal structure."""

        encoded = self.encoder.encode(data, extractor_name)
        adaptation = self.adapter.adapt(encoded["fim"])

        fim_key = f"fim_{self._counter}"
        stats_key = f"statistics_{self._counter}"
        self._counter += 1

        self.memory.add(fim_key, encoded["fim"])
        self.memory.add(stats_key, encoded["statistics"])
        self._last_adaptation = adaptation

        metrics = {"stability": adaptation.stability}
        satisfied = self.goal.is_satisfied(metrics)
        payload = {
            "goal": self.goal.name,
            "satisfied": satisfied,
            "stability": adaptation.stability,
            "metadata": dict(self.metadata),
        }
        return AdaptationSignal(signal_type="semantic_adaptation", payload=payload)

    def last_adaptation(self) -> Optional[AdaptationResult]:
        return self._last_adaptation

    def reconstruct(self, shape) -> np.ndarray:
        """Attempt to reconstruct an approximate signal from the stored FIMs."""

        fim_entries = [
            {"fim": fim, "statistics": stats}
            for fim, stats in self._iter_stored_fims()
        ]
        if not fim_entries:
            raise ValueError("No FIMs stored in memory")
        return self.encoder.decode(fim_entries, shape)

    def _iter_stored_fims(self):
        for key, value in self.memory.items():
            if key.startswith("fim_"):
                stats_key = key.replace("fim_", "statistics_")
                stats = self.memory.get(stats_key)
                if stats is not None:
                    yield value, stats
