"""Higher level orchestration node built on top of :mod:`semantic_smart_node`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np

from .adaptation_signal import AdaptationSignal
from .semantic_smart_node import SemanticSmartNode


@dataclass
class SmartFractalNode:
    """A composable node that can propagate adaptation signals to children."""

    name: str
    node: SemanticSmartNode
    propagation_decay: float = 0.9
    children: Dict[str, "SmartFractalNode"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not 0 < self.propagation_decay <= 1:
            raise ValueError("propagation_decay must be in the interval (0, 1]")

    def add_child(self, child: "SmartFractalNode") -> None:
        self.children[child.name] = child

    def propagate(self, data, extractor_name: str) -> Dict[str, AdaptationSignal]:
        """Process ``data`` and propagate adaptation to child nodes."""

        signal = self.node.process(data, extractor_name)
        outputs = {self.name: signal}

        if not self.children:
            return outputs

        scaled_data = np.asarray(data, dtype=float) * self.propagation_decay
        for child in self.children.values():
            outputs.update(child.propagate(scaled_data, extractor_name))
        return outputs

    def walk(self) -> Iterable[str]:  # pragma: no cover - trivial helper
        yield self.name
        for child in self.children.values():
            yield from child.walk()
