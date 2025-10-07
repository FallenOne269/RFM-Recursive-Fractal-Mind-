"""Fractal processing engine implementing recursive state propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..core.base import BaseSubsystem
from ..core.registry import plugin_registry


@dataclass
class FractalNode:
    """Recursive node used within the fractal hierarchy."""

    level: int
    weights: np.ndarray
    bias: np.ndarray
    children: List["FractalNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the node into JSON-compatible structures."""
        return {
            "level": self.level,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FractalNode":
        """Reconstruct a node from its serialised representation."""
        children = [cls.from_dict(child) for child in payload.get("children", [])]
        return cls(
            level=int(payload["level"]),
            weights=np.array(payload["weights"], dtype=float),
            bias=np.array(payload["bias"], dtype=float),
            children=children,
        )


class FractalEngine(BaseSubsystem):
    """Hierarchical fractal processor producing recursive embeddings."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.depth: int = int(self.config.get("max_depth", 3))
        self.base_dimensions: int = int(self.config.get("base_dimensions", 64))
        self.branching_factor: int = int(self.config.get("branching_factor", 2))
        self.recursion_limit: int = int(self.config.get("recursion_limit", self.depth))
        seed = self.config.get("seed")
        self.rng = np.random.default_rng(seed)
        self.root: FractalNode = self._build_node(0, self.base_dimensions)
        self.output_history: List[List[float]] = []

    def _build_node(self, level: int, dimensions: int) -> FractalNode:
        """Create a hierarchy node with recursively generated children."""
        weights = self.rng.normal(0.0, 0.1, size=(dimensions, dimensions))
        bias = self.rng.normal(0.0, 0.01, size=dimensions)
        if level >= self.depth - 1:
            return FractalNode(level=level, weights=weights, bias=bias, children=[])
        child_dim = max(4, dimensions // 2)
        children = [self._build_node(level + 1, child_dim) for _ in range(self.branching_factor)]
        return FractalNode(level=level, weights=weights, bias=bias, children=children)

    def _prepare_input(self, task: Dict[str, Any]) -> np.ndarray:
        """Normalise task data into a fixed-size vector."""
        data = task.get("data")
        if data is None:
            vector = self.rng.normal(0.0, 1.0, size=self.base_dimensions)
        else:
            vector = np.array(data, dtype=float).ravel()
            if vector.size < self.base_dimensions:
                vector = np.pad(vector, (0, self.base_dimensions - vector.size))
            elif vector.size > self.base_dimensions:
                vector = vector[: self.base_dimensions]
        complexity = float(task.get("complexity", 0.5))
        return vector * (1.0 + complexity)

    def _forward(self, node: FractalNode, vector: np.ndarray, depth_remaining: int) -> np.ndarray:
        """Recursively propagate activations through the hierarchy."""
        activation = np.tanh(vector @ node.weights + node.bias)
        if depth_remaining <= 1 or not node.children:
            return activation
        aggregated = activation.copy()
        for child in node.children:
            child_vector = activation[: child.weights.shape[0]]
            aggregated[: child_vector.size] += self._forward(
                child, child_vector, depth_remaining - 1
            )
        return aggregated / (len(node.children) + 1.0)

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a recursive representation for the provided task."""
        vector = self._prepare_input(task)
        depth = min(self.recursion_limit, self.depth)
        representation = self._forward(self.root, vector, depth)
        self.output_history.append(representation.tolist())
        return {
            "representation": representation.tolist(),
            "depth_processed": depth,
            "activation_energy": float(np.linalg.norm(representation)),
            "history_length": len(self.output_history),
        }

    def get_state(self) -> Dict[str, Any]:
        """Return serialisable fractal engine state."""
        return {
            "config": {
                "max_depth": self.depth,
                "base_dimensions": self.base_dimensions,
                "branching_factor": self.branching_factor,
                "recursion_limit": self.recursion_limit,
            },
            "root": self.root.to_dict(),
            "history": self.output_history[-10:],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore fractal engine state from a persisted payload."""
        config = state.get("config", {})
        self.depth = int(config.get("max_depth", self.depth))
        self.base_dimensions = int(config.get("base_dimensions", self.base_dimensions))
        self.branching_factor = int(config.get("branching_factor", self.branching_factor))
        self.recursion_limit = int(config.get("recursion_limit", self.recursion_limit))
        self.root = FractalNode.from_dict(state["root"])
        self.output_history = [list(entry) for entry in state.get("history", [])]

    def get_status(self) -> Dict[str, Any]:
        """Return lightweight runtime metrics."""
        last_energy = float(np.linalg.norm(self.output_history[-1])) if self.output_history else 0.0
        return {
            "depth": self.depth,
            "base_dimensions": self.base_dimensions,
            "processed_tasks": len(self.output_history),
            "last_activation_energy": last_energy,
        }


def _factory(config: Dict[str, Any]) -> FractalEngine:
    """Factory used for plugin registration."""
    return FractalEngine(config)


plugin_registry.register("fractal_engine", _factory)
