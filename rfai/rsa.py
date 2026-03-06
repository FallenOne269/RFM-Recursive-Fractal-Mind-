from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import time
import numpy as np

from .adaptation_signal import AdaptationSignal
from .semantic_goal import SemanticGoal
from .fractal_graph import FractalGraph
from .dfe import DynamicFractalEncoder
from .smart_fractal_node import SmartFractalNode


@dataclass
class RSAConfig:
    max_depth: int = 5
    max_nodes: int = 50
    adapt_threshold: float = 0.1
    hysteresis: float = 0.05


class RecursiveStructuralAdapter:
    def __init__(self, dfe: DynamicFractalEncoder, graph: FractalGraph, config: RSAConfig | None = None):
        self.dfe = dfe
        self.graph = graph
        self.config = config or RSAConfig()
        self._last_action: Dict[str, str] = {}

    def evaluate_and_adapt(
        self, node_id: str, observed: Dict[str, float], target: Dict[str, float], goal: Optional[SemanticGoal] = None
    ) -> AdaptationSignal:
        error = 0.0
        for key, target_val in target.items():
            obs_val = observed.get(key, 0.0)
            error += (target_val - obs_val)
        goal_similarity = 0.0
        if goal is not None and "semantic_vector" in observed:
            goal_similarity = goal.similarity(np.asarray(observed["semantic_vector"]))
        delta = (error / max(1, len(target))) * (1 + goal_similarity)
        reason = "adaptation" if abs(delta) >= self.config.adapt_threshold else "stable"
        return AdaptationSignal(delta=float(delta), reason=reason, payload={"semantic_vector": observed.get("semantic_vector")})

    def apply_signal(self, node_id: str, signal: AdaptationSignal) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        previous_action = self._last_action.get(node_id, "none")
        node.apply_adaptation(signal)
        action = "none"
        if node.should_bifurcate(signal) and self._can_expand(node_id):
            action = "bifurcate"
            for child in node.bifurcate(self.dfe, signal.payload.get("semantic_vector", np.array([0.0]))):
                if len(self.graph.nodes) < self.config.max_nodes:
                    try:
                        self.graph.add_node(child, parent_id=node_id)
                    except ValueError:
                        break
        elif node.should_consolidate() and previous_action != "consolidate":
            action = "consolidate"
            parent_id = self.graph.parent(node_id)
            if parent_id:
                # merge complexity to parent and remove node
                parent = self.graph.nodes[parent_id]
                parent.complexity = (parent.complexity + node.complexity) / 2
                self._remove_node(node_id)
        self._last_action[node_id] = action

    def _can_expand(self, node_id: str) -> bool:
        depth = self.graph.depth(node_id)
        return depth + 1 <= self.config.max_depth and len(self.graph.nodes) < self.config.max_nodes

    def _remove_node(self, node_id: str) -> None:
        self.graph.nodes.pop(node_id, None)
        parent = self.graph.parent_map.pop(node_id, None)
        self.graph.children_map.pop(node_id, None)
        if parent and node_id in self.graph.children_map.get(parent, []):
            self.graph.children_map[parent].remove(node_id)

