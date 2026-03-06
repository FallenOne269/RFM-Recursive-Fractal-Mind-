from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .smart_fractal_node import SmartFractalNode


class FractalGraph:
    """Simple parent/child graph with depth limits."""

    def __init__(self, max_depth: int = 5):
        self.nodes: Dict[str, SmartFractalNode] = {}
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}
        self.max_depth = max_depth

    def add_node(self, node: SmartFractalNode, parent_id: Optional[str] = None) -> None:
        if parent_id is not None and self.depth(parent_id) + 1 > self.max_depth:
            raise ValueError("max depth exceeded")
        self.nodes[node.node_id] = node
        self.parent_map[node.node_id] = parent_id
        self.children_map.setdefault(node.node_id, [])
        if parent_id is not None:
            self.children_map.setdefault(parent_id, []).append(node.node_id)

    def children(self, node_id: str) -> List[str]:
        return list(self.children_map.get(node_id, []))

    def parent(self, node_id: str) -> Optional[str]:
        return self.parent_map.get(node_id)

    def depth(self, node_id: str) -> int:
        depth = 0
        current = node_id
        while current in self.parent_map and self.parent_map[current] is not None:
            depth += 1
            current = self.parent_map[current]  # type: ignore
        return depth

    def propagate(self, signal, start_node_id: str, mode: str = "updown", max_hops: int = 3):
        visited: List[Tuple[str, object]] = []
        if start_node_id not in self.nodes:
            return visited
        visited.append((start_node_id, signal))
        if mode in {"up", "updown"}:
            parent = self.parent(start_node_id)
            hops = 0
            while parent is not None and hops < max_hops:
                visited.append((parent, signal))
                parent = self.parent(parent)
                hops += 1
        if mode in {"down", "updown"}:
            queue = [(start_node_id, 0)]
            while queue:
                nid, depth = queue.pop(0)
                if depth >= max_hops:
                    continue
                for child in self.children(nid):
                    visited.append((child, signal))
                    queue.append((child, depth + 1))
        return visited

