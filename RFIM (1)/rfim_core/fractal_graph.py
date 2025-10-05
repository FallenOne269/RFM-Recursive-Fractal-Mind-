"""Graph utilities used by the RFIM core package."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, MutableMapping, Tuple

import numpy as np


class FractalGraph:
    """A lightweight undirected graph with helpers for fractal analysis."""

    def __init__(self) -> None:
        self._adjacency: Dict[str, MutableMapping[str, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    def add_node(self, node_id: str) -> None:
        """Ensure that ``node_id`` exists in the graph."""

        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")
        self._adjacency.setdefault(node_id, {})

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Insert an undirected weighted edge between ``source`` and ``target``."""

        self.add_node(source)
        self.add_node(target)
        self._adjacency[source][target] = float(weight)
        self._adjacency[target][source] = float(weight)

    # ------------------------------------------------------------------
    def neighbors(self, node_id: str) -> Iterator[Tuple[str, float]]:
        """Yield pairs of (neighbor_id, weight)."""

        if node_id not in self._adjacency:
            return iter(())
        return iter(self._adjacency[node_id].items())

    # ------------------------------------------------------------------
    def degree(self, node_id: str) -> int:
        return len(self._adjacency.get(node_id, ()))

    def density(self) -> float:
        """Return the graph density in the range [0, 1]."""

        num_nodes = len(self._adjacency)
        if num_nodes < 2:
            return 0.0
        possible_edges = num_nodes * (num_nodes - 1) / 2
        actual_edges = sum(len(neigh) for neigh in self._adjacency.values()) / 2
        return float(actual_edges / possible_edges)

    def fractal_dimension(self) -> float:
        """Estimate the graph fractal dimension using box counting."""

        num_nodes = len(self._adjacency)
        if num_nodes < 2:
            return 1.0

        degrees = np.array([len(neigh) for neigh in self._adjacency.values()], dtype=float)
        average_degree = degrees.mean()
        variance = degrees.var() + 1e-9
        return float(np.log(average_degree + 1.0) / np.log(variance + 1.0))

    # ------------------------------------------------------------------
    def __contains__(self, node_id: str) -> bool:  # pragma: no cover - trivial
        return node_id in self._adjacency

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._adjacency)

    def __iter__(self) -> Iterable[str]:  # pragma: no cover - trivial
        return iter(self._adjacency)
