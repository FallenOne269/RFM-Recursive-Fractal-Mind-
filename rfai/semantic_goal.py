from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _safe_norm(vec: np.ndarray) -> float:
    norm = float(np.linalg.norm(vec))
    return norm if norm > 1e-12 else 1e-12


@dataclass
class SemanticGoal:
    target_vector: np.ndarray

    def similarity(self, vec: np.ndarray) -> float:
        a = np.asarray(self.target_vector, dtype=float)
        b = np.asarray(vec, dtype=float)
        norm_a = _safe_norm(a)
        norm_b = _safe_norm(b)
        return float(np.dot(a, b) / (norm_a * norm_b))

