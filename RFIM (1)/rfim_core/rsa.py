"""Recursive Structural Adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AdaptationResult:
    structure: np.ndarray
    delta: np.ndarray
    stability: float


class RecursiveStructuralAdapter:
    """Incrementally adapt fractal information matrices."""

    def __init__(self, learning_rate: float = 0.2, regularisation: float = 1e-6) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if regularisation < 0:
            raise ValueError("regularisation must be non-negative")
        self.learning_rate = float(learning_rate)
        self.regularisation = float(regularisation)
        self._structure: Optional[np.ndarray] = None

    def adapt(self, fim: np.ndarray) -> AdaptationResult:
        """Blend ``fim`` into the internal structure and return adaptation details."""

        matrix = np.asarray(fim, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("fim must be a square matrix")

        if self._structure is None:
            self._structure = matrix.copy()
            delta = np.zeros_like(matrix)
        else:
            delta = matrix - self._structure
            self._structure += self.learning_rate * delta

        self._structure += np.eye(self._structure.shape[0]) * self.regularisation
        stability = float(np.linalg.norm(delta))
        return AdaptationResult(structure=self._structure.copy(), delta=delta, stability=stability)
