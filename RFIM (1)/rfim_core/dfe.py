"""Dynamic Fractal Encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[float], Sequence[Sequence[float]], np.ndarray]


@dataclass
class FeatureStatistics:
    """Summary statistics extracted by a feature encoder."""

    mean: np.ndarray
    variance: np.ndarray
    energy: float


class DynamicFractalEncoder:
    """Convert data streams into fractal information matrices (FIMs)."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self._extractors: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    # ------------------------------------------------------------------
    def register_feature_extractor(self, name: str, extractor: Callable[[np.ndarray], ArrayLike]) -> None:
        """Register a new feature extractor under ``name``."""

        if not isinstance(name, str) or not name:
            raise ValueError("Extractor name must be a non-empty string")
        if not callable(extractor):
            raise TypeError("extractor must be callable")
        self._extractors[name] = extractor  # Overwriting is intentional for ergonomics.

    # ------------------------------------------------------------------
    def create_fim(self) -> np.ndarray:
        """Return an empty fractal information matrix."""

        return np.zeros((self.dimension, self.dimension), dtype=float)

    # ------------------------------------------------------------------
    def _prepare_features(self, features: ArrayLike) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _statistics(self, array: np.ndarray) -> FeatureStatistics:
        mean = array.mean(axis=0)
        variance = array.var(axis=0)
        energy = float(np.linalg.norm(array))
        return FeatureStatistics(mean=mean, variance=variance, energy=energy)

    # ------------------------------------------------------------------
    def encode(self, data: ArrayLike, extractor_name: str) -> Mapping[str, Union[np.ndarray, FeatureStatistics]]:
        """Encode ``data`` using the extractor referenced by ``extractor_name``."""

        if extractor_name not in self._extractors:
            raise KeyError(f"Unknown feature extractor '{extractor_name}'")

        data_array = np.asarray(data, dtype=float)
        if data_array.size == 0:
            raise ValueError("Cannot encode empty data arrays")

        extractor = self._extractors[extractor_name]
        features = self._prepare_features(extractor(data_array))
        stats = self._statistics(features)

        fim = self.create_fim()
        scaled_features = np.resize(stats.mean, self.dimension)
        fim += np.outer(scaled_features, scaled_features)
        fim += np.diag(np.resize(stats.variance, self.dimension))
        fim /= max(stats.energy, 1e-9)

        return {
            "fim": fim,
            "statistics": stats,
            "extractor": extractor_name,
        }

    # ------------------------------------------------------------------
    def decode(self, encoded_fims: Iterable[Mapping[str, Union[np.ndarray, FeatureStatistics]]], shape: Tuple[int, ...]) -> np.ndarray:
        """Decode a sequence of encoded FIMs back into an approximate data tensor."""

        mats = []
        for entry in encoded_fims:
            fim = np.asarray(entry["fim"], dtype=float)
            if fim.shape != (self.dimension, self.dimension):
                raise ValueError("FIM shape mismatch during decode")
            mats.append(fim)

        if not mats:
            raise ValueError("encoded_fims must contain at least one element")

        averaged = np.mean(mats, axis=0)
        flattened = averaged.mean(axis=0)
        resized = np.resize(flattened, int(np.prod(shape)))
        return resized.reshape(shape)
