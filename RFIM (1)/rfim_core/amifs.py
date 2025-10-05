"""Implementation of the Adaptive Multiscale Iterated Function System (AMIFS)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _ensure_2d_array(data: Optional[Sequence[Sequence[float]]], dimension: int) -> Optional[np.ndarray]:
    """Return ``data`` as a 2-D float array with ``dimension`` columns.

    ``None`` inputs are propagated.  Smaller vectors are tiled to the requested
    dimensionality which keeps the function permissive for simple test cases
    while still being deterministic.
    """

    if data is None:
        return None

    array = np.asarray(data, dtype=float)
    if array.size == 0:
        return np.zeros((1, dimension), dtype=float)

    array = np.atleast_2d(array)
    if array.shape[1] != dimension:
        array = np.resize(array, (array.shape[0], dimension))
    return array


@dataclass(frozen=True)
class _LinearTransform:
    matrix: np.ndarray
    translation: np.ndarray

    def apply(self, vector: np.ndarray) -> np.ndarray:
        return vector @ self.matrix.T + self.translation


class AMIFS:
    """Adaptive Multiscale Iterated Function System implementation.

    The class exposes a compact API tailored for the unit tests contained in the
    repository while still providing behaviour that is useful for lightweight
    experimentation.  The internal representation uses a small collection of
    contractive affine transforms that are deterministic given the constructor
    arguments which keeps the output reproducible across runs.
    """

    def __init__(
        self,
        num_functions: int = 3,
        dimension: int = 2,
        transforms: Optional[Iterable[Tuple[Sequence[float], Sequence[float]]]] = None,
    ) -> None:
        if num_functions <= 0:
            raise ValueError("num_functions must be a positive integer")
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")

        self.dimension = int(dimension)
        self.num_functions = int(num_functions)
        self._transforms = self._build_transforms(transforms)

    # ------------------------------------------------------------------
    # Helper utilities
    def _build_transforms(
        self,
        transforms: Optional[Iterable[Tuple[Sequence[float], Sequence[float]]]],
    ) -> Tuple[_LinearTransform, ...]:
        if transforms is None:
            return self._default_transforms()

        prepared = []
        for entry in transforms:
            matrix, translation = entry
            matrix_arr = np.asarray(matrix, dtype=float)
            translation_arr = np.asarray(translation, dtype=float)
            if matrix_arr.shape != (self.dimension, self.dimension):
                raise ValueError(
                    "Custom transform matrices must be of shape "
                    f"({self.dimension}, {self.dimension})"
                )
            if translation_arr.shape not in {(self.dimension,), (self.dimension, 1)}:
                raise ValueError("Translation vectors must match the system dimension")
            prepared.append(_LinearTransform(matrix_arr, translation_arr.reshape(self.dimension)))

        if len(prepared) != self.num_functions:
            raise ValueError("Number of custom transforms must equal num_functions")

        return tuple(prepared)

    def _default_transforms(self) -> Tuple[_LinearTransform, ...]:
        base = []
        scales = np.linspace(0.35, 0.65, self.num_functions)
        for idx, scale in enumerate(scales):
            matrix = np.eye(self.dimension) * scale
            translation = np.zeros(self.dimension)
            translation[idx % self.dimension] = 1.0 - scale
            base.append(_LinearTransform(matrix, translation))
        return tuple(base)

    def _seed_from_state(
        self,
        features: Optional[np.ndarray],
        system_state: Optional[Mapping[str, float]],
    ) -> int:
        hasher = hashlib.sha256()
        hasher.update(str(self.dimension).encode())
        hasher.update(str(self.num_functions).encode())
        if features is not None:
            hasher.update(features.tobytes())
        if system_state:
            for key in sorted(system_state):
                hasher.update(key.encode())
                hasher.update(str(system_state[key]).encode())
        return int.from_bytes(hasher.digest()[:8], "little", signed=False)

    # ------------------------------------------------------------------
    # Public API
    def generate_attractor(
        self,
        num_points: int,
        features: Optional[Sequence[Sequence[float]]] = None,
        system_state: Optional[Mapping[str, float]] = None,
        iterations: int = 32,
    ) -> np.ndarray:
        """Generate ``num_points`` samples from the attractor defined by the IFS."""

        if num_points <= 0:
            raise ValueError("num_points must be a positive integer")

        prepared_features = _ensure_2d_array(features, self.dimension)
        rng = np.random.default_rng(self._seed_from_state(prepared_features, system_state))
        point = np.zeros(self.dimension, dtype=float)

        # Burn-in phase so that the orbit converges towards the attractor.
        for _ in range(max(iterations, 0)):
            transform = self._transforms[rng.integers(0, len(self._transforms))]
            point = transform.apply(point)

        samples = np.empty((num_points, self.dimension), dtype=float)
        for index in range(num_points):
            transform = self._transforms[rng.integers(0, len(self._transforms))]
            point = transform.apply(point)
            if prepared_features is not None and prepared_features.size:
                feature_vector = prepared_features[index % prepared_features.shape[0]]
                point += 0.05 * np.resize(feature_vector, self.dimension)
            samples[index] = point
        return samples

    def computational_resonance(
        self,
        data: Sequence[Sequence[float]],
        feature_extractor,
        state_generator=None,
        iterations: int = 64,
    ) -> Mapping[str, np.ndarray]:
        """Run the AMIFS resonance loop for ``data``.

        Parameters
        ----------
        data:
            Iterable of data samples.  Each element must be convertible to a
            float vector.
        feature_extractor:
            Callable used to obtain feature vectors from ``data``.
        state_generator:
            Optional callable that returns a mapping representing the system
            state.  The information is folded into the RNG seed which keeps the
            process deterministic.
        iterations:
            Number of warm-up iterations executed before collecting samples.
        """

        if not callable(feature_extractor):
            raise TypeError("feature_extractor must be callable")
        if state_generator is not None and not callable(state_generator):
            raise TypeError("state_generator must be callable when provided")

        data_array = np.asarray(list(data), dtype=float)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, self.dimension)

        features = _ensure_2d_array(feature_extractor(data_array), self.dimension)
        state = state_generator(data_array) if state_generator else None
        attractor = self.generate_attractor(
            num_points=len(data_array),
            features=features,
            system_state=state,
            iterations=iterations,
        )

        mean_feature = np.resize(np.mean(features, axis=0) if features is not None else np.zeros(self.dimension), self.dimension)
        mean_attractor = attractor.mean(axis=0)
        resonance = float(
            np.dot(mean_feature, mean_attractor)
            / (np.linalg.norm(mean_feature) * np.linalg.norm(mean_attractor) + 1e-12)
        )

        return {
            "attractor": attractor,
            "resonance": resonance,
            "state": state,
            "features": features,
        }
