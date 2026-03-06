from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Deque, Iterable, Optional, Tuple

import numpy as np

try:  # Optional GPU acceleration
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - cupy unavailable
    cp = None  # type: ignore


@dataclass
class TransformationParameters:
    """Parameters controlling AMIFS transformations."""

    scale: float = 0.8
    rotation: float = 0.0
    translation: Tuple[float, float] = (0.0, 0.0)
    adaptation_rate: float = 0.1
    sensitivity: float = 0.5
    context_weight: float = 0.2
    history_depth: int = 5
    cache_size: int = 64
    precision: str = "float64"

    def dtype(self) -> np.dtype:
        return np.dtype(self.precision)


class BaseTransformation:
    """Affine transformation supporting CPU/GPU arrays."""

    def __init__(self, params: TransformationParameters):
        self.params = params

    @property
    def xp(self):
        return cp if cp is not None else np

    def apply(self, points, params: Optional[TransformationParameters] = None):
        params = params or self.params
        xp = self.xp
        dtype = getattr(xp, params.dtype().name)
        points_arr = xp.asarray(points, dtype=dtype)
        theta = dtype(params.rotation)
        scale = dtype(params.scale)
        trans = xp.asarray(params.translation, dtype=dtype)
        rot_matrix = xp.asarray(
            [[xp.cos(theta), -xp.sin(theta)], [xp.sin(theta), xp.cos(theta)]],
            dtype=dtype,
        )
        transformed = points_arr @ rot_matrix.T
        transformed = transformed * scale + trans
        return transformed


class AdaptiveComponent:
    """Computes parameter deltas from feature/context summaries using caching."""

    def __init__(self, params: TransformationParameters):
        self.params = params
        self._cached = self._build_cache(params.cache_size)

    def _build_cache(self, size: int):
        @lru_cache(maxsize=size)
        def cached(summary: Tuple[float, float, float, float]):
            feature_mean, feature_var, ctx_mean, ctx_var = summary
            delta_scale = self.params.adaptation_rate * (feature_mean + ctx_mean)
            delta_rot = self.params.adaptation_rate * 0.5 * (feature_var - ctx_var)
            delta_trans = (
                self.params.adaptation_rate
                * self.params.sensitivity
                * (feature_mean - ctx_mean)
            )
            return delta_scale, delta_rot, delta_trans

        return cached

    def _summarize(self, features: np.ndarray, context: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        feat = np.asarray(features)
        ctx = np.asarray(context) if context is not None else np.zeros_like(feat)
        feature_mean = float(feat.mean()) if feat.size else 0.0
        feature_var = float(feat.var()) if feat.size else 0.0
        ctx_mean = float(ctx.mean()) if ctx.size else 0.0
        ctx_var = float(ctx.var()) if ctx.size else 0.0
        return feature_mean, feature_var, ctx_mean, ctx_var

    def compute_delta(self, features: np.ndarray, context: Optional[np.ndarray]) -> Tuple[float, float, float]:
        summary = self._summarize(features, context)
        return self._cached(summary)


class ContextualComponent:
    """Updates parameters based on historical context summaries."""

    def __init__(self, params: TransformationParameters):
        self.params = params
        self.history: Deque[float] = deque(maxlen=params.history_depth)

    def update(self, features: np.ndarray) -> float:
        feat = np.asarray(features)
        if feat.size:
            self.history.append(float(feat.mean()))
        if not self.history:
            return 0.0
        weight = self.params.context_weight
        weighted_avg = sum(val * weight for val in self.history) / len(self.history)
        return weighted_avg


class AMIFS:
    """Adaptive Multiscale Iterated Function System."""

    def __init__(self, params: Optional[TransformationParameters] = None):
        self.params = params or TransformationParameters()
        self.base_transformations = [
            BaseTransformation(self.params),
            BaseTransformation(
                TransformationParameters(
                    scale=self.params.scale * 0.9,
                    rotation=self.params.rotation + 0.2,
                    translation=(self.params.translation[0] + 0.2, self.params.translation[1]),
                    adaptation_rate=self.params.adaptation_rate,
                    sensitivity=self.params.sensitivity,
                    context_weight=self.params.context_weight,
                    history_depth=self.params.history_depth,
                    cache_size=self.params.cache_size,
                    precision=self.params.precision,
                )
            ),
            BaseTransformation(
                TransformationParameters(
                    scale=self.params.scale * 0.7,
                    rotation=self.params.rotation - 0.3,
                    translation=(
                        self.params.translation[0] - 0.1,
                        self.params.translation[1] + 0.3,
                    ),
                    adaptation_rate=self.params.adaptation_rate,
                    sensitivity=self.params.sensitivity,
                    context_weight=self.params.context_weight,
                    history_depth=self.params.history_depth,
                    cache_size=self.params.cache_size,
                    precision=self.params.precision,
                )
            ),
        ]
        self.adaptive_component = AdaptiveComponent(self.params)
        self.contextual_component = ContextualComponent(self.params)

    @property
    def xp(self):
        return cp if cp is not None else np

    def _generator(self, seed: int):
        return np.random.default_rng(seed)

    def generate(self, seed_points: np.ndarray, steps: int, rng_seed: int = 0):
        rng = self._generator(rng_seed)
        xp = self.xp
        dtype = getattr(xp, self.params.dtype().name)
        points = xp.asarray(seed_points, dtype=dtype)
        for _ in range(max(0, steps)):
            idx = rng.integers(0, len(self.base_transformations))
            transform = self.base_transformations[int(idx)]
            points = transform.apply(points, self.params)
        return points

    def adapt(self, features: np.ndarray, context: Optional[np.ndarray] = None) -> TransformationParameters:
        delta_scale, delta_rot, delta_trans = self.adaptive_component.compute_delta(features, context)
        context_delta = self.contextual_component.update(features)
        new_params = TransformationParameters(**vars(self.params))
        new_params.scale = max(0.1, self.params.scale + delta_scale + context_delta)
        new_params.rotation = self.params.rotation + delta_rot * (1 + context_delta)
        new_params.translation = (
            self.params.translation[0] + delta_trans,
            self.params.translation[1] + delta_trans * 0.5,
        )
        self.params = new_params
        return new_params

    def generate_fim_pattern(self, features: np.ndarray, context: Optional[np.ndarray], steps: int = 64):
        updated = self.adapt(features, context)
        seed_points = np.zeros((10, 2), dtype=updated.dtype())
        return self.generate(seed_points, steps=steps, rng_seed=0)


__all__ = [
    "TransformationParameters",
    "BaseTransformation",
    "AdaptiveComponent",
    "ContextualComponent",
    "AMIFS",
]
