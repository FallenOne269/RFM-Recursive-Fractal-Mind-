"""Fractal processing subsystem for the RFAI orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from utils import FractalEngineConfig


@dataclass(frozen=True)
class FractalEngineOutput:
    """Structured result from the fractal engine."""

    depth_reached: int
    aggregated_signal: np.ndarray
    level_outputs: List[np.ndarray]

    def to_dict(self) -> dict[str, float | list[float] | int]:
        """Serialize the output for JSON friendly transports."""

        return {
            "depth_reached": self.depth_reached,
            "aggregated_signal": self.aggregated_signal.tolist(),
            "level_outputs": [level.tolist() for level in self.level_outputs],
        }


class FractalEngine:
    """Hierarchical fractal processor.

    The engine applies recursive averaging to emulate multi-scale feature
    extraction. The implementation keeps the API intentionally small so it can
    be swapped with a production-grade neural system later.
    """

    def __init__(self, config: FractalEngineConfig) -> None:
        self._config = config

    def run(self, signal: Sequence[float]) -> FractalEngineOutput:
        """Process an input signal recursively."""

        vector = self._sanitize_signal(signal)
        levels = self._build_levels(vector)
        padded_levels = self._pad_levels(levels)
        aggregated = np.mean(padded_levels, axis=0)
        return FractalEngineOutput(
            depth_reached=len(levels),
            aggregated_signal=aggregated,
            level_outputs=list(padded_levels),
        )

    def _sanitize_signal(self, signal: Sequence[float]) -> np.ndarray:
        vector = np.asarray(list(signal), dtype=float)
        if vector.ndim != 1:
            raise ValueError("Fractal engine expects a one-dimensional signal.")
        if vector.size == 0:
            raise ValueError("Signal must contain at least one value.")
        return vector

    def _build_levels(self, vector: np.ndarray) -> List[np.ndarray]:
        levels: List[np.ndarray] = []
        current = vector
        for depth in range(self._config.max_depth):
            levels.append(current.copy())
            if current.size <= 1:
                break
            next_size = max(1, current.size // 2)
            reshaped = np.resize(current, next_size * 2).reshape(-1, 2)
            current = reshaped.mean(axis=1) + self._config.noise_scale * np.random.randn(next_size)
        return levels

    def _pad_levels(self, levels: List[np.ndarray]) -> np.ndarray:
        target_length = levels[0].size
        padded = [np.resize(level, target_length) for level in levels]
        return np.stack(padded, axis=0)


def register_plugin(registry: "PluginRegistry") -> None:
    """Register the default fractal engine implementation."""

    from utils import PluginRegistry  # Local import to avoid cycles

    if not isinstance(registry, PluginRegistry):  # pragma: no cover - defensive
        raise TypeError("registry must be a PluginRegistry instance")

    registry.register(
        "fractal_engine",
        "default",
        lambda config: FractalEngine(config),
    )


__all__ = ["FractalEngine", "FractalEngineOutput", "register_plugin"]
