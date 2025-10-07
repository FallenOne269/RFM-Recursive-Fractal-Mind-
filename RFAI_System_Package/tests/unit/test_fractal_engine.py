"""Unit tests for the fractal engine."""

from __future__ import annotations

import numpy as np

from fractal_engine import FractalEngine
from utils import FractalEngineConfig


def test_fractal_engine_produces_levels() -> None:
    engine = FractalEngine(FractalEngineConfig(max_depth=3, base_dimensions=8, noise_scale=0.0))
    output = engine.run([1.0, 2.0, 3.0, 4.0])
    assert output.depth_reached >= 1
    assert len(output.level_outputs) == output.depth_reached
    assert output.aggregated_signal.shape[0] == len(output.level_outputs[0])
    assert not np.isnan(output.aggregated_signal).any()
