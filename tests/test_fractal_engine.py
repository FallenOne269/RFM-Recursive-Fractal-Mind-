"""Tests for the fractal engine subsystem."""

import pytest

from src.fractal_engine import FractalEngine


def test_fractal_engine_respects_recursion_limit() -> None:
    engine = FractalEngine({"max_depth": 5, "recursion_limit": 2, "scale_factor": 0.5})
    result = engine.process([1, 2, 3])
    assert result["iterations"] == 2
    assert len(result["history"]) == 2


@pytest.mark.parametrize("payload", ["invalid", {"values": []}, {"foo": 1}])
def test_fractal_engine_rejects_invalid_input(payload: object) -> None:
    engine = FractalEngine()
    with pytest.raises(ValueError):
        engine.process(payload)
