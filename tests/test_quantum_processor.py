"""Tests for the quantum processor subsystem."""

import pytest

from src.quantum_processor import QuantumProcessor


def test_quantum_processor_generates_normalised_state() -> None:
    processor = QuantumProcessor({"shots": 64, "decoherence": 0.1})
    result = processor.process({"fractal_output": {"final": [1.0, 0.5, -0.5]}})
    amplitudes = result["entangled_state"]
    assert pytest.approx(sum(value * value for value in amplitudes), rel=1e-6) == 1.0
    assert 0.0 <= result["stability"] <= 1.0
    assert result["shots"] == 64


def test_quantum_processor_handles_missing_signal() -> None:
    processor = QuantumProcessor()
    result = processor.process({})
    assert result["entangled_state"] == []
    assert result["stability"] == 0.0


def test_quantum_processor_invalid_config() -> None:
    with pytest.raises(ValueError):
        QuantumProcessor({"shots": 0})
