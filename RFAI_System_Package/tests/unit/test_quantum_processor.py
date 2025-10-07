"""Unit tests for the quantum processor."""

from __future__ import annotations

import numpy as np

from quantum_processor import QuantumProcessor
from utils import QuantumProcessorConfig


def test_quantum_processor_respects_enabled_flag() -> None:
    disabled = QuantumProcessor(QuantumProcessorConfig(enabled=False, qubit_count=0, sampling_runs=1))
    output_disabled = disabled.run(np.array([0.1, 0.2]), 0.5)
    assert output_disabled.active is False

    enabled = QuantumProcessor(QuantumProcessorConfig(enabled=True, qubit_count=8, sampling_runs=16))
    output_enabled = enabled.run(np.array([0.1, 0.2]), 0.5)
    assert output_enabled.active is True
    assert 0 <= output_enabled.confidence <= 1
