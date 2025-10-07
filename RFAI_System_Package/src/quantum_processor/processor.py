"""Quantum processor subsystem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import QuantumProcessorConfig


@dataclass(frozen=True)
class QuantumProcessorOutput:
    """Quantum inference summary."""

    active: bool
    amplitude: float
    confidence: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "active": self.active,
            "amplitude": self.amplitude,
            "confidence": self.confidence,
        }


class QuantumProcessor:
    """Lightweight probabilistic processor."""

    def __init__(self, config: QuantumProcessorConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.qubit_count)

    def run(self, aggregated_signal: np.ndarray, complexity: float) -> QuantumProcessorOutput:
        if not self._config.enabled:
            return QuantumProcessorOutput(active=False, amplitude=0.0, confidence=0.0)

        amplitude = float(np.tanh(float(np.linalg.norm(aggregated_signal)) / (self._config.qubit_count or 1)))
        noise = float(self._rng.normal(0.0, 0.05))
        confidence = float(np.clip(1.0 - complexity / 1.5 + noise, 0.0, 1.0))
        return QuantumProcessorOutput(active=True, amplitude=amplitude, confidence=confidence)


def register_plugin(registry: "PluginRegistry") -> None:
    from utils import PluginRegistry

    if not isinstance(registry, PluginRegistry):  # pragma: no cover - defensive
        raise TypeError("registry must be a PluginRegistry instance")

    registry.register(
        "quantum_processor",
        "default",
        lambda config: QuantumProcessor(config),
    )


__all__ = ["QuantumProcessor", "QuantumProcessorOutput", "register_plugin"]
