"""Quantum-classical hybrid processing simulation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.base import BaseSubsystem
from ..core.registry import plugin_registry


class QuantumProcessor(BaseSubsystem):
    """Light-weight simulation of a quantum enhancement pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled: bool = bool(self.config.get("enabled", True))
        self.qubits: int = int(self.config.get("qubits", 8))
        self.noise_level: float = float(self.config.get("noise_level", 0.02))
        self.entanglement_bias: float = float(self.config.get("entanglement_bias", 0.5))
        seed = self.config.get("seed")
        self.rng = np.random.default_rng(seed)
        self.state_vector: np.ndarray = self._initialise_state()
        self.entropy_history: list[float] = []

    def _initialise_state(self) -> np.ndarray:
        """Initialise a randomised quantum state vector."""
        if not self.enabled:
            return np.zeros(2**self.qubits)
        vector = self.rng.normal(0.0, 1.0, size=2**self.qubits) + 1j * self.rng.normal(
            0.0, 1.0, size=2**self.qubits
        )
        vector /= np.linalg.norm(vector)
        return vector

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply simulated quantum processing to the task."""
        if not self.enabled:
            return {
                "active": False,
                "state_projection": [],
                "entanglement_entropy": 0.0,
            }
        complexity = float(task.get("complexity", 0.5))
        phase_shift = np.exp(1j * complexity * np.pi * self.entanglement_bias)
        noise = self.rng.normal(0.0, self.noise_level, size=self.state_vector.shape)
        updated_state = self.state_vector * phase_shift + noise
        updated_state /= np.linalg.norm(updated_state)
        probabilities = np.abs(updated_state) ** 2
        entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))
        self.entropy_history.append(entropy)
        self.state_vector = updated_state
        sample = probabilities[: min(16, probabilities.size)].tolist()
        return {
            "active": True,
            "state_projection": sample,
            "entanglement_entropy": entropy,
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialise quantum state and configuration."""
        return {
            "config": {
                "enabled": self.enabled,
                "qubits": self.qubits,
                "noise_level": self.noise_level,
                "entanglement_bias": self.entanglement_bias,
            },
            "state_vector": [
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                }
                for value in self.state_vector
            ],
            "entropy_history": list(self.entropy_history[-20:]),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the quantum processor from a saved payload."""
        config = state.get("config", {})
        self.enabled = bool(config.get("enabled", self.enabled))
        self.qubits = int(config.get("qubits", self.qubits))
        self.noise_level = float(config.get("noise_level", self.noise_level))
        self.entanglement_bias = float(config.get("entanglement_bias", self.entanglement_bias))
        vector = [
            complex(entry.get("real", 0.0), entry.get("imag", 0.0))
            for entry in state.get("state_vector", [])
        ]
        self.state_vector = np.array(vector, dtype=complex)
        if self.state_vector.size == 0:
            self.state_vector = self._initialise_state()
        self.entropy_history = list(state.get("entropy_history", []))

    def get_status(self) -> Dict[str, Any]:
        """Return lightweight runtime diagnostics."""
        entropy = self.entropy_history[-1] if self.entropy_history else 0.0
        return {
            "enabled": self.enabled,
            "qubits": self.qubits,
            "last_entropy": float(entropy),
        }


def _factory(config: Dict[str, Any]) -> QuantumProcessor:
    """Factory used for plugin registration."""
    return QuantumProcessor(config)


plugin_registry.register("quantum_processor", _factory)
