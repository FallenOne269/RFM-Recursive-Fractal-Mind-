"""Quantum processor subsystem."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

from ..utils.validation import QuantumProcessorConfig, ValidationError

logger = logging.getLogger(__name__)


class QuantumProcessor:
    """Simulate a lightweight quantum-classical hybrid processor."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        try:
            self.config = QuantumProcessorConfig(**(config or {}))
        except ValidationError as exc:  # pragma: no cover
            # pydantic validation is exercised elsewhere
            logger.error("Invalid quantum processor configuration: %s", exc)
            raise ValueError("Invalid quantum processor configuration") from exc

    def process(self, payload: Any) -> Dict[str, Any]:
        """Generate a probabilistic response from prior component outputs."""

        if not isinstance(payload, dict):
            logger.warning("Quantum processor received unexpected payload: %s", payload)
            return {"entangled_state": [], "stability": 0.0}

        fractal_values: List[float] = [
            float(value)
            for value in payload.get("fractal_output", {}).get("final", [])
        ]
        if not fractal_values:
            return {"entangled_state": [], "stability": 0.0}

        # Compress the classical signal into a pseudo quantum amplitude distribution.
        amplitudes = [math.tanh(value / (idx + 1)) for idx, value in enumerate(fractal_values)]
        norm = math.sqrt(sum(value * value for value in amplitudes)) or 1.0
        entangled_state = [value / norm for value in amplitudes]

        avg_amplitude = sum(entangled_state) / len(entangled_state)
        stability = max(0.0, min(1.0, 1 - self.config.decoherence - abs(avg_amplitude)))

        logger.debug(
            "Quantum processor generated stability %.3f with %s shots",
            stability,
            self.config.shots,
        )

        return {
            "entangled_state": entangled_state,
            "stability": stability,
            "shots": self.config.shots,
        }


__all__ = ["QuantumProcessor"]
