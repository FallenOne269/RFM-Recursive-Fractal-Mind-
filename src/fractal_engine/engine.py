"""Implementation of the fractal processing engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..utils.validation import FractalEngineConfig, ValidationError, validate_input_payload

logger = logging.getLogger(__name__)


class FractalEngine:
    """Apply a recursive fractal transform to numeric sequences.

    The engine contracts each iteration towards the mean of the previous
    sequence, creating a stabilising fractal pattern.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        try:
            self.config = FractalEngineConfig(**(config or {}))
        except ValidationError as exc:  # pragma: no cover - pydantic already tested elsewhere
            logger.error("Invalid fractal engine configuration: %s", exc)
            raise ValueError("Invalid fractal engine configuration") from exc

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Run the fractal transformation pipeline."""

        values = validate_input_payload(input_data)
        depth = min(self.config.max_depth, self.config.recursion_limit)
        logger.debug("Running fractal transform for %s iterations", depth)

        history: List[List[float]] = []
        current = values
        for iteration in range(depth):
            mean = sum(current) / len(current)
            transformed = [
                mean + (value - mean) * self.config.scale_factor for value in current
            ]
            logger.debug("Iteration %s produced values: %s", iteration, transformed)
            history.append(transformed)
            current = transformed

        return {
            "initial": values,
            "history": history,
            "iterations": depth,
            "final": current,
        }


__all__ = ["FractalEngine"]
