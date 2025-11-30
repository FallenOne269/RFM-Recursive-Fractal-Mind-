"""Swarm coordination subsystem."""

from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Any, Dict, List

from ..utils.validation import SwarmCoordinatorConfig, ValidationError

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """Aggregate component results using a virtual swarm of agents."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        try:
            self.config = SwarmCoordinatorConfig(**(config or {}))
        except ValidationError as exc:  # pragma: no cover
            # pydantic validation is exercised elsewhere
            logger.error("Invalid swarm coordinator configuration: %s", exc)
            raise ValueError("Invalid swarm coordinator configuration") from exc

    def process(self, fractal_output: Any) -> Dict[str, Any]:
        """Compute a consensus signal from fractal results."""

        if not isinstance(fractal_output, dict) or "final" not in fractal_output:
            logger.warning("Swarm coordinator received unexpected input: %s", fractal_output)
            return {
                "consensus": None,
                "agent_votes": [],
                "confidence": 0.0,
            }

        signal: List[float] = [float(value) for value in fractal_output["final"]]
        if not signal:
            return {
                "consensus": None,
                "agent_votes": [],
                "confidence": 0.0,
            }

        consensus_value = mean(signal)
        dispersion = pstdev(signal) if len(signal) > 1 else 0.0
        agent_votes = [
            {
                "agent_id": f"agent-{idx}",
                "vote": consensus_value + (value - consensus_value) * 0.1,
                "weight": 1 / self.config.agent_count,
            }
            for idx, value in enumerate(signal[: self.config.agent_count])
        ]

        confidence = max(
            0.0,
            min(
                1.0,
                self.config.consensus_threshold
                * (1 - dispersion / (abs(consensus_value) + 1)),
            ),
        )

        logger.debug(
            "Swarm consensus %.4f with dispersion %.4f and confidence %.2f",
            consensus_value,
            dispersion,
            confidence,
        )

        return {
            "consensus": consensus_value,
            "agent_votes": agent_votes,
            "confidence": confidence,
        }


__all__ = ["SwarmCoordinator"]
