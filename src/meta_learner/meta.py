"""Meta learning subsystem."""

from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Dict

from ..utils.validation import MetaLearnerConfig, ValidationError

logger = logging.getLogger(__name__)


class MetaLearner:
    """Synthesize a holistic view of orchestrator outputs."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        try:
            self.config = MetaLearnerConfig(**(config or {}))
        except ValidationError as exc:  # pragma: no cover - pydantic validation is exercised elsewhere
            logger.error("Invalid meta learner configuration: %s", exc)
            raise ValueError("Invalid meta learner configuration") from exc

    def process(self, payload: Any) -> Dict[str, Any]:
        """Produce adaptive insights from all component outputs."""

        if not isinstance(payload, dict):
            logger.warning("Meta learner received unexpected payload: %s", payload)
            return {"score": 0.0, "recommendations": []}

        scores = []
        if (fractal := payload.get("fractal_output")) and isinstance(fractal, dict):
            final = fractal.get("final") or []
            if final:
                scores.append(mean(final))
        if (swarm := payload.get("swarm_output")) and isinstance(swarm, dict):
            if (confidence := swarm.get("confidence")) is not None:
                scores.append(float(confidence))
        if (quantum := payload.get("quantum_output")) and isinstance(quantum, dict):
            if (stability := quantum.get("stability")) is not None:
                scores.append(float(stability))

        if not scores:
            return {"score": 0.0, "recommendations": ["Collect more data"]}

        aggregate_score = mean(scores)
        adjusted_score = aggregate_score * (1 + self.config.momentum * self.config.learning_rate)

        recommendations = [
            "Maintain current strategy" if adjusted_score >= 0.5 else "Increase exploration",
            "Monitor confidence" if payload.get("swarm_output", {}).get("confidence", 0) < 0.5 else "Leverage consensus",
        ]

        logger.debug("Meta learner score %.3f", adjusted_score)

        return {
            "score": min(1.0, max(0.0, adjusted_score)),
            "recommendations": recommendations,
        }


__all__ = ["MetaLearner"]
