"""Episode records and the metrics computed over them."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Deque, Dict, List, Optional

import numpy as np

__all__ = ["EpisodeRecord", "Telemetry", "FEATURE_NAMES"]

#: Signed deviation features that metacognition perceives.  Positive means
#: "more than the healthy operating point", negative means "less".
#: Scaling on the reward-weighted scale-agreement signal (see ``features``).
_AGREEMENT_GAIN = 4.0

FEATURE_NAMES = (
    "coherence",  # +: field locks hard      -: field stays scattered
    "decisiveness",  # +: clear winner          -: leaders stay tied
    "depth",  # +: recursion used a lot  -: recursion barely needed
    "veto_pressure",  # +: constraints firing    -: constraints quiet
    "field_pressure",  # +: population crowded    -: population sparse
    "reward",  # +: answers landing       -: answers missing
    "scale_balance",  # +: only fine scales work -: only coarse scales work
)


@dataclass
class EpisodeRecord:
    """Everything one ``perceive`` call is willing to say about itself."""

    index: int
    label: str
    confidence: float
    coherence: float
    decisiveness: float
    depth_used: int
    max_depth: int
    steps: int
    field_size: int
    max_field_size: int
    vetoes: int
    symbols: int
    residual_norm: float
    scale_balance: float = 0.0
    scale_agreement: float = 0.0
    reward: Optional[float] = None
    params: Dict[str, float] = dataclass_field(default_factory=dict)

    def features(
        self,
        target_coherence: float = 0.75,
        target_decisiveness: float = 0.35,
        target_veto_rate: float = 0.1,
        reward_baseline: float = 0.5,
    ) -> np.ndarray:
        depth_saturation = self.depth_used / max(1, self.max_depth)
        veto_rate = self.vetoes / max(1, self.field_size)
        field_pressure = self.field_size / max(1, self.max_field_size)
        # Reward is scored against how the system has been doing lately, not
        # against a fixed target.  An absolute target reads "0.66 accuracy" as
        # success and hides the fact that it used to be 0.99.
        reward = reward_baseline if self.reward is None else float(self.reward)
        # Which half of the ladder to believe.  Without feedback all the system
        # can do is notice that the halves disagree; with feedback it can tell
        # which half was agreeing with its mistakes, which is a much sharper
        # signal and the one worth acting on.
        if self.reward is None:
            balance = self.scale_balance
        else:
            balance = float(
                np.clip(
                    _AGREEMENT_GAIN
                    * self.scale_agreement
                    * (2.0 * float(self.reward) - 1.0),
                    -1.0,
                    1.0,
                )
            )
        return np.array(
            [
                self.coherence - target_coherence,
                self.decisiveness - target_decisiveness,
                depth_saturation - 0.5,
                veto_rate - target_veto_rate,
                field_pressure - 0.5,
                reward - reward_baseline,
                balance,
            ],
            dtype=float,
        )

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class Telemetry:
    """Bounded episode history plus summary statistics."""

    def __init__(self, maxlen: int = 256):
        self.records: Deque[EpisodeRecord] = deque(maxlen=maxlen)

    def add(self, record: EpisodeRecord) -> EpisodeRecord:
        self.records.append(record)
        return record

    def __len__(self) -> int:
        return len(self.records)

    def recent(self, count: int) -> List[EpisodeRecord]:
        if count <= 0:
            return []
        return list(self.records)[-count:]

    def feature_matrix(self, count: int) -> np.ndarray:
        baseline = self.mean_reward(count * 4)
        rows = [
            record.features(reward_baseline=baseline) for record in self.recent(count)
        ]
        if not rows:
            return np.zeros((0, len(FEATURE_NAMES)))
        return np.stack(rows)

    def mean_reward(self, count: int) -> float:
        rewards = [r.reward for r in self.recent(count) if r.reward is not None]
        if not rewards:
            return 0.5
        return float(np.mean(rewards))

    def summary(self, count: int = 32) -> Dict[str, float]:
        records = self.recent(count)
        if not records:
            return {}
        return {
            "episodes": float(len(records)),
            "mean_coherence": float(np.mean([r.coherence for r in records])),
            "mean_confidence": float(np.mean([r.confidence for r in records])),
            "mean_decisiveness": float(np.mean([r.decisiveness for r in records])),
            "mean_depth": float(np.mean([r.depth_used for r in records])),
            "mean_reward": self.mean_reward(count),
            "symbols": float(records[-1].symbols),
        }
