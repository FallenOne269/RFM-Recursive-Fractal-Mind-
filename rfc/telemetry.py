"""Episode records and the metrics computed over them."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["EpisodeRecord", "Telemetry", "FEATURE_NAMES"]

#: Signed deviation features that metacognition perceives.  Positive means
#: "more than the healthy operating point", negative means "less".
#: Scaling on the reward-weighted scale-agreement signal (see ``features``).
_AGREEMENT_GAIN = 4.0

#: Floor on the estimated success rate when balancing outcome classes, so a
#: window that is all right or all wrong cannot produce an unbounded weight.
_CLASS_FLOOR = 0.1

#: How many rungs of the scale ladder metacognition carries features for.  The
#: policy signatures are fixed-length, so this fixes the width; ladders with
#: fewer bands leave the spare slots at zero, wider ones are truncated.
BAND_SLOTS = 4

FEATURE_NAMES = (
    "coherence",  # +: field locks hard      -: field stays scattered
    "decisiveness",  # +: clear winner          -: leaders stay tied
    "depth",  # +: recursion used a lot  -: recursion barely needed
    "veto_pressure",  # +: constraints firing    -: constraints quiet
    "field_pressure",  # +: population crowded    -: population sparse
    "reward",  # +: answers landing       -: answers missing
    "scale_balance",  # +: only fine scales work -: only coarse scales work
) + tuple(f"credit_band_{level}" for level in range(BAND_SLOTS))
# The trailing per-band slots carry reward-supervised credit: how much each
# band is associated with this system being right rather than wrong.  They are
# what lets metacognition indict a band without asking the other bands, which
# is the one thing consensus cannot do when the bad bands are the majority.


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
    #: Per band: 1.0 if dropping that band would have changed the answer.
    #: Paired with reward below, this is credit assignment; on its own it is
    #: nothing.
    band_pivotal: Tuple[float, ...] = ()
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
                *self._band_credit(reward_baseline),
            ],
            dtype=float,
        )

    def _band_credit(self, reward_baseline: float) -> List[float]:
        """Counterfactual credit for each band, graded by the reward.

        A band scores positively when the answer was right and would have come
        out differently without it -- it carried a correct decision -- and
        negatively when the answer was wrong and it was what tipped it.  A band
        that made no difference scores zero, which is most of them most of the
        time.

        Two corrections make the raw counterfactual usable:

        *Class balance.*  Most episodes are right, so an unweighted average is
        dominated by successes and ends up rewarding whichever bands swing the
        answer around most -- which flatters an erratic band.  Each episode is
        weighted by the inverse frequency of its outcome, so being decisive for
        a rare error counts as heavily as being decisive for a common success.

        *Centring.*  Errors are exactly where the read-out is contested, so on a
        failing stream *some* band is pivotal almost every time and every band's
        raw score drifts negative together.  Only the ranking carries
        information, so the ladder mean is subtracted.  This removes a common
        offset from a reward-graded statistic; it is not the bands scoring each
        other's opinions, which is the move that inverts once the unreliable
        bands are in the majority.

        The signal goes quiet on its own when the system is healthy: with the
        answers coming out right, no band is ever the one that tipped a wrong
        answer, and a confident read-out rarely turns on any single band.
        """

        credit = [0.0] * BAND_SLOTS
        if self.reward is None or not self.band_pivotal:
            return credit
        pivotal = [float(value) for value in self.band_pivotal[:BAND_SLOTS]]
        if not pivotal:
            return credit

        correct = float(self.reward) >= 0.5
        share = float(np.clip(reward_baseline, _CLASS_FLOOR, 1.0 - _CLASS_FLOOR))
        weight = 0.5 / (share if correct else 1.0 - share)
        sign = 1.0 if correct else -1.0
        centre = float(np.mean(pivotal))

        for level, value in enumerate(pivotal):
            credit[level] = float(np.clip(sign * weight * (value - centre), -1.0, 1.0))
        return credit

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
