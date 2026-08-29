"""Metacognition as the same operator, pointed at the system's own history.

This is the part the architecture exists to make cheap.  Because ``Psi`` does
not care what its evidence *means*, the record of the system's own episodes can
be fed to it directly: the history of episode features is decomposed across
*time* scale exactly as sensory input is decomposed across spatial scale, and a
field of "policy" hypotheses -- each one a bounded nudge to the operator's own
parameters -- resonates with it.  Slow drifts drive coarse-scale policies,
transients drive fine-scale ones, and the winner is applied.

Two guardrails keep this from being a self-amplifying loop:

* every parameter has hard bounds (``TUNABLE_PARAMETERS``) and every nudge is
  small, so no single reflection can restructure the system;
* the best-performing parameter set is remembered, and if reward degrades past
  a tolerance the system rolls back to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from .field import ResonantField, normalize_vector
from .operator import OperatorParams, ScaleInvariantOperator
from .scale_space import temporal_bands
from .telemetry import BAND_SLOTS, FEATURE_NAMES, Telemetry

#: Delta keys of this form route to the per-band trust vector rather than to a
#: scalar operator parameter.
BAND_TRUST_PREFIX = "band_trust:"

#: How far one reflection may move a single band's trust, in log2 units.
BAND_TRUST_STEP = 0.6

__all__ = [
    "MetaPolicy",
    "MetaConfig",
    "MetaReport",
    "MetaResonator",
    "DEFAULT_POLICIES",
]


@dataclass(frozen=True)
class MetaPolicy:
    """A parameter nudge plus the situation in which it is the right move.

    ``signature`` is a point in the same signed feature space the telemetry
    emits, so "does this policy fit what has been happening?" is a dot product
    and needs no extra machinery.
    """

    name: str
    signature: Tuple[float, ...]
    deltas: Dict[str, float]
    rationale: str = ""

    def vector(self) -> np.ndarray:
        """The signature as a unit vector, zero-padded to the feature width."""

        padded = np.zeros(len(FEATURE_NAMES), dtype=float)
        values = np.array(self.signature, dtype=float)[: padded.size]
        padded[: values.size] = values
        return normalize_vector(padded)

    def band(self) -> Optional[int]:
        """The band this policy adjusts, if it is a per-band trust policy."""

        for key in self.deltas:
            if key.startswith(BAND_TRUST_PREFIX):
                return int(key[len(BAND_TRUST_PREFIX) :])
        return None


# Feature order: coherence, decisiveness, depth, veto_pressure, field_pressure,
#                reward, scale_balance
DEFAULT_POLICIES: Tuple[MetaPolicy, ...] = (
    MetaPolicy(
        "hold",
        (0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0),
        {},
        "answers are landing; do not disturb the physics",
    ),
    MetaPolicy(
        "raise_coupling",
        (-1.0, -1.0, 0.0, 0.0, 0.0, -0.6, 0.0),
        {"coupling": 0.05},
        "field never agrees with itself; let hypotheses recruit each other harder",
    ),
    MetaPolicy(
        "lower_coupling",
        (1.0, 0.8, 0.0, 0.0, 0.0, -1.0, 0.0),
        {"coupling": -0.05},
        "field locks fast and is wrong; loosen consensus",
    ),
    MetaPolicy(
        "raise_damping",
        (0.8, 0.0, 0.0, 0.0, 1.0, -0.8, 0.0),
        {"damping": 0.02},
        "crowded and over-committed; forget faster",
    ),
    MetaPolicy(
        "lower_damping",
        (-1.0, -0.3, 0.8, 0.0, -0.6, -0.5, 0.0),
        {"damping": -0.02},
        "evidence keeps decaying before it can add up; hold state longer",
    ),
    MetaPolicy(
        "raise_drive_gain",
        (-0.5, -1.0, 1.0, 0.0, 0.0, -0.8, 0.0),
        {"drive_gain": 0.06},
        "recursion is doing the work the drive should do; listen harder",
    ),
    MetaPolicy(
        "lower_drive_gain",
        (1.0, 0.5, 0.0, 1.0, 0.0, -0.8, 0.0),
        {"drive_gain": -0.06},
        "chasing every input into a veto; listen less credulously",
    ),
    MetaPolicy(
        "raise_band_equalization",
        (0.0, -1.0, -0.8, 0.0, 0.0, -1.0, 0.0),
        {"band_equalization": 0.04},
        "ties are not breaking; give the quiet fine-scale evidence a real vote",
    ),
    MetaPolicy(
        "lower_band_equalization",
        (0.0, 0.4, 0.5, 1.0, 0.5, -0.8, 0.0),
        {"band_equalization": -0.04},
        "amplifying noise into confident nonsense; trust the loud bands again",
    ),
)


def _band_trust_policies() -> Tuple[MetaPolicy, ...]:
    """One policy per band: withdraw trust from the band that tips the errors.

    Each keys off that band's own credit slot and nothing else, so the decision
    is "has this band been deciding my mistakes?" rather than "does this band
    disagree with the others?".  The second question is the one that inverts
    when the unreliable bands are in the majority; the first cannot, because a
    band's credit is signed by the reward and never by a vote.

    Only distrust, deliberately.  The band weights are renormalised, so
    nothing depends on the absolute level of trust and raising every band in
    turn is an expensive no-op -- which is exactly what a trust-and-distrust
    pair produced in practice.  Withdrawing trust from whichever band is
    tipping the errors says the same thing unambiguously, and a band that
    recovers is re-trusted in relative terms as its neighbours are marked down.
    """

    policies: List[MetaPolicy] = []
    base = len(FEATURE_NAMES) - BAND_SLOTS
    for level in range(BAND_SLOTS):
        signature = [0.0] * len(FEATURE_NAMES)
        signature[base + level] = -1.0
        # Mild weight on reward so a healthy system leaves the ladder alone.
        signature[FEATURE_NAMES.index("reward")] = -0.35
        policies.append(
            MetaPolicy(
                f"distrust_band_{level}",
                tuple(signature),
                {f"{BAND_TRUST_PREFIX}{level}": -BAND_TRUST_STEP},
                f"band {level} has been tipping the answers that turned out wrong",
            )
        )
    return tuple(policies)


DEFAULT_POLICIES = DEFAULT_POLICIES + _band_trust_policies()


@dataclass
class MetaConfig:
    window: int = 16
    min_episodes: int = 8
    time_levels: int = 3
    steps: int = 12
    step_size: float = 1.0
    rollback_tolerance: float = 0.05
    expectation_decay: float = 0.03
    min_confidence: float = 0.2
    exploration: float = 0.25
    probe_parameters: Tuple[str, ...] = ("drive_gain", "damping", "coupling")


@dataclass
class MetaReport:
    applied: Optional[str] = None
    rationale: str = ""
    confidence: float = 0.0
    deltas: Dict[str, float] = dataclass_field(default_factory=dict)
    params: Dict[str, float] = dataclass_field(default_factory=dict)
    rolled_back: bool = False
    reason: str = ""
    distribution: Dict[str, float] = dataclass_field(default_factory=dict)


class MetaResonator:
    """Runs ``Psi`` over the system's own telemetry to retune ``Psi``."""

    def __init__(
        self,
        config: Optional[MetaConfig] = None,
        policies: Optional[Tuple[MetaPolicy, ...]] = None,
    ):
        self.config = config or MetaConfig()
        self.policies = tuple(policies or DEFAULT_POLICIES)
        self.best_params: Optional[OperatorParams] = None
        self.best_reward: float = -np.inf
        self.history: List[MetaReport] = []
        self._probe_index = 0
        self._probe_sign = 1.0
        self._probe_reward: Optional[float] = None

    @staticmethod
    def _apply(operator: ScaleInvariantOperator, deltas: Mapping[str, float]) -> None:
        """Route each delta to the scalar parameters or to the trust vector."""

        scalars = {
            key: value
            for key, value in deltas.items()
            if not key.startswith(BAND_TRUST_PREFIX)
        }
        if scalars:
            operator.params = operator.params.with_deltas(scalars)
        for key, value in deltas.items():
            if key.startswith(BAND_TRUST_PREFIX):
                level = int(key[len(BAND_TRUST_PREFIX) :])
                operator.params = operator.params.with_band_trust(level, value)

    def _probe(self, operator: ScaleInvariantOperator, reward: float) -> MetaReport:
        """Change something small and find out, instead of guessing.

        When no policy fits the situation confidently, the honest position is
        that the system does not know what is wrong -- and a bounded experiment
        beats a confident guess.  Direction is kept while it pays and reversed
        when it stops, and the rollback above undoes anything that makes things
        worse, so the worst case is a wasted window rather than a damaged
        system.
        """

        config = self.config
        if self._probe_reward is not None and reward < self._probe_reward:
            self._probe_sign = -self._probe_sign
            self._probe_index = (self._probe_index + 1) % len(config.probe_parameters)
        self._probe_reward = reward
        name = config.probe_parameters[self._probe_index]
        delta = self._probe_sign * config.exploration
        self._apply(operator, {name: delta})
        return MetaReport(
            applied=f"probe:{name}",
            rationale="no policy fitted; running a bounded experiment instead",
            deltas={name: delta},
            params=operator.params.as_dict(),
            reason="exploration",
        )

    def _check_rollback(
        self, operator: ScaleInvariantOperator, reward: float
    ) -> Optional[MetaReport]:
        """Restore the best-known parameters when reward has fallen away.

        The yardstick decays.  A high-water mark set before a regime change is
        unreachable afterwards, and comparing against it forever would make the
        system roll back at every reflection and never adapt again -- which is
        exactly what an undecayed best-ever baseline does.
        """

        config = self.config
        if self.best_params is not None:
            self.best_reward -= config.expectation_decay
        if self.best_params is None or reward > self.best_reward:
            self.best_params, self.best_reward = operator.params, reward
            return None
        if reward >= self.best_reward - config.rollback_tolerance:
            return None
        operator.params = self.best_params
        return MetaReport(
            rolled_back=True,
            reason=f"reward {reward:.3f} fell below best {self.best_reward:.3f}",
            params=operator.params.as_dict(),
        )

    # ------------------------------------------------------------------ logic
    def reflect(
        self, telemetry: Telemetry, operator: ScaleInvariantOperator
    ) -> MetaReport:
        config = self.config
        if len(telemetry) < config.min_episodes:
            return MetaReport(
                reason="not enough history", params=operator.params.as_dict()
            )

        reward = telemetry.mean_reward(config.window)
        rollback = self._check_rollback(operator, reward)
        if rollback is not None:
            self.history.append(rollback)
            return rollback

        features = telemetry.feature_matrix(config.window)
        if features.shape[0] < 2:
            return MetaReport(
                reason="not enough history", params=operator.params.as_dict()
            )

        bands = temporal_bands(
            features,
            min(config.time_levels, max(1, int(np.log2(features.shape[0])) + 1)),
        )
        field = ResonantField(dim=len(FEATURE_NAMES), max_size=4 * len(self.policies))
        for policy in self.policies:
            for band in bands:
                field.spawn(
                    policy.vector(),
                    scale=band.level,
                    label=policy.name,
                    amplitude=0.05,
                    origin="policy",
                    meta={"policy": policy.name},
                )

        # The same operator, the same physics -- only the evidence differs.
        operator.run(field, bands, config.steps)
        distribution = field.label_distribution()
        if not distribution:
            return MetaReport(
                reason="no policy resonated", params=operator.params.as_dict()
            )

        name, confidence = max(
            distribution.items(), key=lambda item: (item[1], item[0])
        )
        policy = next(p for p in self.policies if p.name == name)
        report = MetaReport(
            applied=None,
            rationale=policy.rationale,
            confidence=float(confidence),
            params=operator.params.as_dict(),
            distribution={key: float(value) for key, value in distribution.items()},
        )
        under_performing = reward < self.best_reward - config.expectation_decay
        if confidence < config.min_confidence or (
            not policy.deltas and under_performing
        ):
            if config.exploration > 0.0 and under_performing:
                probe = self._probe(operator, reward)
                probe.distribution = report.distribution
                self.history.append(probe)
                return probe
            report.reason = f"winning policy '{name}' below confidence floor"
            self.history.append(report)
            return report
        if not policy.deltas:
            report.applied = name
            report.reason = "policy is a no-op by design"
            self.history.append(report)
            return report

        scale = config.step_size * float(confidence)
        deltas = {key: value * scale for key, value in policy.deltas.items()}
        self._apply(operator, deltas)
        report.applied = name
        report.deltas = deltas
        report.params = operator.params.as_dict()
        self.history.append(report)
        return report
