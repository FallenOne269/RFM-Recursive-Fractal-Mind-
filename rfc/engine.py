"""Resonant Fractal Cognition -- the assembled system.

``ResonantFractalCognition`` wires the substrate together:

    evidence -> scale bands -> resonant field -> (constraints) -> read-out
                                   |                 |
                                   |                 +-> symbol lattice -> priors
                                   +-> ambiguity? -> recurse on the residual
                                                        with the same operator

and, between episodes, feeds its own telemetry back through the same operator
to retune itself.  Everything is deterministic given a seed: no hidden clocks,
no unseeded randomness, and a stable iteration order everywhere it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .constraints import ConstraintField, Invariant
from .field import ResonantField, normalize_vector
from .lattice import LatticeConfig, SymbolLattice
from .metacognition import MetaConfig, MetaReport, MetaResonator
from .operator import OperatorParams, ScaleInvariantOperator, StepReport
from .scale_space import dyadic_decompose
from .telemetry import EpisodeRecord, Telemetry

__all__ = ["RFCConfig", "Percept", "ResonantFractalCognition"]

_GOLDEN = 0.6180339887498949


def _band_pivotal(field: ResonantField, answer: str) -> Tuple[float, ...]:
    """Per band: would the answer have come out differently without it?

    Recorded raw and unjudged.  It only becomes evidence about a band once the
    episode's reward says whether that answer was worth deciding.
    """

    if not answer:
        return ()
    pivotal = field.band_pivotality(answer)
    return tuple(float(pivotal[level]) for level in sorted(pivotal))


@dataclass
class RFCConfig:
    dim: int = 32
    levels: int = 4
    steps: int = 24
    max_depth: int = 2
    ambiguity_threshold: float = 0.18
    residual_floor: float = 0.05
    recursion_boost: float = 0.6
    recursion_steps: int = 8
    max_field_size: int = 96
    seed_amplitude: float = 0.05
    phase_dispersal: float = 0.0
    exploratory_hypotheses: bool = True
    reflect_every: int = 8
    stochastic_measurement: bool = False
    seed: int = 0
    params: OperatorParams = dataclass_field(default_factory=OperatorParams)
    lattice: LatticeConfig = dataclass_field(default_factory=LatticeConfig)
    meta: MetaConfig = dataclass_field(default_factory=MetaConfig)


@dataclass
class Percept:
    """What one episode concluded, and enough context to argue about it."""

    label: str
    confidence: float
    coherence: float
    decisiveness: float
    depth_used: int
    steps: int
    hypothesis: Optional[str]
    alternatives: Dict[str, float]
    symbols: List[str]
    vetoed: List[Tuple[str, str]]
    residual_norm: float
    scale_balance: float = 0.0
    scale_agreement: float = 0.0
    band_pivotal: Tuple[float, ...] = ()
    trace: List[Dict[str, Any]] = dataclass_field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "coherence": self.coherence,
            "decisiveness": self.decisiveness,
            "depth_used": self.depth_used,
            "steps": self.steps,
            "alternatives": self.alternatives,
            "symbols": self.symbols,
            "vetoed": self.vetoed,
        }


@dataclass
class _Outcome:
    field: ResonantField
    labels: Dict[str, float]
    coherence: float
    decisiveness: float
    steps: int
    depth_used: int
    trace: List[Dict[str, Any]]
    residual: np.ndarray


class ResonantFractalCognition:
    """A small, complete mind built out of one operator applied recursively."""

    def __init__(
        self,
        config: Optional[RFCConfig] = None,
        codebook: Optional[Mapping[str, Sequence[float]]] = None,
        invariants: Optional[Sequence[Invariant]] = None,
    ):
        self.config = config or RFCConfig()
        self.operator = ScaleInvariantOperator(self.config.params)
        self.lattice = SymbolLattice(self.config.lattice)
        self.constraints = ConstraintField(invariants)
        self.telemetry = Telemetry()
        self.meta = MetaResonator(self.config.meta)
        self.rng = np.random.default_rng(self.config.seed)
        self.codebook: Dict[str, np.ndarray] = {}
        self._concept_bands: Dict[str, List[np.ndarray]] = {}
        self.episode = 0
        self._step_clock = 0
        for label, vector in (codebook or {}).items():
            self.add_concept(label, vector)

    # ------------------------------------------------------------- vocabulary
    def add_concept(self, label: str, vector: Sequence[float] | np.ndarray) -> None:
        """Register a named candidate direction in evidence space."""

        arr = normalize_vector(vector)
        if arr.size != self.config.dim:
            raise ValueError(
                f"concept '{label}' has dim {arr.size}, expected {self.config.dim}"
            )
        self.codebook[label] = arr
        # A concept is stored once as a whole and once per scale.  The scale
        # copies are what actually enter the field: a hypothesis says "at this
        # scale, the evidence should look like *this*", so it has to be the
        # concept's own band, not the whole concept viewed through a band.
        self._concept_bands[label] = [
            band.unit() for band in dyadic_decompose(arr, self.config.levels)
        ]

    def add_invariant(self, invariant: Invariant) -> None:
        self.constraints.add(invariant)

    # ----------------------------------------------------------------- seeding
    def _seed_field(self, bands, depth: int) -> ResonantField:
        config = self.config
        field = ResonantField(config.dim, max_size=config.max_field_size)
        index = 0
        for label in sorted(self.codebook):
            concept_bands = self._concept_bands[label]
            for band in bands:
                vector = concept_bands[band.level]
                if not np.any(vector):
                    continue
                # Seeded at zero lag: "no opinion yet".  Phase is the system's
                # agreement variable, so a hypothesis must be *moved* off zero
                # by evidence rather than starting somewhere arbitrary.  The
                # dispersal knob exists to demonstrate exactly that -- turn it
                # up and the read-out degrades into seeded interference.
                phase = (index * _GOLDEN * 2.0 * np.pi * config.phase_dispersal) % (
                    2.0 * np.pi
                )
                field.spawn(
                    vector,
                    scale=band.level,
                    label=label,
                    amplitude=config.seed_amplitude,
                    phase=phase,
                    origin="concept",
                    depth=depth,
                    meta={"claim": self.codebook[label]},
                )
                index += 1
        if config.exploratory_hypotheses:
            # Hypotheses the system proposes itself, straight off the evidence.
            # They carry no label, so they never vote in the read-out -- they
            # exist to shape interference and to catch structure the codebook
            # has no name for yet.
            for band in bands:
                unit = band.unit()
                if float(np.dot(unit, unit)) <= 1e-9:
                    continue
                phase = (index * _GOLDEN * 2.0 * np.pi * config.phase_dispersal) % (
                    2.0 * np.pi
                )
                field.spawn(
                    unit,
                    scale=band.level,
                    label="",
                    amplitude=config.seed_amplitude,
                    phase=phase,
                    origin="evidence",
                    depth=depth,
                )
                index += 1
        return field

    # --------------------------------------------------------------- resolving
    def _resolve(
        self, evidence: np.ndarray, depth: int, context: Mapping[str, Any]
    ) -> _Outcome:
        config = self.config
        bands = dyadic_decompose(evidence, config.levels)
        field = self._seed_field(bands, depth)
        trace: List[Dict[str, Any]] = []
        is_root = depth == 0

        def observe(active_field: ResonantField, report: StepReport) -> None:
            if is_root:
                self.lattice.observe(active_field, self._step_clock)
                self._step_clock += 1
                trace.append(
                    {
                        "step": report.step,
                        "coherence": round(report.coherence, 4),
                        "labels": {
                            k: round(v, 4) for k, v in sorted(report.labels.items())
                        },
                        "vetoes": report.constraints.count,
                    }
                )

        steps = config.steps
        self.operator.run(
            field,
            bands,
            steps,
            constraints=self.constraints,
            context=context,
            prior_fn=self.lattice.priors,
            observer=observe,
        )

        labels = field.label_distribution()
        decisiveness = field.ambiguity()
        depth_used = depth

        # Recursion: only when the field genuinely cannot choose.  The child
        # sees the residual -- what the leading interpretation fails to explain
        # -- so recursion looks at new evidence rather than re-litigating the
        # same evidence at a deeper indentation level.
        residual = np.asarray(evidence, dtype=float).copy()
        leader = (
            max(labels.items(), key=lambda item: (item[1], item[0]))[0]
            if labels
            else ""
        )
        if leader:
            direction = self.codebook.get(leader)
            if direction is not None:
                residual = residual - float(np.dot(residual, direction)) * direction

        if (
            depth < config.max_depth
            and labels
            and decisiveness < config.ambiguity_threshold
            and float(np.linalg.norm(residual)) > config.residual_floor
        ):
            child = self._resolve(residual, depth + 1, context)
            depth_used = max(depth_used, child.depth_used)
            steps += child.steps
            if child.labels:
                winner = max(child.labels.items(), key=lambda item: (item[1], item[0]))
                boost = 1.0 + config.recursion_boost * winner[1]
                for hypothesis in field.active():
                    if hypothesis.label == winner[0]:
                        hypothesis.amplitude *= boost
                field.normalize()
                self.operator.run(
                    field,
                    bands,
                    config.recursion_steps,
                    constraints=self.constraints,
                    context=context,
                    prior_fn=self.lattice.priors,
                    observer=observe,
                )
                steps += config.recursion_steps
                labels = field.label_distribution()
                decisiveness = field.ambiguity()

        return _Outcome(
            field=field,
            labels=labels,
            coherence=field.coherence(),
            decisiveness=decisiveness,
            steps=steps,
            depth_used=depth_used,
            trace=trace,
            residual=residual,
        )

    # ------------------------------------------------------------------- api
    def perceive(
        self,
        evidence: Sequence[float] | np.ndarray,
        context: Optional[Mapping[str, Any]] = None,
        reward: Optional[float] = None,
    ) -> Percept:
        """Run one full episode over ``evidence`` and return the read-out."""

        vector = np.asarray(evidence, dtype=float).reshape(-1)
        if vector.size != self.config.dim:
            raise ValueError(
                f"evidence has dim {vector.size}, expected {self.config.dim}"
            )
        ctx: Dict[str, Any] = dict(context or {})
        outcome = self._resolve(vector, 0, ctx)

        rng = self.rng if self.config.stochastic_measurement else None
        winner = outcome.field.measure(rng)
        labels = outcome.labels
        if labels:
            label, confidence = max(labels.items(), key=lambda item: (item[1], item[0]))
        else:
            label, confidence = "", 0.0

        vetoed = [(h.hid, h.veto_reason) for h in outcome.field.ordered() if h.vetoed]
        percept = Percept(
            label=label,
            confidence=float(confidence),
            coherence=float(outcome.coherence),
            decisiveness=float(outcome.decisiveness),
            depth_used=outcome.depth_used,
            steps=outcome.steps,
            hypothesis=winner.hid if winner else None,
            alternatives={key: float(value) for key, value in sorted(labels.items())},
            symbols=sorted({s.label for s in self.lattice.symbols.values()}),
            vetoed=vetoed,
            residual_norm=float(np.linalg.norm(outcome.residual)),
            scale_balance=outcome.field.scale_balance(),
            scale_agreement=outcome.field.scale_agreement(),
            band_pivotal=_band_pivotal(outcome.field, label),
            trace=outcome.trace,
        )

        self.episode += 1
        self.telemetry.add(
            EpisodeRecord(
                index=self.episode,
                label=percept.label,
                confidence=percept.confidence,
                coherence=percept.coherence,
                decisiveness=percept.decisiveness,
                depth_used=percept.depth_used,
                max_depth=self.config.max_depth,
                steps=percept.steps,
                field_size=len(outcome.field),
                max_field_size=self.config.max_field_size,
                vetoes=len(vetoed),
                symbols=len(self.lattice.symbols),
                residual_norm=percept.residual_norm,
                scale_balance=percept.scale_balance,
                scale_agreement=percept.scale_agreement,
                band_pivotal=percept.band_pivotal,
                reward=reward,
                params=self.operator.params.as_dict(),
            )
        )
        if self.config.reflect_every and self.episode % self.config.reflect_every == 0:
            self.reflect()
        return percept

    def decompose(
        self, evidence: Sequence[float] | np.ndarray, components: int = 2
    ) -> List[str]:
        """Name several sources in one piece of evidence, strongest first.

        Each round resolves the field, takes the leading interpretation, and
        subtracts what that interpretation explains -- the same residual step
        the recursive path uses when a single reading cannot settle the field.
        A one-shot read-out cannot do this: its runner-up is whatever correlates
        best with the *whole* mixture, which is usually the leader's neighbour
        rather than the second source.
        """

        residual = np.asarray(evidence, dtype=float).reshape(-1).copy()
        if residual.size != self.config.dim:
            raise ValueError(
                f"evidence has dim {residual.size}, expected {self.config.dim}"
            )
        found: List[str] = []
        for _ in range(max(1, int(components))):
            if float(np.linalg.norm(residual)) <= self.config.residual_floor:
                break
            bands = dyadic_decompose(residual, self.config.levels)
            field = self._seed_field(bands, 0)
            self.operator.run(
                field,
                bands,
                self.config.steps,
                constraints=self.constraints,
                context={},
                prior_fn=self.lattice.priors,
            )
            labels = {
                label: value
                for label, value in field.label_distribution().items()
                if label not in found
            }
            if not labels:
                break
            winner = max(labels.items(), key=lambda item: (item[1], item[0]))[0]
            found.append(winner)
            direction = self.codebook[winner]
            residual = residual - float(np.dot(residual, direction)) * direction
        return found

    def learn(self, reward: float) -> None:
        """Attach a reward to the most recent episode, after the fact."""

        if not self.telemetry.records:
            raise RuntimeError("no episode to attach a reward to")
        self.telemetry.records[-1].reward = float(reward)

    def reflect(self) -> MetaReport:
        """Apply the operator to the system's own history and retune it."""

        report = self.meta.reflect(self.telemetry, self.operator)
        self.config.params = self.operator.params
        return report

    # -------------------------------------------------------------- reporting
    def explain(self, percept: Percept) -> str:
        """A plain-language account of how the answer was reached."""

        lines = [
            f"label={percept.label or '<none>'} confidence={percept.confidence:.3f} "
            f"coherence={percept.coherence:.3f} margin={percept.decisiveness:.3f}",
            f"resolved at depth {percept.depth_used} over {percept.steps} operator steps",
        ]
        ranked = sorted(
            percept.alternatives.items(), key=lambda item: (-item[1], item[0])
        )
        if len(ranked) > 1:
            runners = ", ".join(f"{name}={value:.3f}" for name, value in ranked[1:4])
            lines.append(f"runners-up: {runners}")
        if percept.vetoed:
            reasons = ", ".join(sorted({reason for _, reason in percept.vetoed}))
            lines.append(f"{len(percept.vetoed)} hypotheses vetoed by: {reasons}")
        if self.lattice.symbols:
            lines.append("crystallised symbols:")
            lines.extend(f"  {rule}" for rule in self.lattice.rules())
        return "\n".join(lines)

    def state(self) -> Dict[str, Any]:
        return {
            "episodes": self.episode,
            "params": self.operator.params.as_dict(),
            "lattice": self.lattice.snapshot(),
            "telemetry": self.telemetry.summary(),
            "concepts": sorted(self.codebook),
            "invariants": [inv.name for inv in self.constraints.invariants],
        }
