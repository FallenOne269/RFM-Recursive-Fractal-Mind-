"""The resonant hypothesis field.

A field is a population of hypotheses, each a unit direction in the evidence
space carrying an oscillator state (amplitude ``r`` and phase ``theta``).  The
field's "belief" is not stored anywhere: it is the interference pattern of
those oscillators.  Agreement shows up as phase locking, disagreement as
cancellation, and the read-out is the squared amplitude, normalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["Hypothesis", "Coalition", "ResonantField", "normalize_vector"]

_EPS = 1e-12

#: Scaling applied to the coarse/fine consensus gap (see ``scale_balance``).
_BALANCE_GAIN = 4.0


def normalize_vector(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS:
        return np.zeros_like(arr)
    return arr / norm


@dataclass
class Hypothesis:
    """A single oscillator in the field.

    ``scale`` selects which band of evidence drives it; the operator turns that
    into a natural frequency on a dyadic ladder, so a hypothesis is only driven
    coherently by evidence arriving at its own scale.
    """

    hid: str
    vector: np.ndarray
    scale: int
    label: str = ""
    amplitude: float = 0.05
    phase: float = 0.0
    lag: float = 0.0
    symbol: Optional[str] = None
    vetoed: bool = False
    veto_reason: str = ""
    origin: str = "seed"
    depth: int = 0
    age: int = 0
    meta: Dict[str, object] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = normalize_vector(self.vector)
        self.amplitude = float(max(0.0, self.amplitude))
        self.phase = float(self.phase % (2.0 * np.pi))
        self.lag = float(self.lag % (2.0 * np.pi))

    @property
    def complex_amplitude(self) -> complex:
        return complex(
            self.amplitude * np.cos(self.phase), self.amplitude * np.sin(self.phase)
        )

    @property
    def locked_amplitude(self) -> complex:
        """Amplitude in the frame co-rotating with this hypothesis' own drive.

        Hypotheses live on a dyadic frequency ladder, so their lab-frame phases
        can never all agree -- that is the point of the ladder.  What *can*
        agree is their phase lag against the band that drives them.  Every
        agreement measure in the system is taken in this frame: it answers "is
        every scale telling the same story?", which lab-frame phase cannot.
        """

        return complex(
            self.amplitude * np.cos(self.lag), self.amplitude * np.sin(self.lag)
        )

    def claim(self) -> np.ndarray:
        """The whole thing this hypothesis is a claim about.

        ``vector`` is only this hypothesis' slice of the scale ladder, so
        anything reasoning about *what* is being proposed -- a guard, an
        explanation -- has to ask for the claim rather than the slice.
        """

        claim = self.meta.get("claim")
        if isinstance(claim, np.ndarray) and claim.size == self.vector.size:
            return claim
        return self.vector

    def clone(self) -> "Hypothesis":
        return Hypothesis(
            hid=self.hid,
            vector=self.vector.copy(),
            scale=self.scale,
            label=self.label,
            amplitude=self.amplitude,
            phase=self.phase,
            lag=self.lag,
            symbol=self.symbol,
            vetoed=self.vetoed,
            veto_reason=self.veto_reason,
            origin=self.origin,
            depth=self.depth,
            age=self.age,
            meta=dict(self.meta),
        )


@dataclass
class Coalition:
    """A phase-locked group of mutually aligned hypotheses."""

    members: List[str]
    centroid: np.ndarray
    amplitude: float
    coherence: float
    label: str
    scales: List[int]


class ResonantField:
    """A mutable population of hypotheses plus the read-outs over it."""

    def __init__(self, dim: int, max_size: int = 96):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.max_size = int(max_size)
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._counter = 0

    # ------------------------------------------------------------------ setup
    def new_id(self, prefix: str = "h") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def add(self, hypothesis: Hypothesis) -> Hypothesis:
        if hypothesis.vector.size != self.dim:
            raise ValueError(
                f"hypothesis {hypothesis.hid} has dim {hypothesis.vector.size}, field expects {self.dim}"
            )
        self.hypotheses[hypothesis.hid] = hypothesis
        return hypothesis

    def spawn(
        self,
        vector: Sequence[float] | np.ndarray,
        scale: int,
        label: str = "",
        amplitude: float = 0.05,
        phase: float = 0.0,
        origin: str = "seed",
        depth: int = 0,
        meta: Optional[Mapping[str, object]] = None,
    ) -> Hypothesis:
        hypothesis = Hypothesis(
            hid=self.new_id(),
            vector=np.asarray(vector, dtype=float),
            scale=int(scale),
            label=label,
            amplitude=amplitude,
            phase=phase,
            lag=phase,
            origin=origin,
            depth=depth,
            meta=dict(meta or {}),
        )
        return self.add(hypothesis)

    def remove(self, hid: str) -> None:
        self.hypotheses.pop(hid, None)

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return len(self.hypotheses)

    def ids(self) -> List[str]:
        return sorted(self.hypotheses)

    def ordered(self) -> List[Hypothesis]:
        """Hypotheses in a stable order -- determinism depends on this."""

        return [self.hypotheses[hid] for hid in self.ids()]

    def active(self) -> List[Hypothesis]:
        return [h for h in self.ordered() if not h.vetoed]

    def vectors(self) -> np.ndarray:
        items = self.ordered()
        if not items:
            return np.zeros((0, self.dim))
        return np.stack([h.vector for h in items])

    def amplitudes(self) -> np.ndarray:
        return np.array([h.amplitude for h in self.ordered()], dtype=float)

    def phases(self) -> np.ndarray:
        return np.array([h.phase for h in self.ordered()], dtype=float)

    # --------------------------------------------------------------- readouts
    def superposition(self) -> np.ndarray:
        """Real projection of the interference pattern in evidence space."""

        state = np.zeros(self.dim, dtype=float)
        for hyp in self.active():
            state += hyp.amplitude * np.cos(hyp.phase) * hyp.vector
        return state

    def coherence(self) -> float:
        """Amplitude-weighted Kuramoto order parameter in ``[0, 1]``.

        1.0 means every active hypothesis is in phase (the field has settled on
        one story); near 0 means the population is cancelling itself out.
        """

        items = self.active()
        if not items:
            return 0.0
        total = sum(h.amplitude for h in items)
        if total <= _EPS:
            return 0.0
        vector_sum = sum(h.locked_amplitude for h in items)
        return float(min(1.0, abs(vector_sum) / total))

    def distribution(self) -> Dict[str, float]:
        """Born-style read-out: probability proportional to squared amplitude."""

        items = self.active()
        weights = {h.hid: h.amplitude**2 for h in items}
        total = sum(weights.values())
        if total <= _EPS:
            return {hid: 0.0 for hid in weights}
        return {hid: value / total for hid, value in weights.items()}

    def label_distribution(self) -> Dict[str, float]:
        """Coherent read-out: scales sharing a label are summed *as waves*.

        Adding the probabilities of a label's copies would make the loud coarse
        scale outvote the quiet fine one, and a label would score well while its
        fine-scale evidence flatly contradicted it.  Summing the complex locked
        amplitudes instead makes the read-out conjunctive: copies that are all
        driven by their own bands sit near zero lag and add constructively,
        while a copy nothing supports keeps its seeded phase and subtracts.
        Agreement across scale is therefore what produces confidence, which is
        the whole reason for the scale ladder.
        """

        sums: Dict[str, complex] = {}
        for hypothesis in self.active():
            if not hypothesis.label:
                continue
            sums[hypothesis.label] = (
                sums.get(hypothesis.label, 0j) + hypothesis.locked_amplitude
            )
        totals = {label: float(abs(value) ** 2) for label, value in sums.items()}
        total = sum(totals.values())
        if total <= _EPS:
            return {label: 0.0 for label in totals}
        return {label: value / total for label, value in totals.items()}

    def scale_distributions(self) -> Dict[int, Dict[str, float]]:
        """The label distribution each scale would report on its own."""

        by_scale: Dict[int, Dict[str, float]] = {}
        for hypothesis in self.active():
            if not hypothesis.label:
                continue
            bucket = by_scale.setdefault(hypothesis.scale, {})
            bucket[hypothesis.label] = (
                bucket.get(hypothesis.label, 0.0) + hypothesis.amplitude**2
            )
        result: Dict[int, Dict[str, float]] = {}
        for scale, weights in by_scale.items():
            total = sum(weights.values())
            if total <= _EPS:
                continue
            result[scale] = {label: value / total for label, value in weights.items()}
        return result

    def scale_consensus(self) -> Dict[int, float]:
        """How far each scale is out of step with the rest of the ladder.

        Not "how confident is this scale" -- interference makes a swamped band
        *more* confident, not less, because noise in a small subspace still
        picks a clear winner.  What gives a bad witness away is that its story
        stops matching everyone else's, so each scale is scored by the cosine
        between its own label distribution and the average of the others'.
        """

        distributions = self.scale_distributions()
        if len(distributions) < 2:
            return {scale: 1.0 for scale in distributions}
        labels = sorted({label for dist in distributions.values() for label in dist})
        vectors = {
            scale: np.array([dist.get(label, 0.0) for label in labels], dtype=float)
            for scale, dist in distributions.items()
        }
        consensus: Dict[int, float] = {}
        for scale, vector in vectors.items():
            others = [other for key, other in vectors.items() if key != scale]
            reference = np.mean(others, axis=0)
            denominator = float(np.linalg.norm(vector) * np.linalg.norm(reference))
            consensus[scale] = (
                float(np.dot(vector, reference) / denominator)
                if denominator > _EPS
                else 0.0
            )
        return consensus

    def scale_balance(self) -> float:
        """``+1`` when only the fine half is still in step, ``-1`` when only the coarse half."""

        consensus = self.scale_consensus()
        if len(consensus) < 2:
            return 0.0
        levels = sorted(consensus)
        midpoint = len(levels) / 2.0
        coarse = [
            consensus[level] for index, level in enumerate(levels) if index < midpoint
        ]
        fine = [
            consensus[level] for index, level in enumerate(levels) if index >= midpoint
        ]
        if not coarse or not fine:
            return 0.0
        # Scaled up: the raw gap between two halves of a consensus score is a
        # few hundredths, and the metacognitive features are meant to be O(1)
        # deviations so that no one of them silently outvotes the rest.
        return float(
            np.clip(_BALANCE_GAIN * (np.mean(fine) - np.mean(coarse)), -1.0, 1.0)
        )

    def scale_complex(self) -> Dict[int, Dict[str, complex]]:
        """Per-scale, per-label complex amplitudes -- the read-out, unsummed."""

        by_scale: Dict[int, Dict[str, complex]] = {}
        for hypothesis in self.active():
            if not hypothesis.label:
                continue
            bucket = by_scale.setdefault(hypothesis.scale, {})
            bucket[hypothesis.label] = (
                bucket.get(hypothesis.label, 0j) + hypothesis.locked_amplitude
            )
        return by_scale

    def scale_agreement(self) -> float:
        """How much more the fine half agreed with the rest of the ladder than the coarse half.

        Each scale is scored against the answer the *other* scales would have
        given, never against the answer it helped produce -- otherwise whichever
        scales dominate would always look like the reliable ones, which is
        circular and, as it happens, exactly backwards when the dominant scales
        are the corrupted ones.  Multiplied by whether the episode was right,
        this becomes credit assignment across scale.
        """

        per_scale = self.scale_complex()
        if len(per_scale) < 2:
            return 0.0
        levels = sorted(per_scale)
        labels = sorted({label for bucket in per_scale.values() for label in bucket})
        if len(labels) < 2:
            return 0.0

        agreement: Dict[int, float] = {}
        for level in levels:
            held_out = {
                label: sum(
                    per_scale[other].get(label, 0j)
                    for other in levels
                    if other != level
                )
                for label in labels
            }
            reference = max(
                held_out.items(), key=lambda item: (abs(item[1]) ** 2, item[0])
            )[0]
            own = {label: abs(per_scale[level].get(label, 0j)) ** 2 for label in labels}
            total = sum(own.values())
            agreement[level] = float(own[reference] / total) if total > _EPS else 0.0

        midpoint = len(levels) / 2.0
        coarse = [
            agreement[level] for index, level in enumerate(levels) if index < midpoint
        ]
        fine = [
            agreement[level] for index, level in enumerate(levels) if index >= midpoint
        ]
        if not coarse or not fine:
            return 0.0
        return float(np.clip(np.mean(fine) - np.mean(coarse), -1.0, 1.0))

    def measure(
        self, rng: Optional[np.random.Generator] = None
    ) -> Optional[Hypothesis]:
        """Collapse the field to one hypothesis.

        Deterministic (arg-max) unless a generator is supplied, in which case
        the Born distribution is sampled.  Vetoed hypotheses are excluded here
        as well as damped by the constraint field, so a vetoed hypothesis can
        never be returned no matter how much amplitude it accumulated first.
        """

        probabilities = self.distribution()
        if not probabilities:
            return None
        if rng is None:
            best = max(probabilities.items(), key=lambda item: (item[1], item[0]))
            return self.hypotheses[best[0]]
        hids = sorted(probabilities)
        weights = np.array([probabilities[hid] for hid in hids], dtype=float)
        if weights.sum() <= _EPS:
            return self.hypotheses[hids[0]]
        weights = weights / weights.sum()
        return self.hypotheses[str(rng.choice(hids, p=weights))]

    def ambiguity(self) -> float:
        """Gap between the two leading labels; small means "still undecided"."""

        labels = sorted(
            self.label_distribution().items(), key=lambda item: (-item[1], item[0])
        )
        if len(labels) < 2:
            return 1.0 if labels else 0.0
        return float(labels[0][1] - labels[1][1])

    # ------------------------------------------------------------- coalitions
    def coalitions(
        self, overlap_threshold: float = 0.6, phase_tolerance: float = 0.9
    ) -> List[Coalition]:
        """Group hypotheses into the things the field is actually considering.

        Hypotheses carrying the same label are copies of one claim seen at
        different scales; their vectors are near-orthogonal by construction (a
        scale ladder is a decomposition), so overlap would never group them.
        They are grouped by label instead, which makes a coalition exactly what
        it should be: "this claim, as supported at every scale at once".
        Unlabelled hypotheses -- the ones the field proposed for itself -- have
        no label to group by and fall back to overlap.

        Deterministic: seeds are taken in descending amplitude with the id as a
        tie-break, so the same field always yields the same coalitions.
        """

        remaining = sorted(self.active(), key=lambda h: (-h.amplitude, h.hid))
        taken: set[str] = set()
        groups: List[Coalition] = []
        for seed in remaining:
            if seed.hid in taken:
                continue
            members = [seed]
            taken.add(seed.hid)
            for other in remaining:
                if other.hid in taken:
                    continue
                if seed.label or other.label:
                    grouped = seed.label == other.label
                else:
                    overlap = float(np.dot(seed.vector, other.vector))
                    phase_gap = abs(np.angle(np.exp(1j * (other.lag - seed.lag))))
                    grouped = (
                        overlap >= overlap_threshold and phase_gap <= phase_tolerance
                    )
                if grouped:
                    members.append(other)
                    taken.add(other.hid)
            amplitude = float(sum(m.amplitude for m in members))
            vector_sum = sum(m.locked_amplitude for m in members)
            coherence = (
                float(min(1.0, abs(vector_sum) / amplitude))
                if amplitude > _EPS
                else 0.0
            )
            centroid = np.zeros(self.dim, dtype=float)
            for member in members:
                centroid += member.amplitude * member.vector
            centroid = normalize_vector(centroid)
            labels = [m.label for m in members if m.label]
            label = (
                max(set(labels), key=lambda value: (labels.count(value), value))
                if labels
                else ""
            )
            groups.append(
                Coalition(
                    members=[m.hid for m in members],
                    centroid=centroid,
                    amplitude=amplitude,
                    coherence=coherence,
                    label=label,
                    scales=sorted({m.scale for m in members}),
                )
            )
        groups.sort(key=lambda c: (-c.amplitude, c.members[0]))
        return groups

    # ----------------------------------------------------------- housekeeping
    def normalize(self) -> None:
        """Conserve probability mass *within each scale*.

        Normalising the field as a whole would let one band out-shout the
        others: the coarse band usually carries the most energy, so whichever
        hypothesis won there would dominate the cross-scale sum and the finer
        scales would never get a vote.  Each scale is a separate witness and
        each witness gets one vote, so mass is conserved per scale and the
        cross-scale read-out compares like with like.
        """

        items = self.ordered()
        if not items:
            return
        groups: Dict[int, List[Hypothesis]] = {}
        for hypothesis in items:
            groups.setdefault(hypothesis.scale, []).append(hypothesis)
        for group in groups.values():
            total = float(np.sqrt(sum(h.amplitude**2 for h in group)))
            if total <= _EPS:
                continue
            for hypothesis in group:
                hypothesis.amplitude /= total

    def limit_energy(self, cap: float) -> bool:
        """Scale the field down if its total energy exceeds ``cap``.

        A guard, not a normaliser: below the cap nothing is touched, so the
        relative loudness of the scales -- which is evidence -- survives.
        """

        if cap <= 0.0:
            return False
        energy = float(np.sqrt(sum(h.amplitude**2 for h in self.ordered())))
        if energy <= cap or energy <= _EPS:
            return False
        factor = cap / energy
        for hypothesis in self.ordered():
            hypothesis.amplitude *= factor
        return True

    def prune(self, threshold: float, keep_min: int = 2) -> List[str]:
        """Drop negligible hypotheses, then cap the population by amplitude.

        ``threshold`` is *relative to the strongest hypothesis*.  An absolute
        floor would empty the field as soon as competition sharpened it, taking
        the quiet fine-scale copies with it -- and those copies are exactly what
        the cross-scale read-out needs to still be there at the end.
        """

        items = sorted(self.ordered(), key=lambda h: (-h.amplitude, h.hid))
        dropped: List[str] = []
        floor = threshold * (items[0].amplitude if items else 0.0)
        for hyp in items[keep_min:]:
            # Vetoed hypotheses are kept deliberately.  They cost nothing (no
            # drive, no positive coupling) and they are the record of what the
            # constraint field refused, which is the part of an answer a
            # reviewer most wants to see.
            if hyp.amplitude < floor and not hyp.vetoed:
                dropped.append(hyp.hid)
        for hid in dropped:
            self.remove(hid)
        survivors = [
            h
            for h in sorted(self.ordered(), key=lambda h: (-h.amplitude, h.hid))
            if not h.vetoed
        ]
        for hyp in survivors[self.max_size :]:
            dropped.append(hyp.hid)
            self.remove(hyp.hid)
        return dropped

    def snapshot(self) -> Dict[str, object]:
        return {
            "size": len(self.hypotheses),
            "coherence": self.coherence(),
            "labels": self.label_distribution(),
            "vetoed": [h.hid for h in self.ordered() if h.vetoed],
        }
