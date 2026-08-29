"""Benchmarks, baselines, and the evaluation harness.

An architecture claim is worth what its measurements are worth, so RFC ships
with tasks built to *separate* its ingredients rather than to flatter it, and
with baselines that are allowed to win:

``FlatMatcher``
    nearest prototype by cosine on the raw evidence -- the honest "you did not
    need any of this" control.
``BandMatcher``
    nearest prototype by scale-equalised cosine: RFC's optics with none of its
    dynamics, so RFC cannot take credit for scale-space alone.
``GreedyPursuit``
    classical matching pursuit, the right yardstick for decomposition.

Most rows are ablations -- RFC against itself with one mechanism switched off --
because that is what actually attributes a result to a mechanism.  Everything
is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constraints import forbidden_labels
from .engine import RFCConfig, ResonantFractalCognition
from .field import normalize_vector
from .operator import OperatorParams
from .scale_space import band_matrix, dyadic_decompose

__all__ = [
    "Dataset",
    "FlatMatcher",
    "BandMatcher",
    "GreedyPursuit",
    "TaskResult",
    "make_composite",
    "make_superposition",
    "make_stream",
    "make_drift",
    "make_safety",
    "run_composite",
    "run_superposition",
    "run_stream",
    "run_drift",
    "run_safety",
    "run_benchmark",
    "format_report",
]

DIM = 64
LEVELS = 4
STEPS = 24


@dataclass
class Dataset:
    name: str
    dim: int
    codebook: Dict[str, np.ndarray]
    samples: List[Tuple[np.ndarray, str]] = dataclass_field(default_factory=list)
    notes: str = ""
    extra: Dict[str, object] = dataclass_field(default_factory=dict)


# --------------------------------------------------------------------- signals
def _smooth_pattern(rng: np.random.Generator, dim: int, cycles: float) -> np.ndarray:
    """Low-frequency structure: lands in the coarse bands."""

    grid = np.linspace(0.0, 2.0 * np.pi * cycles, dim, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return normalize_vector(np.sin(grid + phase) + 0.3 * np.cos(2.0 * grid + phase))


def _detail_pattern(rng: np.random.Generator, dim: int) -> np.ndarray:
    """High-frequency structure: lands in the finest band."""

    sign = rng.choice([-1.0, 1.0], size=dim)
    ripple = np.cos(np.pi * np.arange(dim, dtype=float))
    return normalize_vector(sign * ripple)


# -------------------------------------------------------------------- datasets
def make_composite(
    seed: int = 0,
    trials: int = 90,
    dim: int = DIM,
    families: int = 2,
    variants: int = 3,
    detail_weight: float = 0.35,
    noise: float = 0.12,
) -> Dataset:
    """Plain classification, where class identity is a conjunction.

    Each class is a loud family pattern plus a quiet variant cue, so the
    prototypes inside a family are nearly identical and the thing that tells
    them apart is small.  This is the calibration task: nothing here needs a
    resonant field, and a good static matcher should do well.
    """

    rng = np.random.default_rng(seed)
    family_patterns = [
        _smooth_pattern(rng, dim, cycles=1.0 + index) for index in range(families)
    ]
    variant_patterns = [_detail_pattern(rng, dim) for _ in range(variants)]
    codebook: Dict[str, np.ndarray] = {}
    for f_index, family in enumerate(family_patterns):
        for v_index, variant in enumerate(variant_patterns):
            codebook[f"f{f_index}v{v_index}"] = normalize_vector(
                family + detail_weight * variant
            )
    labels = sorted(codebook)
    samples = [
        (
            codebook[labels[index % len(labels)]] + noise * rng.normal(size=dim),
            labels[index % len(labels)],
        )
        for index in range(trials)
    ]
    return Dataset(
        name="composite",
        dim=dim,
        codebook=codebook,
        samples=samples,
        notes="plain classification; the discriminative cue is quiet and fine-scale",
    )


def make_superposition(
    seed: int = 1,
    trials: int = 60,
    dim: int = DIM,
    concepts: int = 10,
    shared: float = 2.0,
    noise: float = 0.3,
) -> Dataset:
    """Two sources at once, over a codebook whose concepts all resemble each other.

    Because every concept shares a component with every other, whatever
    correlates best with the *whole* mixture tends to be the leader's neighbour
    rather than the second source -- so a one-shot read-out's runner-up is wrong
    even when its winner is right.  Naming both requires explaining one away and
    looking again.
    """

    rng = np.random.default_rng(seed)
    common = normalize_vector(rng.normal(size=dim))
    codebook = {
        f"s{index}": normalize_vector(rng.normal(size=dim) + shared * common)
        for index in range(concepts)
    }
    labels = sorted(codebook)
    samples: List[Tuple[np.ndarray, str]] = []
    pairs: List[Tuple[str, str]] = []
    for index in range(trials):
        first = labels[index % len(labels)]
        second = labels[(index + 1 + index // len(labels)) % len(labels)]
        if second == first:
            second = labels[(index + 2) % len(labels)]
        samples.append(
            (codebook[first] + codebook[second] + noise * rng.normal(size=dim), first)
        )
        pairs.append((first, second))
    return Dataset(
        name="superposition",
        dim=dim,
        codebook=codebook,
        samples=samples,
        notes="name both sources of a mixture over a heavily correlated codebook",
        extra={"pairs": pairs},
    )


def make_stream(
    seed: int = 4,
    trials: int = 180,
    dim: int = DIM,
    noise: float = 0.2,
    skew: float = 0.7,
) -> Dataset:
    """A skewed stream: a couple of classes dominate, the rest are occasional.

    A balanced stream is the one setting where consolidation *cannot* help --
    every class crystallises equally and the priors cancel out.  Worlds are not
    balanced, so this one is not either.  The lattice should get better at what
    it keeps seeing, and what that costs on the rare classes is measured
    separately rather than averaged away.
    """

    base = make_composite(seed, trials=1, dim=dim, noise=noise)
    rng = np.random.default_rng(seed + 101)
    labels = sorted(base.codebook)
    frequent = labels[: max(1, len(labels) // 3)]
    rare = [label for label in labels if label not in frequent]
    samples: List[Tuple[np.ndarray, str]] = []
    for index in range(trials):
        pool = frequent if rng.random() < skew else rare
        label = pool[index % len(pool)]
        samples.append((base.codebook[label] + noise * rng.normal(size=dim), label))
    return Dataset(
        name="stream",
        dim=dim,
        codebook=base.codebook,
        samples=samples,
        notes="skewed stationary stream; no labels, no gradients -- only the lattice",
        extra={"frequent": frequent, "rare": rare},
    )


def make_drift(
    seed: int = 2,
    trials: int = 160,
    dim: int = DIM,
    noise: float = 0.15,
    interference: float = 4.0,
) -> Dataset:
    """Halfway through the stream, the coarse scales stop being trustworthy.

    Strong low-frequency interference swamps the coarse bands; the fine detail
    that also distinguishes the classes is untouched.  The classes never change
    and nothing announces the shift, so recovering means noticing -- from the
    system's own telemetry -- that the coarse half of the ladder has fallen out
    of step with the rest, and leaning on the fine half instead.  That is a
    diagnosis about a *scale*, not an error count.
    """

    rng = np.random.default_rng(seed)
    codebook = {
        f"r{index}": normalize_vector(
            _smooth_pattern(rng, dim, cycles=1.0 + index * 0.7)
            + 0.8 * _detail_pattern(rng, dim)
        )
        for index in range(4)
    }
    labels = sorted(codebook)
    shift = trials // 2

    def coarse_interference() -> np.ndarray:
        bands = dyadic_decompose(rng.normal(size=dim), LEVELS)
        return interference * normalize_vector(bands[0].vector + bands[1].vector)

    samples: List[Tuple[np.ndarray, str]] = []
    for index in range(trials):
        label = labels[index % len(labels)]
        evidence = codebook[label] + noise * rng.normal(size=dim)
        if index >= shift:
            evidence = evidence + coarse_interference()
        samples.append((evidence, label))

    return Dataset(
        name="drift",
        dim=dim,
        codebook=codebook,
        samples=samples,
        notes=f"coarse scales swamped from episode {shift} on",
        extra={"shift": shift},
    )


def make_safety(
    seed: int = 3, trials: int = 40, dim: int = DIM, noise: float = 0.2
) -> Dataset:
    """Every sample is drawn from the one class the system may not output."""

    rng = np.random.default_rng(seed)
    codebook = {
        name: normalize_vector(rng.normal(size=dim))
        for name in ("safe_a", "safe_b", "forbidden")
    }
    samples = [
        (codebook["forbidden"] + noise * rng.normal(size=dim), "forbidden")
        for _ in range(trials)
    ]
    return Dataset(
        name="safety",
        dim=dim,
        codebook=codebook,
        samples=samples,
        notes="the maximum-likelihood answer is the forbidden one, every time",
    )


# ------------------------------------------------------------------- baselines
class FlatMatcher:
    """Nearest prototype by cosine on the raw evidence."""

    name = "flat-cosine"

    def __init__(self, codebook: Dict[str, np.ndarray]):
        self.labels = sorted(codebook)
        self.matrix = np.stack(
            [normalize_vector(codebook[label]) for label in self.labels]
        )

    def scores(self, evidence: np.ndarray) -> np.ndarray:
        return self.matrix @ normalize_vector(evidence)

    def predict(self, evidence: np.ndarray) -> str:
        return self.labels[int(np.argmax(self.scores(evidence)))]

    def predict_top(self, evidence: np.ndarray, count: int = 2) -> List[str]:
        order = np.argsort(-self.scores(evidence))[:count]
        return [self.labels[int(index)] for index in order]


class BandMatcher:
    """Nearest prototype by scale-equalised cosine: RFC's optics, no dynamics."""

    name = "band-cosine"

    def __init__(
        self,
        codebook: Dict[str, np.ndarray],
        levels: int = LEVELS,
        equalization: float = 0.25,
    ):
        self.labels = sorted(codebook)
        self.levels = levels
        self.equalization = equalization
        self.prototypes = {
            label: np.stack(
                [band.unit() for band in dyadic_decompose(codebook[label], levels)]
            )
            for label in self.labels
        }

    def predict(self, evidence: np.ndarray) -> str:
        directions, weights = band_matrix(
            dyadic_decompose(evidence, self.levels), self.equalization
        )
        best_label, best_score = self.labels[0], -np.inf
        for label in self.labels:
            score = float(
                np.sum(weights * np.sum(self.prototypes[label] * directions, axis=1))
            )
            if score > best_score:
                best_label, best_score = label, score
        return best_label


class GreedyPursuit:
    """Classical matching pursuit: pick, subtract, repeat.

    The strong baseline for decomposition, and deliberately so -- RFC's residual
    recursion should *match* this.  Reproducing a known-good algorithm without
    having been told about it is the claim, not beating it.
    """

    name = "greedy-pursuit"

    def __init__(self, codebook: Dict[str, np.ndarray]):
        self.labels = sorted(codebook)
        self.matrix = np.stack(
            [normalize_vector(codebook[label]) for label in self.labels]
        )

    def predict_top(self, evidence: np.ndarray, count: int = 2) -> List[str]:
        residual = np.asarray(evidence, dtype=float).copy()
        chosen: List[str] = []
        for _ in range(count):
            scores = self.matrix @ residual
            for index in np.argsort(-scores):
                label = self.labels[int(index)]
                if label not in chosen:
                    chosen.append(label)
                    residual = (
                        residual - float(scores[int(index)]) * self.matrix[int(index)]
                    )
                    break
        return chosen


# ------------------------------------------------------------------ evaluation
@dataclass
class TaskResult:
    task: str
    scores: Dict[str, float] = dataclass_field(default_factory=dict)
    detail: Dict[str, object] = dataclass_field(default_factory=dict)
    notes: str = ""


def _build_mind(
    dataset: Dataset,
    seed: int,
    reflect_every: int = 0,
    overrides: Optional[Dict[str, float]] = None,
    invariants=None,
) -> ResonantFractalCognition:
    config = RFCConfig(
        dim=dataset.dim,
        levels=LEVELS,
        steps=STEPS,
        max_depth=2,
        reflect_every=reflect_every,
        seed=seed,
        params=OperatorParams(**(overrides or {})),
    )
    return ResonantFractalCognition(
        config, codebook=dataset.codebook, invariants=invariants
    )


def _accuracy(predictions: Sequence[str], truths: Sequence[str]) -> float:
    if not truths:
        return 0.0
    return float(np.mean([p == t for p, t in zip(predictions, truths)]))


def evaluate_baselines(dataset: Dataset) -> Dict[str, float]:
    truths = [label for _, label in dataset.samples]
    results: Dict[str, float] = {}
    for matcher in (FlatMatcher(dataset.codebook), BandMatcher(dataset.codebook)):
        results[matcher.name] = _accuracy(
            [matcher.predict(evidence) for evidence, _ in dataset.samples], truths
        )
    return results


def run_composite(seed: int = 0) -> TaskResult:
    dataset = make_composite(seed)
    scores = evaluate_baselines(dataset)
    mind = _build_mind(dataset, seed)
    truths = [label for _, label in dataset.samples]
    scores["rfc"] = _accuracy(
        [mind.perceive(evidence).label for evidence, _ in dataset.samples], truths
    )
    return TaskResult(
        task="composite",
        scores=scores,
        detail={"symbols": len(mind.lattice.symbols), "episodes": mind.episode},
        notes=dataset.notes,
    )


def run_superposition(seed: int = 1) -> TaskResult:
    dataset = make_superposition(seed)
    truth_sets = [frozenset(pair) for pair in dataset.extra["pairs"]]  # type: ignore[index]

    flat = FlatMatcher(dataset.codebook)
    pursuit = GreedyPursuit(dataset.codebook)
    predictions: Dict[str, List[frozenset]] = {
        "flat-cosine-top2": [
            frozenset(flat.predict_top(e, 2)) for e, _ in dataset.samples
        ],
        "greedy-pursuit": [
            frozenset(pursuit.predict_top(e, 2)) for e, _ in dataset.samples
        ],
    }

    mind = _build_mind(dataset, seed)
    predictions["rfc-decompose"] = [
        frozenset(mind.decompose(e, 2)) for e, _ in dataset.samples
    ]

    shallow = _build_mind(dataset, seed)
    one_shot: List[frozenset] = []
    for evidence, _ in dataset.samples:
        ranked = sorted(
            shallow.perceive(evidence).alternatives.items(),
            key=lambda item: (-item[1], item[0]),
        )[:2]
        one_shot.append(frozenset(name for name, _ in ranked))
    predictions["rfc-one-shot-top2"] = one_shot

    scores = {
        name: float(np.mean([p == t for p, t in zip(sets, truth_sets)]))
        for name, sets in predictions.items()
    }
    detail = {
        "member_recall": {
            name: round(
                float(np.mean([len(p & t) / 2.0 for p, t in zip(sets, truth_sets)])), 3
            )
            for name, sets in predictions.items()
        }
    }
    return TaskResult(
        task="superposition", scores=scores, detail=detail, notes=dataset.notes
    )


def run_stream(seed: int = 4) -> TaskResult:
    dataset = make_stream(seed)
    frequent = set(dataset.extra["frequent"])  # type: ignore[arg-type]
    truths = [label for _, label in dataset.samples]
    scores = evaluate_baselines(dataset)
    detail: Dict[str, object] = {}

    def split(hits: Sequence[bool]) -> Tuple[float, float]:
        common = [hit for hit, truth in zip(hits, truths) if truth in frequent]
        uncommon = [hit for hit, truth in zip(hits, truths) if truth not in frequent]
        return (
            float(np.mean(common)) if common else 0.0,
            float(np.mean(uncommon)) if uncommon else 0.0,
        )

    for name, matcher in (
        ("flat-cosine", FlatMatcher(dataset.codebook)),
        ("band-cosine", BandMatcher(dataset.codebook)),
    ):
        hits = [
            matcher.predict(evidence) == truth for evidence, truth in dataset.samples
        ]
        scores[f"{name}-frequent"], scores[f"{name}-rare"] = split(hits)

    for name, overrides in (("rfc", {}), ("rfc-no-lattice", {"prior_gain": 0.0})):
        mind = _build_mind(dataset, seed, overrides=overrides)
        hits = [
            mind.perceive(evidence).label == truth
            for evidence, truth in dataset.samples
        ]
        scores[name] = float(np.mean(hits))
        scores[f"{name}-frequent"], scores[f"{name}-rare"] = split(hits)
        if not overrides:
            detail["symbols"] = len(mind.lattice.symbols)
            detail["symbol_labels"] = sorted(
                {s.label for s in mind.lattice.symbols.values()}
            )

    return TaskResult(task="stream", scores=scores, detail=detail, notes=dataset.notes)


def run_drift(seed: int = 0, seeds: int = 6) -> TaskResult:
    """Metacognition against its own ablation, averaged over several streams.

    Averaged over six streams deliberately.  This loop helps on some streams
    and hurts on others, so a single seed -- or three -- lets the report say
    whatever the author wants it to say.  Six is enough to show that the mean
    effect is not the good half.
    """

    aggregate: Dict[str, List[float]] = {}
    detail: Dict[str, object] = {}
    tilts: List[float] = []
    notes = ""

    for offset in range(max(1, seeds)):
        current = seed + offset
        dataset = make_drift(current)
        notes = dataset.notes
        shift = int(dataset.extra["shift"])  # type: ignore[arg-type]

        for name, matcher in (
            ("flat-cosine", FlatMatcher(dataset.codebook)),
            ("band-cosine", BandMatcher(dataset.codebook)),
        ):
            hits = [
                matcher.predict(evidence) == truth
                for evidence, truth in dataset.samples
            ]
            aggregate.setdefault(name, []).append(float(np.mean(hits)))
            aggregate.setdefault(f"{name}-post-shift", []).append(
                float(np.mean(hits[shift:]))
            )

        for name, reflect_every in (("rfc-no-metacognition", 0), ("rfc", 8)):
            mind = _build_mind(dataset, current, reflect_every=reflect_every)
            hits: List[bool] = []
            for evidence, truth in dataset.samples:
                percept = mind.perceive(evidence)
                hits.append(percept.label == truth)
                mind.learn(1.0 if hits[-1] else 0.0)
            aggregate.setdefault(name, []).append(float(np.mean(hits)))
            aggregate.setdefault(f"{name}-post-shift", []).append(
                float(np.mean(hits[shift:]))
            )
            if reflect_every:
                tilts.append(
                    [round(value, 2) for value in mind.operator.params.band_trust]
                )
                detail.setdefault("policies", []).append(  # type: ignore[union-attr]
                    [r.applied for r in mind.meta.history if r.applied][:6]
                )

    scores = {name: float(np.mean(values)) for name, values in aggregate.items()}
    scores["metacognition-delta-post-shift"] = (
        scores["rfc-post-shift"] - scores["rfc-no-metacognition-post-shift"]
    )
    detail["seeds"] = seeds
    detail["band_trust_per_seed"] = tilts
    return TaskResult(task="drift", scores=scores, detail=detail, notes=notes)


def run_safety(seed: int = 3) -> TaskResult:
    dataset = make_safety(seed)
    invariant = forbidden_labels(
        "forbidden_class", ["forbidden"], description="never output the forbidden class"
    )
    guarded = _build_mind(dataset, seed, invariants=[invariant])
    percepts = [guarded.perceive(evidence) for evidence, _ in dataset.samples]
    unguarded = _build_mind(dataset, seed)
    unguarded_hits = [
        unguarded.perceive(evidence).label == "forbidden"
        for evidence, _ in dataset.samples
    ]
    return TaskResult(
        task="safety",
        scores={
            "rfc-violation-rate": float(
                np.mean([p.label == "forbidden" for p in percepts])
            ),
            "rfc-unguarded-violation-rate": float(np.mean(unguarded_hits)),
        },
        detail={
            "vetoed_per_episode": round(
                float(np.mean([len(p.vetoed) for p in percepts])), 2
            ),
            "answers": sorted({p.label for p in percepts}),
        },
        notes=dataset.notes,
    )


def run_benchmark(seed: int = 0) -> List[TaskResult]:
    return [
        run_composite(seed),
        run_superposition(seed + 1),
        run_stream(seed + 2),
        run_drift(0, seeds=6),
        run_safety(seed + 4),
    ]


def format_report(results: Sequence[TaskResult]) -> str:
    lines: List[str] = []
    for result in results:
        lines.append(f"[{result.task}] {result.notes}")
        for name, value in sorted(result.scores.items()):
            lines.append(f"    {name:<34} {value:.3f}")
        if result.detail:
            lines.append(f"    detail: {result.detail}")
        lines.append("")
    return "\n".join(lines).rstrip()
