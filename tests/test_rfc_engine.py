"""Engine-level behaviour: determinism, recursion, symbols, safety, self-tuning."""

from __future__ import annotations

import numpy as np
import pytest

from rfc import (
    Invariant,
    LatticeConfig,
    OperatorParams,
    Percept,
    RFCConfig,
    ResonantFractalCognition,
    forbidden_direction,
    forbidden_labels,
)
from rfc.field import normalize_vector
from rfc.metacognition import MetaConfig
from rfc.tasks import make_composite, make_superposition


def _codebook(count: int = 4, dim: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        chr(ord("a") + index): normalize_vector(rng.normal(size=dim))
        for index in range(count)
    }


def _mind(dim: int = 64, seed: int = 0, **kwargs) -> ResonantFractalCognition:
    return ResonantFractalCognition(
        RFCConfig(dim=dim, seed=seed, **kwargs), codebook=_codebook(dim=dim)
    )


def test_perceive_recovers_the_generating_concept():
    codebook = _codebook()
    mind = ResonantFractalCognition(RFCConfig(dim=64, seed=0), codebook=codebook)
    rng = np.random.default_rng(1)
    hits = 0
    for label, vector in codebook.items():
        hits += mind.perceive(vector + 0.1 * rng.normal(size=64)).label == label
    assert hits == len(codebook)


def test_identical_seeds_give_identical_episodes():
    rng = np.random.default_rng(7)
    evidence = [rng.normal(size=64) for _ in range(5)]
    runs = []
    for _ in range(2):
        mind = _mind()
        runs.append(
            [
                (p.label, round(p.confidence, 9), round(p.coherence, 9), p.depth_used)
                for p in (mind.perceive(sample) for sample in evidence)
            ]
        )
    assert runs[0] == runs[1]


def test_probabilities_are_a_distribution():
    mind = _mind()
    percept = mind.perceive(np.random.default_rng(2).normal(size=64))
    assert percept.alternatives
    assert sum(percept.alternatives.values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in percept.alternatives.values())


def test_evidence_of_the_wrong_size_is_rejected():
    mind = _mind()
    with pytest.raises(ValueError):
        mind.perceive(np.zeros(7))
    with pytest.raises(ValueError):
        mind.add_concept("bad", np.zeros(7))


def test_recursion_only_fires_when_the_field_cannot_choose():
    codebook = _codebook()
    decisive = ResonantFractalCognition(
        RFCConfig(dim=64, seed=0, ambiguity_threshold=0.0), codebook=codebook
    )
    eager = ResonantFractalCognition(
        RFCConfig(dim=64, seed=0, ambiguity_threshold=1.0), codebook=codebook
    )
    evidence = codebook["a"] + codebook["b"]
    assert decisive.perceive(evidence).depth_used == 0
    assert eager.perceive(evidence).depth_used > 0


def test_recursion_depth_is_capped():
    codebook = _codebook()
    mind = ResonantFractalCognition(
        RFCConfig(dim=64, seed=0, ambiguity_threshold=1.0, max_depth=2),
        codebook=codebook,
    )
    percept = mind.perceive(codebook["a"] + codebook["b"] + codebook["c"])
    assert percept.depth_used <= 2


def test_decompose_names_both_sources():
    dataset = make_superposition(1, trials=12)
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=1), codebook=dataset.codebook
    )
    hits = 0
    for (evidence, _), pair in zip(dataset.samples, dataset.extra["pairs"]):
        hits += frozenset(mind.decompose(evidence, 2)) == frozenset(pair)
    assert hits >= len(dataset.samples) // 2


def test_decompose_does_not_repeat_itself():
    codebook = _codebook()
    mind = ResonantFractalCognition(RFCConfig(dim=64, seed=0), codebook=codebook)
    found = mind.decompose(codebook["a"] + codebook["c"], 3)
    assert len(found) == len(set(found))


def test_symbols_crystallise_and_carry_provenance():
    dataset = make_composite(0, trials=12)
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=0), codebook=dataset.codebook
    )
    for evidence, _ in dataset.samples:
        mind.perceive(evidence)
    assert mind.lattice.symbols
    for symbol in mind.lattice.symbols.values():
        assert symbol.label in dataset.codebook
        assert symbol.support
        assert symbol.scales
    assert mind.lattice.rules()


def test_one_symbol_per_claim():
    dataset = make_composite(0, trials=20)
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=0), codebook=dataset.codebook
    )
    for evidence, _ in dataset.samples:
        mind.perceive(evidence)
    labels = [symbol.label for symbol in mind.lattice.symbols.values()]
    assert len(labels) == len(set(labels))


def test_unsupported_symbols_dissolve():
    dataset = make_composite(0, trials=6)
    config = RFCConfig(
        dim=dataset.dim, seed=0, lattice=LatticeConfig(decay=0.5, dissolve_strength=0.5)
    )
    mind = ResonantFractalCognition(config, codebook=dataset.codebook)
    for evidence, _ in dataset.samples:
        mind.perceive(evidence)
    before = len(mind.lattice.symbols)
    mind.lattice.decay(step=10**6)
    assert len(mind.lattice.symbols) < before or before == 0


def test_a_vetoed_answer_is_never_returned():
    dataset = make_composite(0, trials=20)
    banned = sorted(dataset.codebook)[0]
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=0),
        codebook=dataset.codebook,
        invariants=[forbidden_labels("banned", [banned])],
    )
    for evidence, truth in dataset.samples:
        percept = mind.perceive(evidence)
        assert percept.label != banned
        if truth == banned:
            assert percept.vetoed


def test_veto_makes_amplitude_non_increasing():
    """The safety property, checked on the physics rather than on the output."""

    from rfc.field import ResonantField
    from rfc.operator import ScaleInvariantOperator
    from rfc.scale_space import dyadic_decompose

    dim = 32
    target = normalize_vector(np.ones(dim))
    bands = dyadic_decompose(target, 4)
    field = ResonantField(dim=dim)
    banned = field.spawn(target, scale=0, label="banned", amplitude=0.5)
    field.spawn(target, scale=1, label="allowed", amplitude=0.5)

    from rfc.constraints import ConstraintField

    constraints = ConstraintField(
        [forbidden_labels("banned", ["banned"], severity=0.5)]
    )
    operator = ScaleInvariantOperator()
    history = [banned.amplitude]
    for index in range(20):
        operator.step(
            field, bands, time=index * 0.35, step_index=index, constraints=constraints
        )
        history.append(banned.amplitude)
    assert banned.vetoed
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(history[1:], history[2:])
    )
    assert field.measure().label != "banned"


def test_direction_invariant_blocks_a_forbidden_region():
    codebook = _codebook()
    mind = ResonantFractalCognition(
        RFCConfig(dim=64, seed=0),
        codebook=codebook,
        invariants=[forbidden_direction("no_a", codebook["a"], threshold=0.9)],
    )
    percept = mind.perceive(codebook["a"])
    assert percept.label != "a"


def test_invariants_can_read_the_context():
    codebook = _codebook()
    invariant = Invariant(
        name="context_gate",
        predicate=lambda hypothesis, context: context.get("mode") == "strict"
        and hypothesis.label == "b",
    )
    mind = ResonantFractalCognition(
        RFCConfig(dim=64, seed=0), codebook=codebook, invariants=[invariant]
    )
    assert mind.perceive(codebook["b"], context={"mode": "strict"}).label != "b"
    assert mind.perceive(codebook["b"], context={"mode": "open"}).label == "b"


def test_reflection_only_moves_tunable_parameters_and_stays_bounded():
    dataset = make_composite(0, trials=40)
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=0, reflect_every=8), codebook=dataset.codebook
    )
    defaults = OperatorParams()
    for evidence, truth in dataset.samples:
        percept = mind.perceive(evidence)
        mind.learn(1.0 if percept.label == truth else 0.0)
    tuned = mind.operator.params
    from rfc.operator import TUNABLE_PARAMETERS

    for name, (low, high) in TUNABLE_PARAMETERS.items():
        assert low <= getattr(tuned, name) <= high
    for name in ("dt", "base_frequency", "prior_gain"):
        assert getattr(tuned, name) == getattr(defaults, name)


def test_reflection_rolls_back_when_reward_collapses():
    dataset = make_composite(0, trials=40)
    mind = ResonantFractalCognition(
        RFCConfig(
            dim=dataset.dim,
            seed=0,
            reflect_every=8,
            meta=MetaConfig(window=8, min_episodes=8, expectation_decay=0.0),
        ),
        codebook=dataset.codebook,
    )
    for index, (evidence, _) in enumerate(dataset.samples):
        mind.perceive(evidence)
        mind.learn(1.0 if index < 16 else 0.0)
    assert any(report.rolled_back for report in mind.meta.history)


def test_reflection_waits_for_enough_history():
    mind = _mind(reflect_every=0)
    report = mind.reflect()
    assert report.applied is None
    assert "history" in report.reason


def test_explain_mentions_the_answer_and_the_refusals():
    dataset = make_composite(0, trials=4)
    banned = sorted(dataset.codebook)[0]
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=0),
        codebook=dataset.codebook,
        invariants=[forbidden_labels("banned", [banned])],
    )
    percept = mind.perceive(dataset.samples[0][0])
    text = mind.explain(percept)
    assert percept.label in text
    assert "vetoed" in text
    assert isinstance(percept, Percept)


def test_state_is_serialisable_and_complete():
    mind = _mind()
    mind.perceive(np.random.default_rng(0).normal(size=64))
    state = mind.state()
    assert state["episodes"] == 1
    assert set(state) == {
        "episodes",
        "params",
        "lattice",
        "telemetry",
        "concepts",
        "invariants",
    }
