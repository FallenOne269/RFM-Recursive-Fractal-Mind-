"""Bridge to the existing RFAI stack, benchmarks, and the command line."""

from __future__ import annotations

import numpy as np
import pytest

from rfai.dfe import DynamicFractalEncoder
from rfai.semantic_goal import SemanticGoal
from rfc import RFCConfig, ResonantFractalCognition
from rfc.bridge import encode_evidence, evidence_from_fim, goal_alignment_invariant
from rfc.cli import main as cli_main
from rfc.tasks import (
    BandMatcher,
    FlatMatcher,
    GreedyPursuit,
    make_composite,
    run_composite,
    run_safety,
    run_superposition,
    format_report,
)


def test_rfai_motifs_become_evidence_vectors():
    encoder = DynamicFractalEncoder()
    vector = encode_evidence(
        {"payload": 3}, encoder, np.array([0.4, 0.1, 0.9]), {"load": 0.2}, dim=32
    )
    assert vector.shape == (32,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_the_same_motif_always_encodes_the_same_way():
    encoder = DynamicFractalEncoder()
    fim = encoder.encode({"payload": 1}, np.array([0.2, 0.5]), {})
    np.testing.assert_allclose(evidence_from_fim(fim, 32), evidence_from_fim(fim, 32))


def test_a_semantic_goal_becomes_an_enforced_invariant():
    dim = 32
    rng = np.random.default_rng(0)
    wanted = rng.normal(size=dim)
    codebook = {"aligned": wanted, "opposed": -wanted, "other": rng.normal(size=dim)}
    invariant = goal_alignment_invariant(SemanticGoal(wanted), dim, min_similarity=-0.5)
    mind = ResonantFractalCognition(
        RFCConfig(dim=dim, seed=0), codebook=codebook, invariants=[invariant]
    )
    percept = mind.perceive(-wanted / np.linalg.norm(wanted))
    assert percept.label != "opposed"
    assert any(reason == "goal_alignment" for _, reason in percept.vetoed)


def test_baselines_agree_with_themselves():
    dataset = make_composite(0, trials=8)
    for matcher in (FlatMatcher(dataset.codebook), BandMatcher(dataset.codebook)):
        first = [matcher.predict(evidence) for evidence, _ in dataset.samples]
        second = [matcher.predict(evidence) for evidence, _ in dataset.samples]
        assert first == second
        assert set(first) <= set(dataset.codebook)


def test_greedy_pursuit_returns_distinct_labels():
    dataset = make_composite(0, trials=4)
    pursuit = GreedyPursuit(dataset.codebook)
    for evidence, _ in dataset.samples:
        picked = pursuit.predict_top(evidence, 3)
        assert len(picked) == len(set(picked)) == 3


def test_safety_task_reports_a_clean_sheet():
    result = run_safety(3)
    assert result.scores["rfc-violation-rate"] == 0.0
    assert result.scores["rfc-unguarded-violation-rate"] > 0.5
    assert "forbidden" not in result.detail["answers"]


def test_recursion_beats_its_own_ablation_on_superposition():
    result = run_superposition(1)
    assert result.scores["rfc-decompose"] > result.scores["rfc-one-shot-top2"]


def test_classification_stays_in_the_neighbourhood_of_the_baselines():
    """RFC should reduce to something sensible where nothing fancy is needed."""

    result = run_composite(0)
    assert result.scores["rfc"] > 0.75
    assert result.scores["rfc"] > result.scores["flat-cosine"] - 0.15


def test_report_formatting_is_readable():
    text = format_report([run_safety(3)])
    assert "safety" in text
    assert "rfc-violation-rate" in text


def test_cli_runs_an_episode(capsys):
    assert cli_main(["episode", "--seed", "0", "--episodes", "4"]) == 0
    assert "accuracy" in capsys.readouterr().out


def test_cli_emits_json(capsys):
    import json

    assert cli_main(["episode", "--seed", "0", "--episodes", "3", "--json"]) == 0
    payload = capsys.readouterr().out
    assert json.loads(payload[payload.index("{") :])["episodes"] == 3


def test_cli_can_forbid_an_answer(capsys):
    dataset = make_composite(0, trials=1)
    banned = sorted(dataset.codebook)[0]
    assert (
        cli_main(
            [
                "episode",
                "--seed",
                "0",
                "--episodes",
                "6",
                "--forbid",
                banned,
                "--verbose",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert f"answer={banned} " not in output
