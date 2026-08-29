"""Substrate-level properties: scale space, field read-outs, operator physics."""

from __future__ import annotations

import numpy as np
import pytest

from rfc.field import ResonantField, normalize_vector
from rfc.operator import BAND_TRUST_BOUNDS, OperatorParams, ScaleInvariantOperator
from rfc.scale_space import band_matrix, dyadic_decompose, temporal_bands


def test_dyadic_bands_reconstruct_the_signal_exactly():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=64)
    bands = dyadic_decompose(signal, levels=4)
    assert [band.level for band in bands] == [0, 1, 2, 3]
    np.testing.assert_allclose(sum(band.vector for band in bands), signal, atol=1e-12)


def test_dyadic_bands_separate_frequencies():
    grid = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    slow = np.sin(grid)
    fast = np.cos(np.pi * np.arange(64, dtype=float))
    slow_bands = dyadic_decompose(slow, 4)
    fast_bands = dyadic_decompose(fast, 4)
    assert slow_bands[0].energy > slow_bands[-1].energy
    assert fast_bands[-1].energy > fast_bands[0].energy


def test_band_equalization_moves_weight_toward_quiet_bands():
    grid = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    signal = np.sin(grid) + 0.05 * np.cos(np.pi * np.arange(64, dtype=float))
    bands = dyadic_decompose(signal, 4)
    _, raw = band_matrix(bands, equalization=0.0)
    _, equalized = band_matrix(bands, equalization=1.0)
    assert raw[0] > raw[-1]
    np.testing.assert_allclose(equalized, np.ones_like(equalized))


def test_temporal_bands_split_slow_from_fast():
    history = np.zeros((8, 3))
    history[:, 0] = 1.0
    history[-2:, 1] = 1.0
    bands = temporal_bands(history, levels=3)
    assert bands[0].vector[0] == pytest.approx(1.0)
    assert bands[-1].vector[1] > 0.0


def test_label_read_out_is_conjunctive_across_scales():
    """A label whose fine scale is in antiphase must not win on coarse alone."""

    field = ResonantField(dim=4)
    direction = normalize_vector([1.0, 0.0, 0.0, 0.0])
    other = normalize_vector([0.0, 1.0, 0.0, 0.0])
    agree = field.spawn(direction, scale=0, label="agree", amplitude=0.6)
    agree_fine = field.spawn(direction, scale=1, label="agree", amplitude=0.6)
    conflict = field.spawn(other, scale=0, label="conflict", amplitude=0.7)
    conflict_fine = field.spawn(other, scale=1, label="conflict", amplitude=0.7)
    for hypothesis in (agree, agree_fine, conflict):
        hypothesis.lag = 0.0
    conflict_fine.lag = np.pi  # the fine scale flatly refutes this reading

    distribution = field.label_distribution()
    assert distribution["agree"] > distribution["conflict"]
    assert distribution["conflict"] < 0.05


def test_measure_is_deterministic_and_prefers_the_loudest():
    field = ResonantField(dim=3)
    field.spawn([1.0, 0.0, 0.0], scale=0, label="a", amplitude=0.2)
    winner = field.spawn([0.0, 1.0, 0.0], scale=0, label="b", amplitude=0.9)
    assert field.measure().hid == winner.hid
    assert field.measure().hid == field.measure().hid


def test_normalize_conserves_mass_within_each_scale():
    field = ResonantField(dim=2)
    field.spawn([1.0, 0.0], scale=0, label="a", amplitude=3.0)
    field.spawn([0.0, 1.0], scale=0, label="b", amplitude=4.0)
    field.spawn([1.0, 0.0], scale=1, label="a", amplitude=0.1)
    field.normalize()
    for scale in (0, 1):
        energy = sum(h.amplitude**2 for h in field.ordered() if h.scale == scale)
        assert energy == pytest.approx(1.0)


def test_resonance_selects_the_matching_scale():
    """A hypothesis is driven by its own band and barely hears the others.

    Compared as shares inside each scale group, because mass is conserved per
    scale: across groups the absolute amplitudes are not comparable by design.
    """

    rng = np.random.default_rng(3)
    dim = 32
    evidence = np.cos(np.pi * np.arange(dim, dtype=float)) + 0.01 * rng.normal(size=dim)
    bands = dyadic_decompose(evidence, 4)
    fine = bands[-1].unit()
    decoy = normalize_vector(rng.normal(size=dim))

    field = ResonantField(dim=dim)
    matched = field.spawn(fine, scale=3, label="matched", amplitude=0.1)
    matched_decoy = field.spawn(decoy, scale=3, label="decoy_fine", amplitude=0.1)
    mistuned = field.spawn(fine, scale=0, label="mistuned", amplitude=0.1)
    mistuned_decoy = field.spawn(decoy, scale=0, label="decoy_coarse", amplitude=0.1)
    ScaleInvariantOperator(OperatorParams(cross_scale_leak=0.05)).run(
        field, bands, steps=24
    )

    matched_share = matched.amplitude / (matched.amplitude + matched_decoy.amplitude)
    mistuned_share = mistuned.amplitude / (
        mistuned.amplitude + mistuned_decoy.amplitude
    )
    assert matched_share > mistuned_share
    assert matched_share > 0.8


def test_contradicted_hypotheses_end_up_in_antiphase():
    dim = 16
    evidence = np.zeros(dim)
    evidence[:8] = 1.0
    bands = dyadic_decompose(evidence, 2)
    field = ResonantField(dim=dim)
    aligned = field.spawn(bands[0].unit(), scale=0, label="with", amplitude=0.2)
    opposed = field.spawn(-bands[0].unit(), scale=0, label="against", amplitude=0.2)
    ScaleInvariantOperator().run(field, bands, steps=20)
    assert np.cos(aligned.lag) > 0.9
    assert np.cos(opposed.lag) < -0.5


def test_field_is_bounded_under_a_long_run():
    rng = np.random.default_rng(11)
    dim = 32
    bands = dyadic_decompose(rng.normal(size=dim), 4)
    field = ResonantField(dim=dim)
    for index in range(12):
        field.spawn(
            rng.normal(size=dim), scale=index % 4, label=f"c{index % 4}", amplitude=0.4
        )
    ScaleInvariantOperator().run(field, bands, steps=200)
    amplitudes = [h.amplitude for h in field.ordered()]
    assert amplitudes
    assert all(np.isfinite(value) for value in amplitudes)
    assert max(amplitudes) < 100.0


def test_operator_parameter_updates_stay_inside_their_bounds():
    params = OperatorParams()
    raised = params.with_deltas({"coupling": 99.0})
    lowered = params.with_deltas({"damping": -99.0})
    assert raised.coupling <= 1.5
    assert lowered.damping >= 0.01
    with pytest.raises(KeyError):
        params.with_deltas({"not_a_parameter": 1.0})


def test_pivotality_finds_the_band_that_decided_the_answer():
    """A band is pivotal when the read-out would land elsewhere without it."""

    field = ResonantField(dim=2)
    quiet = normalize_vector([1.0, 0.0])
    loud = normalize_vector([0.0, 1.0])
    for scale in (0, 1):
        hypothesis = field.spawn(quiet, scale=scale, label="quiet", amplitude=0.5)
        hypothesis.lag = 0.0
    decisive = field.spawn(loud, scale=2, label="loud", amplitude=1.2)
    decisive.lag = 0.0

    assert field.label_distribution()["loud"] > field.label_distribution()["quiet"]
    pivotal = field.band_pivotality("loud")
    assert pivotal[2] is True  # remove it and "quiet" wins
    assert pivotal[0] is False  # remove it and "loud" still wins
    assert pivotal[1] is False


def test_pivotality_is_empty_without_an_answer():
    field = ResonantField(dim=2)
    field.spawn([1.0, 0.0], scale=0, label="a", amplitude=0.5)
    field.spawn([0.0, 1.0], scale=1, label="b", amplitude=0.5)
    assert not any(field.band_pivotality("").values())


def test_band_trust_is_bounded_and_local():
    params = OperatorParams()
    tuned = params.with_band_trust(2, 0.5).with_band_trust(2, 99.0)
    low, high = BAND_TRUST_BOUNDS
    assert tuned.band_trust[2] == pytest.approx(high)
    assert tuned.band_trust[0] == 0.0
    assert tuned.band_trust[1] == 0.0
    floored = params.with_band_trust(0, -99.0)
    assert floored.band_trust[0] == pytest.approx(low)
    with pytest.raises(ValueError):
        params.with_band_trust(-1, 0.1)


def test_band_trust_changes_what_the_read_out_believes():
    """Trust reaches the answer, not just the parameters.

    Trust does not act by handing a band more amplitude -- mass is conserved
    per scale, so the normaliser would take that straight back.  It acts on how
    hard a band's evidence drives, which changes which hypotheses win *inside*
    that scale and how firmly they lock, and leaks across scales from there.
    So it only shows up in a field with something to compete: a pair of lone
    oscillators, each alone at its own scale, is normalised to a dead heat no
    matter what any of this is set to.

    That the shift is in the *right* direction is an outcome claim, measured
    over a whole stream in the integration tests rather than asserted here.
    """

    from rfc.engine import RFCConfig, ResonantFractalCognition
    from rfc.tasks import make_drift

    dataset = make_drift(0, trials=1)
    evidence = dataset.samples[0][0]

    def distribution(trust):
        config = RFCConfig(
            dim=dataset.dim, seed=0, params=OperatorParams(band_trust=trust)
        )
        mind = ResonantFractalCognition(config, codebook=dataset.codebook)
        return mind.perceive(evidence).alternatives

    uniform = distribution(())
    tilted = distribution((-1.0, -1.0, 1.0, 1.0))
    drift = sum(abs(tilted[key] - uniform[key]) for key in uniform)
    assert drift > 0.01
