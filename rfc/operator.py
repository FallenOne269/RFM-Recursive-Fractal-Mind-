"""The scale-invariant cognitive operator, written once and used everywhere.

The research this repository collects describes three stacked layers -- a
quantum-inspired substrate, a neuro-symbolic middle, and a metacognitive top --
and then observes that integrating them is the hard part.  RFC's answer is to
refuse the stack: there is one operator, ``Psi``, and the three "layers" are
that same operator applied at different scales and to different evidence.
Perception applies it to sensory bands.  Symbol formation reads its fixed
points.  Metacognition applies it to a record of its own behaviour.

Mechanically ``Psi`` advances a bank of damped, driven, phase-coupled
oscillators for one tick:

1. **Drive.**  Each hypothesis is pushed by evidence bands.  A hypothesis at
   scale ``s`` has natural frequency ``w0 * 2**-s``, and each band ``l`` drives
   with reference phase ``w_l * t``.  A hypothesis therefore holds a constant
   phase lag against its own band and accumulates amplitude, while its lag
   against any other band rotates and the drive time-averages to nothing.
   Scale selectivity is not a rule anyone wrote down; it falls out of the
   dyadic frequency ladder.
2. **Coupling.**  Two things happen between hypotheses, and they pull in
   opposite directions on purpose.  *Phases* couple Kuramoto-style, weighted by
   vector overlap, so hypotheses that agree lock together into coalitions.
   *Amplitudes* compete: hypotheses that overlap are, by construction, rival
   explanations of the same evidence, so each one suppresses the others in
   proportion to how much they overlap.  Cooperation in phase, competition in
   mass -- the on-center/off-surround arrangement that lets a small margin
   between two near-identical readings sharpen instead of blurring away.
3. **Constraints.**  Violators are inverted and starved (see ``constraints``).
4. **Damping, conservation, pruning.**  Amplitudes decay, the field is
   renormalised so total probability is conserved, and negligible hypotheses
   are dropped so the population cannot explode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .constraints import ConstraintField, ConstraintReport
from .field import ResonantField
from .scale_space import ScaleBand, band_matrix

__all__ = [
    "OperatorParams",
    "StepReport",
    "ScaleInvariantOperator",
    "TUNABLE_PARAMETERS",
]

_EPS = 1e-12

#: Parameters metacognition is allowed to move, with their hard bounds.
TUNABLE_PARAMETERS: Dict[str, tuple[float, float]] = {
    "coupling": (0.0, 1.5),
    "competition": (0.0, 1.5),
    "damping": (0.01, 1.0),
    "drive_gain": (0.05, 2.0),
    "inhibition": (0.0, 2.0),
    "cross_scale_leak": (0.0, 0.6),
    "band_equalization": (0.0, 1.0),
    "band_tilt": (-1.5, 1.5),
}


@dataclass(frozen=True)
class OperatorParams:
    """Physics of the field.  Frozen so adaptation is explicit and auditable."""

    base_frequency: float = 1.0
    coupling: float = 0.45
    competition: float = 0.2
    damping: float = 0.15
    drive_gain: float = 0.75
    inhibition: float = 0.7
    cross_scale_leak: float = 0.1
    band_equalization: float = 0.25
    band_tilt: float = 0.0
    prior_gain: float = 0.35
    dt: float = 0.35
    prune_threshold: float = 0.002
    prune_after: int = 12
    energy_cap: float = 12.0
    max_phase_rate: float = 12.0

    def with_deltas(self, deltas: Mapping[str, float]) -> "OperatorParams":
        """Return a copy with bounded parameter nudges applied."""

        updates: Dict[str, float] = {}
        for name, delta in deltas.items():
            if name not in TUNABLE_PARAMETERS:
                raise KeyError(f"{name} is not a tunable parameter")
            low, high = TUNABLE_PARAMETERS[name]
            updates[name] = float(
                np.clip(getattr(self, name) + float(delta), low, high)
            )
        return replace(self, **updates)

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in TUNABLE_PARAMETERS}


@dataclass
class StepReport:
    step: int
    coherence: float
    mean_amplitude: float
    drive_energy: float
    pruned: List[str]
    constraints: ConstraintReport
    leader: Optional[str]
    labels: Dict[str, float]


class ScaleInvariantOperator:
    """One application of ``Psi``.  Stateless apart from its parameters."""

    def __init__(self, params: Optional[OperatorParams] = None):
        self.params = params or OperatorParams()

    # ------------------------------------------------------------------ utils
    def natural_frequency(self, scale: int) -> float:
        """Dyadic frequency ladder -- the fractal part of the substrate."""

        return float(self.params.base_frequency * (2.0 ** -int(scale)))

    # ------------------------------------------------------------------- step
    def step(
        self,
        field: ResonantField,
        bands: Sequence[ScaleBand],
        time: float,
        step_index: int = 0,
        constraints: Optional[ConstraintField] = None,
        context: Optional[Mapping[str, object]] = None,
        priors: Optional[Mapping[str, float]] = None,
    ) -> StepReport:
        params = self.params
        items = field.ordered()
        if not items:
            return StepReport(
                step_index, 0.0, 0.0, 0.0, [], ConstraintReport(), None, {}
            )

        vectors = np.stack([h.vector for h in items])
        amplitudes = np.array([h.amplitude for h in items], dtype=float)
        phases = np.array([h.phase for h in items], dtype=float)
        scales = np.array([h.scale for h in items], dtype=int)
        vetoed = np.array([h.vetoed for h in items], dtype=bool)

        directions, weights = band_matrix(bands, params.band_equalization)
        band_levels = np.array([band.level for band in bands], dtype=int)
        if params.band_tilt:
            # How much to trust each scale.  Positive tilt leans on fine
            # detail, negative on coarse shape.  Metacognition owns this knob:
            # it is how the system says "my coarse evidence has stopped being
            # reliable" without anyone telling it which scale went bad.
            centre = (band_levels.max() + band_levels.min()) / 2.0
            tilt = 2.0 ** (params.band_tilt * (band_levels.astype(float) - centre))
            weights = weights * tilt
            weights = weights / max(float(weights.mean()), 1e-12)
        band_frequencies = params.base_frequency * (2.0 ** -band_levels.astype(float))
        band_phases = band_frequencies * float(time)

        # --- 1. drive -------------------------------------------------------
        alignment = vectors @ directions.T  # (n, L)
        own_band = scales[:, None] == band_levels[None, :]
        gate = np.where(own_band, 1.0, params.cross_scale_leak)
        gain = gate * weights[None, :]
        lag = phases[:, None] - band_phases[None, :]
        own_reference = self.natural_frequency_vector(scales) * float(time)

        # Only *supporting* evidence can drive a hypothesis into resonance.
        # Without this split a driven oscillator would happily lock in
        # antiphase to evidence that contradicts it and grow just as fast:
        # correct physics, wrong semantics.  Contradicting evidence instead
        # inhibits, in proportion to how strongly it disagrees.
        support = np.maximum(alignment, 0.0)
        conflict = np.maximum(-alignment, 0.0)
        drive_amp = params.drive_gain * np.sum(gain * support * np.cos(lag), axis=1)
        drive_amp -= params.inhibition * np.sum(gain * conflict, axis=1) * amplitudes

        # Phase, unlike amplitude, follows the *signed* alignment.  Supporting
        # evidence pulls a hypothesis to zero lag; contradicting evidence pulls
        # it to antiphase, where whatever amplitude it still has subtracts from
        # its own label in the cross-scale read-out.  So a claim that one scale
        # flatly refutes cannot be rescued by another scale liking it: the two
        # contributions cancel, which is what "the scales have to agree" means
        # when belief is an interference pattern.
        #
        # The lag is pulled toward the target the evidence names rather than by
        # a sinusoidal torque.  A torque vanishes at zero lag for *either* sign
        # of the evidence, so a hypothesis seeded with no opinion would sit
        # there being contradicted and still counting as agreement.
        net_drive = np.sum(gain * alignment, axis=1)
        target_lag = np.where(net_drive >= 0.0, 0.0, np.pi)
        lag_error = np.angle(np.exp(1j * (target_lag - (phases - own_reference))))
        drive_phase = params.drive_gain * np.abs(net_drive) * lag_error

        # A vetoed hypothesis never receives positive drive again, which is what
        # makes "amplitude is non-increasing after a veto" hold exactly.
        drive_amp = np.where(vetoed, np.minimum(drive_amp, 0.0), drive_amp)

        # --- 2. coupling ----------------------------------------------------
        overlap = vectors @ vectors.T
        np.fill_diagonal(overlap, 0.0)
        state = amplitudes * np.exp(1j * phases)
        coupling_phase = params.coupling * np.imag(
            np.exp(-1j * phases) * (overlap @ state)
        )
        rivalry = np.maximum(overlap, 0.0)
        coupling_amp = -params.competition * (rivalry @ amplitudes)

        # --- 3. top-down priors from crystallised symbols -------------------
        prior_vector = np.zeros_like(amplitudes)
        if priors:
            for index, hypothesis in enumerate(items):
                prior_vector[index] = float(priors.get(hypothesis.hid, 0.0))
            prior_vector = np.where(vetoed, 0.0, prior_vector)
            prior_vector *= params.prior_gain

        # --- 4. integrate ---------------------------------------------------
        d_amp = -params.damping * amplitudes + drive_amp + coupling_amp + prior_vector
        d_phase = self.natural_frequency_vector(scales) + drive_phase + coupling_phase
        d_phase = np.clip(d_phase, -params.max_phase_rate, params.max_phase_rate)

        new_amplitudes = np.clip(amplitudes + params.dt * d_amp, 0.0, None)
        new_phases = (phases + params.dt * d_phase) % (2.0 * np.pi)

        # Phase lag against each hypothesis' own drive.  A resonantly locked
        # hypothesis holds this nearly constant while its lab-frame phase keeps
        # turning, which is why every agreement measure is taken here.
        next_time = float(time) + params.dt
        new_lags = (new_phases - self.natural_frequency_vector(scales) * next_time) % (
            2.0 * np.pi
        )

        for index, hypothesis in enumerate(items):
            hypothesis.amplitude = float(new_amplitudes[index])
            hypothesis.phase = float(new_phases[index])
            hypothesis.lag = float(new_lags[index])
            hypothesis.age += 1

        # --- 5. constraints, conservation, pruning --------------------------
        report = (
            constraints.apply(field, context)
            if constraints is not None
            else ConstraintReport()
        )
        # Mass is conserved per scale, so no band can out-shout another, and
        # the energy cap catches anything the normaliser cannot (an empty or
        # degenerate scale group).
        field.normalize()
        field.limit_energy(params.energy_cap)
        pruned = (
            field.prune(params.prune_threshold)
            if step_index >= params.prune_after
            else []
        )

        drive_energy = float(np.sum(gain * support))
        active = field.active()
        mean_amplitude = (
            float(np.mean([h.amplitude for h in active])) if active else 0.0
        )
        leader = field.measure()
        return StepReport(
            step=step_index,
            coherence=field.coherence(),
            mean_amplitude=mean_amplitude,
            drive_energy=drive_energy,
            pruned=pruned,
            constraints=report,
            leader=leader.hid if leader else None,
            labels=field.label_distribution(),
        )

    def natural_frequency_vector(self, scales: np.ndarray) -> np.ndarray:
        return self.params.base_frequency * (2.0 ** -scales.astype(float))

    def run(
        self,
        field: ResonantField,
        bands: Sequence[ScaleBand],
        steps: int,
        constraints: Optional[ConstraintField] = None,
        context: Optional[Mapping[str, object]] = None,
        prior_fn=None,
        observer=None,
        start_time: float = 0.0,
    ) -> List[StepReport]:
        """Advance the field ``steps`` ticks and return the per-step reports."""

        reports: List[StepReport] = []
        time = float(start_time)
        for index in range(int(steps)):
            priors = prior_fn(field) if prior_fn is not None else None
            report = self.step(field, bands, time, index, constraints, context, priors)
            reports.append(report)
            if observer is not None:
                observer(field, report)
            time += self.params.dt
        return reports
