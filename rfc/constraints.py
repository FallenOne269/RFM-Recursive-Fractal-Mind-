"""Constraints as physics rather than as a post-filter.

Most guarded systems generate a candidate and then ask a filter whether the
candidate is allowed.  RFC pushes the guard one level down: a hypothesis that
violates an invariant is phase-inverted, cut off from drive, and excluded from
measurement.  It therefore interferes *destructively* with the coalition it
belongs to, so it cannot recruit support, and its amplitude is non-increasing
from the step it is flagged onward.  That property is checkable, and the test
suite checks it.

The invariants themselves are ordinary Python predicates, so the repository's
existing "ethical DNA" style policies can be lifted in unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .field import Hypothesis, ResonantField, normalize_vector

__all__ = [
    "Invariant",
    "Violation",
    "ConstraintReport",
    "ConstraintField",
    "forbidden_direction",
    "forbidden_labels",
]


@dataclass
class Invariant:
    """A named guard over hypotheses.

    ``predicate`` returns True when the hypothesis *violates* the invariant.
    ``severity`` in ``(0, 1]`` scales how hard the amplitude is cut on the step
    the violation is detected; regardless of severity the hypothesis is vetoed,
    which is what actually makes it unselectable.
    """

    name: str
    predicate: Callable[[Hypothesis, Mapping[str, object]], bool]
    severity: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        self.severity = float(np.clip(self.severity, 1e-3, 1.0))

    def violated_by(
        self, hypothesis: Hypothesis, context: Mapping[str, object]
    ) -> bool:
        return bool(self.predicate(hypothesis, context))


@dataclass
class Violation:
    hid: str
    invariant: str
    severity: float


@dataclass
class ConstraintReport:
    violations: List[Violation] = dataclass_field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.violations)

    def by_invariant(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for violation in self.violations:
            counts[violation.invariant] = counts.get(violation.invariant, 0) + 1
        return counts


class ConstraintField:
    """Applies invariants to a field by inverting and starving violators."""

    def __init__(self, invariants: Optional[Sequence[Invariant]] = None):
        self.invariants: List[Invariant] = list(invariants or [])

    def add(self, invariant: Invariant) -> None:
        self.invariants.append(invariant)

    def apply(
        self, field: ResonantField, context: Optional[Mapping[str, object]] = None
    ) -> ConstraintReport:
        report = ConstraintReport()
        if not self.invariants:
            return report
        ctx: Mapping[str, object] = context or {}
        for hypothesis in field.ordered():
            for invariant in self.invariants:
                if not invariant.violated_by(hypothesis, ctx):
                    continue
                report.violations.append(
                    Violation(hypothesis.hid, invariant.name, invariant.severity)
                )
                if not hypothesis.vetoed:
                    # Phase inversion: from here on the hypothesis subtracts
                    # from any coalition that tries to absorb it.
                    hypothesis.phase = float((hypothesis.phase + np.pi) % (2.0 * np.pi))
                    hypothesis.vetoed = True
                    hypothesis.veto_reason = invariant.name
                hypothesis.amplitude *= 1.0 - invariant.severity
                break
        return report


def forbidden_direction(
    name: str,
    vector: Sequence[float] | np.ndarray,
    threshold: float = 0.75,
    severity: float = 1.0,
    description: str = "",
) -> Invariant:
    """Veto any hypothesis pointing too close to a forbidden direction."""

    target = normalize_vector(vector)

    def predicate(hypothesis: Hypothesis, _context: Mapping[str, object]) -> bool:
        claim = hypothesis.claim()
        if claim.size != target.size:
            return False
        return float(np.dot(normalize_vector(claim), target)) >= threshold

    return Invariant(
        name=name, predicate=predicate, severity=severity, description=description
    )


def forbidden_labels(
    name: str, labels: Sequence[str], severity: float = 1.0, description: str = ""
) -> Invariant:
    """Veto hypotheses carrying one of ``labels``."""

    banned = set(labels)

    def predicate(hypothesis: Hypothesis, _context: Mapping[str, object]) -> bool:
        return hypothesis.label in banned

    return Invariant(
        name=name, predicate=predicate, severity=severity, description=description
    )
