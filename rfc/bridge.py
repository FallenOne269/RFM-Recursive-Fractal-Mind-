"""Adapters between RFC and the existing RFAI/RFIM code in this repository.

RFC is derived from the same research as the ``rfai`` package, so the two share
vocabulary: RFAI encodes data into Fractal Information Motifs, RFC resonates
over evidence vectors.  These helpers move between them, and lift RFAI's
``SemanticGoal`` into an RFC invariant so goal alignment can be enforced by the
constraint field rather than checked after the fact.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .constraints import Invariant
from .field import Hypothesis, normalize_vector

__all__ = ["evidence_from_fim", "encode_evidence", "goal_alignment_invariant"]


def _fit(vector: np.ndarray, dim: int) -> np.ndarray:
    """Deterministically fold/pad a vector to ``dim`` entries."""

    arr = np.asarray(vector, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(dim, dtype=float)
    if arr.size == dim:
        return arr.copy()
    if arr.size > dim:
        folded = np.zeros(dim, dtype=float)
        for index, value in enumerate(arr):
            folded[index % dim] += value
        return folded
    out = np.zeros(dim, dtype=float)
    out[: arr.size] = arr
    return out


def evidence_from_fim(fim: Any, dim: int = 32) -> np.ndarray:
    """Turn an RFAI ``FractalInformationMotif`` into an RFC evidence vector.

    The motif's semantic vector supplies the coarse structure and its AMIFS
    pattern signature (centroid and spread) modulates the fine structure, so a
    FIM lands in the scale ladder the way it was built: meaning at the top,
    generated detail underneath.
    """

    semantic = _fit(np.asarray(getattr(fim, "semantic_vector", []), dtype=float), dim)
    signature: Mapping[str, Any] = (
        getattr(fim, "metadata", {}).get("signature", {}) or {}
    )
    centroid = _fit(np.asarray(signature.get("centroid", []), dtype=float), dim)
    spread = _fit(np.asarray(signature.get("spread", []), dtype=float), dim)
    detail = np.zeros(dim, dtype=float)
    if np.any(centroid) or np.any(spread):
        ripple = np.cos(np.pi * np.arange(dim, dtype=float))
        detail = 0.35 * (centroid + spread * ripple)
    return normalize_vector(semantic + detail)


def encode_evidence(
    data: Any,
    dfe: Any,
    semantic_vector: Sequence[float] | np.ndarray,
    context: Optional[Dict[str, Any]] = None,
    dim: int = 32,
) -> np.ndarray:
    """Encode raw data through an RFAI ``DynamicFractalEncoder`` into evidence."""

    fim = dfe.encode(data, np.asarray(semantic_vector, dtype=float), context or {})
    return evidence_from_fim(fim, dim)


def goal_alignment_invariant(
    goal: Any,
    dim: int,
    min_similarity: float = -0.4,
    name: str = "goal_alignment",
    severity: float = 1.0,
) -> Invariant:
    """Veto hypotheses that point against an RFAI ``SemanticGoal``.

    Goal alignment usually lives in a scoring function that a search can trade
    away.  Here it becomes part of the substrate: a hypothesis pointing away
    from the goal is phase-inverted and starved, so it cannot accumulate
    support in the first place.
    """

    target = _fit(np.asarray(getattr(goal, "target_vector", goal), dtype=float), dim)
    target = normalize_vector(target)

    def predicate(hypothesis: Hypothesis, _context: Mapping[str, object]) -> bool:
        claim = normalize_vector(hypothesis.claim())
        if claim.size != target.size or not np.any(target):
            return False
        return float(np.dot(claim, target)) < min_similarity

    return Invariant(
        name=name,
        predicate=predicate,
        severity=severity,
        description=f"hypotheses below {min_similarity} similarity to the semantic goal",
    )
