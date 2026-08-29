"""Dyadic scale-space decomposition.

The Resonant Fractal Cognition (RFC) substrate never looks at evidence "as a
whole".  Evidence is split into a ladder of dyadic scale bands, and every band
is handed to hypotheses tuned to that band's frequency.  The ladder is what
makes the architecture fractal in a literal, checkable sense: band ``l`` is the
same operation as band ``l - 1`` applied at half the support.

Two decompositions are provided and they are deliberately the same idea applied
to two different axes:

``dyadic_decompose``
    splits a *feature vector* across spatial scale.
``temporal_bands``
    splits a *history of feature vectors* across time scale.

The second one is what metacognition consumes, which is how the system ends up
reflecting on itself with exactly the machinery it uses to perceive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

__all__ = ["ScaleBand", "dyadic_decompose", "temporal_bands", "band_matrix"]


@dataclass(frozen=True)
class ScaleBand:
    """One rung of the scale ladder.

    ``level`` 0 is the coarsest band; larger levels carry finer detail.  The
    vector always has the dimensionality of the original signal so that bands
    from different levels can be compared directly.
    """

    level: int
    vector: np.ndarray
    energy: float

    def unit(self) -> np.ndarray:
        norm = float(np.linalg.norm(self.vector))
        if norm <= 1e-12:
            return np.zeros_like(self.vector)
        return self.vector / norm


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _box_smooth(x: np.ndarray, factor: int) -> np.ndarray:
    """Average ``x`` over non-overlapping blocks of ``factor`` and re-expand."""

    if factor <= 1:
        return x.copy()
    blocks = x.reshape(-1, factor)
    means = blocks.mean(axis=1, keepdims=True)
    return np.repeat(means, factor, axis=1).reshape(-1)


def dyadic_decompose(signal: Sequence[float] | np.ndarray, levels: int = 4) -> List[ScaleBand]:
    """Split ``signal`` into ``levels`` bands that sum back to the original.

    Uses a box (Haar) pyramid: successively coarser block averages, with each
    band holding the detail that the next-coarser band cannot explain.  The
    reconstruction property (``sum(band.vector) == signal``) is exact up to
    floating point and is covered by the test suite -- it is what lets the
    engine subtract an explained component and recurse on the residual.
    """

    if levels < 1:
        raise ValueError("levels must be >= 1")
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("signal must be non-empty")

    original_dim = x.size
    padded = np.zeros(_next_pow2(max(x.size, 1 << (levels - 1))), dtype=float)
    padded[: x.size] = x

    # Approximation ladder: approx[0] is the signal, approx[k] is the block
    # average over 2**k samples.  Box filters compose, so a single smoothing
    # pass per level is equivalent to iterated smoothing.
    approx = [padded]
    for level in range(1, levels):
        factor = min(1 << level, padded.size)
        approx.append(_box_smooth(padded, factor))

    bands: List[ScaleBand] = []
    coarsest = approx[levels - 1][:original_dim]
    bands.append(ScaleBand(0, coarsest, float(np.dot(coarsest, coarsest))))
    for level in range(1, levels):
        detail = (approx[levels - 1 - level] - approx[levels - level])[:original_dim]
        bands.append(ScaleBand(level, detail, float(np.dot(detail, detail))))
    return bands


def temporal_bands(history: Sequence[Sequence[float]] | np.ndarray, levels: int = 3) -> List[ScaleBand]:
    """Decompose a history of feature vectors across *time* scale.

    Band 0 is the long-run average of the whole window.  Band ``l`` is the
    average over the most recent ``T / 2**l`` rows minus the average over the
    window twice that length, i.e. "what changed when you looked twice as
    recently".  Slow drifts land in coarse bands, transients in fine ones.
    """

    if levels < 1:
        raise ValueError("levels must be >= 1")
    rows = np.asarray(history, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("history must be a non-empty 2-D array")

    total = rows.shape[0]

    def window_mean(count: int) -> np.ndarray:
        count = max(1, min(total, count))
        return rows[-count:].mean(axis=0)

    bands: List[ScaleBand] = []
    base = window_mean(total)
    bands.append(ScaleBand(0, base, float(np.dot(base, base))))
    for level in range(1, levels):
        recent = window_mean(total >> level)
        previous = window_mean(total >> (level - 1))
        detail = recent - previous
        bands.append(ScaleBand(level, detail, float(np.dot(detail, detail))))
    return bands


def band_matrix(bands: Sequence[ScaleBand], equalization: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Return unit band directions and their drive weights.

    ``equalization`` interpolates between honouring raw band energy (0.0) and
    giving every band an equal vote (1.0).  Partial equalization is the reason
    RFC can hear a low-energy fine-scale cue that a plain cosine match against
    the raw vector drowns out.
    """

    if not bands:
        raise ValueError("at least one band is required")
    equalization = float(np.clip(equalization, 0.0, 1.0))
    directions = np.stack([band.unit() for band in bands])
    energies = np.array([max(band.energy, 0.0) for band in bands], dtype=float)
    total = float(energies.sum())
    if total <= 1e-12:
        shares = np.full(energies.shape, 1.0 / energies.size)
    else:
        shares = energies / total
    weights = shares ** (1.0 - equalization)
    weight_sum = float(weights.sum())
    if weight_sum > 1e-12:
        weights = weights / weight_sum * weights.size
    return directions, weights
