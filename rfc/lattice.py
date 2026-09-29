"""Symbols as a phase transition, not as a separate subsystem.

The neuro-symbolic literature usually bolts a symbolic module onto a neural
one and then struggles with the seam.  RFC has no seam because it has no second
module: a symbol is simply what a coalition becomes once it has stayed
coherent for long enough.  Sustained resonance crystallises into a durable
``Symbol``; the symbol then feeds back as a top-down prior on the very field
that produced it; and if the world stops supporting it, it dissolves.

Every symbol keeps its provenance -- the hypotheses that supported it, the
scales they lived at, and the episode it formed in -- so any downstream answer
can be traced back to the evidence that crystallised it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional

import numpy as np

from .field import Coalition, ResonantField, normalize_vector

__all__ = ["Symbol", "LatticeConfig", "SymbolLattice"]


@dataclass
class Symbol:
    """A crystallised, named regularity."""

    sid: str
    vector: np.ndarray
    label: str
    strength: float
    scales: List[int]
    support: List[str]
    created_step: int
    last_seen_step: int
    parent: Optional[str] = None
    children: List[str] = dataclass_field(default_factory=list)
    observations: int = 1

    def similarity(self, vector: np.ndarray) -> float:
        return float(np.dot(self.vector, normalize_vector(vector)))


@dataclass
class LatticeConfig:
    crystallize_amplitude: float = 0.55
    crystallize_coherence: float = 0.7
    persistence: int = 3
    merge_similarity: float = 0.9
    link_similarity: float = 0.6
    decay: float = 0.03
    dissolve_strength: float = 0.08
    max_symbols: int = 64


class SymbolLattice:
    """Holds crystallised symbols and the partial order between them."""

    def __init__(self, config: Optional[LatticeConfig] = None):
        self.config = config or LatticeConfig()
        self.symbols: Dict[str, Symbol] = {}
        self._streaks: Dict[str, int] = {}
        self._counter = 0

    # ------------------------------------------------------------------ build
    def _next_id(self) -> str:
        self._counter += 1
        return f"sym{self._counter}"

    @staticmethod
    def _coalition_key(coalition: Coalition) -> str:
        return "|".join(sorted(coalition.members))

    def observe(self, field: ResonantField, step: int) -> List[Symbol]:
        """Advance crystallisation by one step; return newly formed symbols."""

        config = self.config
        formed: List[Symbol] = []
        seen_keys: set[str] = set()
        for coalition in field.coalitions():
            if not coalition.label:
                continue
            key = self._coalition_key(coalition)
            seen_keys.add(key)
            resonant = (
                coalition.amplitude >= config.crystallize_amplitude
                and coalition.coherence >= config.crystallize_coherence
            )
            if not resonant:
                self._streaks[key] = 0
                continue
            streak = self._streaks.get(key, 0) + 1
            self._streaks[key] = streak
            if streak < config.persistence:
                continue
            symbol = self._crystallize(coalition, step)
            if symbol is not None:
                formed.append(symbol)
        for key in list(self._streaks):
            if key not in seen_keys:
                self._streaks.pop(key, None)
        self.decay(step)
        return formed

    def _find_existing(self, coalition: Coalition) -> Optional[Symbol]:
        """A symbol names a claim, so a label it already holds is the same symbol.

        Matching on centroid alone spawns a second copy every time the same
        claim crystallises with a slightly different amplitude profile across
        scales, which fills the lattice with near-duplicates of one concept.
        """

        if coalition.label:
            for symbol in sorted(self.symbols.values(), key=lambda s: s.sid):
                if symbol.label == coalition.label:
                    return symbol
        return self.match(coalition.centroid, self.config.merge_similarity)

    def _crystallize(self, coalition: Coalition, step: int) -> Optional[Symbol]:
        config = self.config
        existing = self._find_existing(coalition)
        if existing is not None:
            existing.strength = float(
                min(1.5, existing.strength + 0.1 * coalition.amplitude)
            )
            existing.last_seen_step = step
            existing.observations += 1
            existing.support = list(coalition.members)
            existing.scales = sorted(set(existing.scales) | set(coalition.scales))
            existing.vector = normalize_vector(
                0.9 * existing.vector + 0.1 * normalize_vector(coalition.centroid)
            )
            return None
        if len(self.symbols) >= config.max_symbols:
            weakest = min(self.symbols.values(), key=lambda s: (s.strength, s.sid))
            self.dissolve(weakest.sid)
        symbol = Symbol(
            sid=self._next_id(),
            vector=normalize_vector(coalition.centroid),
            label=coalition.label,
            strength=float(coalition.amplitude),
            scales=list(coalition.scales),
            support=list(coalition.members),
            created_step=step,
            last_seen_step=step,
        )
        self.symbols[symbol.sid] = symbol
        self._link(symbol)
        return symbol

    def _link(self, symbol: Symbol) -> None:
        """Attach the new symbol under its nearest established relative."""

        best: Optional[Symbol] = None
        best_score = self.config.link_similarity
        for other in self.symbols.values():
            if other.sid == symbol.sid:
                continue
            score = other.similarity(symbol.vector)
            if score > best_score or (
                score == best_score and best is not None and other.sid < best.sid
            ):
                best, best_score = other, score
        if best is not None:
            symbol.parent = best.sid
            best.children.append(symbol.sid)

    # ----------------------------------------------------------------- access
    def match(self, vector: np.ndarray, threshold: float) -> Optional[Symbol]:
        unit = normalize_vector(vector)
        best: Optional[Symbol] = None
        best_score = threshold
        for symbol in sorted(self.symbols.values(), key=lambda s: s.sid):
            score = symbol.similarity(unit)
            if score >= best_score:
                best, best_score = symbol, score
        return best

    def priors(self, field: ResonantField) -> Dict[str, float]:
        """Top-down bias: symbols push the field toward what they encode."""

        if not self.symbols:
            return {}
        priors: Dict[str, float] = {}
        for hypothesis in field.active():
            total = 0.0
            for symbol in self.symbols.values():
                overlap = symbol.similarity(hypothesis.vector)
                if overlap > 0.0:
                    total += symbol.strength * overlap
            if total > 0.0:
                priors[hypothesis.hid] = total
        maximum = max(priors.values(), default=0.0)
        if maximum > 0.0:
            priors = {hid: value / maximum for hid, value in priors.items()}
        return priors

    def decay(self, step: int) -> List[str]:
        """Symbols the world stopped supporting fade out and are removed."""

        removed: List[str] = []
        for symbol in list(self.symbols.values()):
            if symbol.last_seen_step < step:
                symbol.strength -= self.config.decay
            if symbol.strength <= self.config.dissolve_strength:
                removed.append(symbol.sid)
        for sid in removed:
            self.dissolve(sid)
        return removed

    def dissolve(self, sid: str) -> None:
        symbol = self.symbols.pop(sid, None)
        if symbol is None:
            return
        if symbol.parent and symbol.parent in self.symbols:
            parent = self.symbols[symbol.parent]
            parent.children = [child for child in parent.children if child != sid]
        for child in symbol.children:
            if child in self.symbols:
                self.symbols[child].parent = symbol.parent

    def rules(self) -> List[str]:
        """Human-readable rendering of the lattice, for explanations."""

        lines: List[str] = []
        for symbol in sorted(self.symbols.values(), key=lambda s: s.sid):
            parent = (
                f" < {self.symbols[symbol.parent].label}"
                if symbol.parent in self.symbols
                else ""
            )
            scales = ",".join(str(s) for s in symbol.scales)
            lines.append(
                f"{symbol.sid}: {symbol.label}{parent} "
                f"[scales {scales}] strength={symbol.strength:.2f} seen={symbol.observations}"
            )
        return lines

    def snapshot(self) -> Dict[str, object]:
        return {
            "count": len(self.symbols),
            "labels": sorted({s.label for s in self.symbols.values()}),
            "rules": self.rules(),
        }
