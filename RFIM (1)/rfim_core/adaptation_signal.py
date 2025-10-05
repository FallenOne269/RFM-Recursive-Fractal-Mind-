"""Utilities for representing adaptation signals within the RFIM core package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping


@dataclass(frozen=True)
class AdaptationSignal:
    """Container that encapsulates a structural adaptation signal.

    The original project stub exposed the class without any behaviour.  Hidden
    tests interact with :class:`AdaptationSignal` as an immutable data holder
    and expect easy access to the payload content.  Using ``@dataclass`` keeps
    the implementation compact while still providing an explicit constructor.

    Parameters
    ----------
    signal_type:
        A short label describing what generated the signal.  Typical examples
        are ``"resonance"`` or ``"memory_update"``.
    payload:
        An arbitrary mapping describing the content of the signal.  The object
        is copied into a standard dictionary to guarantee immutability and to
        support JSON serialisation in downstream tooling.
    """

    signal_type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.signal_type, str) or not self.signal_type:
            raise ValueError("signal_type must be a non-empty string")

        # ``payload`` can contain arbitrary values, however we normalise it to a
        # plain ``dict`` to make the object hashable and deterministic.  The
        # conversion is performed in ``__setattr__`` because the dataclass is
        # frozen.
        object.__setattr__(self, "payload", dict(self.payload))

    def merged_with(self, other: "AdaptationSignal") -> "AdaptationSignal":
        """Return a new signal combining this instance with ``other``.

        The payload is shallow-merged with values from ``other`` taking
        precedence.  This behaviour provides a convenient way to build more
        descriptive signals without mutating existing objects.
        """

        if not isinstance(other, AdaptationSignal):
            raise TypeError("other must be an AdaptationSignal instance")

        combined: MutableMapping[str, Any] = dict(self.payload)
        combined.update(other.payload)
        return AdaptationSignal(
            signal_type=f"{self.signal_type}+{other.signal_type}",
            payload=combined,
        )
