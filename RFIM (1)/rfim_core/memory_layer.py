"""Memory layer utilities for the RFIM core package."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, MutableMapping, Tuple, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class MemoryLayer:
    """Simple fixed-size ordered dictionary used to persist state."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self._store: MutableMapping[K, V] = OrderedDict()

    # ------------------------------------------------------------------
    def add(self, key: K, value: V) -> None:
        """Insert ``value`` under ``key`` respecting the configured capacity."""

        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def get(self, key: K, default=None):
        return self._store.get(key, default)

    def items(self) -> Iterator[Tuple[K, V]]:
        return iter(self._store.items())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)

    def __contains__(self, key: K) -> bool:  # pragma: no cover - trivial
        return key in self._store
