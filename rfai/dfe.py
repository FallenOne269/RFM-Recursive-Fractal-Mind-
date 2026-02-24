from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import concurrent.futures
import numpy as np

from .amifs import AMIFS


def _featurize(data: Any) -> np.ndarray:
    """Simple deterministic featurizer."""
    if isinstance(data, (int, float)):
        return np.array([float(data)])
    if isinstance(data, dict):
        return np.array([len(data), sum(len(str(k)) for k in data.keys())], dtype=float)
    text = str(data)
    return np.array([len(text), sum(ord(c) for c in text) % 1000], dtype=float)


@dataclass
class FractalInformationMotif:
    fim_id: str
    amifs: AMIFS
    semantic_vector: np.ndarray
    complexity: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())


class DynamicFractalEncoder:
    def __init__(self, amifs: Optional[AMIFS] = None):
        self.amifs = amifs or AMIFS()
        self.registry: Dict[str, FractalInformationMotif] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"fim_{self._counter}"

    def encode(self, data: Any, semantic_vector: np.ndarray, context: Dict) -> FractalInformationMotif:
        features = _featurize(data)
        ctx_vec = np.asarray(list(context.values()), dtype=float) if context else np.zeros_like(features)
        params = self.amifs.adapt(features, ctx_vec)
        pattern = self.amifs.generate_fim_pattern(features, ctx_vec, steps=16)
        signature = self._signature(pattern)
        fim_id = self._next_id()
        fim = FractalInformationMotif(
            fim_id=fim_id,
            amifs=self.amifs,
            semantic_vector=np.asarray(semantic_vector, dtype=float),
            complexity=float(np.linalg.norm(semantic_vector)),
            metadata={"signature": signature, "params": params},
        )
        self.registry[fim_id] = fim
        return fim

    def _signature(self, pattern) -> Dict[str, float]:
        xp = np if not hasattr(pattern, "__array_namespace__") else None
        arr = np.asarray(pattern)
        centroid = arr.mean(axis=0)
        spread = arr.std(axis=0)
        return {"centroid": centroid.tolist(), "spread": spread.tolist()}

    def decode(self, fim_id: str) -> Dict[str, Any]:
        fim = self.registry.get(fim_id)
        if fim is None:
            raise KeyError(f"Unknown FIM {fim_id}")
        return {
            "fim_id": fim.fim_id,
            "semantic_vector": fim.semantic_vector.tolist(),
            "signature": fim.metadata.get("signature", {}),
        }

    def get(self, fim_id: str) -> Optional[FractalInformationMotif]:
        return self.registry.get(fim_id)

    def delete(self, fim_id: str) -> None:
        self.registry.pop(fim_id, None)

    def list(self) -> List[str]:
        return list(self.registry.keys())

    def link_parent_child(self, parent_id: str, child_id: str) -> None:
        parent = self.registry.get(parent_id)
        child = self.registry.get(child_id)
        if parent and child:
            child.parent_id = parent_id
            parent.children_ids.append(child_id)

    def encode_many(self, list_of_items: List[Any], n_workers: int = 0) -> List[FractalInformationMotif]:
        if not list_of_items:
            return []
        workers = n_workers or min(4, len(list_of_items))
        semantic_vector = np.ones(4)

        def _encode_item(item):
            return self.encode(item, semantic_vector, context={})

        if workers == 1:
            return [_encode_item(item) for item in list_of_items]

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_encode_item, list_of_items))
        return results

