from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any
import numpy as np


@dataclass
class FractalState:
    data: Dict[str, Any]

    def copy(self) -> "FractalState":
        return FractalState(data=dict(self.data))


class RecursiveFractalAlgorithm:
    def __init__(self, max_depth: int = 4, transform_gain: float = 1.0):
        self.max_depth = max_depth
        self.transform_gain = transform_gain
        self.adapt_threshold = 0.1

    def transform_func(self, state: FractalState, depth: int) -> FractalState:
        new_data = state.copy().data
        new_data["value"] = new_data.get("value", 0.0) + self.transform_gain / (depth + 1)
        new_data["depth"] = depth
        return FractalState(new_data)

    def recursive_process(self, state: FractalState, depth: int = 0) -> FractalState:
        current = self.transform_func(state, depth)
        if depth >= self.max_depth:
            return current
        return self.recursive_process(current, depth + 1)

    def self_improve(self, feedback: Dict[str, float]) -> None:
        performance = feedback.get("performance", 0.0)
        if performance > self.adapt_threshold:
            self.transform_gain *= 1.05
            self.adapt_threshold *= 1.1
        else:
            self.transform_gain *= 0.95
            self.adapt_threshold *= 0.9

