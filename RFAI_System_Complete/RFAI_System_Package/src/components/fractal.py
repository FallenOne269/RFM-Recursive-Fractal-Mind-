"""Fractal processing components for the RFAI system."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FractalModule:
    """Self-similar processing module with recursive structure."""

    level: int
    dimensions: int
    sub_modules: List["FractalModule"]
    weights: np.ndarray
    activation_state: np.ndarray
    learning_rate: float

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.random.randn(self.dimensions, self.dimensions) * 0.1
        if self.activation_state is None:
            self.activation_state = np.zeros(self.dimensions)


class FractalEngine:
    """Constructs and executes the recursive fractal processing hierarchy."""

    def __init__(self, max_depth: int, base_dimensions: int) -> None:
        self.max_depth = max_depth
        self.base_dimensions = base_dimensions
        self.hierarchy = self._build_fractal_hierarchy()

    def _build_fractal_hierarchy(self) -> Dict[int, List[FractalModule]]:
        hierarchy: Dict[int, List[FractalModule]] = {}

        for level in range(self.max_depth):
            modules_at_level: List[FractalModule] = []
            num_modules = max(1, 2 ** (self.max_depth - level - 1))
            dims = max(4, self.base_dimensions // (2 ** level))

            for _ in range(num_modules):
                sub_modules: List[FractalModule] = []
                if level < self.max_depth - 1:
                    sub_count = max(1, min(4, 2 ** (self.max_depth - level - 2)))
                    sub_dims = max(4, dims // 2)
                    for _ in range(sub_count):
                        sub_modules.append(
                            FractalModule(
                                level=level + 1,
                                dimensions=sub_dims,
                                sub_modules=[],
                                weights=np.random.randn(sub_dims, sub_dims) * 0.1,
                                activation_state=np.zeros(sub_dims),
                                learning_rate=0.001 * (2 ** level),
                            )
                        )

                modules_at_level.append(
                    FractalModule(
                        level=level,
                        dimensions=dims,
                        sub_modules=sub_modules,
                        weights=np.random.randn(dims, dims) * 0.1,
                        activation_state=np.zeros(dims),
                        learning_rate=0.001 * (2 ** level),
                    )
                )

            hierarchy[level] = modules_at_level

        return hierarchy

    def prepare_input(self, raw_data: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Normalise raw input into the base dimensionality for processing."""

        if raw_data.ndim != 1:
            raise ValueError("Input data must be a 1D array")
        if raw_data.size == 0:
            raise ValueError("Input data cannot be empty")
        try:
            if np.isnan(raw_data).any() or np.isinf(raw_data).any():
                raise ValueError("Input data contains NaN or infinite values")
        except TypeError as exc:
            raise ValueError("Input data must be numeric") from exc

        if 0 in self.hierarchy and self.hierarchy[0]:
            target_dims = self.hierarchy[0][0].dimensions
        else:
            target_dims = self.base_dimensions

        if raw_data.size == target_dims:
            return raw_data
        if strict:
            raise ValueError(
                f"Input size {raw_data.size} does not match expected dimensions {target_dims}"
            )
        if raw_data.size > target_dims:
            return raw_data[:target_dims]

        padded = np.zeros(target_dims, dtype=raw_data.dtype)
        padded[: raw_data.size] = raw_data
        return padded

    def _resize_for_module(self, module: FractalModule, data: np.ndarray) -> np.ndarray:
        if data.size > module.dimensions:
            return data[: module.dimensions]
        if data.size < module.dimensions:
            padded = np.zeros(module.dimensions, dtype=data.dtype)
            padded[: data.size] = data
            return padded
        return data

    def process(self, input_data: np.ndarray, level: int = 0) -> np.ndarray:
        """Execute recursive fractal processing for the provided input."""

        if level >= self.max_depth:
            return input_data

        if input_data.ndim != 1:
            raise ValueError("Input data must be a 1D array")
        if input_data.size == 0:
            raise ValueError("Input data cannot be empty")
        if np.isnan(input_data).any() or np.isinf(input_data).any():
            raise ValueError("Input data contains NaN or infinite values")

        modules = self.hierarchy.get(level, [])
        if not modules:
            raise ValueError(f"No modules defined for level {level}")

        expected_size = modules[0].dimensions
        if input_data.size != expected_size:
            raise ValueError(
                f"Input size {input_data.size} does not match expected dimensions {expected_size}"
            )

        level_output = np.zeros(expected_size)

        for module in modules:
            module_input = self._resize_for_module(module, input_data.copy())

            try:
                transformed = np.tanh(module.weights @ module_input.reshape(-1, 1)).flatten()
            except ValueError:
                transformed = np.tanh(module_input * np.diag(module.weights))

            if module.sub_modules and level < self.max_depth - 1:
                next_input = self._resize_for_module(module.sub_modules[0], transformed)
                sub_output = self.process(next_input, level + 1)
                min_size = min(len(transformed), len(sub_output))
                combined = np.zeros(max(len(transformed), len(sub_output)))
                combined[:min_size] = 0.7 * transformed[:min_size] + 0.3 * sub_output[:min_size]
                if len(transformed) > min_size:
                    combined[min_size : min_size + len(transformed) - min_size] = transformed[min_size:]
                elif len(sub_output) > min_size:
                    combined[min_size : min_size + len(sub_output) - min_size] = sub_output[min_size:]
                transformed = combined[: module.dimensions]

            module.activation_state = 0.9 * module.activation_state + 0.1 * transformed

            output_size = min(len(level_output), len(transformed))
            level_output[:output_size] += transformed[:output_size] / len(modules)

        return level_output

    def total_parameters(self) -> int:
        """Return the total number of parameters across the hierarchy."""

        return sum(
            sum(module.weights.size for module in modules)
            for modules in self.hierarchy.values()
        )
