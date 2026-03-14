"""Factorial and fractional factorial experimental designs.

Generates experiment configurations that test multiple variables simultaneously,
enabling detection of main effects and interactions that greedy one-at-a-time
optimization misses entirely.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from causal_optimizer.types import SearchSpace, Variable, VariableType


class FactorialDesigner:
    """Generate factorial experimental designs.

    Supports:
    - Full factorial: all combinations (2^k experiments for k binary factors)
    - Fractional factorial: subset that preserves main effects (2^(k-p) experiments)
    - Latin Hypercube: space-filling design for continuous variables
    """

    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space

    def full_factorial(self, levels: int = 2) -> list[dict[str, Any]]:
        """Generate a full factorial design.

        For k variables at `levels` levels each, produces levels^k experiments.
        """
        variable_levels = self._get_variable_levels(levels)
        combinations = list(itertools.product(*variable_levels.values()))

        designs = []
        for combo in combinations:
            design = {}
            for var_name, value in zip(variable_levels.keys(), combo, strict=False):
                design[var_name] = value
            designs.append(design)

        return designs

    def fractional_factorial(self, resolution: int = 3) -> list[dict[str, Any]]:
        """Generate a fractional factorial design.

        Resolution III: main effects estimable (confounded with 2-way interactions)
        Resolution IV: main effects clear, 2-way interactions confounded with each other
        Resolution V: main effects and 2-way interactions estimable
        """
        try:
            from pyDOE3 import fracfact
        except ImportError:
            # Fallback: generate a resolution III design manually
            return self._manual_fractional_factorial(resolution)

        k = self.search_space.dimensionality
        if k <= 1:
            return self.full_factorial()

        # Build generator string for pyDOE3
        generators = self._build_generator_string(k, resolution)
        design_matrix = fracfact(generators)

        return self._matrix_to_designs(design_matrix)

    def latin_hypercube(self, n_samples: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Generate a Latin Hypercube design for space-filling exploration."""
        from scipy.stats.qmc import LatinHypercube

        k = self.search_space.dimensionality
        sampler = LatinHypercube(d=k, seed=seed)
        samples = sampler.random(n=n_samples)  # values in [0, 1]

        designs = []
        for row in samples:
            design = {}
            for i, var in enumerate(self.search_space.variables):
                design[var.name] = self._scale_sample(var, row[i])
            designs.append(design)

        return designs

    def _get_variable_levels(self, n_levels: int) -> dict[str, list[Any]]:
        """Get discrete levels for each variable."""
        levels: dict[str, list[Any]] = {}
        for var in self.search_space.variables:
            if var.variable_type == VariableType.CATEGORICAL:
                levels[var.name] = (var.choices or [])[:n_levels]
            elif var.variable_type == VariableType.BOOLEAN:
                levels[var.name] = [False, True]
            elif var.lower is not None and var.upper is not None:
                levels[var.name] = list(np.linspace(var.lower, var.upper, n_levels))
            else:
                levels[var.name] = list(range(n_levels))
        return levels

    def _scale_sample(self, var: Variable, unit_value: float) -> Any:
        """Scale a [0, 1] sample to the variable's range."""
        if var.variable_type == VariableType.BOOLEAN:
            return bool(unit_value > 0.5)
        if var.variable_type == VariableType.CATEGORICAL:
            choices = var.choices or []
            idx = int(unit_value * len(choices)) % len(choices)
            return choices[idx]
        if var.lower is not None and var.upper is not None:
            value = var.lower + unit_value * (var.upper - var.lower)
            if var.variable_type == VariableType.INTEGER:
                return int(round(value))
            return value
        return unit_value

    def _build_generator_string(self, k: int, resolution: int) -> str:
        """Build pyDOE3 generator string for fractional factorial."""
        # For small k, just do full factorial
        if k <= 4:
            labels = "abcdefghijklmnop"[:k]
            return " ".join(labels)

        # Resolution III: each added factor = product of two base factors
        base_factors = max(3, k - (k - resolution))
        labels = "abcdefghijklmnop"
        generators = list(labels[:base_factors])

        for i in range(base_factors, k):
            # Generate from products of base factors
            gen = labels[i % base_factors] + labels[(i + 1) % base_factors]
            generators.append(gen)

        return " ".join(generators)

    def _manual_fractional_factorial(self, resolution: int) -> list[dict[str, Any]]:
        """Fallback fractional factorial without pyDOE3."""
        k = self.search_space.dimensionality
        # Use at most 2^(k-1) runs for resolution III
        n_runs = max(4, 2 ** max(2, k - (k // 3)))
        return self.latin_hypercube(n_runs)

    def _matrix_to_designs(self, matrix: np.ndarray) -> list[dict[str, Any]]:
        """Convert a coded design matrix (-1/+1) to actual parameter values."""
        designs = []
        for row in matrix:
            design = {}
            for i, var in enumerate(self.search_space.variables):
                # Map -1 to lower, +1 to upper
                unit = (row[i] + 1) / 2  # convert to [0, 1]
                design[var.name] = self._scale_sample(var, unit)
            designs.append(design)
        return designs
