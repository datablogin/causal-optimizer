"""Toy graph benchmark: X -> Z -> Y with POMIS = [{Z}]."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class ToyGraphBenchmark:
    """X -> Z -> Y. POMIS = [{Z}].

    Structural equations:
        X = epsilon_0
        Z = exp(-X) + epsilon_1
        Y = cos(Z) - exp(-Z/20) + epsilon_2

    The optimal intervention is on Z alone.

    This is a maximization problem internally (higher Y is better).
    The objective is negated so that the engine can minimize: objective = -Y.
    """

    def __init__(self, noise_scale: float = 0.1, rng: np.random.Generator | None = None) -> None:
        self.noise_scale = noise_scale
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def search_space() -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
                Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=20.0),
            ]
        )

    @staticmethod
    def causal_graph() -> CausalGraph:
        return CausalGraph(
            edges=[("x", "z"), ("z", "objective")],
            bidirected_edges=[],
        )

    @staticmethod
    def known_pomis() -> list[frozenset[str]]:
        return [frozenset({"z"})]

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run the SCM with partial intervention semantics.

        If only z is provided, x is sampled. If only x is provided, z follows structural eq.
        If both provided, both are fixed.
        """
        # X's prior is N(0, 1) — separate from structural noise_scale
        x = parameters.get("x", self.rng.normal(0, 1))
        z_structural = np.exp(-x) + self.rng.normal(0, self.noise_scale)
        z = parameters.get("z", z_structural)

        y = np.cos(z) - np.exp(-z / 20) + self.rng.normal(0, self.noise_scale)
        return {"objective": -y}  # negate because we minimize; optimal Y is maximized
