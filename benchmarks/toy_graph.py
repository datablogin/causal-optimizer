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
    """

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

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
        x = parameters.get("x", np.random.normal(0, 1))
        z_structural = np.exp(-x) + np.random.normal(0, self.noise_scale)
        z = parameters.get("z", z_structural)

        y = np.cos(z) - np.exp(-z / 20) + np.random.normal(0, self.noise_scale)
        return {"objective": -y}  # negate because we minimize; optimal Y is maximized
