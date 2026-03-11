"""Interaction benchmark: two boolean variables that hurt alone but help together."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class InteractionBenchmark:
    """Two boolean variables that hurt alone but help together.

    Greedy hill-climbing fails; factorial design succeeds.

    Lower objective is better natively (no negation needed). When both
    ``use_a`` and ``use_b`` are True the objective drops by 3 from baseline,
    whereas activating either alone increases it.
    """

    def __init__(self, noise_scale: float = 0.2, rng: np.random.Generator | None = None) -> None:
        self.noise_scale = noise_scale
        self.baseline = 10.0
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def search_space() -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="use_a", variable_type=VariableType.BOOLEAN),
                Variable(name="use_b", variable_type=VariableType.BOOLEAN),
                Variable(
                    name="c_value",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )

    @staticmethod
    def causal_graph() -> CausalGraph:
        return CausalGraph(
            edges=[("use_a", "objective"), ("use_b", "objective"), ("c_value", "objective")],
            bidirected_edges=[],
        )

    @staticmethod
    def known_pomis() -> list[frozenset[str]]:
        return [frozenset({"use_a", "use_b", "c_value"})]

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run the interaction benchmark."""
        a = parameters.get("use_a", False)
        b = parameters.get("use_b", False)
        c = parameters.get("c_value", 0.5)

        result = self.baseline
        if a and not b:
            result += 2.0
        elif b and not a:
            result += 1.5
        elif a and b:
            result -= 3.0

        result += c * 0.1
        result += self.rng.normal(0, self.noise_scale)
        return {"objective": result}
