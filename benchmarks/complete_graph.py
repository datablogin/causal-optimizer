"""Complete graph benchmark: 7 variables + 2 unobserved confounders."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class CompleteGraphBenchmark:
    """7 variables + 2 unobserved confounders.

    Structural equations:
        U1 ~ N(0,1), U2 ~ N(0,1)  (unobserved)
        F = epsilon
        A = F^2 + U1 + epsilon
        B = U2 + epsilon
        C = exp(-B) + epsilon
        D = exp(-C)/10 + epsilon
        E = cos(A) + C/10 + epsilon
        Y = cos(D) - D/5 + sin(E) - E/4 + U1 + exp(-U2) + epsilon

    Known POMIS = [{b}, {d}, {e}, {b,d}, {d,e}].
    """

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

    @staticmethod
    def search_space() -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="f", variable_type=VariableType.CONTINUOUS, lower=-4.0, upper=4.0),
                Variable(name="a", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=20.0),
                Variable(name="b", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
                Variable(name="c", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=10.0),
                Variable(name="d", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
                Variable(name="e", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            ]
        )

    @staticmethod
    def causal_graph() -> CausalGraph:
        return CausalGraph(
            edges=[
                ("f", "a"),
                ("a", "e"),
                ("b", "c"),
                ("c", "d"),
                ("c", "e"),
                ("d", "objective"),
                ("e", "objective"),
            ],
            bidirected_edges=[
                ("a", "objective"),  # confounded via U1
                ("b", "objective"),  # confounded via U2
            ],
        )

    @staticmethod
    def known_pomis() -> list[frozenset[str]]:
        return [
            frozenset({"b"}),
            frozenset({"d"}),
            frozenset({"e"}),
            frozenset({"b", "d"}),
            frozenset({"d", "e"}),
        ]

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run the SCM with partial intervention semantics."""
        u1 = np.random.normal(0, 1)
        u2 = np.random.normal(0, 1)

        f = parameters.get("f", np.random.uniform(-4, 4))
        a = parameters.get("a", f**2 + u1 + np.random.normal(0, self.noise_scale))
        b = parameters.get("b", u2 + np.random.normal(0, self.noise_scale))
        c = parameters.get("c", np.exp(-b) + np.random.normal(0, self.noise_scale))
        d = parameters.get("d", np.exp(-c) / 10 + np.random.normal(0, self.noise_scale))
        e = parameters.get("e", np.cos(a) + c / 10 + np.random.normal(0, self.noise_scale))

        y = (
            np.cos(d)
            - d / 5
            + np.sin(e)
            - e / 4
            + u1
            + np.exp(-u2)
            + np.random.normal(0, self.noise_scale)
        )
        return {"objective": -y}
