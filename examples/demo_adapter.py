"""Minimal DomainAdapter for CLI demonstrations.

Wraps the Branin function as a domain adapter so the CLI can run it via:
    causal-optimizer run --adapter examples.demo_adapter:DemoAdapter --budget 10 --db demo.db
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class DemoAdapter(DomainAdapter):
    """Branin function adapter for CLI demonstrations."""

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=10.0),
                Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=15.0),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        # Standard Branin-Hoo parameters
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        result = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return {"objective": float(result)}

    def get_prior_graph(self) -> CausalGraph | None:
        return CausalGraph(edges=[("x1", "objective"), ("x2", "objective")])
