"""Toy domain adapter for CLI testing."""

from __future__ import annotations

from typing import Any

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import SearchSpace, Variable, VariableType


class ToyAdapter(DomainAdapter):
    """Simple quadratic adapter for testing."""

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
                Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}
