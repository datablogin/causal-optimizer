"""Harder counterfactual benchmark variants for causal differentiation.

Provides two variants of the base DemandResponseScenario that create
stronger positive-control pressure for causal guidance:

1. HighNoiseDemandResponse — 10+ irrelevant nuisance dimensions
2. ConfoundedDemandResponse — hidden confounder creating Simpson's paradox

Public API
----------
- :class:`HighNoiseDemandResponse`
- :class:`ConfoundedDemandResponse`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
)
from causal_optimizer.types import CausalGraph, SearchSpace


class HighNoiseDemandResponse(DemandResponseScenario):
    """High-dimensional noise variant of the demand-response benchmark.

    Adds 10 irrelevant nuisance dimensions to the search space while
    keeping the same 3 true causal parents.  Causal graph knowledge
    lets the optimizer focus on 3 dimensions instead of 13+.
    """

    @staticmethod
    def search_space() -> SearchSpace:
        raise NotImplementedError

    @staticmethod
    def causal_graph() -> CausalGraph:
        raise NotImplementedError

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        raise NotImplementedError


class ConfoundedDemandResponse(DemandResponseScenario):
    """Confounded treatment assignment variant of the demand-response benchmark.

    Introduces a hidden confounder (grid stress) that affects both
    treatment assignment probability and the outcome, creating a
    Simpson's paradox scenario.
    """

    @staticmethod
    def search_space() -> SearchSpace:
        raise NotImplementedError

    @staticmethod
    def causal_graph() -> CausalGraph:
        raise NotImplementedError

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError

    def naive_policy(self, data: pd.DataFrame, cost: float) -> Any:
        raise NotImplementedError

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        raise NotImplementedError
