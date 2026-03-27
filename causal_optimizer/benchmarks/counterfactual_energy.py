"""Semi-synthetic counterfactual energy benchmark — demand response scenario.

Uses real ERCOT covariates with known treatment effects to enable
counterfactual evaluation of optimization strategies.

Public API
----------
- :class:`DemandResponseScenario` — generates semi-synthetic data.
- :class:`CounterfactualBenchmarkResult` — result container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from causal_optimizer.types import CausalGraph, SearchSpace


@dataclass
class CounterfactualBenchmarkResult:
    """Result of running one strategy on the counterfactual benchmark.

    Attributes:
        strategy: Optimization strategy name.
        budget: Number of experiments in the optimization run.
        seed: Random seed used.
        policy_value: Average outcome under the learned policy.
        oracle_value: Average outcome under the oracle policy.
        regret: oracle_value - policy_value.
        treatment_effect_mae: MAE of estimated vs true effects.
        runtime_seconds: Wall-clock time for the full run.
    """

    strategy: str
    budget: int
    seed: int
    policy_value: float
    oracle_value: float
    regret: float
    treatment_effect_mae: float
    runtime_seconds: float


class DemandResponseScenario:
    """Semi-synthetic demand-response scenario using real covariates.

    Stub — not yet implemented.
    """

    treatment_cost: float

    def __init__(
        self,
        covariates: pd.DataFrame,
        seed: int = 0,
        treatment_cost: float = 50.0,
    ) -> None:
        raise NotImplementedError

    def generate(self) -> pd.DataFrame:
        """Generate semi-synthetic dataset with counterfactual outcomes."""
        raise NotImplementedError

    @staticmethod
    def causal_graph() -> CausalGraph:
        """Return the known causal graph for the demand-response scenario."""
        raise NotImplementedError

    @staticmethod
    def search_space() -> SearchSpace:
        """Return the policy search space."""
        raise NotImplementedError

    def oracle_policy_value(self, data: pd.DataFrame) -> float:
        """Compute the value of the oracle policy on the given data."""
        raise NotImplementedError

    def run_benchmark(
        self,
        budget: int,
        seed: int,
        strategy: str = "random",
    ) -> CounterfactualBenchmarkResult:
        """Run one strategy on this scenario and return results."""
        raise NotImplementedError
