"""Benchmark structural causal models for testing the optimization engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from causal_optimizer.benchmarks.complete_graph import CompleteGraphBenchmark
from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
    PolicyRunner,
    evaluate_policy,
    net_benefit,
    propensity,
    treatment_effect,
)
from causal_optimizer.benchmarks.counterfactual_variants import (
    ConfoundedDemandResponse,
    HighNoiseDemandResponse,
    MediumNoiseDemandResponse,
)
from causal_optimizer.benchmarks.dose_response import (
    DoseResponseBenchmarkResult,
    DoseResponseScenario,
    ProtocolRunner,
    dose_response_effect,
    evaluate_protocol,
)
from causal_optimizer.benchmarks.high_dimensional import HighDimensionalSparseBenchmark
from causal_optimizer.benchmarks.hillstrom import (
    HILLSTROM_FROZEN_PARAMS,
    HILLSTROM_POOLED_PROPENSITY,
    HILLSTROM_PRIMARY_PROPENSITY,
    HillstromBenchmarkResult,
    HillstromPolicyRunner,
    HillstromScenario,
    HillstromSliceType,
    hillstrom_active_search_space,
    hillstrom_null_baseline,
    hillstrom_projected_prior_graph,
    load_hillstrom_slice,
    permute_hillstrom_spend,
)
from causal_optimizer.benchmarks.hillstrom import (
    VALID_STRATEGIES as HILLSTROM_VALID_STRATEGIES,
)
from causal_optimizer.benchmarks.interaction import InteractionBenchmark
from causal_optimizer.benchmarks.interaction_policy import (
    InteractionPolicyScenario,
    evaluate_interaction_policy,
    interaction_propensity,
    interaction_treatment_effect,
)
from causal_optimizer.benchmarks.interaction_scm import InteractionSCM
from causal_optimizer.benchmarks.null_predictive_energy import (
    NullSignalResult,
    check_null_signal,
    permute_target,
    run_null_strategy,
)
from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    ValidationEnergyRunner,
    evaluate_on_test,
    load_energy_frame,
    split_time_frame,
)
from causal_optimizer.benchmarks.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    sample_random_params,
)
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, SearchSpace


class BenchmarkSCM(Protocol):
    """Protocol that all benchmark SCMs must satisfy.

    Provides a search space, causal graph, known POMIS sets, and a run method
    that executes the structural equations under (partial) intervention.

    All implementations should accept ``noise_scale`` and optional ``rng``
    in ``__init__``, but Protocol cannot enforce constructor signatures.
    """

    noise_scale: float

    @staticmethod
    def search_space() -> SearchSpace: ...
    @staticmethod
    def causal_graph() -> CausalGraph: ...
    @staticmethod
    def known_pomis() -> list[frozenset[str]]: ...
    def run(self, parameters: dict[str, Any]) -> dict[str, float]: ...


__all__ = [
    "HILLSTROM_FROZEN_PARAMS",
    "HILLSTROM_POOLED_PROPENSITY",
    "HILLSTROM_PRIMARY_PROPENSITY",
    "HILLSTROM_VALID_STRATEGIES",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSCM",
    "CompleteGraphBenchmark",
    "ConfoundedDemandResponse",
    "CounterfactualBenchmarkResult",
    "DemandResponseScenario",
    "DoseResponseBenchmarkResult",
    "DoseResponseScenario",
    "HighDimensionalSparseBenchmark",
    "HighNoiseDemandResponse",
    "HillstromBenchmarkResult",
    "HillstromPolicyRunner",
    "HillstromScenario",
    "HillstromSliceType",
    "InteractionBenchmark",
    "InteractionPolicyScenario",
    "InteractionSCM",
    "MediumNoiseDemandResponse",
    "NullSignalResult",
    "PolicyRunner",
    "PredictiveBenchmarkResult",
    "ProtocolRunner",
    "ToyGraphBenchmark",
    "ValidationEnergyRunner",
    "check_null_signal",
    "dose_response_effect",
    "evaluate_interaction_policy",
    "evaluate_on_test",
    "evaluate_policy",
    "evaluate_protocol",
    "hillstrom_active_search_space",
    "hillstrom_null_baseline",
    "hillstrom_projected_prior_graph",
    "interaction_propensity",
    "interaction_treatment_effect",
    "load_energy_frame",
    "load_hillstrom_slice",
    "net_benefit",
    "permute_hillstrom_spend",
    "permute_target",
    "propensity",
    "run_null_strategy",
    "sample_random_params",
    "split_time_frame",
    "treatment_effect",
]
