"""Benchmark structural causal models for testing the optimization engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from causal_optimizer.benchmarks.complete_graph import CompleteGraphBenchmark
from causal_optimizer.benchmarks.high_dimensional import HighDimensionalSparseBenchmark
from causal_optimizer.benchmarks.interaction import InteractionBenchmark
from causal_optimizer.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, SearchSpace


class BenchmarkSCM(Protocol):
    """Protocol that all benchmark SCMs must satisfy.

    Provides a search space, causal graph, known POMIS sets, and a run method
    that executes the structural equations under (partial) intervention.

    All implementations should accept noise_scale and optional rng
    in __init__, but Protocol cannot enforce constructor signatures.
    """

    @staticmethod
    def search_space() -> SearchSpace: ...
    @staticmethod
    def causal_graph() -> CausalGraph: ...
    @staticmethod
    def known_pomis() -> list[frozenset[str]]: ...
    def run(self, parameters: dict[str, Any]) -> dict[str, float]: ...


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSCM",
    "CompleteGraphBenchmark",
    "HighDimensionalSparseBenchmark",
    "InteractionBenchmark",
    "ToyGraphBenchmark",
]
