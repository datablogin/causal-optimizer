"""Integration tests for the benchmark runner."""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.complete_graph import CompleteGraphBenchmark
from causal_optimizer.benchmarks.high_dimensional import HighDimensionalSparseBenchmark
from causal_optimizer.benchmarks.interaction import InteractionBenchmark
from causal_optimizer.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
from causal_optimizer.types import CausalGraph, SearchSpace

ALL_BENCHMARK_CLASSES = [
    ToyGraphBenchmark,
    CompleteGraphBenchmark,
    InteractionBenchmark,
    HighDimensionalSparseBenchmark,
]

ALL_STRATEGIES = ["causal", "random", "surrogate_only"]


class TestHighDimensionalBenchmark:
    """Tests for the HighDimensionalSparseBenchmark."""

    def test_search_space_has_20_variables(self) -> None:
        space = HighDimensionalSparseBenchmark.search_space()
        assert isinstance(space, SearchSpace)
        assert len(space.variables) == 20

    def test_causal_graph_chain(self) -> None:
        graph = HighDimensionalSparseBenchmark.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert ("x1", "x2") in graph.edges
        assert ("x2", "x3") in graph.edges
        assert ("x3", "objective") in graph.edges
        assert len(graph.edges) == 3

    def test_known_pomis(self) -> None:
        """Chain without confounders: only the direct parent of objective is in POMIS."""
        pomis = HighDimensionalSparseBenchmark.known_pomis()
        assert pomis == [frozenset({"x3"})]

    def test_known_pomis_matches_algorithm(self) -> None:
        """Verify hardcoded POMIS matches the POMIS algorithm output."""
        from causal_optimizer.optimizer.pomis import compute_pomis

        graph = HighDimensionalSparseBenchmark.causal_graph()
        computed = compute_pomis(graph, "objective")
        hardcoded = HighDimensionalSparseBenchmark.known_pomis()
        assert set(computed) == set(hardcoded)

    def test_run_returns_objective(self) -> None:
        bench = HighDimensionalSparseBenchmark(rng=np.random.default_rng(42))
        result = bench.run({"x1": 1.0, "x2": 0.5, "x3": 0.1})
        assert "objective" in result
        assert isinstance(result["objective"], (float, np.floating))

    def test_irrelevant_variables_ignored(self) -> None:
        """Changing x4-x20 should not affect the objective (noise aside)."""
        bench = HighDimensionalSparseBenchmark(noise_scale=0.0, rng=np.random.default_rng(42))
        r1 = bench.run({"x1": 1.0, "x2": 0.5, "x3": 0.1, "x10": -5.0})

        bench2 = HighDimensionalSparseBenchmark(noise_scale=0.0, rng=np.random.default_rng(42))
        r2 = bench2.run({"x1": 1.0, "x2": 0.5, "x3": 0.1, "x10": 5.0})

        # With noise_scale=0 and same seed, only causal vars matter.
        # x10 is irrelevant so results should be identical.
        assert r1["objective"] == r2["objective"]

    def test_determinism_with_rng(self) -> None:
        bench1 = HighDimensionalSparseBenchmark(rng=np.random.default_rng(99))
        bench2 = HighDimensionalSparseBenchmark(rng=np.random.default_rng(99))
        r1 = bench1.run({"x1": 2.0})
        r2 = bench2.run({"x1": 2.0})
        assert r1["objective"] == r2["objective"]


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass fields."""

    def test_fields_present(self) -> None:
        result = BenchmarkResult(
            convergence_curve=[1.0, 0.9, 0.8],
            final_best=0.8,
            experiments_to_threshold=2,
            strategy="causal",
            benchmark_name="ToyGraphBenchmark",
            seed=0,
        )
        assert result.convergence_curve == [1.0, 0.9, 0.8]
        assert result.final_best == 0.8
        assert result.experiments_to_threshold == 2
        assert result.strategy == "causal"
        assert result.benchmark_name == "ToyGraphBenchmark"
        assert result.seed == 0

    def test_threshold_can_be_none(self) -> None:
        result = BenchmarkResult(
            convergence_curve=[1.0],
            final_best=1.0,
            experiments_to_threshold=None,
            strategy="random",
            benchmark_name="Test",
            seed=0,
        )
        assert result.experiments_to_threshold is None


@pytest.mark.parametrize(
    "benchmark_cls",
    ALL_BENCHMARK_CLASSES,
    ids=[cls.__name__ for cls in ALL_BENCHMARK_CLASSES],
)
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
class TestRunnerSmoke:
    """Smoke tests: each benchmark x strategy with a tiny budget."""

    def test_runner_executes(self, benchmark_cls: type, strategy: str) -> None:
        bench = benchmark_cls(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        result = runner.run(strategy=strategy, budget=5, seed=0)

        assert isinstance(result, BenchmarkResult)
        assert len(result.convergence_curve) == 5
        assert result.strategy == strategy
        assert result.benchmark_name == benchmark_cls.__name__
        assert result.seed == 0
        assert isinstance(result.final_best, float)
        # final_best must equal the last (and smallest) element of the curve
        assert result.final_best == result.convergence_curve[-1]


class TestConvergenceCurveMonotonicity:
    """Convergence curve should be monotonically non-increasing for minimization."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_monotonic_non_increasing(self, strategy: str) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        result = runner.run(strategy=strategy, budget=10, seed=0)

        curve = result.convergence_curve
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1], (
                f"Convergence curve is not monotonically non-increasing at step {i}: "
                f"{curve[i - 1]} -> {curve[i]}"
            )


class TestCompare:
    """Tests for the compare() method."""

    def test_compare_returns_all_results(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        results = runner.compare(
            strategies=["causal", "random"],
            budget=5,
            n_seeds=2,
        )
        # 2 strategies x 2 seeds = 4 results
        assert len(results) == 4

        strategy_counts = {"causal": 0, "random": 0}
        for r in results:
            strategy_counts[r.strategy] += 1
        assert strategy_counts["causal"] == 2
        assert strategy_counts["random"] == 2

    def test_compare_different_seeds(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        results = runner.compare(strategies=["random"], budget=5, n_seeds=3)

        seeds = [r.seed for r in results]
        assert seeds == [0, 1, 2]


class TestRunnerEdgeCases:
    """Edge cases and error handling for BenchmarkRunner."""

    def test_zero_budget_raises(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        with pytest.raises(ValueError, match="budget must be a positive integer"):
            runner.run(strategy="random", budget=0)

    def test_negative_budget_raises(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        with pytest.raises(ValueError, match="budget must be a positive integer"):
            runner.run(strategy="random", budget=-1)

    def test_invalid_strategy_raises(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        with pytest.raises(ValueError, match="Unknown strategy"):
            runner.run(strategy="invalid", budget=5)

    def test_zero_n_seeds_raises(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        with pytest.raises(ValueError, match="n_seeds must be a positive integer"):
            runner.compare(strategies=["random"], budget=5, n_seeds=0)

    def test_negative_n_seeds_raises(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        with pytest.raises(ValueError, match="n_seeds must be a positive integer"):
            runner.compare(strategies=["random"], budget=5, n_seeds=-1)

    def test_budget_one(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench)
        result = runner.run(strategy="random", budget=1, seed=0)
        assert len(result.convergence_curve) == 1

    def test_threshold_calculation(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench, threshold_pct=0.10)
        # With a known optimum, threshold should be computed
        result = runner.run(strategy="random", budget=5, seed=0, known_optimum=-1.0)
        # experiments_to_threshold is either an int or None
        assert result.experiments_to_threshold is None or isinstance(
            result.experiments_to_threshold, int
        )

    def test_threshold_step_exact(self) -> None:
        """Verify threshold step calculation with a known convergence curve."""
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench, threshold_pct=0.10)
        # With known_optimum=1.0 and threshold_pct=0.10: threshold = 0.1
        # One-sided check: val <= optimum + threshold = 1.1
        # Curve: [5.0, 3.0, 1.05, 0.5] — step 3 has 1.05 <= 1.1
        step = runner._compute_threshold_step([5.0, 3.0, 1.05, 0.5], known_optimum=1.0)
        assert step == 3

    def test_threshold_step_never_reached(self) -> None:
        """Threshold is never reached if all values are far from optimum."""
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench, threshold_pct=0.10)
        step = runner._compute_threshold_step([5.0, 4.0, 3.0], known_optimum=0.0)
        assert step is None

    def test_threshold_step_overshoot(self) -> None:
        """Values better than optimum should still count as converged."""
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench, threshold_pct=0.10)
        # known_optimum=-1.0, threshold=0.10. One-sided: val <= -1.0 + 0.10 = -0.90
        # Step 2 has -1.5 <= -0.90, so it counts as converged.
        step = runner._compute_threshold_step([5.0, -1.5], known_optimum=-1.0)
        assert step == 2

    def test_threshold_step_zero_optimum(self) -> None:
        """When optimum is 0.0, fallback scale of 1.0 is used."""
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        runner = BenchmarkRunner(bench, threshold_pct=0.10)
        # threshold = 0.10 * 1.0 = 0.10; 0.05 <= 0.0 + 0.10 → step 2
        step = runner._compute_threshold_step([5.0, 0.05], known_optimum=0.0)
        assert step == 2
