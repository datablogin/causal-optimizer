"""Convergence regression tests on HighDimensionalSparseBenchmark.

The high-dimensional sparse benchmark has 20 variables but only 3 causal
ancestors (x1 -> x2 -> x3 -> objective). The remaining 17 are distractors.
This is the marquee test for causal guidance: random search wastes budget
exploring irrelevant variables, while causal search focuses on x1-x3.

Same philosophy as test_convergence.py: directional smoke tests with
generous tolerances, not statistical proofs.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.high_dimensional import HighDimensionalSparseBenchmark
from causal_optimizer.benchmarks.runner import BenchmarkResult, BenchmarkRunner

from .helpers import (
    assert_causal_beats_random,
    assert_curve_lengths,
    assert_monotonic_curves,
    finals_for_strategy,
)

BUDGET = 40
N_SEEDS = 5
TOLERANCE_FRACTION = 0.20


@pytest.mark.slow
class TestHighDimConvergence:
    """Convergence regression on HighDimensionalSparseBenchmark (20 vars, 3 causal)."""

    @pytest.fixture(scope="class")
    def results(self) -> list[BenchmarkResult]:
        """Run the benchmark comparison once for the whole test class.

        Note: engine-based strategies (causal, surrogate_only) have unseeded
        internal RNG (suggestions, bootstrap). This is acceptable because we
        use generous tolerances and multiple seeds to absorb variance. See
        BenchmarkRunner.run docstring for details.
        """
        bench = HighDimensionalSparseBenchmark(noise_scale=0.1)
        runner = BenchmarkRunner(bench)
        return runner.compare(
            strategies=["causal", "random", "surrogate_only"],
            budget=BUDGET,
            n_seeds=N_SEEDS,
        )

    def test_causal_beats_random_mean_final(self, results: list[BenchmarkResult]) -> None:
        """Causal strategy's mean final objective beats random with 20% tolerance.

        With 17 distractor variables, the advantage of causal guidance should
        be more pronounced than on the toy graph. Random search distributes
        budget across all 20 dimensions, while causal focuses on the 3 that
        matter.
        """
        assert_causal_beats_random(results, N_SEEDS, TOLERANCE_FRACTION, "HighDimensionalSparse")

    def test_surrogate_only_runs_without_error(self, results: list[BenchmarkResult]) -> None:
        """Surrogate-only strategy completes all runs without crashing."""
        surrogate_results = finals_for_strategy(results, "surrogate_only")
        assert len(surrogate_results) == N_SEEDS

    def test_convergence_curves_are_monotonic(self, results: list[BenchmarkResult]) -> None:
        """All convergence curves should be monotonically non-increasing."""
        assert_monotonic_curves(results)

    def test_all_strategies_produce_budget_length_curves(
        self, results: list[BenchmarkResult]
    ) -> None:
        """Each result's convergence curve should have exactly ``BUDGET`` entries."""
        assert_curve_lengths(results, BUDGET)

    def test_causal_not_drastically_worse_than_random(self, results: list[BenchmarkResult]) -> None:
        """Loose sanity check: causal should not be catastrophically worse than random.

        This is a weaker guard than test_causal_beats_random_mean_final (which
        uses 20% tolerance). Here we use 50% tolerance as a safety net to catch
        severe regressions even when the tighter test is noisy.

        See test_convergence.py for the complementary worst-seed check used on
        the ToyGraph benchmark (the asymmetry is explained in that file's
        module docstring).
        """
        causal_finals = finals_for_strategy(results, "causal")
        random_finals = finals_for_strategy(results, "random")

        avg_causal = float(np.mean(causal_finals))
        avg_random = float(np.mean(random_finals))

        # Very generous: just check causal is not worse than random + 50%
        tolerance = 0.50 * max(abs(avg_random), 1.0)
        assert avg_causal <= avg_random + tolerance, (
            f"Causal ({avg_causal:.4f}) is significantly worse than "
            f"random ({avg_random:.4f}) on high-dim sparse benchmark. "
            f"This should not happen -- causal has the ground-truth graph."
        )
