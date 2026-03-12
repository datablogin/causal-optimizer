"""Convergence regression tests on ToyGraphBenchmark.

These are directional smoke tests: the causal strategy (with ground-truth
causal graph) should beat random search on average, given enough budget and
seeds. We use generous tolerances because:

1. Budget is modest (40 experiments per run).
2. The engine has stochastic internals (exploration, off-policy gating).
3. We only need to confirm the *direction* of the advantage, not its magnitude.

If these tests become flaky, increase ``BUDGET`` or ``N_SEEDS`` rather than
weakening assertions — the underlying claim (causal > random) should hold
with enough samples.

Note: this file tests worst-seed variance (test_causal_beats_random_worst_seed)
while test_high_dim_convergence.py tests a loose sanity-check bound instead.
The asymmetry is intentional — worst-seed checks are most useful on the simpler
benchmark where variance is lower, while the high-dim benchmark benefits more
from a catastrophic-regression guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.runner import BenchmarkResult, BenchmarkRunner
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark

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
class TestToyGraphConvergence:
    """Convergence regression on ToyGraphBenchmark (X -> Z -> objective)."""

    @pytest.fixture(scope="class")
    def results(self) -> list[BenchmarkResult]:
        """Run the benchmark comparison once for the whole test class.

        Note: engine-based strategies (causal, surrogate_only) have unseeded
        internal RNG (suggestions, bootstrap). This is acceptable because we
        use generous tolerances and multiple seeds to absorb variance. See
        BenchmarkRunner.run docstring for details.
        """
        bench = ToyGraphBenchmark(noise_scale=0.1)
        runner = BenchmarkRunner(bench)
        return runner.compare(
            strategies=["causal", "random", "surrogate_only"],
            budget=BUDGET,
            n_seeds=N_SEEDS,
        )

    def test_causal_beats_random_mean_final(self, results: list[BenchmarkResult]) -> None:
        """Causal strategy's mean final objective beats random with 20% tolerance.

        Since we minimize, lower is better. The assertion is:
            avg_causal <= avg_random + TOLERANCE_FRACTION * |avg_random|

        This gives causal a generous cushion -- it just needs to not be
        catastrophically worse than random.
        """
        assert_causal_beats_random(results, N_SEEDS, TOLERANCE_FRACTION, "ToyGraph")

    def test_surrogate_only_runs_without_error(self, results: list[BenchmarkResult]) -> None:
        """Surrogate-only strategy completes all runs without crashing."""
        surrogate_results = finals_for_strategy(results, "surrogate_only")
        assert len(surrogate_results) == N_SEEDS

    def test_convergence_curves_are_monotonic(self, results: list[BenchmarkResult]) -> None:
        """All convergence curves should be monotonically non-increasing (best-so-far)."""
        assert_monotonic_curves(results)

    def test_all_strategies_produce_budget_length_curves(
        self, results: list[BenchmarkResult]
    ) -> None:
        """Each result's convergence curve should have exactly ``BUDGET`` entries."""
        assert_curve_lengths(results, BUDGET)

    def test_causal_beats_random_worst_seed(self, results: list[BenchmarkResult]) -> None:
        """Causal strategy's worst seed should not be drastically worse than random's mean.

        This guards against high variance: even the worst causal seed should
        be within a generous margin of random's average performance.
        """
        causal_finals = finals_for_strategy(results, "causal")
        random_finals = finals_for_strategy(results, "random")

        worst_causal = max(causal_finals)  # max because lower is better
        avg_random = float(np.mean(random_finals))

        # Absolute tolerance of 1.0: on ToyGraph the objective (negated Y)
        # typically ranges from about -1.2 to +0.5, so 1.0 covers roughly
        # the full range. This matches the existing integration test tolerance
        # in tests/integration/test_causal_beats_naive.py.
        tolerance = 1.0
        assert worst_causal <= avg_random + tolerance, (
            f"Worst causal seed ({worst_causal:.4f}) is drastically worse than "
            f"random mean ({avg_random:.4f}) on ToyGraph. "
            f"Causal finals: {causal_finals}"
        )
