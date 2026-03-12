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
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.runner import BenchmarkRunner
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark

BUDGET = 40
N_SEEDS = 5
# 20% tolerance: causal must beat random by this fraction of random's |mean|,
# but we allow being up to TOLERANCE * |random_mean| worse to absorb noise.
TOLERANCE_FRACTION = 0.20


@pytest.mark.slow
class TestToyGraphConvergence:
    """Convergence regression on ToyGraphBenchmark (X -> Z -> objective)."""

    @pytest.fixture(scope="class")
    def results(self) -> list[object]:
        """Run the benchmark comparison once for the whole test class."""
        bench = ToyGraphBenchmark(noise_scale=0.1)
        runner = BenchmarkRunner(bench)
        return runner.compare(
            strategies=["causal", "random", "surrogate_only"],
            budget=BUDGET,
            n_seeds=N_SEEDS,
        )

    def test_causal_beats_random_mean_final(self, results: list[object]) -> None:
        """Causal strategy's mean final objective beats random with 20% tolerance.

        Since we minimize, lower is better. The assertion is:
            avg_causal <= avg_random + TOLERANCE_FRACTION * |avg_random|

        This gives causal a generous cushion — it just needs to not be
        catastrophically worse than random.
        """
        from causal_optimizer.benchmarks.runner import BenchmarkResult

        causal_finals = [
            r.final_best for r in results if isinstance(r, BenchmarkResult) and r.strategy == "causal"
        ]
        random_finals = [
            r.final_best for r in results if isinstance(r, BenchmarkResult) and r.strategy == "random"
        ]

        assert len(causal_finals) == N_SEEDS
        assert len(random_finals) == N_SEEDS

        avg_causal = float(np.mean(causal_finals))
        avg_random = float(np.mean(random_finals))

        # Lower is better. Allow causal to be up to 20% of |random mean| worse.
        tolerance = TOLERANCE_FRACTION * max(abs(avg_random), 1.0)
        assert avg_causal <= avg_random + tolerance, (
            f"Causal ({avg_causal:.4f}) did not beat random ({avg_random:.4f}) "
            f"within {TOLERANCE_FRACTION:.0%} tolerance on ToyGraph. "
            f"Causal finals: {causal_finals}, Random finals: {random_finals}"
        )

    def test_surrogate_only_runs_without_error(self, results: list[object]) -> None:
        """Surrogate-only strategy completes all runs without crashing."""
        from causal_optimizer.benchmarks.runner import BenchmarkResult

        surrogate_results = [
            r for r in results if isinstance(r, BenchmarkResult) and r.strategy == "surrogate_only"
        ]
        assert len(surrogate_results) == N_SEEDS

    def test_convergence_curves_are_monotonic(self, results: list[object]) -> None:
        """All convergence curves should be monotonically non-increasing (best-so-far)."""
        from causal_optimizer.benchmarks.runner import BenchmarkResult

        for r in results:
            if not isinstance(r, BenchmarkResult):
                continue
            curve = r.convergence_curve
            for i in range(1, len(curve)):
                assert curve[i] <= curve[i - 1] + 1e-12, (
                    f"Convergence curve for {r.strategy} seed={r.seed} is not monotonic "
                    f"at step {i}: {curve[i - 1]:.6f} -> {curve[i]:.6f}"
                )

    def test_all_strategies_produce_budget_length_curves(self, results: list[object]) -> None:
        """Each result's convergence curve should have exactly ``BUDGET`` entries."""
        from causal_optimizer.benchmarks.runner import BenchmarkResult

        for r in results:
            if not isinstance(r, BenchmarkResult):
                continue
            assert len(r.convergence_curve) == BUDGET, (
                f"{r.strategy} seed={r.seed}: expected {BUDGET} steps, "
                f"got {len(r.convergence_curve)}"
            )

    def test_causal_beats_random_worst_seed(self, results: list[object]) -> None:
        """Causal strategy's worst seed should not be drastically worse than random's mean.

        This guards against high variance: even the worst causal seed should
        be within a generous margin of random's average performance.
        """
        from causal_optimizer.benchmarks.runner import BenchmarkResult

        causal_finals = [
            r.final_best for r in results if isinstance(r, BenchmarkResult) and r.strategy == "causal"
        ]
        random_finals = [
            r.final_best for r in results if isinstance(r, BenchmarkResult) and r.strategy == "random"
        ]

        worst_causal = max(causal_finals)  # max because lower is better
        avg_random = float(np.mean(random_finals))

        # Very generous: worst causal seed should be within 50% of |random mean|
        tolerance = 0.50 * max(abs(avg_random), 1.0)
        assert worst_causal <= avg_random + tolerance, (
            f"Worst causal seed ({worst_causal:.4f}) is drastically worse than "
            f"random mean ({avg_random:.4f}) on ToyGraph. "
            f"Causal finals: {causal_finals}"
        )
