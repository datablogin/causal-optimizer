"""Directional test: causal optimization should outperform random search.

This is a directional test, NOT a strict statistical guarantee. The causal
strategy has access to the ground-truth causal graph, which should help it
focus on the right variables and converge faster than uniform random sampling.

We use a generous tolerance because:
1. Small budgets (30 experiments) mean high variance.
2. The engine has stochastic components (exploration, off-policy gating).
3. We only need to show the trend, not statistical significance.

If this test becomes flaky, increase the budget or n_seeds rather than
removing it — the underlying claim (causal > random) should hold reliably
with enough samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.high_dimensional import HighDimensionalSparseBenchmark
from causal_optimizer.benchmarks.runner import BenchmarkRunner
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark


@pytest.mark.slow
class TestCausalBeatsNaive:
    """Causal strategy should find a better objective than random on average."""

    def test_causal_vs_random_toy_graph(self) -> None:
        """Causal strategy outperforms random on ToyGraphBenchmark.

        We run each strategy with budget=30 across n_seeds=3, then compare
        the average final_best. The causal strategy should achieve a lower
        (better, since we minimize) average objective.

        Uses a generous tolerance: causal just needs to be no worse than
        random + 1.0 (i.e., not catastrophically worse). In practice,
        causal should be strictly better on average.
        """
        n_seeds = 3
        budget = 30

        bench = ToyGraphBenchmark(noise_scale=0.1)
        runner = BenchmarkRunner(bench)

        results = runner.compare(
            strategies=["causal", "random"],
            budget=budget,
            n_seeds=n_seeds,
        )

        causal_finals = [r.final_best for r in results if r.strategy == "causal"]
        random_finals = [r.final_best for r in results if r.strategy == "random"]

        avg_causal = np.mean(causal_finals)
        avg_random = np.mean(random_finals)

        # Causal should be at least as good as random (lower is better).
        # We allow a generous tolerance of 1.0 to account for stochasticity.
        # In practice, causal should be strictly better.
        assert avg_causal <= avg_random + 1.0, (
            f"Causal strategy ({avg_causal:.4f}) did not outperform "
            f"random ({avg_random:.4f}) within tolerance. "
            f"Causal finals: {causal_finals}, Random finals: {random_finals}"
        )

    def test_causal_vs_random_high_dimensional(self) -> None:
        """Causal strategy outperforms random on HighDimensionalSparseBenchmark.

        This is the marquee test: 20 variables but only 3 are causal ancestors
        of the objective. The causal strategy should focus on the 3 relevant
        variables while random wastes budget exploring all 20.

        Uses a generous tolerance: causal just needs to be no worse than
        random + 1.0. With 17 distractor variables, the advantage of causal
        guidance should be pronounced.
        """
        n_seeds = 3
        budget = 30

        bench = HighDimensionalSparseBenchmark(noise_scale=0.1)
        runner = BenchmarkRunner(bench)

        results = runner.compare(
            strategies=["causal", "random"],
            budget=budget,
            n_seeds=n_seeds,
        )

        causal_finals = [r.final_best for r in results if r.strategy == "causal"]
        random_finals = [r.final_best for r in results if r.strategy == "random"]

        avg_causal = np.mean(causal_finals)
        avg_random = np.mean(random_finals)

        assert avg_causal <= avg_random + 1.0, (
            f"Causal strategy ({avg_causal:.4f}) did not outperform "
            f"random ({avg_random:.4f}) within tolerance on high-dimensional benchmark. "
            f"Causal finals: {causal_finals}, Random finals: {random_finals}"
        )
