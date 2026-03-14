"""Integration test: CausalGP vs Ax on ToyGraph.

Compares CausalGP strategy against standard Ax Bayesian optimization
over multiple seeds to verify competitive performance.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("botorch")

from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
from causal_optimizer.engine.loop import ExperimentEngine


@pytest.mark.slow
def test_causal_gp_matches_ax_on_toygraph() -> None:
    """CausalGP mean final objective <= Ax mean * 1.15 over 3 seeds on ToyGraph (30 steps)."""
    n_seeds = 3
    n_steps = 30
    causal_gp_finals: list[float] = []
    ax_finals: list[float] = []

    for seed in range(n_seeds):
        # CausalGP strategy
        bench_cgp = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(seed))
        engine_cgp = ExperimentEngine(
            search_space=ToyGraphBenchmark.search_space(),
            runner=bench_cgp,
            causal_graph=ToyGraphBenchmark.causal_graph(),
            objective_name="objective",
            minimize=True,
            seed=seed,
            max_skips=0,
            strategy="causal_gp",
        )
        for _ in range(n_steps):
            engine_cgp.step()

        best_cgp = engine_cgp.log.best_result("objective", minimize=True)
        assert best_cgp is not None
        causal_gp_finals.append(best_cgp.metrics["objective"])

        # Ax strategy (default bayesian)
        bench_ax = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(seed))
        engine_ax = ExperimentEngine(
            search_space=ToyGraphBenchmark.search_space(),
            runner=bench_ax,
            causal_graph=ToyGraphBenchmark.causal_graph(),
            objective_name="objective",
            minimize=True,
            seed=seed,
            max_skips=0,
        )
        for _ in range(n_steps):
            engine_ax.step()

        best_ax = engine_ax.log.best_result("objective", minimize=True)
        assert best_ax is not None
        ax_finals.append(best_ax.metrics["objective"])

    cgp_mean = float(np.mean(causal_gp_finals))
    ax_mean = float(np.mean(ax_finals))

    # For minimization, lower is better.
    # Allow CausalGP to be at most 15% worse than Ax in absolute terms.
    assert cgp_mean <= ax_mean + abs(ax_mean) * 0.15, (
        f"CausalGP (mean={cgp_mean:.4f}) not competitive with Ax (mean={ax_mean:.4f}); "
        f"seeds: cgp={causal_gp_finals}, ax={ax_finals}"
    )
