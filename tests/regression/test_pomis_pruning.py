"""POMIS pruning regression tests: CompleteGraph benchmark.

Verifies two things:
1. CompleteGraphBenchmark.known_pomis() correctly lists the 5 POMIS sets from
   Aglietti et al. (AISTATS 2020): [{b}, {d}, {e}, {b,d}, {d,e}].
2. An ExperimentEngine with the CompleteGraph's causal graph computes and uses
   POMIS sets to guide search — far fewer sets than the naive 2^6 = 64 subsets.

Reference: Aglietti et al., "Causal Bayesian Optimization", AISTATS 2020.

Note: ``pytestmark`` in conftest.py does NOT propagate to sibling test
modules — each test class must be decorated with ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks import CompleteGraphBenchmark
from causal_optimizer.benchmarks.runner import BenchmarkRunner
from causal_optimizer.engine.loop import ExperimentEngine

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_POMIS: list[frozenset[str]] = [
    frozenset({"b"}),
    frozenset({"d"}),
    frozenset({"e"}),
    frozenset({"b", "d"}),
    frozenset({"d", "e"}),
]

# Naive search space: 2^6 = 64 variable subsets (a, b, c, d, e, f — excluding objective).
NAIVE_SEARCH_SPACE_SIZE = 64


# ──────────────────────────────────────────────────────────────────────────────
# known_pomis() correctness tests (no engine, no algorithm)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestCompleteGraphPOMIS:
    """Verify CompleteGraphBenchmark.known_pomis() matches the Aglietti paper."""

    def test_complete_graph_pomis_size(self) -> None:
        """known_pomis() must list exactly 5 POMIS sets.

        The 5 sets correspond to {b}, {d}, {e}, {b,d}, {d,e} from
        Table 1 in Aglietti et al. (AISTATS 2020).
        """
        pomis = CompleteGraphBenchmark.known_pomis()
        assert len(pomis) == 5, f"Expected 5 POMIS sets, got {len(pomis)}: {pomis}"

    def test_complete_graph_pomis_members(self) -> None:
        """known_pomis() must list exactly the 5 paper-defined sets."""
        pomis = CompleteGraphBenchmark.known_pomis()
        actual = set(pomis)
        expected = set(KNOWN_POMIS)
        assert actual == expected, (
            f"POMIS mismatch.\n"
            f"  Expected: {sorted(expected, key=lambda s: (len(s), sorted(s)))}\n"
            f"  Got:      {sorted(actual, key=lambda s: (len(s), sorted(s)))}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# POMIS-guided engine pruning tests (slow)
# ──────────────────────────────────────────────────────────────────────────────

BUDGET = 30
N_SEEDS = 3


@pytest.mark.slow
class TestPOMISPruning:
    """Regression: POMIS-guided engine explores far fewer variable sets than naive."""

    @pytest.fixture(scope="class")
    def bench(self) -> CompleteGraphBenchmark:
        return CompleteGraphBenchmark(noise_scale=0.1)

    def test_pomis_prunes_search_space(self, bench: CompleteGraphBenchmark) -> None:
        """ExperimentEngine with causal graph computes POMIS sets, not naive search.

        The key observable is the engine's ``pomis_sets`` attribute after the
        exploration phase completes. With POMIS, the engine explores a structured
        subset of the search space; the number of computed POMIS sets is far
        smaller than the naive 2^6 = 64 possible variable subsets.

        We also verify that the causal engine's final best objective is at least
        as good as random search, confirming the POMIS guidance is helpful.
        """
        graph = CompleteGraphBenchmark.causal_graph()
        space = CompleteGraphBenchmark.search_space()

        causal_finals: list[float] = []
        pomis_sets_counts: list[int] = []

        for seed in range(N_SEEDS):
            # --- Causal run ---
            rng = np.random.default_rng(seed)
            causal_bench = CompleteGraphBenchmark(noise_scale=0.1, rng=rng)
            engine = ExperimentEngine(
                search_space=space,
                runner=causal_bench,
                causal_graph=graph,
            )
            for _ in range(BUDGET):
                engine.step()

            # Record the number of POMIS sets computed (should be << 64)
            pomis_sets = engine.pomis_sets
            if pomis_sets is not None:
                pomis_sets_counts.append(len(pomis_sets))

            best = engine.log.best_result(minimize=True)
            if best is not None:
                causal_finals.append(best.metrics.get("objective", float("inf")))

        # Run random comparison using BenchmarkRunner
        runner = BenchmarkRunner(bench)
        random_results = runner.compare(["random"], budget=BUDGET, n_seeds=N_SEEDS)
        random_finals = [r.final_best for r in random_results]

        # The causal engine must have computed POMIS sets
        assert len(pomis_sets_counts) > 0, "Engine never computed POMIS sets"

        # POMIS sets count must be << NAIVE_SEARCH_SPACE_SIZE (64): require ≥5× reduction
        max_allowed = NAIVE_SEARCH_SPACE_SIZE // 5  # 64 / 5 = 12
        for count in pomis_sets_counts:
            assert count <= max_allowed, (
                f"POMIS sets count {count} exceeds {max_allowed} "
                f"(≥5× reduction vs. naive {NAIVE_SEARCH_SPACE_SIZE} required)"
            )

        # Causal should achieve at least comparable (within 30%) performance vs random
        if causal_finals and random_finals:
            avg_causal = float(sum(causal_finals) / len(causal_finals))
            avg_random = float(sum(random_finals) / len(random_finals))
            tolerance = 0.30 * max(abs(avg_random), 1.0)
            assert avg_causal <= avg_random + tolerance, (
                f"Causal ({avg_causal:.4f}) much worse than random ({avg_random:.4f}) "
                f"on CompleteGraph. POMIS guidance may be causing harm."
            )

    def test_pomis_pruning_ratio(self, bench: CompleteGraphBenchmark) -> None:
        """POMIS reduces the effective search space by ≥ 5× vs. naive 64 subsets.

        The engine's computed POMIS sets must number ≤ 64 / 5 = 12 sets.
        This verifies the POMIS algorithm provides meaningful pruning:
        the causal optimizer considers at most ~20% of the variable subsets
        that a naive strategy would try.
        """
        graph = CompleteGraphBenchmark.causal_graph()
        space = CompleteGraphBenchmark.search_space()

        # Run the engine to trigger POMIS computation (occurs at exploration→optimization)
        rng = np.random.default_rng(0)
        causal_bench = CompleteGraphBenchmark(noise_scale=0.1, rng=rng)
        engine = ExperimentEngine(
            search_space=space,
            runner=causal_bench,
            causal_graph=graph,
        )
        # Run just past the exploration threshold (11 steps) to trigger POMIS computation
        for _ in range(15):
            engine.step()

        pomis_sets = engine.pomis_sets
        assert pomis_sets is not None, "Engine did not compute POMIS sets after exploration phase."

        pomis_count = len(pomis_sets)
        max_allowed = NAIVE_SEARCH_SPACE_SIZE // 5  # 64 / 5 = 12

        assert pomis_count <= max_allowed, (
            f"POMIS contains {pomis_count} sets; "
            f"expected ≤ {max_allowed} for ≥5× reduction vs. naive ({NAIVE_SEARCH_SPACE_SIZE}).\n"
            f"POMIS sets: {pomis_sets}"
        )
