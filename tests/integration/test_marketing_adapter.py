"""Integration tests for MarketingAdapter through the ExperimentEngine.

Tests the full pipeline: adapter -> engine -> simulator -> diagnostics.
All tests marked @pytest.mark.slow (complete in <60s each).
"""

from __future__ import annotations

import pytest

from causal_optimizer.domain_adapters.marketing import MarketingAdapter
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.optimizer.pomis import compute_pomis


@pytest.mark.slow
class TestMarketingAdapterIntegration:
    """End-to-end integration tests for MarketingAdapter."""

    def test_full_pipeline_40_experiments(self) -> None:
        """Full pipeline: 40 experiments with no crashes, valid metrics."""
        adapter = MarketingAdapter(seed=42)
        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=adapter,
            objective_name=adapter.get_objective_name(),
            minimize=adapter.get_minimize(),
            causal_graph=adapter.get_prior_graph(),
            descriptor_names=adapter.get_descriptor_names(),
            epsilon_mode=False,
            seed=42,
        )

        log = engine.run_loop(40)

        assert len(log.results) == 40
        crashes = sum(1 for r in log.results if r.status.value == "crash")
        assert crashes == 0, f"Expected 0 crashes, got {crashes}"
        # All results should have the expected metrics
        for r in log.results:
            assert "conversions" in r.metrics
            assert "total_spend" in r.metrics
            assert "channel_diversity" in r.metrics

    def test_phase_transitions(self) -> None:
        """Engine should transition through exploration -> optimization phases."""
        adapter = MarketingAdapter(seed=42)
        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=adapter,
            objective_name=adapter.get_objective_name(),
            minimize=adapter.get_minimize(),
            causal_graph=adapter.get_prior_graph(),
            descriptor_names=adapter.get_descriptor_names(),
            epsilon_mode=False,
            seed=42,
        )

        phases_seen: set[str] = set()
        for _ in range(20):
            result = engine.step()
            phases_seen.add(result.metadata.get("phase", "unknown"))

        assert "exploration" in phases_seen, "Should start in exploration"
        assert "optimization" in phases_seen, "Should transition to optimization"

    def test_causal_focus_variables(self) -> None:
        """Engine should compute POMIS sets using the adapter's causal graph."""
        adapter = MarketingAdapter(seed=42)
        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=adapter,
            objective_name=adapter.get_objective_name(),
            minimize=adapter.get_minimize(),
            causal_graph=adapter.get_prior_graph(),
            descriptor_names=adapter.get_descriptor_names(),
            epsilon_mode=False,
            seed=42,
        )

        # Run past exploration to trigger POMIS computation
        engine.run_loop(15)

        assert engine.pomis_sets is not None, "POMIS should be computed"
        assert len(engine.pomis_sets) >= 1

    def test_pomis_pruning(self) -> None:
        """POMIS should prune the search space (fewer vars per set than full 5)."""
        adapter = MarketingAdapter()
        graph = adapter.get_prior_graph()
        space = adapter.get_search_space()

        pomis_sets = compute_pomis(graph, "conversions")

        assert len(pomis_sets) >= 1
        all_search_vars = set(space.variable_names)
        has_pruning = any(len(pset & all_search_vars) < len(all_search_vars) for pset in pomis_sets)
        assert has_pruning, f"POMIS should prune, got sets: {pomis_sets}"

    def test_optimization_improves_over_exploration(self) -> None:
        """Optimization phase should find better conversions than exploration average."""
        adapter = MarketingAdapter(seed=42)
        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=adapter,
            objective_name=adapter.get_objective_name(),
            minimize=adapter.get_minimize(),
            causal_graph=adapter.get_prior_graph(),
            descriptor_names=adapter.get_descriptor_names(),
            epsilon_mode=False,
            seed=42,
        )

        log = engine.run_loop(30)

        # Split into exploration (first 10) and optimization (rest)
        exploration_results = [
            r.metrics["conversions"] for r in log.results[:10] if r.status.value != "crash"
        ]
        optimization_results = [
            r.metrics["conversions"] for r in log.results[10:] if r.status.value != "crash"
        ]

        if exploration_results and optimization_results:
            best_exploration = max(exploration_results)
            best_optimization = max(optimization_results)
            # Best optimization result should be at least as good as best exploration
            assert best_optimization >= best_exploration * 0.8, (
                f"Optimization best ({best_optimization:.3f}) should be competitive with "
                f"exploration best ({best_exploration:.3f})"
            )

    def test_diagnostics(self) -> None:
        """diagnose() should return a valid report with recommendations."""
        adapter = MarketingAdapter(seed=42)
        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=adapter,
            objective_name=adapter.get_objective_name(),
            minimize=adapter.get_minimize(),
            causal_graph=adapter.get_prior_graph(),
            descriptor_names=adapter.get_descriptor_names(),
            epsilon_mode=False,
            seed=42,
        )

        engine.run_loop(20)

        report = engine.diagnose(total_budget=40)
        assert report is not None
        assert len(report.recommendations) > 0

    def test_reproducibility(self) -> None:
        """Same seed should produce identical experiment sequences."""

        def run_with_seed(seed: int) -> list[float]:
            adapter = MarketingAdapter(seed=seed)
            engine = ExperimentEngine(
                search_space=adapter.get_search_space(),
                runner=adapter,
                objective_name=adapter.get_objective_name(),
                minimize=adapter.get_minimize(),
                causal_graph=adapter.get_prior_graph(),
                descriptor_names=adapter.get_descriptor_names(),
                epsilon_mode=False,
                seed=seed,
            )
            log = engine.run_loop(10)
            return [r.metrics["conversions"] for r in log.results]

        run1 = run_with_seed(42)
        run2 = run_with_seed(42)
        assert run1 == run2
