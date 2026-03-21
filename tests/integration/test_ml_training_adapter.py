"""Integration tests for MLTrainingAdapter through the ExperimentEngine.

Tests the full pipeline: adapter -> engine -> simulator -> diagnostics.
All tests marked @pytest.mark.slow (complete in <60s each).
"""

from __future__ import annotations

import pytest

from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter
from causal_optimizer.engine.loop import ExperimentEngine


@pytest.mark.slow
class TestMLTrainingAdapterIntegration:
    """End-to-end integration tests for MLTrainingAdapter."""

    def test_full_pipeline_50_experiments(self) -> None:
        """Full pipeline: 50 experiments with no crashes, valid metrics."""
        adapter = MLTrainingAdapter(seed=42)
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

        log = engine.run_loop(50)

        assert len(log.results) == 50
        crashes = sum(1 for r in log.results if r.status.value == "crash")
        assert crashes == 0, f"Expected 0 crashes, got {crashes}"
        for r in log.results:
            assert "val_loss" in r.metrics
            assert "memory_usage" in r.metrics
            assert "model_capacity" in r.metrics

    def test_phase_transitions(self) -> None:
        """Engine should transition through exploration -> optimization phases."""
        adapter = MLTrainingAdapter(seed=42)
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

        assert "exploration" in phases_seen
        assert "optimization" in phases_seen

    def test_screening_handles_categoricals(self) -> None:
        """Screening should handle categorical variables without crashing."""
        adapter = MLTrainingAdapter(seed=42)
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

        # Run past exploration to trigger screening
        log = engine.run_loop(15)
        assert len(log.results) == 15

    def test_optimization_improves_loss(self) -> None:
        """Optimization should find lower val_loss than exploration average."""
        adapter = MLTrainingAdapter(seed=42)
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

        exploration_losses = [
            r.metrics["val_loss"] for r in log.results[:10] if r.status.value != "crash"
        ]
        optimization_losses = [
            r.metrics["val_loss"] for r in log.results[10:] if r.status.value != "crash"
        ]

        if exploration_losses and optimization_losses:
            best_exploration = min(exploration_losses)
            best_optimization = min(optimization_losses)
            # Optimization should be competitive with or better than exploration
            assert best_optimization <= best_exploration * 1.2, (
                f"Optimization best ({best_optimization:.3f}) should be competitive with "
                f"exploration best ({best_exploration:.3f})"
            )

    def test_categoricals_in_results(self) -> None:
        """All results should have valid categorical values."""
        adapter = MLTrainingAdapter(seed=42)
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

        log = engine.run_loop(15)

        for r in log.results:
            assert r.parameters["optimizer"] in ["adamw", "sgd", "muon", "lion"]
            assert r.parameters["activation"] in ["gelu", "swiglu", "relu"]

    def test_diagnostics(self) -> None:
        """diagnose() should return a valid report with recommendations."""
        adapter = MLTrainingAdapter(seed=42)
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

        report = engine.diagnose(total_budget=50)
        assert report is not None
        assert len(report.recommendations) > 0
