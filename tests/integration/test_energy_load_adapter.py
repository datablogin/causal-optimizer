"""Integration tests for EnergyLoadAdapter through the ExperimentEngine.

Tests the full pipeline: adapter -> engine -> model training -> diagnostics.
All tests marked @pytest.mark.slow (complete in <60s each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter
from causal_optimizer.engine.loop import ExperimentEngine

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


@pytest.mark.slow
class TestEnergyLoadAdapterIntegration:
    """End-to-end integration tests for EnergyLoadAdapter."""

    def test_full_pipeline_15_experiments(self, fixture_df: pd.DataFrame) -> None:
        """Full pipeline: 15 experiments with no crashes, valid metrics."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
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

        assert len(log.results) == 15
        crashes = sum(1 for r in log.results if r.status.value == "crash")
        assert crashes == 0, f"Expected 0 crashes, got {crashes}"
        # All results should have the expected metrics
        for r in log.results:
            assert "mae" in r.metrics
            assert "rmse" in r.metrics
            assert "runtime_seconds" in r.metrics
            assert all(isinstance(v, float) for v in r.metrics.values())

    def test_phase_transitions(self, fixture_df: pd.DataFrame) -> None:
        """Engine should transition through exploration -> optimization phases."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
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
        for _ in range(15):
            result = engine.step()
            phases_seen.add(result.metadata.get("phase", "unknown"))

        assert "exploration" in phases_seen, "Should start in exploration"
        assert "optimization" in phases_seen, "Should transition to optimization"

    def test_diagnostics(self, fixture_df: pd.DataFrame) -> None:
        """diagnose() should return a valid report with recommendations."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
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

        engine.run_loop(15)

        report = engine.diagnose(total_budget=30)
        assert report is not None
        assert len(report.recommendations) > 0

    def test_reproducibility(self, fixture_df: pd.DataFrame) -> None:
        """Same seed should produce identical experiment sequences."""

        def run_with_seed(seed: int) -> list[float]:
            adapter = EnergyLoadAdapter(data=fixture_df, seed=seed)
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
            return [r.metrics["mae"] for r in log.results]

        run1 = run_with_seed(42)
        run2 = run_with_seed(42)
        assert run1 == run2

    def test_optimization_improves_over_exploration(self, fixture_df: pd.DataFrame) -> None:
        """Best MAE in optimization phase should be < median MAE in exploration."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
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

        log = engine.run_loop(20)

        # Split into exploration (first 10) and optimization (rest)
        exploration_maes = [r.metrics["mae"] for r in log.results[:10] if r.status.value != "crash"]
        optimization_maes = [
            r.metrics["mae"] for r in log.results[10:] if r.status.value != "crash"
        ]

        if exploration_maes and optimization_maes:
            median_exploration = float(np.median(exploration_maes))
            best_optimization = min(optimization_maes)
            assert best_optimization < median_exploration, (
                f"Best optimization MAE ({best_optimization:.3f}) should be < "
                f"median exploration MAE ({median_exploration:.3f})"
            )
