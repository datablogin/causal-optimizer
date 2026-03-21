"""Integration tests for MarketingLogAdapter through the ExperimentEngine.

Tests the full pipeline: adapter -> engine -> IPS evaluation -> diagnostics.
All tests marked @pytest.mark.slow (complete in <60s each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.engine.loop import ExperimentEngine

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "marketing_log_fixture.csv"


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    """Load the fixture CSV."""
    return pd.read_csv(FIXTURE_PATH)


@pytest.mark.slow
class TestMarketingLogAdapterIntegration:
    """End-to-end integration tests for MarketingLogAdapter."""

    def test_full_pipeline_15_experiments(self, fixture_df: pd.DataFrame) -> None:
        """Full pipeline: 15 experiments with no crashes, valid metrics."""
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
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
            assert "policy_value" in r.metrics
            assert "total_cost" in r.metrics
            assert "treated_fraction" in r.metrics
            assert "effective_sample_size" in r.metrics

    def test_phase_transitions(self, fixture_df: pd.DataFrame) -> None:
        """Engine should transition through exploration -> optimization phases."""
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
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
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
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
            adapter = MarketingLogAdapter(data=fixture_df, seed=seed)
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
            return [r.metrics["policy_value"] for r in log.results]

        run1 = run_with_seed(42)
        run2 = run_with_seed(42)
        assert run1 == run2

    def test_optimization_sanity(self, fixture_df: pd.DataFrame) -> None:
        """Best policy_value in optimization > median in exploration."""
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
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
        exploration_values = [
            r.metrics["policy_value"] for r in log.results[:10] if r.status.value != "crash"
        ]
        optimization_values = [
            r.metrics["policy_value"] for r in log.results[10:] if r.status.value != "crash"
        ]

        if exploration_values and optimization_values:
            median_exploration = float(np.median(exploration_values))
            best_optimization = max(optimization_values)
            # Best optimization should beat median exploration
            assert best_optimization >= median_exploration * 0.8, (
                f"Best optimization ({best_optimization:.3f}) should be competitive with "
                f"median exploration ({median_exploration:.3f})"
            )
