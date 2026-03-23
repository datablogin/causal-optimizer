"""Wall-clock regression test for MarketingLogAdapter pipeline.

Ensures the off-policy predictor hot-path optimizations keep the marketing
fixture pipeline within acceptable wall-clock budgets.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.engine.loop import ExperimentEngine

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "marketing_log_fixture.csv"


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


@pytest.mark.slow
class TestMarketingPerformance:
    """Wall-clock performance regression for marketing pipeline."""

    def test_15_experiments_under_15_seconds(self, fixture_df: pd.DataFrame) -> None:
        """15-experiment marketing pipeline should complete in < 15s wall-clock."""
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

        start = time.monotonic()
        log = engine.run_loop(15)
        elapsed = time.monotonic() - start

        assert len(log.results) == 15
        assert elapsed < 15.0, (
            f"Marketing pipeline took {elapsed:.1f}s, expected < 15s. "
            f"Off-policy hot path may need optimization."
        )
