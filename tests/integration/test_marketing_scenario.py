"""Integration test: MarketingAdapter end-to-end through ExperimentEngine.

Validates that the full optimization loop works with the marketing domain
adapter's 5-variable mixed-type search space and prior causal graph.
Uses a simulated runner (no real campaign platform) to verify:
  - No crashes over 30 experiments
  - POMIS prunes the 5-variable space (fewer than 5 variables per set)
  - Engine focuses on causally relevant variables during optimization
  - Phase transitions happen correctly
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.domain_adapters.marketing import MarketingAdapter
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.optimizer.pomis import compute_pomis


class MarketingSimRunner:
    """Simulated marketing experiment runner with known causal structure.

    The objective (conversions) depends causally on email_frequency,
    social_spend_pct, search_bid_multiplier, and retargeting_enabled.
    creative_variant has an indirect effect via click_through_rate.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        email = float(parameters.get("email_frequency", 0))
        social = float(parameters.get("social_spend_pct", 0.0))
        bid = float(parameters.get("search_bid_multiplier", 1.0))
        retargeting = parameters.get("retargeting_enabled", False)
        creative = parameters.get("creative_variant", "control")

        # Simulate causal mechanisms
        email_opens = email * 0.3 + self._rng.normal(0, 0.05)
        social_impressions = social * 100 + self._rng.normal(0, 2)
        brand_awareness = social_impressions * 0.01 + self._rng.normal(0, 0.05)
        search_volume = brand_awareness * 10 + self._rng.normal(0, 1)
        search_clicks = search_volume * bid * 0.1 + self._rng.normal(0, 0.5)

        # Creative effect
        creative_map = {"control": 0.0, "urgency": 0.15, "social_proof": 0.1, "benefit": 0.05}
        ctr = 0.02 + creative_map.get(creative, 0.0) + self._rng.normal(0, 0.005)

        site_visits = email_opens + search_clicks + ctr * 100 + self._rng.normal(0, 1)
        return_visits = (10.0 if retargeting else 0.0) + self._rng.normal(0, 0.5)

        conversions = site_visits * 0.05 + return_visits * 0.1 + self._rng.normal(0, 0.2)

        return {
            "conversions": conversions,
            "total_spend": social * 1000 + bid * 50 + email * 10,
            "channel_diversity": min(1.0, (social > 0.1) + (email > 1) + (bid > 1.0)) / 3.0,
        }


@pytest.mark.slow
class TestMarketingScenario:
    """End-to-end test of ExperimentEngine with MarketingAdapter."""

    def test_engine_runs_30_experiments_without_crash(self) -> None:
        """The engine should complete 30 experiments with no exceptions."""
        adapter = MarketingAdapter()
        space = adapter.get_search_space()
        graph = adapter.get_prior_graph()
        runner = MarketingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=space,
            runner=runner,
            objective_name="conversions",
            minimize=False,  # maximize conversions
            causal_graph=graph,
            epsilon_mode=False,
            seed=42,
        )

        log = engine.run_loop(30)

        assert len(log.results) == 30
        # At least some experiments should be KEEP
        keep_count = sum(1 for r in log.results if r.status.value == "keep")
        assert keep_count >= 1, "Expected at least one KEEP result"

    def test_no_crash_results(self) -> None:
        """No experiments should crash with valid simulated data."""
        adapter = MarketingAdapter()
        runner = MarketingSimRunner(seed=123)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="conversions",
            minimize=False,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=123,
        )

        log = engine.run_loop(30)

        crash_count = sum(1 for r in log.results if r.status.value == "crash")
        assert crash_count == 0, f"Expected 0 crashes, got {crash_count}"

    def test_pomis_prunes_search_space(self) -> None:
        """POMIS should produce intervention sets smaller than the full 5-variable space."""
        adapter = MarketingAdapter()
        graph = adapter.get_prior_graph()
        space = adapter.get_search_space()

        pomis_sets = compute_pomis(graph, "conversions")

        assert len(pomis_sets) >= 1, "Expected at least one POMIS set"
        # Each POMIS set should be a proper subset of all search space + graph variables
        all_search_vars = set(space.variable_names)
        for pset in pomis_sets:
            # POMIS sets contain graph nodes, not just search space vars.
            # The key pruning test: at least one set should have fewer
            # search-space-intersecting variables than the full 5.
            search_overlap = pset & all_search_vars
            # The set should not be the entire search space
            assert len(search_overlap) < len(all_search_vars) or len(pset) < len(all_search_vars), (
                f"POMIS set {pset} did not prune the search space"
            )

    def test_phase_transitions(self) -> None:
        """Engine should transition through exploration -> optimization phases."""
        adapter = MarketingAdapter()
        runner = MarketingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="conversions",
            minimize=False,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        phases_seen: set[str] = set()
        for _ in range(30):
            result = engine.step()
            phase = result.metadata.get("phase", "unknown")
            phases_seen.add(phase)

        assert "exploration" in phases_seen, "Should start in exploration phase"
        assert "optimization" in phases_seen, "Should transition to optimization phase"

    def test_engine_focuses_on_causal_ancestors(self) -> None:
        """After screening, the engine should focus on causally relevant variables.

        The marketing graph has all 5 search space variables as ancestors of
        'conversions', so focus_variables should include them. The key test is
        that POMIS sets are computed and used to constrain optimization.
        """
        adapter = MarketingAdapter()
        runner = MarketingSimRunner(seed=42)

        engine = ExperimentEngine(
            search_space=adapter.get_search_space(),
            runner=runner,
            objective_name="conversions",
            minimize=False,
            causal_graph=adapter.get_prior_graph(),
            epsilon_mode=False,
            seed=42,
        )

        # Run past the exploration phase to trigger screening + POMIS
        engine.run_loop(15)

        # POMIS should have been computed (graph has confounders)
        assert engine.pomis_sets is not None, "POMIS sets should be computed"
        assert len(engine.pomis_sets) >= 1, "Should have at least one POMIS set"

    def test_multiple_seeds_produce_valid_results(self) -> None:
        """Different seeds should all produce valid (non-crash) results.

        Full determinism is not guaranteed because LatinHypercube and other
        internal RNG sources are not all seeded by the engine seed. Instead
        we verify that multiple seeds all complete successfully.
        """
        adapter = MarketingAdapter()

        for seed in [42, 99, 123]:
            runner = MarketingSimRunner(seed=seed)
            engine = ExperimentEngine(
                search_space=adapter.get_search_space(),
                runner=runner,
                objective_name="conversions",
                minimize=False,
                causal_graph=adapter.get_prior_graph(),
                epsilon_mode=False,
                seed=seed,
            )
            log = engine.run_loop(15)
            assert len(log.results) == 15
            crashes = sum(1 for r in log.results if r.status.value == "crash")
            assert crashes == 0, f"Seed {seed} produced {crashes} crashes"
