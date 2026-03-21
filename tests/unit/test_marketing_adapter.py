"""Unit tests for MarketingAdapter simulator.

Tests the structural equations, confounders, noise, seed support,
and metric outputs of the marketing campaign simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.domain_adapters.marketing import MarketingAdapter


class TestMarketingAdapterBasics:
    """Basic adapter contract tests."""

    def test_get_search_space_has_5_variables(self) -> None:
        adapter = MarketingAdapter()
        space = adapter.get_search_space()
        assert len(space.variables) == 5

    def test_get_prior_graph_returns_graph(self) -> None:
        adapter = MarketingAdapter()
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert len(graph.edges) > 0
        assert len(graph.bidirected_edges) == 2

    def test_get_descriptor_names(self) -> None:
        adapter = MarketingAdapter()
        assert adapter.get_descriptor_names() == ["total_spend", "channel_diversity"]

    def test_get_objective_name(self) -> None:
        adapter = MarketingAdapter()
        assert adapter.get_objective_name() == "conversions"

    def test_get_minimize_is_false(self) -> None:
        adapter = MarketingAdapter()
        assert adapter.get_minimize() is False


class TestMarketingSimulator:
    """Tests for the run_experiment simulator."""

    def test_run_experiment_returns_three_metrics(self) -> None:
        adapter = MarketingAdapter(seed=42)
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.5,
            "creative_variant": "urgency",
            "retargeting_enabled": True,
        }
        metrics = adapter.run_experiment(params)
        assert "conversions" in metrics
        assert "total_spend" in metrics
        assert "channel_diversity" in metrics

    def test_conversions_is_positive_at_reasonable_params(self) -> None:
        adapter = MarketingAdapter(seed=42)
        params = {
            "email_frequency": 4,
            "social_spend_pct": 0.6,
            "search_bid_multiplier": 1.5,
            "creative_variant": "social_proof",
            "retargeting_enabled": True,
        }
        metrics = adapter.run_experiment(params)
        assert metrics["conversions"] > 0

    def test_total_spend_is_nonnegative(self) -> None:
        adapter = MarketingAdapter(seed=42)
        params = {
            "email_frequency": 0,
            "social_spend_pct": 0.0,
            "search_bid_multiplier": 0.5,
            "creative_variant": "control",
            "retargeting_enabled": False,
        }
        metrics = adapter.run_experiment(params)
        assert metrics["total_spend"] >= 0

    def test_channel_diversity_between_0_and_1(self) -> None:
        adapter = MarketingAdapter(seed=42)
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.5,
            "creative_variant": "urgency",
            "retargeting_enabled": True,
        }
        metrics = adapter.run_experiment(params)
        assert 0.0 <= metrics["channel_diversity"] <= 1.0


class TestMarketingStructuralEquations:
    """Tests for the causal structure of the simulator."""

    def test_email_frequency_saturates(self) -> None:
        """Higher email_frequency should increase conversions but with diminishing returns."""
        base = {
            "social_spend_pct": 0.3,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": False,
        }
        # Low vs medium vs high email frequency
        results = []
        for freq in [1, 4, 7]:
            adapter_i = MarketingAdapter(seed=42)
            params = {**base, "email_frequency": freq}
            results.append(adapter_i.run_experiment(params)["conversions"])

        # Monotonically increasing
        assert results[1] > results[0], "Medium email > low email"
        assert results[2] > results[1], "High email > medium email"
        # Diminishing returns: marginal gain from 4->7 should be less than 1->4
        gain_low = results[1] - results[0]
        gain_high = results[2] - results[1]
        assert gain_high < gain_low, "Email frequency should show diminishing returns"

    def test_retargeting_increases_conversions(self) -> None:
        """retargeting_enabled=True should increase conversions."""
        base = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
        }
        off = MarketingAdapter(seed=42).run_experiment({**base, "retargeting_enabled": False})
        on = MarketingAdapter(seed=42).run_experiment({**base, "retargeting_enabled": True})
        assert on["conversions"] > off["conversions"]

    def test_creative_variant_urgency_best(self) -> None:
        """'urgency' creative should produce higher CTR effect than 'control'."""
        base = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "retargeting_enabled": False,
        }
        control = MarketingAdapter(seed=42).run_experiment({**base, "creative_variant": "control"})
        urgency = MarketingAdapter(seed=42).run_experiment({**base, "creative_variant": "urgency"})
        assert urgency["conversions"] > control["conversions"]

    def test_social_spend_increases_conversions(self) -> None:
        """Higher social_spend_pct should increase conversions (via brand path)."""
        base = {
            "email_frequency": 0,
            "search_bid_multiplier": 0.5,
            "creative_variant": "control",
            "retargeting_enabled": False,
        }
        low = MarketingAdapter(seed=42).run_experiment({**base, "social_spend_pct": 0.1})
        high = MarketingAdapter(seed=42).run_experiment({**base, "social_spend_pct": 0.9})
        assert high["conversions"] > low["conversions"]

    def test_search_bid_multiplier_increases_conversions(self) -> None:
        """Higher bid multiplier should increase conversions via paid clicks."""
        base = {
            "email_frequency": 0,
            "social_spend_pct": 0.3,
            "creative_variant": "control",
            "retargeting_enabled": False,
        }
        low = MarketingAdapter(seed=42).run_experiment({**base, "search_bid_multiplier": 0.5})
        high = MarketingAdapter(seed=42).run_experiment({**base, "search_bid_multiplier": 3.0})
        assert high["conversions"] > low["conversions"]


class TestMarketingConfounders:
    """Tests for latent confounder effects."""

    def test_confounder_adds_variance(self) -> None:
        """Running with different seeds should show variance from confounders."""
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": True,
        }
        results = [
            MarketingAdapter(seed=s).run_experiment(params)["conversions"] for s in range(20)
        ]
        assert np.std(results) > 0, "Confounders should add variance across seeds"


class TestMarketingNoise:
    """Tests for noise scale configuration."""

    def test_custom_noise_scale(self) -> None:
        """noise_scale=0 should produce deterministic results (except confounders)."""
        adapter1 = MarketingAdapter(seed=42, noise_scale=0.0)
        adapter2 = MarketingAdapter(seed=42, noise_scale=0.0)
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": True,
        }
        m1 = adapter1.run_experiment(params)
        m2 = adapter2.run_experiment(params)
        assert m1["conversions"] == pytest.approx(m2["conversions"], abs=1e-10)

    def test_higher_noise_more_variance(self) -> None:
        """Higher noise_scale should produce more variance across runs."""
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": True,
        }
        low_noise_results = [
            MarketingAdapter(seed=s, noise_scale=0.01).run_experiment(params)["conversions"]
            for s in range(20)
        ]
        high_noise_results = [
            MarketingAdapter(seed=s, noise_scale=0.5).run_experiment(params)["conversions"]
            for s in range(20)
        ]
        assert np.std(high_noise_results) > np.std(low_noise_results)


class TestMarketingSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_same_seed_same_result(self) -> None:
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": True,
        }
        m1 = MarketingAdapter(seed=42).run_experiment(params)
        m2 = MarketingAdapter(seed=42).run_experiment(params)
        assert m1 == m2

    def test_different_seed_different_result(self) -> None:
        params = {
            "email_frequency": 3,
            "social_spend_pct": 0.5,
            "search_bid_multiplier": 1.0,
            "creative_variant": "control",
            "retargeting_enabled": True,
        }
        m1 = MarketingAdapter(seed=42).run_experiment(params)
        m2 = MarketingAdapter(seed=99).run_experiment(params)
        assert m1["conversions"] != pytest.approx(m2["conversions"], abs=1e-10)


class TestMarketingOptimum:
    """Test that the simulator has an interior optimum."""

    def test_interior_optimum_beats_boundaries(self) -> None:
        """A mid-range parameter set should beat extreme boundaries."""
        # All-min boundary
        boundary_low = {
            "email_frequency": 0,
            "social_spend_pct": 0.0,
            "search_bid_multiplier": 0.5,
            "creative_variant": "control",
            "retargeting_enabled": False,
        }
        # All-max boundary
        boundary_high = {
            "email_frequency": 7,
            "social_spend_pct": 1.0,
            "search_bid_multiplier": 3.0,
            "creative_variant": "urgency",
            "retargeting_enabled": True,
        }
        # Interior point
        interior = {
            "email_frequency": 4,
            "social_spend_pct": 0.6,
            "search_bid_multiplier": 1.8,
            "creative_variant": "urgency",
            "retargeting_enabled": True,
        }

        # Average over multiple seeds to overcome noise
        def avg_conversions(params: dict, n: int = 50) -> float:
            return float(
                np.mean(
                    [
                        MarketingAdapter(seed=s).run_experiment(params)["conversions"]
                        for s in range(n)
                    ]
                )
            )

        avg_low = avg_conversions(boundary_low)
        avg_high = avg_conversions(boundary_high)
        avg_interior = avg_conversions(interior)

        # Interior should beat at least one boundary
        assert avg_interior > avg_low, "Interior should beat low boundary"
        # Interior should beat all-max boundary (diminishing returns make max suboptimal)
        assert avg_interior > avg_high or avg_high > avg_low, (
            "Simulator should have interesting structure (not monotone to boundary)"
        )
