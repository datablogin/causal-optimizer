"""Marketing campaign optimization adapter.

Simulates a marketing campaign system with realistic causal structure:
email campaigns, social ads, search ads, creative variants, and retargeting.

Structural equations follow the prior causal graph:
  email_frequency -> email_opens -> site_visits -> conversions
  social_spend_pct -> social_impressions -> brand_awareness -> search_volume -> conversions
  search_bid_multiplier -> paid_clicks -> conversions
  creative_variant -> ctr -> conversions
  retargeting_enabled -> repeat_visits -> conversions
  Bidirected: U_purchase_intent <-> (social_impressions, conversions)
  Bidirected: U_seasonality <-> (brand_awareness, search_volume)

Approximate known optimum (interior, NOT at boundary):
  email_frequency ~= 5, social_spend_pct ~= 0.6, search_bid_multiplier ~= 1.8,
  creative_variant = "urgency", retargeting_enabled = True
  -> conversions ~= 5.5 (noise-free)

The optimum is interior because:
  - email_opens saturates (log), so email_frequency=7 is wasteful
  - social_spend_pct has diminishing returns (sqrt)
  - search_bid_multiplier has diminishing returns (sqrt)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class MarketingAdapter(DomainAdapter):
    """Adapter for marketing campaign optimization.

    Optimizes marketing interventions (channel mix, bid strategy, creative,
    audience targeting) to maximize incremental conversions.

    Args:
        seed: Random seed for reproducibility.
        noise_scale: Standard deviation of Gaussian noise added to each
            intermediate variable. Default 0.1.
    """

    def __init__(self, seed: int | None = None, noise_scale: float = 0.1) -> None:
        self._seed = seed
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """ExperimentRunner protocol: delegates to run_experiment."""
        return self.run_experiment(parameters)

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(
                    name="email_frequency",
                    variable_type=VariableType.INTEGER,
                    lower=0,
                    upper=7,
                ),
                Variable(
                    name="social_spend_pct",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="search_bid_multiplier",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.5,
                    upper=3.0,
                ),
                Variable(
                    name="creative_variant",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["control", "urgency", "social_proof", "benefit"],
                ),
                Variable(
                    name="retargeting_enabled",
                    variable_type=VariableType.BOOLEAN,
                ),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run a simulated marketing experiment respecting the prior causal graph.

        Each intermediate variable is computed from its causal parents plus
        Gaussian noise. Two latent confounders (U_purchase_intent and
        U_seasonality) inject correlated variation into their respective
        bidirected-edge endpoints.
        """
        sigma = self._noise_scale

        # Extract parameters
        email_freq = float(parameters.get("email_frequency", 0))
        social_spend = float(parameters.get("social_spend_pct", 0.0))
        bid_mult = float(parameters.get("search_bid_multiplier", 1.0))
        creative = parameters.get("creative_variant", "control")
        retargeting = parameters.get("retargeting_enabled", False)

        # --- Latent confounders (per-experiment draws) ---
        u_purchase_intent = self._rng.normal(0, 1)
        u_seasonality = self._rng.normal(0, 1)

        # --- Structural equations ---

        # email_frequency -> email_opens (saturating: log)
        email_opens = np.log1p(email_freq) * 2.0 + self._rng.normal(0, sigma)
        email_opens = max(0.0, email_opens)

        # social_spend_pct -> social_impressions (diminishing returns: sqrt)
        social_impressions = (
            np.sqrt(social_spend) * 80.0
            + u_purchase_intent * 5.0  # confounder affects ad exposure
            + self._rng.normal(0, sigma * 2)
        )
        social_impressions = max(0.0, social_impressions)

        # social_impressions -> brand_awareness (+ seasonality confounder)
        brand_awareness = (
            social_impressions * 0.02
            + u_seasonality * 0.3  # seasonality affects brand awareness
            + self._rng.normal(0, sigma)
        )
        brand_awareness = max(0.0, brand_awareness)

        # brand_awareness -> search_volume (+ seasonality confounder)
        search_volume = (
            brand_awareness * 8.0
            + u_seasonality * 2.0  # seasonality also affects search volume
            + self._rng.normal(0, sigma)
        )
        search_volume = max(0.0, search_volume)

        # search_volume + search_bid_multiplier -> paid_clicks (diminishing returns on bid)
        paid_clicks = search_volume * np.sqrt(bid_mult) * 0.15 + self._rng.normal(0, sigma)
        paid_clicks = max(0.0, paid_clicks)

        # creative_variant -> click_through_rate
        creative_map = {
            "control": 0.02,
            "urgency": 0.05,
            "social_proof": 0.04,
            "benefit": 0.03,
        }
        ctr = creative_map.get(creative, 0.02) + self._rng.normal(0, sigma * 0.1)
        ctr = max(0.001, ctr)

        # email_opens + paid_clicks + ctr -> site_visits
        site_visits = (
            email_opens * 0.8
            + paid_clicks
            + ctr * 60.0  # CTR applied to a base traffic pool
            + self._rng.normal(0, sigma)
        )
        site_visits = max(0.0, site_visits)

        # retargeting_enabled -> repeat_visits
        repeat_visits = (8.0 if retargeting else 0.0) + self._rng.normal(0, sigma)
        repeat_visits = max(0.0, repeat_visits)

        # conversions = f(site_visits, repeat_visits, U_purchase_intent) — additive
        conversions = (
            site_visits * 0.08
            + repeat_visits * 0.15
            + u_purchase_intent * 0.3  # confounder directly affects conversions
            + self._rng.normal(0, sigma)
        )

        # --- Descriptor metrics ---
        total_spend = social_spend * 1000 + bid_mult * 50 + email_freq * 10
        active_channels = (
            (1.0 if social_spend > 0.1 else 0.0)
            + (1.0 if email_freq >= 1 else 0.0)
            + (1.0 if bid_mult > 0.6 else 0.0)
        )
        channel_diversity = min(1.0, active_channels / 3.0)

        return {
            "conversions": float(conversions),
            "total_spend": float(total_spend),
            "channel_diversity": float(channel_diversity),
        }

    def get_prior_graph(self) -> CausalGraph:
        """Marketing-specific causal graph based on domain knowledge."""
        return CausalGraph(
            edges=[
                ("email_frequency", "email_opens"),
                ("email_opens", "site_visits"),
                ("social_spend_pct", "social_impressions"),
                ("social_impressions", "brand_awareness"),
                ("brand_awareness", "search_volume"),
                ("search_volume", "search_clicks"),
                ("search_bid_multiplier", "search_clicks"),
                ("search_clicks", "site_visits"),
                ("site_visits", "conversions"),
                ("creative_variant", "click_through_rate"),
                ("click_through_rate", "site_visits"),
                ("retargeting_enabled", "return_visits"),
                ("return_visits", "conversions"),
            ],
            bidirected_edges=[
                # Purchase intent confounds both ad exposure and conversions
                ("social_impressions", "conversions"),
                # Seasonality confounds brand awareness and search volume
                ("brand_awareness", "search_volume"),
            ],
        )

    def get_descriptor_names(self) -> list[str]:
        return ["total_spend", "channel_diversity"]

    def get_objective_name(self) -> str:
        return "conversions"

    def get_minimize(self) -> bool:
        return False  # maximize conversions
