"""Marketing campaign optimization adapter.

Connects to the causal-inference-marketing library for treatment effect
estimation and uses marketing-specific causal graphs.
"""

from __future__ import annotations

from typing import Any

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class MarketingAdapter(DomainAdapter):
    """Adapter for marketing campaign optimization.

    Optimizes marketing interventions (channel mix, bid strategy, creative,
    audience targeting) to maximize incremental conversions.
    """

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(variables=[
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
        ])

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run a marketing experiment (simulation or live A/B test)."""
        raise NotImplementedError(
            "MarketingAdapter.run_experiment requires integration with your "
            "campaign platform. Override this method with your implementation."
        )

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
