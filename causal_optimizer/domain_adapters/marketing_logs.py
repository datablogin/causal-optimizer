"""Marketing log-based policy evaluation adapter.

Evaluates marketing treatment policies using logged observational data
with inverse propensity score (IPS/IPW) weighting. Unlike the simulation-based
MarketingAdapter, this adapter works with real or pre-generated log data
and evaluates counterfactual policies offline.

Policy variables control:
  - eligibility_threshold: minimum predicted uplift score to treat a user
  - email_share / social_share_of_remainder: budget allocation across channels
  - min_propensity_clip: floor for IPS weights to control variance
  - regularization: uplift model regularization strength
  - treatment_budget_pct: fraction of eligible users to treat

The adapter computes IPS-weighted policy value, total cost, treated fraction,
and effective sample size for each candidate policy configuration.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType

logger = logging.getLogger(__name__)


class MarketingLogAdapter(DomainAdapter):
    """Adapter for offline marketing policy evaluation from logged data.

    Evaluates treatment policies using inverse propensity score (IPS) weighting
    on observational marketing data.

    Args:
        data: DataFrame with logged marketing data.  Must contain at least
            the treatment, outcome, and cost columns.
        data_path: Path to a CSV file to load as the data.  Exactly one of
            ``data`` or ``data_path`` must be provided.
        seed: Accepted for API consistency with other adapters. Unused
            because evaluation is fully deterministic given fixed data
            and parameters (no stochastic elements).
        propensity_col: Column name for logging-policy propensity scores.
            If the column does not exist in the data, uniform propensity
            (treatment rate) is assumed.
        treatment_col: Column name for binary treatment indicator (0/1).
        outcome_col: Column name for the outcome (e.g., revenue).
        cost_col: Column name for per-observation cost.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_path: str | None = None,
        seed: int | None = None,
        propensity_col: str = "propensity",
        treatment_col: str = "treatment",
        outcome_col: str = "outcome",
        cost_col: str = "cost",
    ) -> None:
        if data is not None and data_path is not None:
            raise ValueError("Exactly one of 'data' or 'data_path' must be provided, not both.")
        if data is None and data_path is None:
            raise ValueError("Exactly one of 'data' or 'data_path' must be provided.")

        if data_path is not None:
            data = pd.read_csv(data_path)

        assert data is not None  # for type narrowing
        self._seed = seed
        self._data = data.copy()
        self._propensity_col = propensity_col
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._cost_col = cost_col

        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist and data is non-empty."""
        if self._data.empty:
            raise ValueError("Empty DataFrame: data must contain at least one row.")

        missing = []
        for col_name, col_attr in [
            ("treatment", self._treatment_col),
            ("outcome", self._outcome_col),
            ("cost", self._cost_col),
        ]:
            if col_attr not in self._data.columns:
                missing.append(f"{col_name} (column '{col_attr}')")

        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        nan_cols = []
        for col in [self._treatment_col, self._outcome_col, self._cost_col]:
            if col in self._data.columns and self._data[col].isna().any():
                nan_cols.append(col)
        if (
            self._propensity_col in self._data.columns
            and self._data[self._propensity_col].isna().any()
        ):
            nan_cols.append(self._propensity_col)
        if nan_cols:
            raise ValueError(f"Columns contain NaN values: {', '.join(nan_cols)}")

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(
                    name="eligibility_threshold",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="email_share",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="social_share_of_remainder",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="min_propensity_clip",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.01,
                    upper=0.5,
                ),
                Variable(
                    name="regularization",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.001,
                    upper=10.0,
                ),
                Variable(
                    name="treatment_budget_pct",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.1,
                    upper=1.0,
                ),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate a treatment policy on the logged data using IPS weighting.

        Steps:
        1. Compute uplift scores for each observation using a simple model.
        2. Apply eligibility threshold and budget constraints.
        3. Clip propensities and compute IPS weights.
        4. Compute IPS-weighted policy value, total cost, treated fraction, ESS.
        """
        eligibility_threshold = float(parameters.get("eligibility_threshold", 0.3))
        email_share = float(parameters.get("email_share", 0.4))
        social_share_of_remainder = float(parameters.get("social_share_of_remainder", 0.5))

        # Simplex parameterization: shares always sum to 1.0
        social_share = social_share_of_remainder * (1.0 - email_share)
        min_propensity_clip = float(parameters.get("min_propensity_clip", 0.05))
        regularization = float(parameters.get("regularization", 1.0))
        treatment_budget_pct = float(parameters.get("treatment_budget_pct", 0.5))

        df = self._data
        n = len(df)
        treatment = df[self._treatment_col].values.astype(int)
        outcome = df[self._outcome_col].values.astype(float)
        cost = df[self._cost_col].values.astype(float)

        # Get or estimate propensities
        if self._propensity_col in df.columns:
            propensity = df[self._propensity_col].values.astype(float)
        else:
            # Fallback: use marginal treatment rate as uniform propensity
            treat_rate = np.clip(treatment.mean(), 0.01, 0.99)
            propensity = np.full(n, treat_rate)

        # Clip propensities for stability
        propensity_clipped = np.clip(propensity, min_propensity_clip, 1.0 - min_propensity_clip)

        # Compute uplift scores using a simple approach:
        # Score = (channel-weighted outcome expectation) * regularization factor
        # This creates heterogeneity based on policy parameters

        # Build per-observation scores based on policy parameters
        # Channel allocation affects which observations are prioritized
        channel_weight = np.ones(n)
        if "channel" in df.columns:
            channels = df["channel"].values
            email_mask = channels == "email"
            social_mask = channels == "social"
            search_share = max(0.0, 1.0 - email_share - social_share)
            channel_weight = np.where(
                email_mask,
                email_share + 0.1,
                np.where(
                    social_mask,
                    social_share + 0.1,
                    search_share + 0.1,
                ),
            )

        # Segment-based uplift scoring (if segment info available)
        segment_score = np.ones(n)
        if "segment" in df.columns:
            segments = df["segment"].values
            segment_score = np.where(
                segments == "high_value",
                0.8,
                np.where(segments == "medium", 0.5, 0.2),
            )

        # Compute uplift score: combines channel weight, segment, and regularization
        # Regularization smooths scores toward uniform (higher reg = more uniform)
        raw_score = channel_weight * segment_score
        reg_factor = 1.0 / (1.0 + regularization)
        uplift_score = reg_factor * raw_score + (1.0 - reg_factor) * np.mean(raw_score)

        # Normalize to [0, 1]
        score_min = uplift_score.min()
        score_max = uplift_score.max()
        if score_max > score_min:
            uplift_score = (uplift_score - score_min) / (score_max - score_min)
        else:
            uplift_score = np.full(n, 0.5)

        # Apply eligibility: only treat users with uplift score >= threshold
        eligible = uplift_score >= eligibility_threshold

        # Apply budget constraint: treat at most treatment_budget_pct of eligible
        n_eligible = eligible.sum()
        n_to_treat = int(n_eligible * treatment_budget_pct)

        # Select top-scoring eligible observations
        policy_treat = np.zeros(n, dtype=bool)
        if n_to_treat > 0:
            eligible_indices = np.where(eligible)[0]
            eligible_scores = uplift_score[eligible_indices]
            # Sort by score descending, take top n_to_treat
            top_indices = eligible_indices[np.argsort(-eligible_scores)[:n_to_treat]]
            policy_treat[top_indices] = True

        # IPS-weighted policy value
        # For treated observations matching policy: weight by 1/p(T=1)
        # For control observations matching policy: weight by 1/p(T=0) = 1/(1-p)
        weights = np.zeros(n)
        # Policy says treat, and actually treated
        match_treat = policy_treat & (treatment == 1)
        weights[match_treat] = 1.0 / propensity_clipped[match_treat]
        # Policy says don't treat, and actually not treated
        match_control = (~policy_treat) & (treatment == 0)
        weights[match_control] = 1.0 / (1.0 - propensity_clipped[match_control])

        # Self-normalized IPS for stability
        weight_sum = weights.sum()
        if weight_sum > 0:
            normalized_weights = weights / weight_sum * n
        else:
            # Fallback: uniform weights if no matches
            normalized_weights = np.ones(n)
            logger.warning("No policy-matching observations; using uniform weights.")

        # Policy value: weighted average outcome
        policy_value = float(np.sum(normalized_weights * outcome) / n)

        # Total cost: sum of IPS-weighted costs for treated observations under policy
        treated_cost_weights = np.zeros(n)
        treated_cost_weights[match_treat] = normalized_weights[match_treat]
        total_cost = float(np.sum(treated_cost_weights * cost) / n)

        treated_fraction = float(policy_treat.mean())

        # Effective sample size (Kish's ESS)
        ess = float(weight_sum**2 / np.sum(weights**2)) if weight_sum > 0 else 0.0

        return {
            "policy_value": policy_value,
            "total_cost": total_cost,
            "treated_fraction": treated_fraction,
            "effective_sample_size": ess,
        }

    def get_prior_graph(self) -> CausalGraph:
        """Prior causal graph for marketing policy evaluation."""
        return CausalGraph(
            edges=[
                ("eligibility_threshold", "treated_fraction"),
                ("treatment_budget_pct", "treated_fraction"),
                ("treated_fraction", "total_cost"),
                ("treated_fraction", "policy_value"),
                ("treated_fraction", "effective_sample_size"),
                ("email_share", "total_cost"),
                ("email_share", "policy_value"),
                ("social_share_of_remainder", "total_cost"),
                ("social_share_of_remainder", "policy_value"),
                ("regularization", "treated_fraction"),
                ("regularization", "policy_value"),
                ("min_propensity_clip", "total_cost"),
                ("min_propensity_clip", "effective_sample_size"),
                ("min_propensity_clip", "policy_value"),
            ],
        )

    def get_descriptor_names(self) -> list[str]:
        return ["total_cost", "treated_fraction"]

    def get_objective_name(self) -> str:
        return "policy_value"

    def get_minimize(self) -> bool:
        return False  # maximize policy value
