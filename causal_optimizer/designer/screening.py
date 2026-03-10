"""Screening designs for identifying important variables and interactions.

Uses fANOVA-style analysis and fractional factorial designs to determine
which variables and interactions dominate performance variation before
committing to full optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    import numpy as np

    from causal_optimizer.types import ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Result of variable importance screening."""

    main_effects: dict[str, float]  # variable -> importance score
    interactions: dict[tuple[str, str], float]  # (var1, var2) -> interaction strength
    important_variables: list[str]  # variables above threshold, sorted by importance

    @property
    def summary(self) -> str:
        lines = ["Variable Importance (main effects):"]
        for var, imp in sorted(self.main_effects.items(), key=lambda x: -x[1]):
            lines.append(f"  {var}: {imp:.4f}")
        if self.interactions:
            lines.append("\nTop Interactions:")
            sorted_interactions = sorted(self.interactions.items(), key=lambda x: -x[1])[:5]
            for (v1, v2), strength in sorted_interactions:
                lines.append(f"  {v1} x {v2}: {strength:.4f}")
        return "\n".join(lines)


class ScreeningDesigner:
    """Identify which variables and interactions matter most.

    Uses random forest-based functional ANOVA decomposition to estimate
    main effects and interaction effects from experiment logs.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        importance_threshold: float = 0.05,
    ) -> None:
        self.search_space = search_space
        self.importance_threshold = importance_threshold

    def screen(
        self,
        experiment_log: ExperimentLog,
        objective_name: str = "objective",
    ) -> ScreeningResult:
        """Analyze experiment history to identify important variables."""
        df = experiment_log.to_dataframe()
        var_names = self.search_space.variable_names
        available = [v for v in var_names if v in df.columns]

        if len(available) < 1 or objective_name not in df.columns:
            return ScreeningResult(
                main_effects={},
                interactions={},
                important_variables=[],
            )

        features = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[objective_name].values

        # Main effects via random forest feature importance
        main_effects = self._compute_main_effects(features, y)

        # Interaction effects via pairwise residual analysis
        interactions = self._compute_interactions(features, y, main_effects)

        important = [
            var
            for var, imp in sorted(main_effects.items(), key=lambda x: -x[1])
            if imp > self.importance_threshold
        ]

        return ScreeningResult(
            main_effects=main_effects,
            interactions=interactions,
            important_variables=important,
        )

    def _compute_main_effects(self, features: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Compute main effect importance via random forest."""
        if len(features) < 5:
            # Not enough data — return uniform importance
            return {col: 1.0 / len(features.columns) for col in features.columns}

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=min(5, len(features) // 2),
            random_state=42,
        )
        rf.fit(features, y)

        importances = rf.feature_importances_
        return {col: float(imp) for col, imp in zip(features.columns, importances, strict=False)}

    def _compute_interactions(
        self,
        features: pd.DataFrame,
        y: np.ndarray,
        main_effects: dict[str, float],
    ) -> dict[tuple[str, str], float]:
        """Estimate pairwise interaction strengths.

        Uses the H-statistic approach: fit a model with interaction terms
        and measure how much the interaction adds beyond main effects.
        """
        if len(features) < 10 or len(features.columns) < 2:
            return {}

        interactions: dict[tuple[str, str], float] = {}
        cols = features.columns.tolist()

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue

                # Create interaction feature
                pair_with_inter = features[[c1, c2]].copy()
                pair_with_inter["interaction"] = features[c1] * features[c2]

                # Fit with and without interaction
                rf_without = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
                rf_with = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)

                rf_without.fit(features[[c1, c2]], y)
                rf_with.fit(pair_with_inter, y)

                score_without = max(0, rf_without.score(features[[c1, c2]], y))
                score_with = max(0, rf_with.score(pair_with_inter, y))

                interaction_strength = max(0, score_with - score_without)
                if interaction_strength > 0.01:
                    interactions[(c1, c2)] = interaction_strength

        return interactions
