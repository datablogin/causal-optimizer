"""Off-policy evaluation — predict experiment outcomes without running them.

Uses historical experiment data to estimate what a candidate configuration
would achieve, enabling cheap pre-screening before expensive execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

if TYPE_CHECKING:
    from causal_optimizer.types import ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Predicted outcome for a candidate experiment."""

    expected_value: float
    uncertainty: float  # standard deviation of prediction
    confidence_interval: tuple[float, float]
    model_quality: float  # cross-validated R² of the surrogate


class OffPolicyPredictor:
    """Predict experiment outcomes from historical data.

    Fits a surrogate model to past experiment results, then predicts
    outcomes for candidate configurations. High uncertainty triggers
    actual experimentation (the observation-intervention tradeoff).
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        min_history: int = 5,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.min_history = min_history
        self._model: RandomForestRegressor | None = None
        self._var_names: list[str] = []
        self._model_quality: float = 0.0

    def fit(
        self,
        experiment_log: ExperimentLog,
        search_space: SearchSpace,
        objective_name: str = "objective",
    ) -> None:
        """Fit the surrogate model on experiment history."""
        df = experiment_log.to_dataframe()
        self._var_names = [v.name for v in search_space.variables if v.name in df.columns]

        if len(df) < self.min_history or not self._var_names:
            self._model = None
            return

        features = (
            df[self._var_names]
            .apply(
                lambda x: x.astype(float, errors="ignore"),
            )
            .fillna(0)
            .values
        )
        y = df[objective_name].values

        self._model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        self._model.fit(features, y)

        # Estimate model quality
        if len(df) >= 10:
            scores = cross_val_score(
                RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
                features,
                y,
                cv=min(5, len(df)),
                scoring="r2",
            )
            self._model_quality = float(max(0, np.mean(scores)))
        else:
            self._model_quality = 0.0

    def predict(self, parameters: dict[str, Any]) -> Prediction | None:
        """Predict the outcome of a candidate experiment."""
        if self._model is None:
            return None

        x = np.array([parameters.get(v, 0) for v in self._var_names]).reshape(1, -1)

        # Get predictions from individual trees for uncertainty
        tree_predictions = np.array([tree.predict(x)[0] for tree in self._model.estimators_])

        expected = float(np.mean(tree_predictions))
        uncertainty = float(np.std(tree_predictions))
        ci = (
            float(np.percentile(tree_predictions, 2.5)),
            float(np.percentile(tree_predictions, 97.5)),
        )

        return Prediction(
            expected_value=expected,
            uncertainty=uncertainty,
            confidence_interval=ci,
            model_quality=self._model_quality,
        )

    def should_run_experiment(self, parameters: dict[str, Any]) -> bool:
        """Decide whether to run the experiment or trust the prediction.

        This implements the observation-intervention tradeoff from CBO:
        - Low uncertainty + good model → trust prediction, skip experiment
        - High uncertainty or poor model → run the experiment
        """
        prediction = self.predict(parameters)
        if prediction is None:
            return True  # No model, must run

        if prediction.model_quality < 0.3:
            return True  # Model too poor to trust

        return prediction.uncertainty > self.uncertainty_threshold
