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

from causal_optimizer.predictor.epsilon import compute_epsilon

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
        epsilon_mode: bool = False,
        n_max: int = 100,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.min_history = min_history
        self.epsilon_mode = epsilon_mode
        self.n_max = n_max
        self._model: RandomForestRegressor | None = None
        self._var_names: list[str] = []
        self._model_quality: float = 0.0
        self._experiment_log: ExperimentLog | None = None
        self._search_space: SearchSpace | None = None
        self._rng: np.random.Generator = np.random.default_rng()

    def fit(
        self,
        experiment_log: ExperimentLog,
        search_space: SearchSpace,
        objective_name: str = "objective",
    ) -> None:
        """Fit the surrogate model on experiment history."""
        self._experiment_log = experiment_log
        self._search_space = search_space
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

        When epsilon_mode is False (default), uses the original heuristic:
        - Low uncertainty + good model -> trust prediction, skip experiment
        - High uncertainty or poor model -> run the experiment

        When epsilon_mode is True, uses the CBO epsilon controller:
        - Compute epsilon from coverage of the search space
        - With probability epsilon: OBSERVE (skip experiment, trust surrogate)
        - With probability 1-epsilon: INTERVENE (run experiment)
        - Even when observing, fall back to intervening if uncertainty is high
        """
        if self.epsilon_mode:
            return self._should_run_epsilon(parameters)

        return self._should_run_heuristic(parameters)

    def _should_run_heuristic(self, parameters: dict[str, Any]) -> bool:
        """Original heuristic-based decision logic."""
        prediction = self.predict(parameters)
        if prediction is None:
            return True  # No model, must run

        if prediction.model_quality < 0.3:
            return True  # Model too poor to trust

        return prediction.uncertainty > self.uncertainty_threshold

    def _should_run_epsilon(self, parameters: dict[str, Any]) -> bool:
        """Epsilon controller decision logic from CBO (Aglietti et al.)."""
        observed_data = self._get_observed_data()
        domain_bounds = self._get_domain_bounds()

        if observed_data is None or domain_bounds is None:
            return True  # Not enough data, must run

        n_current = observed_data.shape[0]
        epsilon = compute_epsilon(observed_data, domain_bounds, n_current, self.n_max)

        # With probability epsilon, observe (skip experiment)
        if self._rng.random() < epsilon:
            # Even when the epsilon controller says observe, check if
            # uncertainty is too high — if so, fall back to intervening
            prediction = self.predict(parameters)
            if prediction is not None and prediction.uncertainty > self.uncertainty_threshold:
                logger.debug(
                    "Epsilon controller chose observe, but uncertainty too high "
                    "(%.3f > %.3f); intervening instead",
                    prediction.uncertainty,
                    self.uncertainty_threshold,
                )
                return True
            logger.debug(
                "Epsilon controller chose observe (epsilon=%.3f); skipping experiment",
                epsilon,
            )
            return False

        # With probability 1-epsilon, intervene (run experiment)
        return True

    def _get_observed_data(self) -> np.ndarray | None:
        """Extract the feature matrix from experiment history.

        Returns:
            Array of shape (n_samples, n_dims) or None if insufficient data.
        """
        if self._experiment_log is None or self._search_space is None:
            return None

        if not self._var_names:
            return None

        df = self._experiment_log.to_dataframe()
        if len(df) < 3:
            return None

        features = (
            df[self._var_names].apply(lambda x: x.astype(float, errors="ignore")).fillna(0).values
        )
        return np.asarray(features, dtype=np.float64)

    def _get_domain_bounds(self) -> list[tuple[float, float]] | None:
        """Extract bounds from the search space for numeric variables.

        Returns:
            List of (lower, upper) tuples for each variable, or None if
            the search space is not available.
        """
        if self._search_space is None:
            return None

        bounds: list[tuple[float, float]] = []
        for var in self._search_space.variables:
            if var.name not in self._var_names:
                continue
            lower = var.lower if var.lower is not None else 0.0
            upper = var.upper if var.upper is not None else 1.0
            bounds.append((lower, upper))

        return bounds if bounds else None
