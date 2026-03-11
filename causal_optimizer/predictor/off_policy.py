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
from causal_optimizer.types import VariableType

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
        seed: int | None = None,
    ) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.min_history = min_history
        self.epsilon_mode = epsilon_mode
        self.n_max = n_max
        self._model: RandomForestRegressor | None = None
        self._var_names: list[str] = []
        self._numeric_var_names: list[str] = []
        self._model_quality: float = 0.0
        self._cached_features: np.ndarray | None = None
        self._cached_epsilon: float = 0.0
        self._search_space: SearchSpace | None = None
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def fit(
        self,
        experiment_log: ExperimentLog,
        search_space: SearchSpace,
        objective_name: str = "objective",
    ) -> None:
        """Fit the surrogate model on experiment history.

        When ``epsilon_mode`` is enabled, this method also caches the numeric
        feature matrix for the epsilon controller. The cache is a point-in-time
        snapshot that updates each time ``fit()`` is called. Since the engine
        calls ``fit()`` after every experiment (see ``run_experiment`` in
        ``loop.py``), the cache stays current.
        """
        self._search_space = search_space
        df = experiment_log.to_dataframe()
        self._var_names = [v.name for v in search_space.variables if v.name in df.columns]

        # Epsilon mode: track numeric variables, cache features, precompute epsilon
        if self.epsilon_mode:
            self._numeric_var_names = [
                v.name
                for v in search_space.variables
                if v.name in df.columns
                and v.variable_type in (VariableType.CONTINUOUS, VariableType.INTEGER)
            ]

            # Cache numeric features (even before min_history, since the
            # convex hull only needs 3+ rows, not min_history rows).
            # Drop rows with NaN rather than imputing 0.0 — imputing can
            # place points outside domain bounds and inflate hull coverage.
            if self._numeric_var_names and len(df) >= 1:
                numeric_df = df[self._numeric_var_names].dropna()
                if len(numeric_df) >= 1:
                    self._cached_features = numeric_df.to_numpy(dtype=np.float64)
                else:
                    self._cached_features = None
            else:
                self._cached_features = None

            # Precompute epsilon to avoid repeated ConvexHull work
            domain_bounds = self._get_domain_bounds()
            if self._cached_features is not None and domain_bounds is not None:
                n_current = self._cached_features.shape[0]
                self._cached_epsilon = compute_epsilon(
                    self._cached_features, domain_bounds, n_current, self.n_max
                )
            else:
                self._cached_epsilon = 0.0
        else:
            self._numeric_var_names = []
            self._cached_features = None
            self._cached_epsilon = 0.0

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
        """Epsilon controller decision logic from CBO (Aglietti et al.).

        Uses the epsilon value cached during the last ``fit()`` call to avoid
        recomputing the ConvexHull on every decision.
        """
        epsilon = self._cached_epsilon
        if epsilon <= 0.0:
            return True  # No coverage data or zero epsilon, must run

        # With probability epsilon, observe (skip experiment); otherwise intervene
        if self._rng.random() < epsilon:
            # Even when the epsilon controller says observe, check if
            # uncertainty is too high — if so, fall back to intervening.
            # Also intervene if there's no model (prediction is None),
            # since we can't trust a nonexistent surrogate.
            prediction = self.predict(parameters)
            if prediction is None:
                logger.debug(
                    "Epsilon controller chose observe (epsilon=%.3f), but no model "
                    "available; intervening instead",
                    epsilon,
                )
                return True
            if prediction.model_quality < 0.3:
                logger.debug(
                    "Epsilon controller chose observe (epsilon=%.3f), but model quality "
                    "too low (%.3f < 0.3); intervening instead",
                    epsilon,
                    prediction.model_quality,
                )
                return True
            if prediction.uncertainty > self.uncertainty_threshold:
                logger.debug(
                    "Epsilon controller chose observe (epsilon=%.3f), but uncertainty "
                    "too high (%.3f > %.3f); intervening instead",
                    epsilon,
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

    def _get_domain_bounds(self) -> list[tuple[float, float]] | None:
        """Extract bounds from the search space for numeric variables only.

        Only includes continuous and integer variables (not categorical or
        boolean).

        Returns:
            List of (lower, upper) tuples for each numeric variable, or None
            if the search space is not available or has no numeric variables.
        """
        if self._search_space is None:
            return None

        bounds: list[tuple[float, float]] = []
        for var in self._search_space.variables:
            if var.name not in self._numeric_var_names:
                continue
            if var.lower is None or var.upper is None:
                logger.warning(
                    "Variable '%s' has no bounds; defaulting to [0.0, 1.0] for epsilon "
                    "coverage estimate. Set explicit bounds for accurate results.",
                    var.name,
                )
            lower = var.lower if var.lower is not None else 0.0
            upper = var.upper if var.upper is not None else 1.0
            bounds.append((lower, upper))

        return bounds if bounds else None
