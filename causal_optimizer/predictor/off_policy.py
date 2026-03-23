"""Off-policy evaluation — predict experiment outcomes without running them.

Uses historical experiment data to estimate what a candidate configuration
would achieve, enabling cheap pre-screening before expensive execution.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from causal_optimizer.predictor.encoding import encode_dataframe_for_rf, encode_params_for_rf
from causal_optimizer.predictor.epsilon import compute_epsilon
from causal_optimizer.types import VariableType

if TYPE_CHECKING:
    from causal_optimizer.estimator.observational import ObservationalEstimate
    from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)

# Sentinel distinguishing "not yet tried" from "tried and failed"
_NOT_TRIED: object = object()


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
        causal_graph: CausalGraph | None = None,
        objective_name: str | None = None,
        obs_min_history: int = 20,
    ) -> None:
        """Initialize the off-policy predictor.

        Args:
            seed: Seed for the epsilon controller's internal RNG. Controls
                reproducibility of observe/intervene coin flips in
                ``_should_run_epsilon``. Has no effect when
                ``epsilon_mode=False``.
            causal_graph: Optional causal graph for observational estimation.
                When provided alongside ``objective_name``, the predictor
                will attempt to combine RF predictions with observational
                causal effect estimates for improved prediction.
            objective_name: Name of the objective metric. Required when
                ``causal_graph`` is provided.
            obs_min_history: Minimum number of experiments in the log before
                attempting observational (DoWhy) prediction. Prevents
                expensive DoWhy calls when there is insufficient data for
                reliable causal estimation.  Defaults to 20.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.min_history = min_history
        self.epsilon_mode = epsilon_mode
        self.n_max = n_max
        self.obs_min_history = obs_min_history
        self._model: RandomForestRegressor | None = None
        self._var_names: list[str] = []
        self._numeric_var_names: list[str] = []
        self._model_quality: float = 0.0
        self._cached_features: np.ndarray | None = None
        self._cached_epsilon: float = 0.0
        self._search_space: SearchSpace | None = None
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._warned_unbound_vars: set[str] = set()
        self._causal_graph: CausalGraph | None = causal_graph
        self._objective_name: str | None = objective_name
        self._obs_estimator: type | object | None = _NOT_TRIED
        self._experiment_log: ExperimentLog | None = None
        self._last_prediction: Prediction | None = None

    @property
    def last_prediction(self) -> Prediction | None:
        """The prediction cached by the most recent ``should_run_experiment()`` call."""
        return self._last_prediction

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
                # Use total experiment count for budget progress (not the
                # NaN-dropped row count used for the hull data).
                n_current = len(df)
                self._cached_epsilon = compute_epsilon(
                    self._cached_features, domain_bounds, n_current, self.n_max
                )
            else:
                self._cached_epsilon = 0.0
        else:
            self._numeric_var_names = []
            self._cached_features = None
            self._cached_epsilon = 0.0

        # Cache experiment log for observational prediction
        self._experiment_log = experiment_log

        # Check DoWhy availability and cache estimator class for
        # _observational_predict (which tries backdoor, frontdoor, IV per
        # variable — mirroring _analyze_variable in observational.py).
        if self._causal_graph is not None and self._obs_estimator is _NOT_TRIED:
            try:
                from causal_optimizer.estimator.observational import ObservationalEstimator

                # Store the *class* so _observational_predict can instantiate
                # per-method estimators (backdoor, frontdoor, IV).
                self._obs_estimator = ObservationalEstimator
            except ImportError:
                logger.debug("DoWhy not available; observational estimation disabled")
                self._obs_estimator = None
            except (AttributeError, TypeError) as exc:
                logger.debug("Failed to load ObservationalEstimator: %s", exc)
                self._obs_estimator = None

        if len(df) < self.min_history or not self._var_names:
            self._model = None
            return

        # _search_space is set at the top of fit(), but mypy needs narrowing
        search_space = self._search_space
        if search_space is None:  # pragma: no cover
            self._model = None
            return
        features = encode_dataframe_for_rf(df, self._var_names, search_space)
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
        if self._model is None or self._search_space is None:
            return None

        x = encode_params_for_rf(parameters, self._var_names, self._search_space)

        # Get predictions from individual trees for uncertainty
        tree_predictions = np.array([tree.predict(x)[0] for tree in self._model.estimators_])

        expected = float(np.mean(tree_predictions))
        uncertainty = float(np.std(tree_predictions))
        ci = (
            float(np.percentile(tree_predictions, 2.5)),
            float(np.percentile(tree_predictions, 97.5)),
        )

        rf_prediction = Prediction(
            expected_value=expected,
            uncertainty=uncertainty,
            confidence_interval=ci,
            model_quality=self._model_quality,
        )

        # Try to combine with observational estimate (skip if not tried or failed)
        if (
            self._obs_estimator is not None
            and self._obs_estimator is not _NOT_TRIED
            and self._experiment_log is not None
        ):
            obs_estimate = self._observational_predict(parameters)
            if obs_estimate is not None:
                return self._combine_predictions(rf_prediction, obs_estimate)

        return rf_prediction

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

        The resulting ``Prediction`` (if any) is cached as ``_last_prediction``
        so callers can reuse it without a redundant ``predict()`` call.
        """
        self._last_prediction = None

        if self.epsilon_mode:
            return self._should_run_epsilon(parameters)

        return self._should_run_heuristic(parameters)

    def _should_run_heuristic(self, parameters: dict[str, Any]) -> bool:
        """Original heuristic-based decision logic.

        Cheap guards (model availability, model quality) are checked *before*
        calling ``predict()`` to avoid expensive RF + observational inference
        when the result would be discarded anyway.
        """
        # Cheap guard: no model fitted yet
        if self._model is None:
            return True  # No model, must run

        # Cheap guard: model quality too low to trust
        if self._model_quality < 0.3:
            return True  # Model too poor to trust

        prediction = self.predict(parameters)
        self._last_prediction = prediction
        if prediction is None:
            return True  # predict() failed (e.g. search space not set)

        return prediction.uncertainty > self.uncertainty_threshold

    def _should_run_epsilon(self, parameters: dict[str, Any]) -> bool:
        """Epsilon controller decision logic from CBO (Aglietti et al.).

        Uses the epsilon value cached during the last ``fit()`` call to avoid
        recomputing the ConvexHull on every decision.

        Note:
            The epsilon controller is effectively dormant until model quality
            exceeds 0.3 (requires >= 10 experiments for cross-validation).
            During experiments 5--9 the model exists but has quality 0.0, so
            the model-quality guard always forces intervention.
        """
        epsilon = self._cached_epsilon
        if epsilon <= 0.0:
            return True  # No coverage data or zero epsilon, must run

        # With probability epsilon, observe (skip experiment); otherwise intervene
        if self._rng.random() < epsilon:
            # Check model availability and quality cheaply before running
            # RF inference — avoids redundant predict() calls when the model
            # is known to be unreliable (e.g., experiments 5–9 where
            # model_quality is 0.0 because cross-validation hasn't run yet).
            if self._model is None or self._model_quality < 0.3:
                logger.debug(
                    "Epsilon controller chose observe (epsilon=%.3f), but model "
                    "unavailable or quality too low (%.3f); intervening instead",
                    epsilon,
                    self._model_quality,
                )
                return True
            prediction = self.predict(parameters)
            self._last_prediction = prediction
            if prediction is None:
                logger.debug(
                    "Epsilon controller chose observe (epsilon=%.3f), but predict() "
                    "returned None; intervening instead",
                    epsilon,
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

    def _observational_predict(self, parameters: dict[str, Any]) -> ObservationalEstimate | None:
        """Predict outcome using observational causal estimation.

        Tries all causal-ancestor parameters with all identification methods
        (backdoor, frontdoor, IV) — mirroring ``_analyze_variable`` in
        ``observational.py`` — and selects the identified estimate with the
        tightest finite confidence interval.  Returns ``None`` when no
        ancestor yields an identified estimate with a finite CI.

        Gated by ``obs_min_history``: returns ``None`` when the experiment log
        has fewer than ``obs_min_history`` results, avoiding expensive DoWhy
        calls on insufficient data.
        """
        if (
            self._obs_estimator is None
            or self._obs_estimator is _NOT_TRIED
            or self._experiment_log is None
        ):
            return None

        if self._causal_graph is None:
            return None

        # Gate: skip expensive DoWhy calls when history is too short
        if len(self._experiment_log.results) < self.obs_min_history:
            return None

        # _obs_estimator stores the *class* (set during fit())
        estimator_cls = self._obs_estimator
        if not callable(estimator_cls):
            return None

        objective_name = self._objective_name or "objective"

        if objective_name in self._causal_graph.nodes:
            ancestors = self._causal_graph.ancestors(objective_name)
        else:
            return None

        # Collect all identified estimates, then pick the tightest CI
        from causal_optimizer.diagnostics.observational import IDENTIFICATION_METHODS

        candidates: list[ObservationalEstimate] = []

        for var_name, var_value in parameters.items():
            if var_name not in ancestors:
                continue
            if not isinstance(var_value, (int, float)):
                continue
            # Try each identification method (backdoor, frontdoor, IV)
            for method in IDENTIFICATION_METHODS:
                try:
                    # Suppress repeated statsmodels/pygraphviz warnings from DoWhy
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", module=r"statsmodels.*")
                        warnings.filterwarnings("ignore", module=r"pygraphviz.*")
                        est = estimator_cls(
                            causal_graph=self._causal_graph,
                            method=method,
                        )
                        result = est.estimate_intervention(
                            experiment_log=self._experiment_log,
                            treatment_var=var_name,
                            treatment_value=float(var_value),
                            objective_name=objective_name,
                        )
                    if result.identified:
                        candidates.append(result)
                except Exception:
                    logger.debug(
                        "Observational prediction (%s) failed for %s",
                        method,
                        var_name,
                        exc_info=True,
                    )
                    continue

        if not candidates:
            return None

        # Select the estimate with the smallest finite CI width
        def _ci_width(est: ObservationalEstimate) -> float:
            ci = est.confidence_interval
            width = ci[1] - ci[0]
            if not np.isfinite(width) or width <= 0:
                return float("inf")
            return width

        best = min(candidates, key=_ci_width)

        # Guard: if the best candidate still has a non-finite CI, fall back
        # to pure RF prediction rather than polluting _combine_predictions.
        if not np.isfinite(_ci_width(best)):
            return None

        return best

    def _combine_predictions(
        self, rf_pred: Prediction, obs_estimate: ObservationalEstimate
    ) -> Prediction:
        """Combine RF and observational predictions via heuristic ensemble.

        This is a heuristic combination, not a principled Bayesian fusion.
        The RF uncertainty (tree ensemble std) and observational CI (DoWhy
        linear regression SE) come from fundamentally different models, so
        the weighting and thresholds below are pragmatic approximations:

        - **Agreement**: RF and obs means within 1 RF std.  The combined
          mean is a precision-weighted average (inverse CI width), and the
          combined CI is the overlap of the two CIs clamped to contain the
          combined mean.
        - **Disagreement**: conservative union — the CI spans both estimates
          and uncertainty is inflated.

        Known limitations:
        - Inverse-CI-width weighting assumes comparable calibration across
          the two uncertainty models.
        - The ``/4.0`` and ``/2.0`` divisors are heuristic scale factors
          without formal justification.
        - When obs CI is much tighter than RF CI, the combined mean is
          dominated by the observational estimate.

        Despite these limitations, empirically the combination is safe:
        disagreement always *increases* uncertainty (triggering more
        experiments), and agreement only *modestly* tightens the CI.
        """
        rf_mean = rf_pred.expected_value
        obs_mean = obs_estimate.expected_outcome
        difference = abs(rf_mean - obs_mean)

        rf_ci_width = rf_pred.confidence_interval[1] - rf_pred.confidence_interval[0]
        obs_ci_width = obs_estimate.confidence_interval[1] - obs_estimate.confidence_interval[0]

        # Agreement threshold: predictions agree if they're within 1 RF std
        agree = difference <= rf_pred.uncertainty

        if agree:
            # Agree → weighted average, CI from overlap of the two CIs
            # Weight by inverse CI width (tighter CI gets more weight)
            rf_weight = 1.0 / max(rf_ci_width, 1e-10)
            obs_weight = 1.0 / max(obs_ci_width, 1e-10)
            total_weight = rf_weight + obs_weight
            combined_mean = (rf_mean * rf_weight + obs_mean * obs_weight) / total_weight
            # Use the overlap (intersection) of the two CIs, clamped to
            # always contain the weighted mean so the CI is never empty.
            inner_lo = max(rf_pred.confidence_interval[0], obs_estimate.confidence_interval[0])
            inner_hi = min(rf_pred.confidence_interval[1], obs_estimate.confidence_interval[1])
            if inner_lo <= inner_hi:
                combined_ci = (
                    min(inner_lo, combined_mean),
                    max(inner_hi, combined_mean),
                )
            else:
                # No overlap: span a CI that covers combined_mean and both CIs
                all_bounds = [
                    rf_pred.confidence_interval[0],
                    rf_pred.confidence_interval[1],
                    obs_estimate.confidence_interval[0],
                    obs_estimate.confidence_interval[1],
                    combined_mean,
                ]
                combined_ci = (min(all_bounds), max(all_bounds))
            combined_uncertainty = min(rf_pred.uncertainty, obs_ci_width / 4.0)
        else:
            # Disagree → conservative, wider CI
            combined_mean = (rf_mean + obs_mean) / 2.0
            # Expand CI to cover both predictions
            all_bounds = [
                rf_pred.confidence_interval[0],
                rf_pred.confidence_interval[1],
                obs_estimate.confidence_interval[0],
                obs_estimate.confidence_interval[1],
            ]
            combined_ci = (min(all_bounds), max(all_bounds))
            combined_uncertainty = max(rf_pred.uncertainty, difference, obs_ci_width / 2.0)

        return Prediction(
            expected_value=combined_mean,
            uncertainty=combined_uncertainty,
            confidence_interval=combined_ci,
            model_quality=rf_pred.model_quality,
        )

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
            if (
                var.lower is None or var.upper is None
            ) and var.name not in self._warned_unbound_vars:
                logger.warning(
                    "Variable '%s' has no bounds; defaulting to [0.0, 1.0] for "
                    "epsilon coverage estimate. Set explicit bounds for accurate "
                    "results.",
                    var.name,
                )
                self._warned_unbound_vars.add(var.name)
            lower = var.lower if var.lower is not None else 0.0
            upper = var.upper if var.upper is not None else 1.0
            bounds.append((lower, upper))

        return bounds if bounds else None
