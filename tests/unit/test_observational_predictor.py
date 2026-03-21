"""Tests for observational-enhanced off-policy prediction."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np

from causal_optimizer.predictor.off_policy import OffPolicyPredictor
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name=f"x{i}", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)
            for i in range(3)
        ]
    )


def _result(params: dict, objective: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=str(uuid.uuid4()),
        parameters=params,
        metrics={"objective": objective},
        status=ExperimentStatus.KEEP,
    )


def _make_log(n: int = 15) -> ExperimentLog:
    rng = np.random.default_rng(42)
    log = ExperimentLog()
    for _ in range(n):
        params = {f"x{i}": float(rng.uniform(0, 10)) for i in range(3)}
        obj = params["x0"] * 0.5 + rng.normal(0, 0.1)
        log.results.append(_result(params, obj))
    return log


def _causal_graph() -> CausalGraph:
    return CausalGraph(
        edges=[("x0", "objective"), ("x1", "objective")],
        nodes=["x0", "x1", "x2", "objective"],
    )


# ---------------------------------------------------------------------------
# Test: no graph — pure RF behavior
# ---------------------------------------------------------------------------


class TestNoGraph:
    def test_no_graph_uses_pure_rf(self) -> None:
        """Without causal graph, predictor should work as before (pure RF)."""
        predictor = OffPolicyPredictor(min_history=5)
        log = _make_log(15)
        ss = _search_space()

        predictor.fit(log, ss, "objective")

        pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert pred is not None
        assert isinstance(pred.expected_value, float)
        assert isinstance(pred.uncertainty, float)

    def test_should_run_without_graph(self) -> None:
        """should_run_experiment works without causal graph."""
        predictor = OffPolicyPredictor(min_history=5)
        log = _make_log(15)
        ss = _search_space()

        predictor.fit(log, ss, "objective")

        result = predictor.should_run_experiment({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test: observational agree → tighter CI
# ---------------------------------------------------------------------------


class TestObsAgree:
    def test_agreement_tightens_confidence_interval(self) -> None:
        """When RF and obs estimates agree, combined CI should be tighter or equal."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = _make_log(15)
        ss = _search_space()
        predictor.fit(log, ss, "objective")

        # Get RF-only prediction first
        rf_pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert rf_pred is not None

        # Mock observational estimate that agrees (same mean, tighter CI)
        mock_obs = MagicMock()
        mock_obs.expected_outcome = rf_pred.expected_value
        mock_obs.confidence_interval = (
            rf_pred.expected_value - 0.5,
            rf_pred.expected_value + 0.5,
        )
        mock_obs.identified = True

        combined = predictor._combine_predictions(rf_pred, mock_obs)

        # Verify combined CI is non-degenerate
        assert combined is not None
        combined_width = combined.confidence_interval[1] - combined.confidence_interval[0]
        rf_width = rf_pred.confidence_interval[1] - rf_pred.confidence_interval[0]
        assert combined_width >= 0
        # When obs CI is tighter, combined should not exceed RF width
        assert combined_width <= rf_width + 1e-10
        # Expected value must be contained in the CI
        assert combined.confidence_interval[0] <= combined.expected_value
        assert combined.expected_value <= combined.confidence_interval[1]


# ---------------------------------------------------------------------------
# Test: observational disagree → wider CI
# ---------------------------------------------------------------------------


class TestObsDisagree:
    def test_disagreement_widens_confidence_interval(self) -> None:
        """When RF and obs estimates disagree, combined CI should be wider."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = _make_log(15)
        ss = _search_space()
        predictor.fit(log, ss, "objective")

        rf_pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert rf_pred is not None

        # Mock observational estimate that strongly disagrees
        mock_obs = MagicMock()
        mock_obs.expected_outcome = rf_pred.expected_value + 100.0
        mock_obs.confidence_interval = (
            rf_pred.expected_value + 95.0,
            rf_pred.expected_value + 105.0,
        )
        mock_obs.identified = True

        with patch.object(predictor, "_observational_predict", return_value=mock_obs):
            combined = predictor._combine_predictions(rf_pred, mock_obs)

        assert combined is not None
        rf_width = rf_pred.confidence_interval[1] - rf_pred.confidence_interval[0]
        combined_width = combined.confidence_interval[1] - combined.confidence_interval[0]
        assert combined_width > rf_width


# ---------------------------------------------------------------------------
# Test: DoWhy not installed
# ---------------------------------------------------------------------------


class TestDoWhyNotInstalled:
    def test_works_without_dowhy(self) -> None:
        """When DoWhy is not installed, predictor falls back to pure RF."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = _make_log(15)
        ss = _search_space()

        with patch(
            "causal_optimizer.estimator.observational.ObservationalEstimator",
            side_effect=ImportError("no dowhy"),
        ):
            predictor.fit(log, ss, "objective")

        # The obs estimator should be None, but RF should still work
        pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert pred is not None


# ---------------------------------------------------------------------------
# Test: estimation failure
# ---------------------------------------------------------------------------


class TestEstimationFailure:
    def test_obs_failure_falls_back_to_rf(self) -> None:
        """If observational estimation fails, should fall back to RF prediction."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = _make_log(15)
        ss = _search_space()
        predictor.fit(log, ss, "objective")

        with patch.object(predictor, "_observational_predict", return_value=None):
            pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})

        assert pred is not None


# ---------------------------------------------------------------------------
# Test: should_run_experiment behavior
# ---------------------------------------------------------------------------


class TestShouldRunExperiment:
    def test_should_run_with_obs_integration(self) -> None:
        """should_run_experiment should still work with observational integration."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = _make_log(15)
        ss = _search_space()
        predictor.fit(log, ss, "objective")

        result = predictor.should_run_experiment({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test: edge cases for _combine_predictions and overall predictor
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_predict_before_fit(self) -> None:
        """predict() returns None before fit() is called."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert pred is None

    def test_empty_experiment_log(self) -> None:
        """Predictor handles empty experiment log gracefully."""
        predictor = OffPolicyPredictor(
            min_history=5,
            causal_graph=_causal_graph(),
            objective_name="objective",
        )
        log = ExperimentLog()
        ss = _search_space()
        predictor.fit(log, ss, "objective")

        pred = predictor.predict({"x0": 5.0, "x1": 3.0, "x2": 1.0})
        assert pred is None

    def test_combine_disagree_expected_value_in_ci(self) -> None:
        """In disagree case, expected_value must be within combined CI."""
        from causal_optimizer.predictor.off_policy import Prediction

        predictor = OffPolicyPredictor(min_history=5)

        rf = Prediction(
            expected_value=5.0,
            uncertainty=0.5,
            confidence_interval=(4.0, 6.0),
            model_quality=0.8,
        )
        obs = MagicMock()
        obs.expected_outcome = 50.0  # very different
        obs.confidence_interval = (45.0, 55.0)

        combined = predictor._combine_predictions(rf, obs)
        assert combined.confidence_interval[0] <= combined.expected_value
        assert combined.expected_value <= combined.confidence_interval[1]
        # Disagree: CI should be wider than RF alone
        rf_width = rf.confidence_interval[1] - rf.confidence_interval[0]
        combined_width = combined.confidence_interval[1] - combined.confidence_interval[0]
        assert combined_width > rf_width

    def test_combine_no_overlap_expected_value_in_ci(self) -> None:
        """When CIs don't overlap but agree, expected_value must be in CI."""
        from causal_optimizer.predictor.off_policy import Prediction

        predictor = OffPolicyPredictor(min_history=5)

        # Two tight CIs that agree (within uncertainty) but don't overlap
        rf = Prediction(
            expected_value=5.0,
            uncertainty=0.5,
            confidence_interval=(4.95, 5.05),
            model_quality=0.8,
        )
        obs = MagicMock()
        obs.expected_outcome = 4.8
        obs.confidence_interval = (4.75, 4.85)

        combined = predictor._combine_predictions(rf, obs)
        assert combined.confidence_interval[0] <= combined.expected_value
        assert combined.expected_value <= combined.confidence_interval[1]

    def test_combine_weighted_mean_correctness(self) -> None:
        """Verify weighted mean calculation in _combine_predictions."""
        from causal_optimizer.predictor.off_policy import Prediction

        predictor = OffPolicyPredictor(min_history=5)

        rf = Prediction(
            expected_value=10.0,
            uncertainty=1.0,
            confidence_interval=(8.0, 12.0),
            model_quality=0.8,
        )
        obs = MagicMock()
        obs.expected_outcome = 10.0
        obs.confidence_interval = (9.5, 10.5)  # tighter -> higher weight

        combined = predictor._combine_predictions(rf, obs)
        # With same mean, combined should be close to 10.0
        assert abs(combined.expected_value - 10.0) < 0.1
        # CI must contain the expected value
        assert combined.confidence_interval[0] <= combined.expected_value
        assert combined.expected_value <= combined.confidence_interval[1]
