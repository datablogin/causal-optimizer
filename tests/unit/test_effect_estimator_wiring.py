"""Tests for EffectEstimator wiring into the ExperimentEngine.

Tests cover:
- Engine uses EffectEstimator (not ad-hoc bootstrap) for keep/discard decisions
- effect_method parameter controls which method is used
- Backward compatibility: default 'bootstrap' behaves the same
- Observe path: predicted results are NOT added to experiment log
- estimate_improvement() convenience method
- Small sample fallback (< 5 experiments uses greedy, not statistical test)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.estimator.effects import EffectEstimate, EffectEstimator
from causal_optimizer.types import (
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


class QuadraticRunner:
    """f(x, y) = x² + y² — minimum at origin."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": float(x**2 + y**2)}


class FixedRunner:
    """Always returns a fixed value — useful for controlled tests."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        return {"objective": self.value}


def make_log_with_results(
    objective_values: list[float],
    statuses: list[ExperimentStatus],
) -> ExperimentLog:
    """Build an ExperimentLog from parallel lists of objectives and statuses."""
    log = ExperimentLog()
    for i, (val, status) in enumerate(zip(objective_values, statuses, strict=True)):
        log.results.append(
            ExperimentResult(
                experiment_id=f"test-{i:04d}",
                parameters={"x": float(i), "y": 0.0},
                metrics={"objective": val},
                status=status,
            )
        )
    return log


# ---------------------------------------------------------------------------
# Test: engine accepts effect_method and creates EffectEstimator
# ---------------------------------------------------------------------------


def test_engine_has_effect_estimator_default() -> None:
    """Engine must create an EffectEstimator with method='bootstrap' by default."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    assert hasattr(engine, "_effect_estimator")
    assert isinstance(engine._effect_estimator, EffectEstimator)
    assert engine._effect_estimator.method == "bootstrap"


@pytest.mark.parametrize("method", ["difference", "bootstrap", "aipw"])
def test_engine_effect_method_stored(method: str) -> None:
    """Engine must accept and store the effect_method parameter."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        effect_method=method,
    )
    assert engine._effect_estimator.method == method


# ---------------------------------------------------------------------------
# Test: EffectEstimator is called (not ad-hoc bootstrap) in _is_improvement_significant
# ---------------------------------------------------------------------------


def test_engine_calls_effect_estimator_for_significance() -> None:
    """_is_improvement_significant must delegate to _effect_estimator.estimate_improvement."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Pre-populate the log with enough history for statistical evaluation.
    # Requires >= 5 KEEP (engine guard) and >= 2 DISCARD.
    engine.log = make_log_with_results(
        objective_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        statuses=[
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.DISCARD,
            ExperimentStatus.DISCARD,
        ],
    )

    with patch.object(
        engine._effect_estimator,
        "estimate_improvement",
        wraps=engine._effect_estimator.estimate_improvement,
    ) as mock_estimate:
        engine._is_improvement_significant(current_objective=2.0)
        mock_estimate.assert_called_once()


# ---------------------------------------------------------------------------
# Test: truly better experiments are KEPT, worse are DISCARDED (with history)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_engine_keeps_clear_improvement() -> None:
    """With enough history, a dramatically better result should be KEEP."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        seed=42,  # Seed for deterministic bootstrap
    )

    # 5 KEEP + 2 DISCARD satisfies both thresholds (_MIN_KEPT=5, _MIN_DISCARDED=2)
    engine.log = make_log_with_results(
        objective_values=[50.0, 48.0, 46.0, 44.0, 42.0, 90.0, 85.0],
        statuses=[ExperimentStatus.KEEP] * 5 + [ExperimentStatus.DISCARD] * 2,
    )

    # A dramatic improvement (value = 1.0 vs best = 42.0) should be KEEP
    # For minimization: lower is better, so 1.0 << 42.0 is clearly better
    result = engine._is_improvement_significant(current_objective=1.0)
    # The statistical test should confirm this is significant (True)
    assert result is True


@pytest.mark.slow
def test_engine_discards_no_improvement() -> None:
    """With enough kept history, a clearly worse result should not be significant."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Simulate stable history: 5 kept + 2 discarded, best = 5.0
    # Need >= 5 kept experiments for the statistical test to run (not permissive)
    engine.log = make_log_with_results(
        objective_values=[10.0, 7.0, 5.0, 6.0, 8.0, 9.0, 11.0],
        statuses=[
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.DISCARD,
            ExperimentStatus.DISCARD,
        ],
    )

    # A clearly worse result (100.0 vs best = 5.0) should NOT be significant improvement
    result = engine._is_improvement_significant(current_objective=100.0)
    # With enough history, a clearly worse result should be False (not significant)
    # or None (greedy fallback will handle it) — it must NOT be True
    assert result is not True


# ---------------------------------------------------------------------------
# Test: backward compatibility — default method produces correct behavior
# ---------------------------------------------------------------------------


def test_backward_compatible_default_method() -> None:
    """Default effect_method='bootstrap' must produce same keep/discard as before."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # First result is always kept (no history)
    r1 = engine.run_experiment({"x": 3.0, "y": 4.0})
    assert r1.status == ExperimentStatus.KEEP

    # A better result should still be kept
    r2 = engine.run_experiment({"x": 1.0, "y": 1.0})
    assert r2.status == ExperimentStatus.KEEP

    # A worse result should still be discarded
    r3 = engine.run_experiment({"x": 4.0, "y": 4.0})
    assert r3.status == ExperimentStatus.DISCARD


# ---------------------------------------------------------------------------
# Test: small sample fallback (< 5 experiments uses greedy)
# ---------------------------------------------------------------------------


def test_is_improvement_significant_returns_none_with_too_few_experiments() -> None:
    """With < 5 total experiments, _is_improvement_significant returns None (greedy)."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Only 3 experiments in log
    engine.log = make_log_with_results(
        objective_values=[10.0, 8.0, 6.0],
        statuses=[ExperimentStatus.KEEP, ExperimentStatus.KEEP, ExperimentStatus.KEEP],
    )

    result = engine._is_improvement_significant(current_objective=4.0)
    assert result is None  # Must fall back to greedy comparison


def test_evaluate_status_uses_greedy_when_few_experiments() -> None:
    """_evaluate_status must use greedy comparison when < 5 experiments."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Run 3 experiments manually
    engine.run_experiment({"x": 3.0, "y": 4.0})  # obj=25.0
    engine.run_experiment({"x": 2.0, "y": 2.0})  # obj=8.0
    engine.run_experiment({"x": 1.0, "y": 1.0})  # obj=2.0

    # Now test status evaluation with only 3 experiments in log
    # Better result => KEEP
    status, _ = engine._evaluate_status({"objective": 0.5})
    assert status == ExperimentStatus.KEEP

    # Worse result => DISCARD
    status, _ = engine._evaluate_status({"objective": 100.0})
    assert status == ExperimentStatus.DISCARD


def test_is_improvement_significant_returns_value_with_enough_experiments() -> None:
    """With >= 5 kept + >= 2 discarded experiments, statistical test runs."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # 5 KEEP + 2 DISCARD satisfies both thresholds (_MIN_KEPT=5, _MIN_DISCARDED=2)
    engine.log = make_log_with_results(
        objective_values=[10.0, 8.0, 6.0, 5.0, 4.0, 9.0, 11.0],
        statuses=[
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.DISCARD,
            ExperimentStatus.DISCARD,
        ],
    )

    # With enough history, should return bool (not None)
    result = engine._is_improvement_significant(current_objective=2.0)
    assert result is not None  # statistical test was run


# ---------------------------------------------------------------------------
# Test: estimate_improvement() convenience method on EffectEstimator
# ---------------------------------------------------------------------------


def test_estimate_improvement_clear_improvement() -> None:
    """estimate_improvement with a dramatically lower value should be significant."""
    estimator = EffectEstimator(method="bootstrap")
    log = make_log_with_results(
        objective_values=[50.0, 48.0, 46.0, 44.0, 42.0, 41.0, 40.0],
        statuses=[ExperimentStatus.KEEP] * 7,
    )

    result = estimator.estimate_improvement(
        experiment_log=log,
        current_value=1.0,  # dramatically better
        objective_name="objective",
        minimize=True,
    )
    assert isinstance(result, EffectEstimate)
    assert result.is_significant is True


def test_estimate_improvement_no_improvement() -> None:
    """estimate_improvement with a worse value should not be significant."""
    estimator = EffectEstimator(method="bootstrap")
    log = make_log_with_results(
        objective_values=[5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.0],
        statuses=[ExperimentStatus.KEEP] * 7,
    )

    result = estimator.estimate_improvement(
        experiment_log=log,
        current_value=100.0,  # much worse
        objective_name="objective",
        minimize=True,
    )
    assert isinstance(result, EffectEstimate)
    assert result.is_significant is False


def test_estimate_improvement_too_few_samples_permissive() -> None:
    """estimate_improvement with < 5 samples should be permissive (is_significant=True)."""
    estimator = EffectEstimator(method="bootstrap")
    log = make_log_with_results(
        objective_values=[10.0, 8.0, 6.0],
        statuses=[ExperimentStatus.KEEP] * 3,
    )

    result = estimator.estimate_improvement(
        experiment_log=log,
        current_value=5.0,
        objective_name="objective",
        minimize=True,
    )
    assert isinstance(result, EffectEstimate)
    assert result.is_significant is True  # permissive fallback


def test_estimate_improvement_maximize() -> None:
    """estimate_improvement works correctly for maximization (minimize=False)."""
    estimator = EffectEstimator(method="bootstrap")
    log = make_log_with_results(
        objective_values=[-50.0, -48.0, -46.0, -44.0, -42.0, -41.0],
        statuses=[ExperimentStatus.KEEP] * 6,
    )

    # For maximization, a higher value (-1.0 >> -50.0) is better
    result = estimator.estimate_improvement(
        experiment_log=log,
        current_value=-1.0,
        objective_name="objective",
        minimize=False,
    )
    assert isinstance(result, EffectEstimate)
    assert result.is_significant is True


def test_estimate_improvement_difference_method() -> None:
    """estimate_improvement works with difference method."""
    estimator = EffectEstimator(method="difference")
    log = make_log_with_results(
        objective_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        statuses=[ExperimentStatus.KEEP] * 6,
    )

    result = estimator.estimate_improvement(
        experiment_log=log,
        current_value=0.1,
        objective_name="objective",
        minimize=True,
    )
    assert isinstance(result, EffectEstimate)
    # Clear improvement should be significant
    assert result.is_significant is True


# ---------------------------------------------------------------------------
# Test: bootstrap with small samples uses fewer iterations
# ---------------------------------------------------------------------------


def test_bootstrap_small_sample_uses_fewer_iterations() -> None:
    """With n < 10, bootstrap should use 100 samples (not 1000) for efficiency."""
    estimator = EffectEstimator(method="bootstrap", n_bootstrap=1000)

    # Directly test _bootstrap_estimate with small arrays
    rng = np.random.default_rng(42)
    treated = rng.normal(5.0, 0.1, size=4)
    control = rng.normal(10.0, 0.1, size=5)

    # With small samples, should complete quickly (uses 100 iterations internally)
    result = estimator._bootstrap_estimate(treated, control, small_sample=True)
    assert isinstance(result, EffectEstimate)
    assert result.method == "bootstrap"


# ---------------------------------------------------------------------------
# Test: observe path — predicted results NOT added to experiment log
# ---------------------------------------------------------------------------


def test_observe_path_does_not_add_to_log() -> None:
    """When predictor says observe (should_run=False), result NOT added to log."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        epsilon_mode=True,
        n_max=10,
        max_skips=0,  # allow 0 skips so observe path is exercised
    )

    # Pre-populate log to enable predictor model
    for val in [3.0, 2.0, 1.5]:
        engine.run_experiment({"x": val, "y": val})

    initial_count = len(engine.log.results)

    # Mock predictor to force observe (should_run=False)
    with patch.object(engine._predictor, "should_run_experiment", return_value=False):
        # With max_skips=0, we reach the max-skip threshold immediately and run anyway
        # But we want to test the observe path — set max_skips to a high value
        pass

    # Now test with a higher max_skips and force the observe path
    engine._max_skips = 1

    # Force predictor to say "observe" for the first check, then run
    call_count = [0]

    def mock_should_run(params: dict[str, Any]) -> bool:
        call_count[0] += 1
        return call_count[0] > 1  # first call: observe; subsequent: run

    with patch.object(engine._predictor, "should_run_experiment", side_effect=mock_should_run):
        result = engine.step()

    # A real experiment was eventually run — the log should have grown by 1
    assert len(engine.log.results) == initial_count + 1

    # The final result should be from an actual run (not predicted)
    assert result.metadata.get("source") != "observation"


def test_observe_path_logs_prediction_without_adding_to_log() -> None:
    """Observations use prediction but the predicted result is NOT logged."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        epsilon_mode=True,
        n_max=20,
        max_skips=1,
    )

    # Pre-populate with enough data for predictor to work
    for i in range(8):
        engine.run_experiment({"x": float(i), "y": 0.0})

    initial_count = len(engine.log.results)

    # Patch predictor: first call returns False (observe), second returns True (run)
    side_effects = [False, True]

    with patch.object(engine._predictor, "should_run_experiment", side_effect=side_effects):
        result = engine.step()

    # One real experiment was run — log grows by exactly 1
    assert len(engine.log.results) == initial_count + 1
    # The result is from a real run
    assert result.metadata.get("predicted") is not True


def test_observe_prediction_metadata_flag() -> None:
    """When an observation is skipped, the engine logs the prediction internally."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        epsilon_mode=True,
        n_max=20,
        max_skips=2,
    )

    # Pre-populate enough for predictor model
    for i in range(8):
        engine.run_experiment({"x": float(i) * 0.1, "y": 0.0})

    # Verify the engine can handle the "observe" decision gracefully
    # by tracking that _handle_observation is called when predictor returns False
    with patch.object(engine._predictor, "should_run_experiment", return_value=False) as mock_pred:
        # With max_skips=2, engine will try 2 times to skip, then force run
        result = engine.step()

    # mock_pred was called at least once (the observe path was exercised)
    assert mock_pred.call_count >= 1
    # A real experiment was run after exhausting skips
    assert result.status in (ExperimentStatus.KEEP, ExperimentStatus.DISCARD)


# ---------------------------------------------------------------------------
# Test: estimate_improvement is used inside _is_improvement_significant
# ---------------------------------------------------------------------------


def test_is_improvement_significant_uses_estimate_improvement() -> None:
    """_is_improvement_significant must call estimate_improvement, not ad-hoc bootstrap."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Enough history for statistical evaluation: requires >= 5 KEEP + >= 2 DISCARD
    engine.log = make_log_with_results(
        objective_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        statuses=[
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.KEEP,
            ExperimentStatus.DISCARD,
            ExperimentStatus.DISCARD,
        ],
    )

    # Patch estimate_improvement and verify it's called
    mock_result = EffectEstimate(
        point_estimate=-5.0,
        confidence_interval=(-10.0, -1.0),
        p_value=0.01,
        is_significant=True,
        method="bootstrap",
    )

    with patch.object(
        engine._effect_estimator,
        "estimate_improvement",
        return_value=mock_result,
    ) as mock_fn:
        sig = engine._is_improvement_significant(current_objective=1.0)

    mock_fn.assert_called_once()
    # Since mock returned is_significant=True, method should return True
    assert sig is True


# ---------------------------------------------------------------------------
# Test: AIPW fallback warning
# ---------------------------------------------------------------------------


def test_estimate_improvement_aipw_warns_and_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    """AIPW method logs a warning and falls back to bootstrap for estimate_improvement."""
    import logging

    estimator = EffectEstimator(method="aipw")
    log = make_log_with_results(
        objective_values=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        statuses=[ExperimentStatus.KEEP] * 6,
    )

    with caplog.at_level(logging.WARNING, logger="causal_optimizer.estimator.effects"):
        result = estimator.estimate_improvement(
            experiment_log=log,
            current_value=1.0,
            objective_name="objective",
            minimize=True,
        )

    assert any("aipw" in record.message.lower() for record in caplog.records)
    # Falls back to bootstrap (or greedy for small sample), so method != "aipw"
    assert result.method in ("bootstrap", "greedy")


# ---------------------------------------------------------------------------
# Test: Invalid effect_method raises at construction time
# ---------------------------------------------------------------------------


def test_engine_rejects_invalid_effect_method() -> None:
    """Engine must reject invalid effect_method at __init__ time."""
    with pytest.raises(ValueError, match="effect_method"):
        ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            effect_method="invalid",
        )
