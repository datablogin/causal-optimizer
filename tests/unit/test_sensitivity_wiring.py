"""Tests for SensitivityValidator wiring into ExperimentEngine."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    SearchSpace,
    Variable,
    VariableType,
)
from causal_optimizer.validator.sensitivity import RobustnessReport


def _make_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


class _QuadRunner:
    """f(x,y) = x^2 + y^2, minimization."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


def test_engine_has_validation_records() -> None:
    """Engine should expose validation_records and a validation_results property."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    assert isinstance(engine.validation_records, list)
    assert len(engine.validation_records) == 0
    # Property should also work
    assert isinstance(engine.validation_results, list)
    assert len(engine.validation_results) == 0


def test_validation_results_contains_robustness_reports() -> None:
    """After enough experiments to trigger a phase transition, validation_results
    should contain RobustnessReport instances."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    # Run 11 experiments to trigger exploration -> optimization transition
    for _ in range(11):
        engine.step()

    # At least one validation report should have been generated at the transition
    assert len(engine.validation_results) >= 1
    assert all(isinstance(r, RobustnessReport) for r in engine.validation_results)


def test_validation_does_not_block_phase_transition() -> None:
    """Even if is_robust=False, the engine should still transition phases."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    # Run enough experiments to get past exploration
    for _ in range(11):
        engine.step()

    # After 11 steps, phase must have reached optimization or beyond at least once.
    # Screening may have reverted it, but validation must never be the cause.
    phases_seen = {r.metadata.get("phase") for r in engine.log.results}
    assert "optimization" in phases_seen or "exploitation" in phases_seen, (
        "Phase never advanced past exploration — validation may have incorrectly blocked it"
    )


def test_validation_at_exploitation_transition() -> None:
    """A RobustnessReport should also be generated at the optimization->exploitation
    phase transition."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    # Run 51 experiments to trigger both transitions
    for _ in range(51):
        engine.step()

    # Should have reports from both transitions (exploration->optimization
    # and optimization->exploitation). Screening retries may add extra
    # exploration->optimization transitions, but there must be at least 2.
    assert len(engine.validation_results) >= 2
    assert all(isinstance(r, RobustnessReport) for r in engine.validation_results)


def test_validation_skipped_with_few_results() -> None:
    """Validation should be skipped (no report) if fewer than 4 experiments."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    # Manually call _run_validation with a tiny log (< 4 results)
    engine.run_experiment({"x": 1.0, "y": 1.0})
    engine.run_experiment({"x": 0.5, "y": 0.5})
    engine._run_validation("exploration", "optimization")
    assert len(engine.validation_results) == 0


def test_validation_graceful_on_error() -> None:
    """If SensitivityValidator raises, validation is skipped gracefully."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    # Add enough experiments for validation
    for _ in range(11):
        engine.run_experiment({"x": 1.0, "y": 1.0})

    with patch.object(engine._validator, "validate_improvement", side_effect=RuntimeError("boom")):
        engine._run_validation("exploration", "optimization")

    # Should have no report since the validator raised
    assert len(engine.validation_results) == 0


def test_validation_wrong_direction_treated_as_non_robust() -> None:
    """If the effect direction is wrong for the optimization goal, is_robust is overridden."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        minimize=True,
        seed=42,
    )
    # Add enough experiments for validation
    for _ in range(11):
        engine.run_experiment({"x": 1.0, "y": 1.0})

    # Mock a report where is_robust=True but effect_size > 0 (wrong for minimize)
    mock_report = RobustnessReport(
        effect_size=1.5,  # Positive = regression for minimization
        noise_estimate=0.5,
        signal_to_noise=3.0,
        e_value=5.0,
        is_robust=True,
        summary="Robust improvement",
    )
    with patch.object(engine._validator, "validate_improvement", return_value=mock_report):
        engine._run_validation("exploration", "optimization")

    assert len(engine.validation_results) == 1
    report = engine.validation_results[0]
    # Should be overridden to non-robust because direction is wrong
    assert report.is_robust is False
    assert "Effect direction is wrong" in report.summary
    assert "originally:" in report.summary


def test_validation_wrong_direction_maximize() -> None:
    """Direction override also works for maximize: negative effect is wrong."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        minimize=False,
        seed=42,
    )
    for _ in range(11):
        engine.run_experiment({"x": 1.0, "y": 1.0})

    # effect_size < 0 is wrong for maximization
    mock_report = RobustnessReport(
        effect_size=-2.0,
        noise_estimate=0.5,
        signal_to_noise=4.0,
        e_value=6.0,
        is_robust=True,
        summary="Robust improvement",
    )
    with patch.object(engine._validator, "validate_improvement", return_value=mock_report):
        engine._run_validation("exploration", "optimization")

    assert len(engine.validation_results) == 1
    report = engine.validation_results[0]
    assert report.is_robust is False
    assert "Effect direction is wrong" in report.summary
