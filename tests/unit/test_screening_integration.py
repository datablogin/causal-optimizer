"""Tests for screening integration at phase transitions."""

import logging
from typing import Any

from causal_optimizer.designer.screening import ScreeningResult
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


class QuadraticRunner:
    """f(x, y, z) = x^2 + y^2 + 0.001*z (z is unimportant)."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        z = parameters.get("z", 0.0)
        return {"objective": x**2 + y**2 + 0.001 * z}


def test_screening_runs_at_exploration_to_optimization_transition():
    """Screening should run when transitioning from exploration to optimization."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    assert engine._screening_result is None
    assert engine._screened_focus_variables is None

    # Run 10 experiments to trigger transition
    engine.run_loop(n_experiments=10)

    # Screening should have run
    assert engine._screening_result is not None
    assert isinstance(engine._screening_result, ScreeningResult)


def test_screened_focus_variables_are_stored():
    """Screened focus variables should be stored on the engine."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    engine.run_loop(n_experiments=10)

    # Should have screening result with at least the main_effects dict populated
    assert engine._screening_result is not None
    assert isinstance(engine._screening_result.main_effects, dict)

    # If important variables were found, they should be stored
    if engine._screening_result.important_variables:
        assert engine._screened_focus_variables is not None
        assert len(engine._screened_focus_variables) > 0


def test_screening_with_insufficient_data_returns_empty():
    """Screening with very few results should return empty gracefully."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Manually trigger screening with no data
    engine._run_screening()

    assert engine._screening_result is not None
    # With no data, should return empty main_effects
    assert isinstance(engine._screening_result.main_effects, dict)
    # Phase should remain exploration (extended) since no important vars found
    assert engine._phase == "exploration"


def test_screening_results_are_logged(caplog):
    """Screening results should be logged."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    with caplog.at_level(logging.INFO, logger="causal_optimizer.engine.loop"):
        engine.run_loop(n_experiments=10)

    # Should log screening summary
    screening_logs = [r for r in caplog.records if "creening" in r.message]
    assert len(screening_logs) > 0


def test_screening_does_not_run_without_phase_transition():
    """Screening should not run if we stay in the same phase."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Run only 5 experiments — still in exploration
    engine.run_loop(n_experiments=5)
    assert engine._phase == "exploration"
    assert engine._screening_result is None


def test_no_important_vars_extends_exploration():
    """When screening finds no important variables, exploration should be extended."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    # Manually set log with insufficient variation to find important vars
    for i in range(10):
        result = ExperimentResult(
            experiment_id=f"test_{i}",
            parameters={"x": 0.0, "y": 0.0, "z": 0.0},
            metrics={"objective": 0.0},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        engine.log.results.append(result)

    # Trigger screening — all variables should have ~0 importance
    engine._run_screening()

    # Since no important variables found, phase should stay exploration
    assert engine._phase == "exploration"
    assert engine._screened_focus_variables is None
