"""Tests for the experiment engine."""

from typing import Any

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
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
        ]
    )


class QuadraticRunner:
    """Simple test runner: f(x, y) = x² + y² (minimum at origin)."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


class CrashingRunner:
    """Runner that always crashes."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        raise RuntimeError("Simulated crash")


def test_engine_run_experiment():
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    result = engine.run_experiment({"x": 1.0, "y": 2.0})
    assert result.metrics["objective"] == 5.0
    assert result.status == ExperimentStatus.KEEP  # first result is always kept
    assert len(engine.log.results) == 1


def test_engine_keep_discard():
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    # First experiment: far from optimum
    r1 = engine.run_experiment({"x": 3.0, "y": 4.0})
    assert r1.status == ExperimentStatus.KEEP

    # Second experiment: closer to optimum — should keep
    r2 = engine.run_experiment({"x": 1.0, "y": 1.0})
    assert r2.status == ExperimentStatus.KEEP

    # Third experiment: worse — should discard
    r3 = engine.run_experiment({"x": 4.0, "y": 4.0})
    assert r3.status == ExperimentStatus.DISCARD


def test_engine_crash_handling():
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=CrashingRunner(),
    )
    result = engine.run_experiment({"x": 1.0, "y": 1.0})
    assert result.status == ExperimentStatus.CRASH


def test_engine_run_loop():
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    log = engine.run_loop(n_experiments=5)
    assert len(log.results) == 5
    assert log.best_result is not None


def test_engine_phase_transitions():
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    assert engine._phase == "exploration"

    # Run 10 experiments to transition
    engine.run_loop(n_experiments=10)
    assert engine._phase == "optimization"

    # Run 40 more to transition to exploitation
    engine.run_loop(n_experiments=40)
    assert engine._phase == "exploitation"
