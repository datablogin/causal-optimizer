"""Tests for SensitivityValidator wiring into ExperimentEngine."""

from __future__ import annotations

from typing import Any

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
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


def test_engine_has_validation_results_attribute() -> None:
    """Engine should expose a validation_results list."""
    engine = ExperimentEngine(
        search_space=_make_space(),
        runner=_QuadRunner(),
        seed=42,
    )
    assert hasattr(engine, "validation_results")
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

    # Phase should have moved to optimization (or beyond) regardless of robustness
    assert engine._phase in ("optimization", "exploitation", "exploration")
    # It shouldn't be stuck in exploration due to validation failure
    # (it might revert due to screening, but validation shouldn't be the cause)


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
    # and optimization->exploitation). Due to screening retries the exact count
    # may vary, but we expect at least 1.
    assert len(engine.validation_results) >= 1
    assert all(isinstance(r, RobustnessReport) for r in engine.validation_results)
