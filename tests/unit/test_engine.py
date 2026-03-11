"""Tests for the experiment engine."""

from typing import Any
from unittest.mock import patch

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
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
    assert log.best_result() is not None


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


# --- POMIS integration tests ---


def test_engine_pomis_computed_at_optimization_transition():
    """POMIS is computed when transitioning to the optimization phase."""
    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[("x", "y")],  # confounder -> has_confounders=True
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    with patch("causal_optimizer.engine.loop.ExperimentEngine._compute_pomis") as mock_compute:
        # Run 10 experiments to trigger exploration -> optimization transition
        engine.run_loop(n_experiments=10)
        # _compute_pomis should have been called exactly once at the transition
        mock_compute.assert_called_once()


def test_engine_pomis_cached_not_recomputed():
    """POMIS sets are cached and not recomputed on every step."""
    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[("x", "y")],
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    mock_pomis = [frozenset({"x"}), frozenset({"y"})]
    call_count = 0

    def counting_compute() -> None:
        nonlocal call_count
        call_count += 1
        engine._pomis_sets = mock_pomis

    with patch.object(engine, "_compute_pomis", side_effect=counting_compute):
        engine.run_loop(n_experiments=15)

    # _compute_pomis is only called once at the exploration->optimization transition
    assert call_count == 1
    # The cached value should persist
    assert engine._pomis_sets == mock_pomis


def test_engine_pomis_not_computed_without_confounders():
    """Without confounders in the causal graph, POMIS is not computed."""
    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        # No bidirected_edges -> has_confounders=False
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    # Run 10 experiments to trigger transition
    engine.run_loop(n_experiments=10)

    # POMIS should not have been computed
    assert engine._pomis_sets is None


def test_engine_pomis_not_computed_without_graph():
    """Without a causal graph, POMIS is not computed."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )

    engine.run_loop(n_experiments=10)
    assert engine._pomis_sets is None


def test_engine_pomis_success_path():
    """POMIS success path stores computed sets."""
    import types

    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[("x", "y")],
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    expected = [frozenset({"x"}), frozenset({"y"})]
    fake_pomis = types.ModuleType("causal_optimizer.optimizer.pomis")
    fake_pomis.compute_pomis = lambda *a, **kw: expected  # type: ignore[attr-defined]

    import sys

    sys.modules["causal_optimizer.optimizer.pomis"] = fake_pomis
    try:
        engine.run_loop(n_experiments=10)
    finally:
        del sys.modules["causal_optimizer.optimizer.pomis"]

    assert engine._pomis_sets == expected


def test_engine_pomis_failure_sets_none():
    """If compute_pomis raises, _pomis_sets should be None."""
    import types

    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[("x", "y")],
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    # Create a fake pomis module that raises ValueError
    fake_pomis = types.ModuleType("causal_optimizer.optimizer.pomis")

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise ValueError("test error")

    fake_pomis.compute_pomis = _raise  # type: ignore[attr-defined]

    import sys

    sys.modules["causal_optimizer.optimizer.pomis"] = fake_pomis
    try:
        engine.run_loop(n_experiments=10)
    finally:
        del sys.modules["causal_optimizer.optimizer.pomis"]

    assert engine._pomis_sets is None


def test_engine_pomis_not_computed_when_screening_resets_phase():
    """POMIS should not be computed if screening resets phase to exploration."""
    graph = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[("x", "y")],
    )
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=graph,
    )

    def screening_resets_phase() -> None:
        """Simulate screening finding no important variables."""
        engine._phase = "exploration"
        engine._screened_focus_variables = None

    with (
        patch.object(engine, "_run_screening", side_effect=screening_resets_phase),
        patch.object(engine, "_compute_pomis") as mock_pomis,
    ):
        engine.run_loop(n_experiments=10)

    # _compute_pomis should NOT have been called because screening reverted phase
    mock_pomis.assert_not_called()


# --- Screening retry guard tests ---


def test_screening_max_retries_limits_reattempts():
    """Screening that finds no important variables should only retry up to max attempts."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        max_screening_attempts=3,
    )

    # Use the real _run_screening but patch ScreeningDesigner to return no important vars
    from causal_optimizer.designer.screening import ScreeningResult

    empty_result = ScreeningResult(
        main_effects={},
        important_variables=[],
        interactions={},
    )

    with (
        patch(
            "causal_optimizer.engine.loop.ScreeningDesigner.screen",
            return_value=empty_result,
        ),
        patch.object(engine, "_compute_pomis"),
    ):
        engine.run_loop(n_experiments=50)

    # The real _run_screening increments the counter; verify it hit the max
    assert engine._screening_attempts == engine._max_screening_attempts


def test_screening_max_retries_proceeds_to_optimization():
    """After max screening retries, engine proceeds to optimization with all variables."""
    from causal_optimizer.designer.screening import ScreeningResult

    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        max_screening_attempts=1,
    )

    empty_result = ScreeningResult(
        main_effects={},
        important_variables=[],
        interactions={},
    )

    with (
        patch(
            "causal_optimizer.engine.loop.ScreeningDesigner.screen",
            return_value=empty_result,
        ),
        patch.object(engine, "_compute_pomis"),
    ):
        engine.run_loop(n_experiments=20)

    # After max retries, focus variables should be set to all variables
    assert engine._screened_focus_variables == engine.search_space.variable_names
    # Phase should have progressed past exploration
    assert engine._phase in ("optimization", "exploitation")


def test_screening_increments_attempt_counter():
    """Each call to _run_screening increments the attempt counter."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
    )
    assert engine._screening_attempts == 0

    # Run 10 experiments to trigger the exploration->optimization transition
    engine.run_loop(n_experiments=10)

    # _run_screening should have been called, incrementing the counter
    assert engine._screening_attempts >= 1


# --- MAP-Elites crash guard tests ---


def test_crashed_experiment_not_added_to_archive():
    """Crashed experiments should not be added to the MAP-Elites archive."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=CrashingRunner(),
        descriptor_names=["objective"],
    )
    assert engine._archive is not None

    engine.run_experiment({"x": 1.0, "y": 1.0})
    assert len(engine.log.results) == 1
    assert engine.log.results[0].status == ExperimentStatus.CRASH
    # Archive should be empty — crashed experiment must not be added
    assert len(engine._archive.archive) == 0


def test_successful_experiment_added_to_archive():
    """Successful experiments should still be added to the MAP-Elites archive."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        descriptor_names=["objective"],
    )
    assert engine._archive is not None

    engine.run_experiment({"x": 1.0, "y": 2.0})
    assert len(engine.log.results) == 1
    assert engine.log.results[0].status == ExperimentStatus.KEEP
    # Archive should have the result
    assert len(engine._archive.archive) == 1


# --- Maximize engine tests ---


class MaximizeRunner:
    """Runner where higher values are better: f(x, y) = -(x^2 + y^2)."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": -(x**2 + y**2)}


def test_evaluate_status_maximize_keeps_better():
    """With minimize=False, a higher objective should be KEEP."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=MaximizeRunner(),
        minimize=False,
    )
    # First experiment: far from optimum (large negative value)
    r1 = engine.run_experiment({"x": 3.0, "y": 4.0})
    assert r1.status == ExperimentStatus.KEEP
    assert r1.metrics["objective"] == -25.0

    # Second experiment: closer to optimum (less negative = higher value)
    r2 = engine.run_experiment({"x": 1.0, "y": 1.0})
    assert r2.status == ExperimentStatus.KEEP
    assert r2.metrics["objective"] == -2.0

    # Third experiment: worse (more negative = lower value)
    r3 = engine.run_experiment({"x": 4.0, "y": 4.0})
    assert r3.status == ExperimentStatus.DISCARD

    # best_result should return the highest value (closest to 0)
    best = engine.log.best_result("objective", minimize=False)
    assert best is not None
    assert best.metrics["objective"] == -2.0
