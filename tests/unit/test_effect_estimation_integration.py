"""Tests for bootstrap-based statistical evaluation in keep/discard decisions."""

from typing import Any

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
        ]
    )


class QuadraticRunner:
    """f(x, y) = x^2 + y^2, minimum at origin."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


class FixedRunner:
    """Runner that returns a pre-set sequence of objectives."""

    def __init__(self, objectives: list[float]) -> None:
        self._objectives = list(objectives)
        self._index = 0

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        val = self._objectives[self._index % len(self._objectives)]
        self._index += 1
        return {"objective": val}


def _seed_engine_with_history(
    engine: ExperimentEngine,
    kept_values: list[float],
    discarded_values: list[float],
) -> None:
    """Add synthetic history to the engine log without running experiments."""
    for i, val in enumerate(kept_values):
        result = ExperimentResult(
            experiment_id=f"kept-{i}",
            parameters={"x": float(i), "y": 0.0},
            metrics={"objective": val},
            status=ExperimentStatus.KEEP,
            metadata={"phase": "exploration"},
        )
        engine.log.results.append(result)

    for i, val in enumerate(discarded_values):
        result = ExperimentResult(
            experiment_id=f"disc-{i}",
            parameters={"x": float(i + 10), "y": 0.0},
            metrics={"objective": val},
            status=ExperimentStatus.DISCARD,
            metadata={"phase": "exploration"},
        )
        engine.log.results.append(result)


class TestGreedyFallback:
    """With too few experiments, greedy behavior should be preserved."""

    def test_first_experiment_always_kept(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        result = engine.run_experiment({"x": 1.0, "y": 2.0})
        assert result.status == ExperimentStatus.KEEP

    def test_greedy_with_few_experiments(self):
        """With < 5 experiments, _is_improvement_significant returns None (greedy)."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        # Run 3 experiments — not enough for statistical evaluation
        r1 = engine.run_experiment({"x": 3.0, "y": 4.0})  # 25
        r2 = engine.run_experiment({"x": 1.0, "y": 1.0})  # 2 — better
        r3 = engine.run_experiment({"x": 4.0, "y": 4.0})  # 32 — worse

        assert r1.status == ExperimentStatus.KEEP
        assert r2.status == ExperimentStatus.KEEP
        assert r3.status == ExperimentStatus.DISCARD

    def test_insufficient_kept_or_discarded(self):
        """Even with >= 5 experiments, need at least 2 KEEP and 2 DISCARD."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        # All improving so all KEEP — no DISCARD examples
        vals = [50.0, 40.0, 30.0, 20.0, 10.0]
        runner = FixedRunner(vals)
        engine.runner = runner

        for _ in range(5):
            engine.run_experiment({"x": 0.0, "y": 0.0})

        # _is_improvement_significant should return None
        result = engine._is_improvement_significant(5.0)
        assert result is None

    def test_is_improvement_significant_returns_none_with_few_data(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        # Only 2 experiments — nowhere near enough
        engine.run_experiment({"x": 3.0, "y": 0.0})
        engine.run_experiment({"x": 4.0, "y": 0.0})

        result = engine._is_improvement_significant(1.0)
        assert result is None


class TestStatisticalEvaluation:
    """With enough history, statistical evaluation should be used."""

    def test_statistical_path_used_with_enough_history(self):
        """With >= 5 KEEP and >= 2 DISCARD, statistical check fires."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )

        # Seed history: 5 KEEP, 3 DISCARD (satisfies _MIN_KEPT=5, _MIN_DISCARDED=2)
        _seed_engine_with_history(
            engine,
            kept_values=[1.0, 2.0, 3.0, 4.0, 5.0],
            discarded_values=[10.0, 15.0, 20.0],
        )

        # _is_improvement_significant should return a bool, not None
        result = engine._is_improvement_significant(0.1)
        assert result is not None

    def test_clear_improvement_is_kept(self):
        """A clear improvement (much better than best) should be kept."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=True,
        )

        # Seed with kept values around 5-10 and discarded around 15-20
        _seed_engine_with_history(
            engine,
            kept_values=[5.0, 6.0, 7.0],
            discarded_values=[15.0, 18.0, 20.0],
        )

        # A value of 0.5 is clearly better than best (5.0)
        metrics = {"objective": 0.5}
        status = engine._evaluate_status(metrics)
        assert status == ExperimentStatus.KEEP

    def test_noise_level_change_discarded(self):
        """A change within noise level should be discarded."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=True,
        )

        # Seed with very tight kept values and some discarded
        _seed_engine_with_history(
            engine,
            kept_values=[5.0, 5.0, 5.0],
            discarded_values=[10.0, 12.0, 15.0],
        )

        # A value of 12.0 is worse than best (5.0) — should be discarded
        metrics = {"objective": 12.0}
        status = engine._evaluate_status(metrics)
        assert status == ExperimentStatus.DISCARD

    def test_maximization_clear_improvement(self):
        """For maximization, higher values should be kept."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=False,
        )

        _seed_engine_with_history(
            engine,
            kept_values=[10.0, 12.0, 15.0],
            discarded_values=[2.0, 3.0, 5.0],
        )

        # 50.0 is clearly better for maximization
        metrics = {"objective": 50.0}
        status = engine._evaluate_status(metrics)
        assert status == ExperimentStatus.KEEP

    def test_maximization_worse_value_discarded(self):
        """For maximization, lower values should be discarded."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=False,
        )

        _seed_engine_with_history(
            engine,
            kept_values=[10.0, 12.0, 15.0],
            discarded_values=[2.0, 3.0, 5.0],
        )

        # 1.0 is clearly worse for maximization
        metrics = {"objective": 1.0}
        status = engine._evaluate_status(metrics)
        assert status == ExperimentStatus.DISCARD

    def test_adaptive_alpha_early(self):
        """With < 20 experiments but enough kept/discarded, statistical test runs."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=True,
        )

        # 5 KEEP + 3 DISCARD = 8 total (< 20) — satisfies statistical thresholds
        _seed_engine_with_history(
            engine,
            kept_values=[5.0, 6.0, 7.0, 8.0, 9.0],
            discarded_values=[15.0, 18.0, 20.0],
        )

        assert len(engine.log.results) < 20
        # The method should work without errors and return a bool
        result = engine._is_improvement_significant(0.1)
        assert isinstance(result, bool)

    def test_adaptive_alpha_late(self):
        """With >= 20 experiments, alpha=0.05 (stricter)."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            minimize=True,
        )

        # Create 20+ experiments
        _seed_engine_with_history(
            engine,
            kept_values=[5.0 + i * 0.1 for i in range(10)],
            discarded_values=[15.0 + i * 0.5 for i in range(12)],
        )

        assert len(engine.log.results) >= 20
        result = engine._is_improvement_significant(0.1)
        assert isinstance(result, bool)

    def test_crash_still_returned_for_missing_objective(self):
        """Missing objective should still return CRASH regardless of history."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )

        _seed_engine_with_history(
            engine,
            kept_values=[5.0, 6.0, 7.0],
            discarded_values=[15.0, 18.0, 20.0],
        )

        status = engine._evaluate_status({"other_metric": 1.0})
        assert status == ExperimentStatus.CRASH
