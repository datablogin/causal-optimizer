"""Tests for off-policy predictor integration into the engine loop."""

from typing import Any
from unittest.mock import MagicMock

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.predictor.off_policy import OffPolicyPredictor
from causal_optimizer.types import (
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


class TestPredictorInitialization:
    """Test that the predictor is properly initialized."""

    def test_predictor_created_on_init(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        assert isinstance(engine._predictor, OffPolicyPredictor)

    def test_max_skips_default(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        assert engine._max_skips == 3

    def test_max_skips_custom(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            max_skips=5,
        )
        assert engine._max_skips == 5


class TestPredictorFitting:
    """Test that the predictor is fitted after each experiment."""

    def test_predictor_fit_called_after_run_experiment(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        engine._predictor.should_run_experiment.return_value = True

        engine.run_experiment({"x": 1.0, "y": 2.0})

        engine._predictor.fit.assert_called_once_with(engine.log, engine.search_space, "objective")

    def test_predictor_fit_called_multiple_times(self):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        engine._predictor.should_run_experiment.return_value = True

        engine.run_experiment({"x": 1.0, "y": 2.0})
        engine.run_experiment({"x": 2.0, "y": 3.0})
        engine.run_experiment({"x": 0.5, "y": 0.5})

        assert engine._predictor.fit.call_count == 3


class TestPredictorSkipping:
    """Test that the predictor can recommend skipping experiments."""

    def test_predictor_allows_experiment(self):
        """When predictor says run, experiment runs normally."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        engine._predictor.should_run_experiment.return_value = True
        engine._predictor.fit.return_value = None

        result = engine.step()
        assert result is not None
        assert len(engine.log.results) == 1

    def test_predictor_skips_then_runs(self):
        """When predictor skips once then allows, second suggestion runs."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        # First call: skip. Second call: run.
        engine._predictor.should_run_experiment.side_effect = [False, True]
        engine._predictor.fit.return_value = None

        result = engine.step()
        assert result is not None
        assert len(engine.log.results) == 1
        # should_run_experiment called twice (once skipped, once allowed)
        assert engine._predictor.should_run_experiment.call_count == 2

    def test_max_skips_prevents_infinite_loop(self):
        """When predictor always says skip, max_skips limits the loop."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            max_skips=3,
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        # Always skip — but max_skips should break the loop
        engine._predictor.should_run_experiment.return_value = False
        engine._predictor.fit.return_value = None

        result = engine.step()
        assert result is not None
        assert len(engine.log.results) == 1
        # should_run_experiment called max_skips times, then we run anyway
        # (3 skips + the final call is not checked because we break)
        assert engine._predictor.should_run_experiment.call_count == 3

    def test_max_skips_zero_always_runs(self):
        """With max_skips=0, experiments always run regardless of predictor."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            max_skips=0,
        )
        engine._predictor = MagicMock(spec=OffPolicyPredictor)
        engine._predictor.should_run_experiment.return_value = False
        engine._predictor.fit.return_value = None

        result = engine.step()
        assert result is not None
        # Predictor not even consulted since max_skips=0
        assert engine._predictor.should_run_experiment.call_count == 0


class TestEarlyExperimentsAlwaysRun:
    """Before min_history, the predictor has no model and always says run."""

    def test_early_experiments_always_run(self):
        """With < min_history experiments, predictor returns True (run)."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        # The real predictor with min_history=5 should always say run
        # when we have fewer than 5 experiments
        for _ in range(4):
            result = engine.step()
            assert result is not None

        assert len(engine.log.results) == 4

    def test_predictor_model_is_none_early(self):
        """Before enough data, the predictor model should be None."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        # Initially, no model fitted
        assert engine._predictor._model is None

        # Run 2 experiments (below min_history of 5)
        engine.run_experiment({"x": 1.0, "y": 2.0})
        engine.run_experiment({"x": 2.0, "y": 3.0})
        assert engine._predictor._model is None


class TestIntegrationStepLoop:
    """Integration tests for the full step loop with predictor."""

    def test_full_loop_with_predictor(self):
        """Run a full loop and verify predictor doesn't break anything."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        log = engine.run_loop(n_experiments=8)
        assert len(log.results) == 8
        assert log.best_result is not None
