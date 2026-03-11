"""Tests for the epsilon controller (observation-intervention tradeoff)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.predictor.epsilon import compute_epsilon, should_observe
from causal_optimizer.predictor.off_policy import OffPolicyPredictor
from causal_optimizer.types import SearchSpace, Variable, VariableType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
    )


class QuadraticRunner:
    """f(x, y) = x^2 + y^2, minimum at origin."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


# ---------------------------------------------------------------------------
# compute_epsilon tests
# ---------------------------------------------------------------------------


class TestComputeEpsilon:
    """Test compute_epsilon with known data."""

    def test_full_coverage_unit_square(self) -> None:
        """2D unit square fully covered should give epsilon close to 1.0."""
        # Corners of the unit square — convex hull volume = 1.0
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        # coverage_ratio = 1.0, rescale = 50/100 = 0.5
        # epsilon = 1.0 / 0.5 = 2.0, clamped to 1.0
        eps = compute_epsilon(data, bounds, n_current=50, n_max=100)
        assert eps == pytest.approx(1.0)

    def test_sparse_data_large_domain(self) -> None:
        """Sparse data in a large domain should give epsilon close to 0.0."""
        # Tiny triangle in a huge domain
        data = np.array(
            [
                [0.0, 0.0],
                [0.01, 0.0],
                [0.0, 0.01],
            ]
        )
        bounds = [(0.0, 100.0), (0.0, 100.0)]
        eps = compute_epsilon(data, bounds, n_current=3, n_max=100)
        assert eps < 0.01

    def test_empty_data(self) -> None:
        """Empty data should return 0.0."""
        data = np.empty((0, 2))
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=0, n_max=100)
        assert eps == 0.0

    def test_single_point(self) -> None:
        """One point is not enough for a convex hull."""
        data = np.array([[0.5, 0.5]])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=1, n_max=100)
        assert eps == 0.0

    def test_two_points(self) -> None:
        """Two points are not enough for a convex hull in 2D."""
        data = np.array([[0.0, 0.0], [1.0, 1.0]])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=2, n_max=100)
        assert eps == 0.0

    def test_collinear_points(self) -> None:
        """Collinear points produce a degenerate hull."""
        data = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        )
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=3, n_max=100)
        # Collinear points either raise QhullError or give zero-volume hull
        assert eps == 0.0

    def test_n_current_zero(self) -> None:
        """n_current=0 should return 0.0."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=0, n_max=100)
        assert eps == 0.0

    def test_n_max_zero(self) -> None:
        """n_max=0 should return 0.0."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=10, n_max=0)
        assert eps == 0.0

    def test_partial_coverage(self) -> None:
        """Partial coverage of the domain should give intermediate epsilon."""
        # Triangle covering half the unit square
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        # Triangle area = 0.5, domain area = 1.0
        # coverage = 0.5, rescale = 50/100 = 0.5
        # epsilon = 0.5 / 0.5 = 1.0
        eps = compute_epsilon(data, bounds, n_current=50, n_max=100)
        assert eps == pytest.approx(1.0)

    def test_epsilon_increases_with_coverage(self) -> None:
        """Epsilon should increase as more of the domain is covered."""
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        # Small triangle
        small = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        # Larger triangle
        large = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])

        eps_small = compute_epsilon(small, bounds, n_current=10, n_max=100)
        eps_large = compute_epsilon(large, bounds, n_current=10, n_max=100)
        assert eps_large > eps_small

    def test_zero_width_bound(self) -> None:
        """A zero-width bound should return 0.0."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        bounds = [(0.0, 0.0), (0.0, 1.0)]
        eps = compute_epsilon(data, bounds, n_current=3, n_max=100)
        assert eps == 0.0

    def test_mismatched_dims(self) -> None:
        """Mismatched dimensions between data and bounds should return 0.0."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        bounds = [(0.0, 1.0)]  # Only 1 bound for 2D data
        eps = compute_epsilon(data, bounds, n_current=3, n_max=100)
        assert eps == 0.0


# ---------------------------------------------------------------------------
# should_observe tests
# ---------------------------------------------------------------------------


class TestShouldObserve:
    """Test should_observe with fixed seed for determinism."""

    def test_deterministic_with_seed(self) -> None:
        """should_observe should be reproducible with fixed seed."""
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        bounds = [(0.0, 1.0), (0.0, 1.0)]

        results_a = [
            should_observe(data, bounds, 50, 100, rng=np.random.default_rng(42)) for _ in range(10)
        ]
        results_b = [
            should_observe(data, bounds, 50, 100, rng=np.random.default_rng(42)) for _ in range(10)
        ]
        assert results_a == results_b

    def test_zero_epsilon_never_observes(self) -> None:
        """With epsilon=0, should never observe."""
        data = np.array([[0.5, 0.5]])  # Too few points -> epsilon=0
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        for seed in range(20):
            result = should_observe(data, bounds, 1, 100, rng=np.random.default_rng(seed))
            assert result is False

    def test_high_epsilon_mostly_observes(self) -> None:
        """With high coverage, should observe most of the time."""
        data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        # epsilon should be 1.0 here (full coverage, late in budget)
        observe_count = sum(
            should_observe(data, bounds, 90, 100, rng=np.random.default_rng(seed))
            for seed in range(100)
        )
        # With epsilon=1.0, all should observe
        assert observe_count == 100


# ---------------------------------------------------------------------------
# OffPolicyPredictor epsilon_mode tests
# ---------------------------------------------------------------------------


class TestOffPolicyPredictorEpsilonMode:
    """Test OffPolicyPredictor with epsilon_mode enabled."""

    def test_epsilon_mode_disabled_by_default(self) -> None:
        """Default OffPolicyPredictor should not use epsilon mode."""
        predictor = OffPolicyPredictor()
        assert predictor.epsilon_mode is False

    def test_epsilon_mode_enabled(self) -> None:
        """OffPolicyPredictor should accept epsilon_mode parameter."""
        predictor = OffPolicyPredictor(epsilon_mode=True, n_max=50)
        assert predictor.epsilon_mode is True
        assert predictor.n_max == 50

    def test_backward_compatible_when_disabled(self) -> None:
        """When epsilon_mode=False, behavior should be identical to original."""
        predictor = OffPolicyPredictor(epsilon_mode=False)
        # No model fitted, should return True (run experiment)
        assert predictor.should_run_experiment({"x": 0.5, "y": 0.5}) is True

    def test_epsilon_mode_no_data_runs_experiment(self) -> None:
        """With epsilon_mode but no data, should always run experiment."""
        predictor = OffPolicyPredictor(epsilon_mode=True, n_max=100)
        assert predictor.should_run_experiment({"x": 0.5, "y": 0.5}) is True

    def test_epsilon_mode_calls_compute_epsilon(self) -> None:
        """Verify epsilon computation is called when in epsilon mode."""
        from causal_optimizer.types import ExperimentLog, ExperimentResult, ExperimentStatus

        predictor = OffPolicyPredictor(epsilon_mode=True, n_max=100)

        # Build enough experiment history
        log = ExperimentLog()
        search_space = make_search_space()
        rng = np.random.default_rng(42)
        for i in range(10):
            x_val = float(rng.random())
            y_val = float(rng.random())
            log.results.append(
                ExperimentResult(
                    experiment_id=str(i),
                    parameters={"x": x_val, "y": y_val},
                    metrics={"objective": x_val**2 + y_val**2},
                    status=ExperimentStatus.KEEP,
                )
            )
        predictor.fit(log, search_space, "objective")

        with patch("causal_optimizer.predictor.off_policy.compute_epsilon") as mock_compute:
            mock_compute.return_value = 0.0  # Always intervene
            result = predictor.should_run_experiment({"x": 0.5, "y": 0.5})
            mock_compute.assert_called_once()
            assert result is True


# ---------------------------------------------------------------------------
# ExperimentEngine integration tests
# ---------------------------------------------------------------------------


class TestEngineEpsilonIntegration:
    """Integration tests for ExperimentEngine with epsilon_mode."""

    def test_engine_accepts_epsilon_params(self) -> None:
        """Engine should accept epsilon_mode and n_max parameters."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            epsilon_mode=True,
            n_max=50,
        )
        assert engine._predictor.epsilon_mode is True
        assert engine._predictor.n_max == 50

    def test_engine_default_no_epsilon(self) -> None:
        """Default engine should not use epsilon mode."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
        )
        assert engine._predictor.epsilon_mode is False

    def test_engine_step_with_epsilon_does_not_crash(self) -> None:
        """Engine with epsilon_mode should complete step() without error."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            epsilon_mode=True,
            n_max=20,
        )
        result = engine.step()
        assert result is not None
        assert len(engine.log.results) == 1

    def test_engine_loop_with_epsilon(self) -> None:
        """Engine with epsilon_mode should complete a full loop."""
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            epsilon_mode=True,
            n_max=10,
        )
        log = engine.run_loop(n_experiments=8)
        assert len(log.results) == 8
        assert log.best_result is not None

    def test_epsilon_mode_can_skip_experiments(self) -> None:
        """With small n_max and good coverage, some experiments may be skipped.

        We mock the predictor to verify the skip path is reachable.
        """
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            epsilon_mode=True,
            n_max=10,
        )
        # Mock the predictor to simulate epsilon skipping
        mock_predictor = MagicMock(spec=OffPolicyPredictor)
        mock_predictor.epsilon_mode = True
        mock_predictor.n_max = 10
        # First call: skip (False means "don't run"), second call: run
        mock_predictor.should_run_experiment.side_effect = [False, True]
        mock_predictor.fit.return_value = None
        engine._predictor = mock_predictor

        result = engine.step()
        assert result is not None
        assert mock_predictor.should_run_experiment.call_count == 2
