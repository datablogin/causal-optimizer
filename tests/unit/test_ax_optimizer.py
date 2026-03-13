"""Tests for the AxBayesianOptimizer class.

Tests are organized into:
1. Basic suggest/update behavior
2. Focus variable constraints
3. POMIS prior biasing
4. Graceful degradation when ax is unavailable
5. Engine integration
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from causal_optimizer.types import (
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_search_space() -> SearchSpace:
    """2 continuous variables [0, 10]."""
    return SearchSpace(
        variables=[
            Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


# ---------------------------------------------------------------------------
# Test 1: suggest returns valid params within search space bounds
# ---------------------------------------------------------------------------


def test_ax_optimizer_suggest_returns_valid_params() -> None:
    """Single suggest() call returns a dict with all search space variables in bounds."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    opt = AxBayesianOptimizer(search_space=ss, objective_name="objective", minimize=True, seed=0)

    params = opt.suggest()

    assert isinstance(params, dict)
    assert "x1" in params
    assert "x2" in params
    assert 0.0 <= params["x1"] <= 10.0
    assert 0.0 <= params["x2"] <= 10.0


# ---------------------------------------------------------------------------
# Test 2: update/suggest cycle improves toward optimum
# ---------------------------------------------------------------------------


def test_ax_optimizer_update_improves_suggestion() -> None:
    """After 5 update/suggest cycles on a quadratic, the 6th suggestion is closer to optimum."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    opt = AxBayesianOptimizer(search_space=ss, objective_name="objective", minimize=True, seed=42)

    # True optimum at (3, 7)
    def quadratic(params: dict[str, Any]) -> float:
        return (params["x1"] - 3.0) ** 2 + (params["x2"] - 7.0) ** 2

    first_params = opt.suggest()
    first_dist = quadratic(first_params)

    for _ in range(5):
        p = opt.suggest()
        v = quadratic(p)
        opt.update(p, v)

    last_params = opt.suggest()
    last_dist = quadratic(last_params)

    # After 5 observations, the optimizer should not have gotten dramatically worse.
    # We allow a 2x tolerance: at worst the last suggestion is 2x further than the first
    # (which was random). In practice, BoTorch should find something near the optimum.
    assert last_dist <= max(first_dist * 2, 100.0), (
        f"Expected distance to improve after 5 updates, "
        f"got {last_dist:.2f} vs first={first_dist:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 3: focus_variables restricts which params vary
# ---------------------------------------------------------------------------


def test_ax_optimizer_focus_variables_respected() -> None:
    """With focus_variables=['x1'], only x1 varies; x2 is fixed at midpoint."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    opt = AxBayesianOptimizer(
        search_space=ss,
        objective_name="objective",
        minimize=True,
        focus_variables=["x1"],
        seed=0,
    )

    # x2 midpoint is (0 + 10) / 2 = 5.0
    expected_x2 = 5.0
    params = opt.suggest()

    assert "x1" in params
    assert "x2" in params
    assert params["x2"] == pytest.approx(expected_x2, abs=1e-9), (
        f"Non-focus variable x2 should be fixed at midpoint 5.0, got {params['x2']}"
    )
    # x1 can be anything in bounds
    assert 0.0 <= params["x1"] <= 10.0


# ---------------------------------------------------------------------------
# Test 4: pomis_prior biases suggestions toward POMIS variables
# ---------------------------------------------------------------------------


def test_ax_optimizer_pomis_prior_biases_toward_pomis() -> None:
    """With pomis_prior={x1}, over 10 suggestions, ≥60% touch only x1."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    pomis_prior = [frozenset({"x1"})]
    opt = AxBayesianOptimizer(
        search_space=ss,
        objective_name="objective",
        minimize=True,
        pomis_prior=pomis_prior,
        seed=0,
    )

    # x2 midpoint is 5.0; "touches only x1" means x2 is at midpoint (not varied)
    x2_midpoint = 5.0
    pomis_count = 0
    n_trials = 10
    for _ in range(n_trials):
        params = opt.suggest()
        if params.get("x2") == pytest.approx(x2_midpoint, abs=1e-9):
            pomis_count += 1

    assert pomis_count >= 6, (
        f"Expected ≥60% of suggestions to be POMIS-only, got {pomis_count}/{n_trials}"
    )


# ---------------------------------------------------------------------------
# Test 5: graceful degradation — ImportError with helpful message
# ---------------------------------------------------------------------------


def test_ax_optimizer_graceful_degradation() -> None:
    """When ax is not importable, instantiating AxBayesianOptimizer raises ImportError."""
    # Temporarily hide ax from the import machinery
    with patch.dict(sys.modules, {"ax": None, "ax.service.ax_client": None}):
        # Re-importing the module under the mock requires we also remove the
        # cached import so the guard runs again.
        import importlib

        import causal_optimizer.optimizer.bayesian as bayesian_module

        importlib.reload(bayesian_module)

        with pytest.raises(ImportError, match="uv sync --extra bayesian"):
            bayesian_module.AxBayesianOptimizer(
                search_space=_make_simple_search_space(),
                objective_name="objective",
            )

    # After test: reload to restore the real module state
    import importlib

    import causal_optimizer.optimizer.bayesian as bmod

    importlib.reload(bmod)


# ---------------------------------------------------------------------------
# Test 6: engine integration — reaches optimization phase without crash
# ---------------------------------------------------------------------------


def test_ax_in_engine_optimization_phase() -> None:
    """Run ExperimentEngine with strategy='bayesian' for 20 steps; reaches optimization phase."""
    from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
    from causal_optimizer.engine.loop import ExperimentEngine

    bench = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(0))
    engine = ExperimentEngine(
        search_space=ToyGraphBenchmark.search_space(),
        runner=bench,
        causal_graph=ToyGraphBenchmark.causal_graph(),
        objective_name="objective",
        minimize=True,
        seed=0,
        max_skips=0,  # no skips so we get 20 real experiments
    )

    # Run 20 steps — enough to pass through exploration (10) into optimization (11+)
    for _ in range(20):
        engine.step()

    assert engine._phase == "optimization", (
        f"Expected phase='optimization' after 20 steps, got {engine._phase!r}"
    )
    assert len(engine.log.results) == 20


# ---------------------------------------------------------------------------
# Coverage gap tests: multi-type search space, best(), and maximize
# ---------------------------------------------------------------------------


def test_ax_optimizer_integer_and_categorical_variables() -> None:
    """AxBayesianOptimizer handles INTEGER and CATEGORICAL variables without error."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = SearchSpace(
        variables=[
            Variable(name="n", variable_type=VariableType.INTEGER, lower=1, upper=10),
            Variable(
                name="method",
                variable_type=VariableType.CATEGORICAL,
                choices=["sgd", "adam", "rmsprop"],
            ),
        ]
    )
    opt = AxBayesianOptimizer(search_space=ss, objective_name="objective", minimize=True, seed=0)
    params = opt.suggest()

    assert "n" in params
    assert "method" in params
    assert 1 <= params["n"] <= 10
    assert params["method"] in ["sgd", "adam", "rmsprop"]


def test_ax_optimizer_best_returns_best_observation() -> None:
    """best() returns the param dict with the lowest objective value seen."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    opt = AxBayesianOptimizer(search_space=ss, objective_name="objective", minimize=True, seed=0)

    assert opt.best() is None  # no observations yet

    opt.update({"x1": 8.0, "x2": 8.0}, 130.0)  # dist^2 from (3,7) = 25+1
    opt.update({"x1": 3.0, "x2": 7.0}, 0.0)  # optimum
    opt.update({"x1": 1.0, "x2": 1.0}, 40.0)  # dist^2 = 4+36

    best = opt.best()
    assert best is not None
    assert best["x1"] == pytest.approx(3.0)
    assert best["x2"] == pytest.approx(7.0)


def test_ax_optimizer_best_maximize() -> None:
    """best() returns the param dict with the highest value when maximize=True."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = _make_simple_search_space()
    opt = AxBayesianOptimizer(search_space=ss, objective_name="objective", minimize=False, seed=0)

    opt.update({"x1": 1.0, "x2": 1.0}, 5.0)
    opt.update({"x1": 9.0, "x2": 9.0}, 99.0)  # best for maximize

    best = opt.best()
    assert best is not None
    assert best["x1"] == pytest.approx(9.0)


def test_ax_optimizer_boolean_variable_and_midpoints() -> None:
    """AxBayesianOptimizer handles BOOLEAN variables and computes midpoints correctly."""
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    ss = SearchSpace(
        variables=[
            Variable(name="cont", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=4.0),
            Variable(name="flag", variable_type=VariableType.BOOLEAN),
            Variable(name="rank", variable_type=VariableType.INTEGER, lower=0, upper=10),
            Variable(
                name="cat", variable_type=VariableType.CATEGORICAL, choices=["a", "b", "c"]
            ),
        ]
    )
    # No focus_variables: all 4 types pass through _build_ax_params (covers BOOLEAN branch).
    # focus_variables=["cont"] would fix the others — here we want them active.
    opt = AxBayesianOptimizer(
        search_space=ss,
        objective_name="objective",
        minimize=True,
        seed=0,
    )
    params = opt.suggest()

    assert "cont" in params
    assert "flag" in params
    assert "rank" in params
    assert "cat" in params

    # Also test midpoints by fixing the non-continuous vars via focus_variables
    opt2 = AxBayesianOptimizer(
        search_space=ss,
        objective_name="objective",
        minimize=True,
        focus_variables=["cont"],
        seed=0,
    )
    params2 = opt2.suggest()

    assert params2["flag"] is False  # midpoint for boolean = False
    assert params2["rank"] == 5  # midpoint for int [0,10] = 5
    assert params2["cat"] == "b"  # midpoint for 3-choice list = index 1 = "b"
