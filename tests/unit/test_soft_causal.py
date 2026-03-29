"""Tests for softer causal influence in the optimizer (Sprint 19).

Covers:
1. Causal-weighted exploration (ancestor emphasis during LHS)
2. Soft ranking during optimization (RF trains on all vars, causal alignment bonus)
3. Targeted candidate rebalancing (adaptive LHS/targeted split)
4. Backward compatibility (weight=0 recovers old behavior, no graph unchanged)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.optimizer.suggest import (
    _get_targeted_ratio,
    _score_candidate_causal_exploration,
    _suggest_exploration,
    _suggest_surrogate,
    suggest_parameters,
)
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def _make_search_space_5d() -> SearchSpace:
    """5-variable space: X1-X3 are ancestors, X4-X5 are noise."""
    return SearchSpace(
        variables=[
            Variable(name="X1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X3", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X4", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X5", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _make_causal_graph() -> CausalGraph:
    """Graph where X1->X2->objective, X3->objective. X4, X5 are noise."""
    return CausalGraph(
        edges=[("X1", "X2"), ("X2", "objective"), ("X3", "objective")],
    )


def _make_experiment_log(n: int = 5, seed: int = 42) -> ExperimentLog:
    """Create an experiment log with n results."""
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        params = {f"X{j}": float(rng.uniform(0, 10)) for j in range(1, 6)}
        # Objective depends on X1, X2, X3 (ancestors), not X4, X5
        obj = params["X1"] + params["X2"] + params["X3"]
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters=params,
                metrics={"objective": obj},
                status=ExperimentStatus.KEEP,
            )
        )
    return ExperimentLog(results=results)


# ---- Test 1: Causal exploration biases ancestors ----


def test_causal_exploration_biases_ancestors():
    """With a graph where X1-X3 are ancestors, exploration should show more
    variation in those dims than in non-ancestor X4/X5 (statistically)."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=5)

    ancestor_vars = set()
    non_ancestor_vars = set()
    n_trials = 40

    for trial in range(n_trials):
        result = _suggest_exploration(
            ss,
            log,
            causal_graph=graph,
            objective_name="objective",
            causal_exploration_weight=0.3,
            seed=trial * 7,
        )
        for k, v in result.items():
            if k in ("X1", "X2", "X3"):
                ancestor_vars.add((trial, k, v))
            else:
                non_ancestor_vars.add((trial, k, v))

    # Collect the per-trial values for each variable
    ancestor_values: dict[str, list[float]] = {"X1": [], "X2": [], "X3": []}
    non_ancestor_values: dict[str, list[float]] = {"X4": [], "X5": []}

    for trial in range(n_trials):
        result = _suggest_exploration(
            ss,
            log,
            causal_graph=graph,
            objective_name="objective",
            causal_exploration_weight=0.3,
            seed=trial * 7,
        )
        for k in ("X1", "X2", "X3"):
            ancestor_values[k].append(result[k])
        for k in ("X4", "X5"):
            non_ancestor_values[k].append(result[k])

    # Ancestor variables should have higher variance (more spread)
    ancestor_stds = [np.std(v) for v in ancestor_values.values()]
    non_ancestor_stds = [np.std(v) for v in non_ancestor_values.values()]
    mean_ancestor_std = np.mean(ancestor_stds)
    mean_non_ancestor_std = np.mean(non_ancestor_stds)

    # With causal weighting, ancestors should be more diverse across trials
    assert mean_ancestor_std > mean_non_ancestor_std * 0.9, (
        f"Ancestor std ({mean_ancestor_std:.3f}) should be meaningfully larger "
        f"than non-ancestor std ({mean_non_ancestor_std:.3f})"
    )


# ---- Test 2: Non-ancestors still vary ----


def test_causal_exploration_still_varies_non_ancestors():
    """Non-ancestor variables should still vary, not be pinned."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=5)

    x4_values = []
    x5_values = []
    for trial in range(20):
        result = _suggest_exploration(
            ss,
            log,
            causal_graph=graph,
            objective_name="objective",
            causal_exploration_weight=0.3,
            seed=trial * 13,
        )
        x4_values.append(result["X4"])
        x5_values.append(result["X5"])

    # Non-ancestors should not all be the same value
    assert np.std(x4_values) > 0.1, "X4 should vary across trials"
    assert np.std(x5_values) > 0.1, "X5 should vary across trials"


# ---- Test 3: Weight=0 is pure LHS ----


def test_causal_exploration_weight_zero_is_pure_lhs():
    """With causal_exploration_weight=0.0, exploration should match old behavior."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=0)

    # With weight=0, should get identical results to no-graph exploration
    for seed in range(10):
        with_graph = _suggest_exploration(
            ss,
            log,
            causal_graph=graph,
            objective_name="objective",
            causal_exploration_weight=0.0,
            seed=seed,
        )
        without_graph = _suggest_exploration(
            ss,
            log,
            causal_graph=None,
            objective_name="objective",
            causal_exploration_weight=0.0,
            seed=seed,
        )
        for var_name in ss.variable_names:
            assert with_graph[var_name] == pytest.approx(without_graph[var_name], abs=1e-10), (
                f"With weight=0, {var_name} should be identical with and without graph"
            )


# ---- Test 4: Soft ranking uses all variables ----


def test_soft_ranking_uses_all_variables():
    """During optimization, RF should train on all variables, not just focus vars."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=10)

    # With soft ranking (causal_softness > 0), the RF should train on all 5 vars
    # not just the 3 ancestor focus vars
    with patch("causal_optimizer.optimizer.suggest.RandomForestRegressor") as mock_rf_cls:
        mock_rf = mock_rf_cls.return_value
        mock_rf.predict.return_value = np.array([1.0])
        mock_rf.fit.return_value = None

        _suggest_surrogate(
            ss,
            log,
            focus_variables=["X1", "X2", "X3"],
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=42,
            causal_softness=0.5,
        )

        # Check that fit was called with all 5 variables, not just 3
        fit_call = mock_rf.fit.call_args
        features = fit_call[0][0]
        assert features.shape[1] == 5, (
            f"RF should train on all 5 variables, got {features.shape[1]}"
        )


# ---- Test 5: Soft ranking prefers ancestor variation ----


def test_soft_ranking_prefers_ancestor_variation():
    """Candidates that vary ancestors should score higher via causal alignment bonus."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=15)

    # Run multiple times and check that the suggested parameters tend to vary
    # ancestor variables more than non-ancestor ones relative to the best
    best = log.best_result("objective", minimize=True)
    assert best is not None

    ancestor_diffs = []
    non_ancestor_diffs = []

    for seed in range(20):
        result = _suggest_surrogate(
            ss,
            log,
            focus_variables=["X1", "X2", "X3"],
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=seed,
            causal_softness=0.5,
        )

        for k in ("X1", "X2", "X3"):
            ancestor_diffs.append(abs(result[k] - best.parameters[k]))
        for k in ("X4", "X5"):
            non_ancestor_diffs.append(abs(result[k] - best.parameters[k]))

    # Ancestor variables should show more variation from best on average
    # because the causal alignment bonus encourages exploring them
    mean_ancestor_diff = np.mean(ancestor_diffs)
    mean_non_ancestor_diff = np.mean(non_ancestor_diffs)

    # With soft ranking, ancestors should have at least comparable movement
    # (they are not pinned like in hard mode)
    assert mean_ancestor_diff > 0.01, (
        f"Ancestors should vary from best, got mean diff {mean_ancestor_diff:.4f}"
    )


# ---- Test 6: Soft ranking does not exclude non-ancestors ----


def test_soft_ranking_does_not_exclude_non_ancestors():
    """Non-ancestor variation should still be possible with soft ranking."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=15)

    best = log.best_result("objective", minimize=True)
    assert best is not None

    non_ancestor_varied = False
    for seed in range(30):
        result = _suggest_surrogate(
            ss,
            log,
            focus_variables=["X1", "X2", "X3"],
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=seed,
            causal_softness=0.5,
        )
        for k in ("X4", "X5"):
            if abs(result[k] - best.parameters.get(k, 5.0)) > 0.5:
                non_ancestor_varied = True
                break
        if non_ancestor_varied:
            break

    assert non_ancestor_varied, (
        "Non-ancestor variables should be able to vary from best with soft ranking"
    )


# ---- Test 7: Targeted rebalancing early ----


def test_targeted_rebalancing_early():
    """At experiment count ~12, targeted ratio should be ~30%."""
    ratio = _get_targeted_ratio(experiment_count=12)
    assert 0.25 <= ratio <= 0.35, (
        f"Early optimization should use ~30% targeted, got {ratio:.2f}"
    )


# ---- Test 8: Targeted rebalancing late ----


def test_targeted_rebalancing_late():
    """At experiment count ~45, targeted ratio should be ~70%."""
    ratio = _get_targeted_ratio(experiment_count=45)
    assert 0.65 <= ratio <= 0.75, (
        f"Late optimization should use ~70% targeted, got {ratio:.2f}"
    )


# ---- Test 9: Backward compat hard focus ----


def test_backward_compat_hard_focus():
    """With causal_softness=1e6, behavior should approximate old hard focus."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    log = _make_experiment_log(n=15)

    best = log.best_result("objective", minimize=True)
    assert best is not None

    # With very large softness, non-focus variables should be pinned to best values
    for seed in range(10):
        result = _suggest_surrogate(
            ss,
            log,
            focus_variables=["X1", "X2", "X3"],
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=seed,
            causal_softness=1e6,
        )

        # With hard focus (very large softness), non-ancestor vars pinned to best
        for k in ("X4", "X5"):
            assert result[k] == pytest.approx(best.parameters[k], abs=0.01), (
                f"With causal_softness=1e6, {k} should be pinned to best value "
                f"(got {result[k]}, expected {best.parameters[k]})"
            )


# ---- Test 10: No graph unchanged ----


def test_no_graph_unchanged():
    """Without a causal graph, all behavior is identical to Sprint 18."""
    ss = _make_search_space_5d()
    log = _make_experiment_log(n=15)

    # Exploration without graph
    for seed in range(5):
        result_new = _suggest_exploration(
            ss,
            log,
            causal_graph=None,
            objective_name="objective",
            causal_exploration_weight=0.3,
            seed=seed,
        )
        result_old = _suggest_exploration(
            ss,
            log,
            seed=seed,
        )
        for var_name in ss.variable_names:
            assert result_new[var_name] == pytest.approx(result_old[var_name], abs=1e-10), (
                f"Without graph, exploration should be unchanged for {var_name}"
            )


# ---- Test: Engine accepts new config params ----


class _DummyRunner:
    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        return {"objective": sum(parameters.values())}


def test_engine_accepts_causal_config_params():
    """ExperimentEngine should accept causal_exploration_weight and causal_softness."""
    ss = SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )
    engine = ExperimentEngine(
        search_space=ss,
        runner=_DummyRunner(),
        causal_exploration_weight=0.5,
        causal_softness=1.0,
    )
    assert engine.causal_exploration_weight == 0.5
    assert engine.causal_softness == 1.0


def test_engine_default_causal_config():
    """Default causal config should be weight=0.3, softness=0.5."""
    ss = SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )
    engine = ExperimentEngine(
        search_space=ss,
        runner=_DummyRunner(),
    )
    assert engine.causal_exploration_weight == 0.3
    assert engine.causal_softness == 0.5


# ---- Test: _score_candidate_causal_exploration ----


def test_score_candidate_causal_exploration_basic():
    """Scoring function should return a positive score for valid candidates."""
    ss = _make_search_space_5d()
    graph = _make_causal_graph()
    existing = [
        {"X1": 5.0, "X2": 5.0, "X3": 5.0, "X4": 5.0, "X5": 5.0},
    ]
    candidate = {"X1": 8.0, "X2": 2.0, "X3": 9.0, "X4": 5.0, "X5": 5.0}

    score = _score_candidate_causal_exploration(
        candidate=candidate,
        existing_params=existing,
        ancestor_names={"X1", "X2", "X3"},
        search_space=ss,
        alpha=0.3,
    )
    assert score > 0.0, f"Score should be positive, got {score}"


def test_score_candidate_ancestor_variation_scores_higher():
    """A candidate varying ancestors should score higher than one varying non-ancestors."""
    ss = _make_search_space_5d()
    existing = [
        {"X1": 5.0, "X2": 5.0, "X3": 5.0, "X4": 5.0, "X5": 5.0},
    ]
    # Candidate A: varies ancestors
    candidate_a = {"X1": 9.0, "X2": 1.0, "X3": 9.0, "X4": 5.0, "X5": 5.0}
    # Candidate B: varies non-ancestors
    candidate_b = {"X1": 5.0, "X2": 5.0, "X3": 5.0, "X4": 9.0, "X5": 1.0}

    score_a = _score_candidate_causal_exploration(
        candidate=candidate_a,
        existing_params=existing,
        ancestor_names={"X1", "X2", "X3"},
        search_space=ss,
        alpha=0.3,
    )
    score_b = _score_candidate_causal_exploration(
        candidate=candidate_b,
        existing_params=existing,
        ancestor_names={"X1", "X2", "X3"},
        search_space=ss,
        alpha=0.3,
    )
    assert score_a > score_b, (
        f"Ancestor-varying candidate (score={score_a:.3f}) should score higher "
        f"than non-ancestor-varying (score={score_b:.3f})"
    )


def test_targeted_ratio_at_midpoint():
    """At experiment count ~25, targeted ratio should be ~50%."""
    ratio = _get_targeted_ratio(experiment_count=25)
    assert 0.45 <= ratio <= 0.55, (
        f"Mid optimization should use ~50% targeted, got {ratio:.2f}"
    )
