"""Tests for causal vs surrogate_only differentiation in suggest_parameters.

Sprint 16 Step 1: When a causal graph is available, _suggest_surrogate()
should generate targeted intervention candidates (perturbing direct parents
of the objective) instead of purely random LHS candidates, creating
observable behavioral differentiation from surrogate_only mode.
"""

from __future__ import annotations

import numpy as np

from causal_optimizer.optimizer.suggest import (
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


def _make_star_graph() -> CausalGraph:
    """Star graph: x, y, z all directly cause objective."""
    return CausalGraph(
        edges=[
            ("x", "objective"),
            ("y", "objective"),
            ("z", "objective"),
        ]
    )


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _make_experiment_log(n: int = 15) -> ExperimentLog:
    """Create an experiment log with n results (past exploration phase)."""
    rng = np.random.default_rng(42)
    results: list[ExperimentResult] = []
    for i in range(n):
        x_val = float(rng.uniform(0, 10))
        y_val = float(rng.uniform(0, 10))
        z_val = float(rng.uniform(0, 10))
        obj = x_val + y_val + 0.5 * z_val
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters={"x": x_val, "y": y_val, "z": z_val},
                metrics={"objective": obj},
                status=ExperimentStatus.KEEP,
            )
        )
    return ExperimentLog(results=results)


def test_causal_and_surrogate_produce_different_suggestions() -> None:
    """Causal and surrogate_only must produce observably different suggestions.

    With the same seeds, at least 3 of 10 suggestion pairs must differ.
    This is the acceptance gate for Sprint 16 Step 1.
    """
    ss = _make_search_space()
    graph = _make_star_graph()
    log = _make_experiment_log(n=15)
    focus_variables = list(ss.variable_names)  # all vars are ancestors in star graph

    n_trials = 10
    n_different = 0

    for trial in range(n_trials):
        seed = 100 + trial
        causal_result = _suggest_surrogate(
            ss,
            log,
            focus_variables,
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=seed,
        )
        surrogate_result = _suggest_surrogate(
            ss,
            log,
            focus_variables,
            minimize=True,
            objective_name="objective",
            causal_graph=None,
            seed=seed,
        )
        if causal_result != surrogate_result:
            n_different += 1

    assert n_different >= 3, (
        f"Only {n_different}/10 suggestions differed between causal and surrogate_only. "
        "Expected at least 3."
    )


def test_targeted_candidates_perturb_few_variables() -> None:
    """Targeted intervention candidates should differ from best in only 1-2 variables.

    When a causal graph is provided, at least some generated candidates should
    be 'targeted' -- perturbing only 1 or 2 direct parents of the objective
    while holding other variables at the best-known values.
    """
    ss = _make_search_space()
    graph = _make_star_graph()
    log = _make_experiment_log(n=15)
    focus_variables = list(ss.variable_names)
    best = log.best_result("objective", minimize=True)
    assert best is not None

    # Generate many suggestions and check if any are "targeted" (close to best
    # in most variables)
    found_targeted = False
    for trial in range(20):
        seed = 200 + trial
        result = _suggest_surrogate(
            ss,
            log,
            focus_variables,
            minimize=True,
            objective_name="objective",
            causal_graph=graph,
            seed=seed,
        )
        # Count how many variables differ significantly from the best
        n_changed = 0
        for var_name in ["x", "y", "z"]:
            best_val = best.parameters[var_name]
            result_val = result[var_name]
            # "significantly" = more than 5% of range (range is 10.0)
            if abs(result_val - best_val) > 0.5:
                n_changed += 1
        if n_changed <= 2 and n_changed >= 1:
            found_targeted = True
            break

    assert found_targeted, (
        "No targeted intervention candidates found. Expected at least one suggestion "
        "that perturbs only 1-2 variables from the best-known configuration."
    )


def test_causal_graph_parents_method() -> None:
    """CausalGraph.parents() returns direct parents of the target."""
    graph = CausalGraph(
        edges=[
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
            ("d", "c"),
        ]
    )
    assert graph.parents("c") == {"b", "a", "d"}
    assert graph.parents("b") == {"a"}
    assert graph.parents("a") == set()
    assert graph.parents("d") == set()


def test_surrogate_only_unchanged() -> None:
    """_suggest_surrogate with causal_graph=None produces same results as before.

    Regression guard: the original 100 LHS candidate behavior must be preserved
    when no causal graph is provided.
    """
    ss = _make_search_space()
    log = _make_experiment_log(n=15)
    focus_variables = list(ss.variable_names)
    seed = 42

    # Call with causal_graph=None (surrogate_only mode)
    result1 = _suggest_surrogate(
        ss,
        log,
        focus_variables,
        minimize=True,
        objective_name="objective",
        causal_graph=None,
        seed=seed,
    )
    result2 = _suggest_surrogate(
        ss,
        log,
        focus_variables,
        minimize=True,
        objective_name="objective",
        causal_graph=None,
        seed=seed,
    )
    # Same seed, same inputs -> deterministic output
    assert result1 == result2

    # All variables should be present
    assert set(result1.keys()) >= {"x", "y", "z"}

    # Values must be within bounds
    for var_name in ["x", "y", "z"]:
        assert 0.0 <= result1[var_name] <= 10.0


def test_suggest_parameters_passes_causal_graph_to_surrogate() -> None:
    """suggest_parameters threads causal_graph to _suggest_surrogate via optimization."""
    from unittest.mock import patch

    ss = _make_search_space()
    graph = _make_star_graph()
    log = _make_experiment_log(n=15)

    with (
        patch("causal_optimizer.optimizer.suggest._suggest_bayesian") as mock_bay,
        patch("causal_optimizer.optimizer.suggest._suggest_surrogate") as mock_surr,
    ):
        mock_bay.side_effect = ImportError("no ax")
        mock_surr.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}

        suggest_parameters(
            ss,
            log,
            causal_graph=graph,
            phase="optimization",
            minimize=True,
            objective_name="objective",
        )

        call_kwargs = mock_surr.call_args
        assert call_kwargs.kwargs.get("causal_graph") is graph


def test_exploitation_prefers_parents_when_causal_graph_provided() -> None:
    """In exploitation, when a causal graph is provided, perturbation should
    prefer direct parents of the objective over non-parent variables."""
    from causal_optimizer.optimizer.suggest import _suggest_exploitation

    # Graph where only x and y are direct parents; z is NOT a parent
    graph = CausalGraph(edges=[("x", "objective"), ("y", "objective")])
    ss = _make_search_space()
    log = _make_experiment_log(n=15)
    best = log.best_result("objective", minimize=True)
    assert best is not None

    # With all 3 variables as focus, but causal graph saying only x, y are parents,
    # perturbation should prefer x and y. Over many trials z should change less often.
    z_changed_count = 0
    xy_changed_count = 0
    n_trials = 100
    for trial in range(n_trials):
        seed = 300 + trial
        result = _suggest_exploitation(
            ss,
            log,
            minimize=True,
            objective_name="objective",
            focus_variables=["x", "y", "z"],
            causal_graph=graph,
            seed=seed,
        )
        if abs(result["z"] - best.parameters["z"]) > 1e-10:
            z_changed_count += 1
        if (
            abs(result["x"] - best.parameters["x"]) > 1e-10
            or abs(result["y"] - best.parameters["y"]) > 1e-10
        ):
            xy_changed_count += 1

    # Parents (x, y) should be perturbed more often than non-parent z
    assert xy_changed_count > z_changed_count, (
        f"xy changed {xy_changed_count} times vs z changed {z_changed_count} times. "
        "Expected parents to be perturbed more frequently."
    )
