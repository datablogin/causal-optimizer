"""Tests for parameter suggestion strategies with causal focus variables."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from causal_optimizer.optimizer.suggest import (
    _get_focus_variables,
    _select_pomis_set,
    _suggest_exploitation,
    _suggest_optimization,
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


def _make_search_space() -> SearchSpace:
    """Create a simple 3-variable search space for testing."""
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _make_experiment_log(n: int = 5, phase: str | None = None) -> ExperimentLog:
    """Create an experiment log with n results."""
    import numpy as np

    rng = np.random.default_rng(42)
    results = []
    for i in range(n):
        x_val = float(rng.uniform(0, 10))
        y_val = float(rng.uniform(0, 10))
        z_val = float(rng.uniform(0, 10))
        obj = x_val + y_val  # objective depends on x and y, not z
        metadata: dict[str, Any] = {}
        if phase is not None:
            metadata["phase"] = phase
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters={"x": x_val, "y": y_val, "z": z_val},
                metrics={"objective": obj},
                status=ExperimentStatus.KEEP,
                metadata=metadata,
            )
        )
    return ExperimentLog(results=results)


def test_get_focus_variables_with_graph():
    """Focus variables are ancestors of the objective in the DAG."""
    # Graph: x -> y -> objective, z is disconnected
    graph = CausalGraph(edges=[("x", "y"), ("y", "objective")])
    ss = _make_search_space()

    focus = _get_focus_variables(ss, graph, "objective")

    assert "x" in focus
    assert "y" in focus
    assert "z" not in focus


def test_get_focus_variables_no_graph():
    """Without a causal graph, all variables are returned."""
    ss = _make_search_space()

    focus = _get_focus_variables(ss, None, "objective")

    assert set(focus) == {"x", "y", "z"}


def test_get_focus_variables_no_ancestors_in_space():
    """If no ancestors are in the search space, fall back to all variables."""
    # Graph has ancestors but none match search space variable names
    graph = CausalGraph(edges=[("a", "b"), ("b", "objective")])
    ss = _make_search_space()

    focus = _get_focus_variables(ss, graph, "objective")

    # Should fall back to all variables since none of a, b are in search space
    assert set(focus) == {"x", "y", "z"}


def test_suggest_surrogate_focus_variables_only_vary_focus():
    """Surrogate should only vary focus variables; non-focus held at best values."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    best = log.best_result()
    assert best is not None

    result = _suggest_surrogate(
        ss, log, focus_variables=["x", "y"], minimize=True, objective_name="objective"
    )

    # Non-focus variable z should be held at the best-known value
    assert result["z"] == best.parameters["z"]
    # Result should contain all variables
    assert "x" in result
    assert "y" in result
    assert "z" in result


def test_suggest_surrogate_empty_focus_uses_all():
    """Empty focus_variables should fall back to using all variables."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)

    result = _suggest_surrogate(
        ss, log, focus_variables=[], minimize=True, objective_name="objective"
    )

    # Should still produce a valid result with all variables
    assert "x" in result
    assert "y" in result
    assert "z" in result


def test_suggest_exploitation_focus_variables():
    """Exploitation should only perturb focus variables."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    best = log.best_result()
    assert best is not None

    # Run exploitation many times; z should never change from best
    z_changed = False
    for _ in range(50):
        result = _suggest_exploitation(
            ss,
            log,
            minimize=True,
            objective_name="objective",
            focus_variables=["x", "y"],
        )
        if result["z"] != best.parameters["z"]:
            z_changed = True
            break

    assert not z_changed, "Non-focus variable z should not be perturbed"


def test_suggest_exploitation_no_focus_perturbs_any():
    """Without focus_variables, exploitation can perturb any variable."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    best = log.best_result()
    assert best is not None

    result = _suggest_exploitation(
        ss,
        log,
        minimize=True,
        objective_name="objective",
        focus_variables=None,
    )

    # Should still produce a valid result
    assert "x" in result
    assert "y" in result
    assert "z" in result


def test_suggest_parameters_exploitation_passes_focus():
    """suggest_parameters with exploitation phase threads focus_variables through."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    graph = CausalGraph(edges=[("x", "y"), ("y", "objective")])

    with patch("causal_optimizer.optimizer.suggest._suggest_exploitation") as mock_exploit:
        mock_exploit.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        suggest_parameters(
            ss,
            log,
            causal_graph=graph,
            phase="exploitation",
            minimize=True,
            objective_name="objective",
        )
        # Verify focus_variables was passed
        call_kwargs = mock_exploit.call_args
        assert "focus_variables" in call_kwargs.kwargs
        focus = call_kwargs.kwargs["focus_variables"]
        assert "x" in focus
        assert "y" in focus
        assert "z" not in focus


# --- POMIS integration tests ---


def test_suggest_optimization_with_pomis_constrains_focus():
    """When pomis_sets is provided, _suggest_optimization constrains focus variables."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5, phase="optimization")

    # With 5 optimization-phase experiments, round-robin index = 5 % 2 = 1,
    # so second set {"y", "z"} is chosen
    pomis_sets = [frozenset({"x"}), frozenset({"y", "z"})]

    with patch("causal_optimizer.optimizer.suggest._suggest_surrogate") as mock_surrogate:
        mock_surrogate.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        _suggest_optimization(
            ss,
            log,
            causal_graph=None,
            minimize=True,
            objective_name="objective",
            pomis_sets=pomis_sets,
        )
        # The focus_variables passed to the surrogate should be from the chosen POMIS set
        call_args = mock_surrogate.call_args
        focus = (
            call_args[0][2]
            if len(call_args[0]) > 2
            else call_args.kwargs.get("focus_variables", [])
        )
        # Round-robin picks set at index 1: {"y", "z"}
        # No graph, so existing focus is all vars — intersection with POMIS = POMIS set
        focus_set = frozenset(focus)
        assert focus_set == frozenset({"y", "z"})


def test_suggest_optimization_pomis_intersects_with_graph_focus():
    """POMIS focus is intersected with graph+screening focus variables."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    # Graph: x -> objective (only x is ancestor)
    graph = CausalGraph(edges=[("x", "objective")])

    # POMIS set includes x and y, but graph focus is only x
    # With 5 experiments, round-robin index = 5 % 1 = 0
    pomis_sets = [frozenset({"x", "y"})]

    with patch("causal_optimizer.optimizer.suggest._suggest_surrogate") as mock_surrogate:
        mock_surrogate.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        _suggest_optimization(
            ss,
            log,
            causal_graph=graph,
            minimize=True,
            objective_name="objective",
            pomis_sets=pomis_sets,
        )
        call_args = mock_surrogate.call_args
        focus = (
            call_args[0][2]
            if len(call_args[0]) > 2
            else call_args.kwargs.get("focus_variables", [])
        )
        # POMIS focus {x, y} intersected with graph focus {x} = {x}
        assert set(focus) == {"x"}


def test_suggest_optimization_pomis_fallback_when_no_intersection():
    """When POMIS and graph focus don't intersect, fall back to POMIS-only."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    # Graph: z -> objective (only z is ancestor)
    graph = CausalGraph(edges=[("z", "objective")])

    # POMIS set has x and y, which don't overlap with graph focus {z}
    pomis_sets = [frozenset({"x", "y"})]

    with patch("causal_optimizer.optimizer.suggest._suggest_surrogate") as mock_surrogate:
        mock_surrogate.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        _suggest_optimization(
            ss,
            log,
            causal_graph=graph,
            minimize=True,
            objective_name="objective",
            pomis_sets=pomis_sets,
        )
        call_args = mock_surrogate.call_args
        focus = (
            call_args[0][2]
            if len(call_args[0]) > 2
            else call_args.kwargs.get("focus_variables", [])
        )
        # No intersection, so fall back to POMIS-only {x, y}
        assert set(focus) == {"x", "y"}


def test_suggest_optimization_without_pomis_unchanged():
    """Without pomis_sets, _suggest_optimization behaves as before."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)

    with patch("causal_optimizer.optimizer.suggest._suggest_surrogate") as mock_surrogate:
        mock_surrogate.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        _suggest_optimization(
            ss,
            log,
            causal_graph=None,
            minimize=True,
            objective_name="objective",
            pomis_sets=None,
        )
        call_args = mock_surrogate.call_args
        focus = (
            call_args[0][2]
            if len(call_args[0]) > 2
            else call_args.kwargs.get("focus_variables", [])
        )
        # Without POMIS, all variables should be in focus (no graph either)
        assert set(focus) == {"x", "y", "z"}


def test_suggest_parameters_passes_pomis_to_optimization():
    """suggest_parameters passes pomis_sets through to _suggest_optimization."""
    ss = _make_search_space()
    log = _make_experiment_log(n=5)
    pomis_sets = [frozenset({"x"})]

    with patch("causal_optimizer.optimizer.suggest._suggest_optimization") as mock_opt:
        mock_opt.return_value = {"x": 1.0, "y": 2.0, "z": 3.0}
        suggest_parameters(
            ss,
            log,
            phase="optimization",
            pomis_sets=pomis_sets,
        )
        call_kwargs = mock_opt.call_args
        assert call_kwargs.kwargs.get("pomis_sets") == pomis_sets


def test_select_pomis_set_round_robin():
    """_select_pomis_set cycles through sets based on experiment count."""
    set_a = frozenset({"x"})
    set_b = frozenset({"y"})
    set_c = frozenset({"z"})
    pomis_sets = [set_a, set_b, set_c]

    # With 0 optimization-phase experiments, should pick index 0
    log_0 = ExperimentLog(results=[])
    assert _select_pomis_set(pomis_sets, log_0) == set_a

    # With 1 optimization-phase experiment, should pick index 1
    log_1 = _make_experiment_log(n=1, phase="optimization")
    assert _select_pomis_set(pomis_sets, log_1) == set_b

    # With 2 optimization-phase experiments, should pick index 2
    log_2 = _make_experiment_log(n=2, phase="optimization")
    assert _select_pomis_set(pomis_sets, log_2) == set_c

    # With 3 optimization-phase experiments, should wrap to index 0
    log_3 = _make_experiment_log(n=3, phase="optimization")
    assert _select_pomis_set(pomis_sets, log_3) == set_a

    # Non-optimization experiments should not affect the round-robin offset
    log_exploration = _make_experiment_log(n=5, phase="exploration")
    assert _select_pomis_set(pomis_sets, log_exploration) == set_a


def test_select_pomis_set_empty_returns_none():
    """_select_pomis_set returns None for empty pomis_sets."""
    log = _make_experiment_log(n=3)
    result = _select_pomis_set([], log)
    assert result is None


def test_select_pomis_set_single_set():
    """With a single POMIS set, it is always returned."""
    log = _make_experiment_log(n=5)
    set_a = frozenset({"x"})
    result = _select_pomis_set([set_a], log)
    assert result == set_a
