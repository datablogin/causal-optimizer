"""Tests for parameter suggestion strategies with causal focus variables."""

from unittest.mock import patch

from causal_optimizer.optimizer.suggest import (
    _get_focus_variables,
    _suggest_exploitation,
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


def _make_experiment_log(n: int = 5) -> ExperimentLog:
    """Create an experiment log with n results."""
    import numpy as np

    rng = np.random.default_rng(42)
    results = []
    for i in range(n):
        x_val = float(rng.uniform(0, 10))
        y_val = float(rng.uniform(0, 10))
        z_val = float(rng.uniform(0, 10))
        obj = x_val + y_val  # objective depends on x and y, not z
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters={"x": x_val, "y": y_val, "z": z_val},
                metrics={"objective": obj},
                status=ExperimentStatus.KEEP,
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
    best = log.best_result
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
    best = log.best_result
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
    best = log.best_result
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
