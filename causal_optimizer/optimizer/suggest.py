"""Parameter suggestion strategies for the optimization loop.

Implements a progression of strategies:
- Exploration phase: DoE-based designs for screening
- Optimization phase: Bayesian optimization (with optional causal graph guidance)
- Exploitation phase: local search around best known configuration
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace, VariableType

logger = logging.getLogger(__name__)


def suggest_parameters(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph | None = None,
    phase: str = "exploration",
    minimize: bool = True,
    objective_name: str = "objective",
    screened_variables: list[str] | None = None,
    base_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest next experiment parameters based on current phase and history.

    Args:
        screened_variables: Variables identified as important by screening.
            Complements graph-based focus variables (intersection if both available).
        base_parameters: Base parameters to perturb from (e.g., from MAP-Elites elite).
            Used in exploitation phase instead of the overall best.
    """
    if phase == "exploration":
        return _suggest_exploration(search_space, experiment_log)
    elif phase == "optimization":
        return _suggest_optimization(
            search_space, experiment_log, causal_graph, minimize, objective_name,
            screened_variables=screened_variables,
        )
    elif phase == "exploitation":
        return _suggest_exploitation(
            search_space, experiment_log, minimize, objective_name,
            base_parameters=base_parameters,
        )
    else:
        return _suggest_exploration(search_space, experiment_log)


def _suggest_exploration(
    search_space: SearchSpace, experiment_log: ExperimentLog
) -> dict[str, Any]:
    """Exploration: Latin Hypercube sampling for space-filling coverage."""
    from causal_optimizer.designer.factorial import FactorialDesigner

    designer = FactorialDesigner(search_space)
    designs = designer.latin_hypercube(n_samples=1)
    return designs[0] if designs else _random_sample(search_space)


def _suggest_optimization(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph | None,
    minimize: bool,
    objective_name: str,
    screened_variables: list[str] | None = None,
) -> dict[str, Any]:
    """Optimization: Bayesian optimization with optional causal guidance.

    If a causal graph is available, uses it to identify which variables
    to prioritize (ancestors of the objective in the DAG). Screening results
    complement the graph-based focus.
    """
    df = experiment_log.to_dataframe()
    if len(df) < 3:
        return _suggest_exploration(search_space, experiment_log)

    # Identify which variables to focus on
    graph_focus = _get_focus_variables(search_space, causal_graph, objective_name)

    # Combine graph-based and screening-based focus variables
    if screened_variables is not None and causal_graph is not None:
        # Both available: use intersection (variables both sources agree on)
        focus_variables = [v for v in graph_focus if v in screened_variables]
        # Fall back to union if intersection is empty
        if not focus_variables:
            focus_variables = list(set(graph_focus) | set(screened_variables))
    elif screened_variables is not None:
        # Only screening available (no graph)
        focus_variables = screened_variables
    else:
        # Only graph (or default)
        focus_variables = graph_focus

    # Try Bayesian optimization via Ax
    try:
        return _suggest_bayesian(search_space, experiment_log, minimize, objective_name)
    except ImportError:
        logger.info("Ax/BoTorch not available, using surrogate-guided sampling")
        return _suggest_surrogate(
            search_space, experiment_log, focus_variables, minimize, objective_name
        )


def _suggest_exploitation(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    minimize: bool,
    objective_name: str,
    base_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exploitation: perturb the best known configuration or a provided base."""
    if base_parameters is not None:
        params = dict(base_parameters)
    else:
        best = experiment_log.best_result
        if best is None:
            return _random_sample(search_space)
        params = dict(best.parameters)

    rng = np.random.default_rng()

    # Perturb one or two variables slightly
    n_perturb = rng.integers(1, min(3, len(search_space.variables)) + 1)
    indices = rng.choice(len(search_space.variables), size=n_perturb, replace=False)

    for idx in indices:
        var = search_space.variables[idx]
        if var.variable_type == VariableType.CONTINUOUS and var.lower is not None and var.upper is not None:
            current = params.get(var.name, (var.lower + var.upper) / 2)
            scale = (var.upper - var.lower) * 0.1  # 10% perturbation
            new_val = current + rng.normal(0, scale)
            params[var.name] = float(np.clip(new_val, var.lower, var.upper))
        elif var.variable_type == VariableType.INTEGER and var.lower is not None and var.upper is not None:
            current = params.get(var.name, int((var.lower + var.upper) / 2))
            new_val = current + rng.integers(-2, 3)
            params[var.name] = int(np.clip(new_val, var.lower, var.upper))
        elif var.variable_type == VariableType.BOOLEAN:
            if rng.random() < 0.3:  # flip with 30% probability
                params[var.name] = not params.get(var.name, False)
        elif var.variable_type == VariableType.CATEGORICAL and var.choices:
            if rng.random() < 0.3:
                params[var.name] = rng.choice(var.choices)

    return params


def _suggest_bayesian(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    minimize: bool,
    objective_name: str,
) -> dict[str, Any]:
    """Bayesian optimization via Ax/BoTorch."""
    from ax.service.ax_client import AxClient

    ax_client = AxClient()

    # Build Ax parameter list
    ax_params = []
    for var in search_space.variables:
        if var.variable_type == VariableType.CONTINUOUS:
            ax_params.append({
                "name": var.name,
                "type": "range",
                "bounds": [var.lower or 0.0, var.upper or 1.0],
                "value_type": "float",
            })
        elif var.variable_type == VariableType.INTEGER:
            ax_params.append({
                "name": var.name,
                "type": "range",
                "bounds": [int(var.lower or 0), int(var.upper or 10)],
                "value_type": "int",
            })
        elif var.variable_type == VariableType.CATEGORICAL:
            ax_params.append({
                "name": var.name,
                "type": "choice",
                "values": var.choices or [],
            })
        elif var.variable_type == VariableType.BOOLEAN:
            ax_params.append({
                "name": var.name,
                "type": "choice",
                "values": [True, False],
            })

    ax_client.create_experiment(
        name="causal_optimizer",
        parameters=ax_params,
        minimize=minimize,
    )

    # Feed historical data
    for result in experiment_log.results:
        if objective_name in result.metrics:
            _, trial_index = ax_client.attach_trial(result.parameters)
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={objective_name: result.metrics[objective_name]},
            )

    params, _ = ax_client.get_next_trial()
    return dict(params)


def _suggest_surrogate(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    focus_variables: list[str],
    minimize: bool,
    objective_name: str,
) -> dict[str, Any]:
    """Surrogate-guided sampling using random forest (fallback when Ax unavailable)."""
    from sklearn.ensemble import RandomForestRegressor

    df = experiment_log.to_dataframe()
    var_names = [v.name for v in search_space.variables if v.name in df.columns]

    if len(var_names) == 0 or len(df) < 3:
        return _random_sample(search_space)

    X = df[var_names].apply(lambda x: x.astype(float, errors="ignore")).fillna(0).values
    y = df[objective_name].values

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)

    # Generate candidates and pick the best predicted
    from causal_optimizer.designer.factorial import FactorialDesigner
    designer = FactorialDesigner(search_space)
    candidates = designer.latin_hypercube(n_samples=100)

    best_candidate = None
    best_pred = float("inf") if minimize else float("-inf")

    for candidate in candidates:
        x = np.array([candidate.get(v, 0) for v in var_names]).reshape(1, -1)
        pred = rf.predict(x)[0]
        if (minimize and pred < best_pred) or (not minimize and pred > best_pred):
            best_pred = pred
            best_candidate = candidate

    return best_candidate or candidates[0]


def _get_focus_variables(
    search_space: SearchSpace,
    causal_graph: CausalGraph | None,
    objective_name: str,
) -> list[str]:
    """Identify variables to focus on, using causal graph if available."""
    if causal_graph is None:
        return search_space.variable_names

    # Find ancestors of the objective in the DAG
    ancestors: set[str] = set()
    frontier = {objective_name}
    while frontier:
        node = frontier.pop()
        for u, v in causal_graph.edges:
            if v == node and u not in ancestors:
                ancestors.add(u)
                frontier.add(u)

    # Intersect with search space variables
    focus = [v for v in search_space.variable_names if v in ancestors]
    return focus if focus else search_space.variable_names


def _random_sample(search_space: SearchSpace) -> dict[str, Any]:
    """Generate a random sample from the search space."""
    rng = np.random.default_rng()
    params: dict[str, Any] = {}
    for var in search_space.variables:
        if var.variable_type == VariableType.CONTINUOUS and var.lower is not None and var.upper is not None:
            params[var.name] = float(rng.uniform(var.lower, var.upper))
        elif var.variable_type == VariableType.INTEGER and var.lower is not None and var.upper is not None:
            params[var.name] = int(rng.integers(int(var.lower), int(var.upper) + 1))
        elif var.variable_type == VariableType.BOOLEAN:
            params[var.name] = bool(rng.choice([True, False]))
        elif var.variable_type == VariableType.CATEGORICAL and var.choices:
            params[var.name] = rng.choice(var.choices)
    return params
