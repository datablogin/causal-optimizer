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

from causal_optimizer.predictor.encoding import encode_dataframe_for_rf, encode_params_for_rf
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ObjectiveSpec,
    SearchSpace,
    Variable,
    VariableType,
)

logger = logging.getLogger(__name__)

#: Reserved metric key for scalarized multi-objective values.
#: Uses a leading underscore to avoid collision with user-defined metric names.
_SCALARIZED_KEY = "__scalarized_objective__"


def _derive_seed(seed: int | None, step: int) -> int | None:
    """Derive a step-specific seed from a base seed.

    Returns ``seed + step`` when *seed* is not ``None``, ensuring each call
    gets unique but deterministic randomness.  When *seed* is ``None`` the
    result is ``None`` (unseeded / non-deterministic).
    """
    return (seed + step) if seed is not None else None


def suggest_parameters(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph | None = None,
    phase: str = "exploration",
    minimize: bool = True,
    objective_name: str = "objective",
    screened_variables: list[str] | None = None,
    base_parameters: dict[str, Any] | None = None,
    pomis_sets: list[frozenset[str]] | None = None,
    objectives: list[ObjectiveSpec] | None = None,
    strategy: str = "bayesian",
    seed: int | None = None,
) -> dict[str, Any]:
    """Suggest next experiment parameters based on current phase and history.

    Args:
        screened_variables: Variables identified as important by screening.
            Complements graph-based focus variables (intersection if both available).
        base_parameters: Base parameters to perturb from (e.g., from MAP-Elites elite).
            Used in exploitation phase instead of the overall best.
        pomis_sets: POMIS intervention sets from causal graph analysis.
            When provided, constrains optimization to intervene on POMIS members.
        objectives: Multi-objective specifications. When provided with >1 entry,
            a scalarized objective is used for surrogate-based suggestion.
        strategy: Optimization strategy for the optimization phase.
            ``"bayesian"`` (default) or ``"causal_gp"`` (experimental CBO).
        seed: Random seed forwarded to strategy-specific optimizers for
            reproducibility.
    """
    if phase == "exploration":
        step_seed = _derive_seed(seed, len(experiment_log.results))
        return _suggest_exploration(search_space, experiment_log, seed=step_seed)

    # When multi-objective, scalarize the experiment log so the surrogate
    # has a single target to optimize.  The scalarized value is written to a
    # reserved key to avoid overwriting any user-defined objective metric.
    # Placed after the exploration early-return to avoid unnecessary work.
    # The key is cleaned up after suggestion to avoid leaking internal state
    # into user-visible data (e.g., ExperimentLog.to_dataframe()).
    surrogate_objective = objective_name
    # When using the scalarized key, the surrogate must always *minimize*
    # because _scalarize_log negates maximize objectives so lower = better.
    surrogate_minimize = minimize
    needs_cleanup = False
    if objectives is not None and len(objectives) > 1:
        _scalarize_log(experiment_log, objectives, _SCALARIZED_KEY)
        surrogate_objective = _SCALARIZED_KEY
        surrogate_minimize = True
        needs_cleanup = True

    try:
        if phase == "optimization":
            return _suggest_optimization(
                search_space,
                experiment_log,
                causal_graph,
                surrogate_minimize,
                surrogate_objective,
                screened_variables=screened_variables,
                pomis_sets=pomis_sets,
                strategy=strategy,
                seed=seed,
            )
        elif phase == "exploitation":
            focus_variables = _get_focus_variables(search_space, causal_graph, objective_name)
            step_seed = _derive_seed(seed, len(experiment_log.results))
            return _suggest_exploitation(
                search_space,
                experiment_log,
                surrogate_minimize,
                surrogate_objective,
                focus_variables=focus_variables,
                base_parameters=base_parameters,
                causal_graph=causal_graph,
                seed=step_seed,
            )
        else:
            step_seed = _derive_seed(seed, len(experiment_log.results))
            return _suggest_exploration(search_space, experiment_log, seed=step_seed)
    finally:
        if needs_cleanup:
            for result in experiment_log.results:
                result.metrics.pop(_SCALARIZED_KEY, None)


def _suggest_exploration(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    seed: int | None = None,
) -> dict[str, Any]:
    """Exploration: Latin Hypercube sampling for space-filling coverage."""
    from causal_optimizer.designer.factorial import FactorialDesigner

    designer = FactorialDesigner(search_space)
    designs = designer.latin_hypercube(n_samples=1, seed=seed)
    return designs[0] if designs else _random_sample(search_space, seed=seed)


def _suggest_optimization(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph | None,
    minimize: bool,
    objective_name: str,
    screened_variables: list[str] | None = None,
    pomis_sets: list[frozenset[str]] | None = None,
    strategy: str = "bayesian",
    seed: int | None = None,
) -> dict[str, Any]:
    """Optimization: Bayesian optimization with optional causal guidance.

    If a causal graph is available, uses it to identify which variables
    to prioritize (ancestors of the objective in the DAG). Screening results
    complement the graph-based focus.

    If POMIS sets are provided (from graphs with confounders), the optimizer
    selects the least-explored POMIS set and constrains suggestions to those
    variables.
    """
    # Derive seed once for all paths within this function.
    step_seed = _derive_seed(seed, len(experiment_log.results))

    df = experiment_log.to_dataframe()
    if len(df) < 3:
        return _suggest_exploration(search_space, experiment_log, seed=step_seed)

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
        focus_variables = screened_variables
    else:
        focus_variables = graph_focus

    # If POMIS sets available, use them to constrain intervention variables
    if pomis_sets is not None and len(pomis_sets) > 0:
        chosen_set = _select_pomis_set(pomis_sets, experiment_log)
        if chosen_set is not None:
            pomis_focus = [v for v in search_space.variable_names if v in chosen_set]
            if pomis_focus:
                # Intersect with existing focus (graph + screening)
                intersected = [v for v in pomis_focus if v in focus_variables]
                focus_variables = intersected if intersected else pomis_focus
                logger.info("POMIS constraining focus to: %s", focus_variables)

    # Route to causal_gp strategy if requested and a causal graph is available
    if strategy == "causal_gp" and causal_graph is not None:
        if pomis_sets:
            logger.info(
                "causal_gp strategy uses graph topology directly; "
                "POMIS focus constraints are not applied"
            )
        try:
            return _suggest_causal_gp(
                search_space,
                experiment_log,
                causal_graph,
                minimize,
                objective_name,
                seed=step_seed,
            )
        except ImportError:
            logger.info("botorch/gpytorch not available for causal_gp, falling back to bayesian")
        except Exception as exc:
            logger.warning("causal_gp surrogate failed (%s), falling back to bayesian", exc)
            # Fall through to bayesian / RF surrogate

    # Try Bayesian optimization via Ax
    try:
        return _suggest_bayesian(
            search_space,
            experiment_log,
            minimize,
            objective_name,
            focus_variables=focus_variables,
            pomis_sets=pomis_sets,
        )
    except ImportError:
        logger.info("Ax/BoTorch not available, using surrogate-guided sampling")
        return _suggest_surrogate(
            search_space,
            experiment_log,
            focus_variables,
            minimize,
            objective_name,
            causal_graph=causal_graph,
            seed=step_seed,
        )


def _suggest_exploitation(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    minimize: bool,
    objective_name: str,
    focus_variables: list[str] | None = None,
    base_parameters: dict[str, Any] | None = None,
    causal_graph: CausalGraph | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Exploitation: perturb the best known configuration or a provided base.

    If focus_variables is provided and non-empty, only perturb variables in
    that set, keeping others at their best-known values.
    If base_parameters is provided (e.g., from MAP-Elites), perturb those
    instead of the overall best.
    When *causal_graph* is provided, perturbation is biased toward direct
    parents of the objective (parents are weighted 7:3 over non-parents
    before normalization).
    """
    if base_parameters is not None:
        params = dict(base_parameters)
    else:
        best = experiment_log.best_result(objective_name, minimize)
        if best is None:
            return _random_sample(search_space, seed=seed)
        params = dict(best.parameters)

    rng = np.random.default_rng(seed)

    # Determine which variables are eligible for perturbation
    if focus_variables:
        eligible_vars = [
            (i, v) for i, v in enumerate(search_space.variables) if v.name in focus_variables
        ]
    else:
        eligible_vars = list(enumerate(search_space.variables))

    if not eligible_vars:
        eligible_vars = list(enumerate(search_space.variables))

    # Build weighted selection probabilities when a causal graph is available.
    # Direct parents of the objective get higher weight (70/30 split).
    if causal_graph is not None:
        parent_names = causal_graph.parents(objective_name)
        parent_focus = {name for name in parent_names if name in (focus_variables or [])}
        if parent_focus and len(parent_focus) < len(eligible_vars):
            weights = np.array([0.7 if v.name in parent_focus else 0.3 for _, v in eligible_vars])
            weights = weights / weights.sum()
        else:
            weights = None
    else:
        weights = None

    # Perturb one or two variables slightly
    n_perturb = rng.integers(1, min(3, len(eligible_vars)) + 1)
    chosen = rng.choice(
        len(eligible_vars),
        size=n_perturb,
        replace=False,
        p=weights,
    )

    for choice_idx in chosen:
        _idx, var = eligible_vars[choice_idx]
        is_continuous = var.variable_type == VariableType.CONTINUOUS
        is_integer = var.variable_type == VariableType.INTEGER
        has_bounds = var.lower is not None and var.upper is not None
        if is_continuous and has_bounds:
            current = params.get(var.name, (var.lower + var.upper) / 2)
            scale = (var.upper - var.lower) * 0.1  # 10% perturbation
            new_val = current + rng.normal(0, scale)
            params[var.name] = float(np.clip(new_val, var.lower, var.upper))
        elif is_integer and has_bounds:
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
    focus_variables: list[str] | None = None,
    pomis_sets: list[frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Bayesian optimization via Ax/BoTorch using :class:`AxBayesianOptimizer`.

    If focus_variables is provided and non-empty, only those variables are
    optimized; non-focus variables are fixed at their midpoint values.

    If pomis_sets is provided, the POMIS prior is forwarded to the optimizer
    so that candidates biased toward POMIS-only interventions are preferred.

    Raises
    ------
    ImportError
        Propagated from :class:`AxBayesianOptimizer` when ax-platform is not
        installed.  The caller (``_suggest_optimization``) catches this and
        falls back to the RF surrogate.

    Notes
    -----
    **Scalability**: a fresh ``AxBayesianOptimizer`` is created on every call
    and all historical results are replayed via ``update()`` in O(N) time.
    For typical experiment budgets (N < 200) this is negligible, but at larger
    scales consider caching the optimizer instance on the engine so history is
    loaded incrementally (one ``update()`` per new result rather than all N).

    **Double POMIS enforcement**: The caller (``_suggest_optimization``) may
    already constrain ``focus_variables`` to the POMIS intersection (hard
    constraint — non-focus vars always at midpoint).  Passing ``pomis_sets``
    here adds a second, *soft* layer: 80% of the time non-POMIS *active*
    variables are clamped to midpoint, and the remaining 20% explore the full
    active space.  The two levels are complementary: the hard constraint from
    ``focus_variables`` eliminates clearly irrelevant variables, while the
    soft POMIS prior guides Ax's acquisition function toward causally
    identified intervention sets.
    """
    from causal_optimizer.optimizer.bayesian import AxBayesianOptimizer

    optimizer = AxBayesianOptimizer(
        search_space=search_space,
        objective_name=objective_name,
        minimize=minimize,
        focus_variables=focus_variables if focus_variables else None,
        pomis_prior=pomis_sets,
    )

    # Feed historical data into the optimizer (O(N) replay — see Notes above)
    for result in experiment_log.results:
        if objective_name in result.metrics:
            optimizer.update(result.parameters, result.metrics[objective_name])

    return optimizer.suggest()


def _suggest_surrogate(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    focus_variables: list[str],
    minimize: bool,
    objective_name: str,
    causal_graph: CausalGraph | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Surrogate-guided sampling using random forest (fallback when Ax unavailable).

    If focus_variables is provided and non-empty, trains the RF model only on
    those features and only varies focus variables in candidates. Non-focus
    variables are held at their best-known values.

    When *causal_graph* is provided, half of the candidate budget (50 of 100)
    is replaced with "targeted intervention" candidates that perturb only 1-2
    direct parents of the objective, starting from the best-known configuration.
    This creates behavioral differentiation from surrogate_only mode (100 LHS).
    """
    from sklearn.ensemble import RandomForestRegressor

    df = experiment_log.to_dataframe()
    all_var_names = [v.name for v in search_space.variables if v.name in df.columns]

    if len(all_var_names) == 0 or len(df) < 3:
        return _random_sample(search_space, seed=seed)

    # Filter to focus variables for RF training; fall back to all if empty
    focus_var_names = [v for v in all_var_names if v in focus_variables] if focus_variables else []

    if not focus_var_names:
        focus_var_names = all_var_names

    # Get best-known values for non-focus variables
    best = experiment_log.best_result(objective_name, minimize)
    best_params = dict(best.parameters) if best else {}

    features = encode_dataframe_for_rf(df, focus_var_names, search_space)
    y = df[objective_name].values

    # Hardcoded random_state for RF training stability — the seed param
    # controls candidate generation (LHS) and fallback sampling, not the
    # surrogate model itself.
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(features, y)

    # Generate candidates
    from causal_optimizer.designer.factorial import FactorialDesigner

    designer = FactorialDesigner(search_space)

    if causal_graph is not None and best_params:
        # Causal mode: 50 LHS + 50 targeted intervention candidates
        candidates = designer.latin_hypercube(n_samples=50, seed=seed)
        targeted = _generate_targeted_candidates(
            search_space,
            best_params,
            causal_graph,
            objective_name,
            n_candidates=50,
            seed=seed,
        )
        candidates.extend(targeted)
    else:
        # Surrogate-only mode: 100 LHS candidates (original behavior)
        candidates = designer.latin_hypercube(n_samples=100, seed=seed)

    # For each candidate, hold non-focus variables at best-known values
    non_focus_vars = set(all_var_names) - set(focus_var_names)
    if non_focus_vars and best_params:
        for candidate in candidates:
            for var_name in non_focus_vars:
                if var_name in best_params:
                    candidate[var_name] = best_params[var_name]

    best_candidate = None
    best_pred = float("inf") if minimize else float("-inf")

    for candidate in candidates:
        x = encode_params_for_rf(candidate, focus_var_names, search_space)
        pred = rf.predict(x)[0]
        if (minimize and pred < best_pred) or (not minimize and pred > best_pred):
            best_pred = pred
            best_candidate = candidate

    return best_candidate or candidates[0]


def _generate_targeted_candidates(
    search_space: SearchSpace,
    best_params: dict[str, Any],
    causal_graph: CausalGraph,
    objective_name: str,
    n_candidates: int = 50,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate targeted intervention candidates by perturbing direct parents.

    Each candidate starts from *best_params*, randomly selects 1 or 2 direct
    parents of the objective, and perturbs only those variables.  This mirrors
    the perturbation logic in :func:`_suggest_exploitation`.

    Falls back to random samples if the graph has no parents of the objective
    that overlap with the search space.
    """
    rng = np.random.default_rng(seed if seed is None else seed + 7919)

    parents = causal_graph.parents(objective_name)
    # Filter to parents that are in the search space
    var_map = {v.name: v for v in search_space.variables}
    eligible_parents = sorted(name for name in parents if name in var_map)

    if not eligible_parents:
        # No usable parents — fall back to random samples
        from causal_optimizer.designer.factorial import FactorialDesigner

        designer = FactorialDesigner(search_space)
        return designer.latin_hypercube(n_samples=n_candidates, seed=seed)

    candidates: list[dict[str, Any]] = []
    for _ in range(n_candidates):
        candidate = dict(best_params)
        n_perturb = rng.integers(1, min(3, len(eligible_parents)) + 1)
        chosen_parents = rng.choice(eligible_parents, size=n_perturb, replace=False)

        for parent_name in chosen_parents:
            var = var_map[parent_name]
            _perturb_variable(candidate, var, rng)

        candidates.append(candidate)
    return candidates


def _perturb_variable(
    params: dict[str, Any],
    var: Variable,
    rng: np.random.Generator,
) -> None:
    """Perturb a single variable in-place.

    Similar to exploitation perturbation but with a wider continuous range
    (10-30% of range vs exploitation's fixed 10%) to ensure diversity among
    the 50 targeted candidates.  Integer: +/-1-2.  Boolean: flip with 30%
    probability.  Categorical: random choice with 30%.
    """
    is_continuous = var.variable_type == VariableType.CONTINUOUS
    is_integer = var.variable_type == VariableType.INTEGER
    has_bounds = var.lower is not None and var.upper is not None
    if is_continuous and has_bounds:
        assert var.lower is not None and var.upper is not None
        current = params.get(var.name, (var.lower + var.upper) / 2)
        # 10-30% of range perturbation
        pct = rng.uniform(0.1, 0.3)
        scale = (var.upper - var.lower) * pct
        new_val = current + rng.normal(0, scale)
        params[var.name] = float(np.clip(new_val, var.lower, var.upper))
    elif is_integer and has_bounds:
        assert var.lower is not None and var.upper is not None
        current = params.get(var.name, int((var.lower + var.upper) / 2))
        new_val = current + rng.integers(-2, 3)
        params[var.name] = int(np.clip(new_val, var.lower, var.upper))
    elif var.variable_type == VariableType.BOOLEAN:
        if rng.random() < 0.3:
            params[var.name] = not params.get(var.name, False)
    elif var.variable_type == VariableType.CATEGORICAL and var.choices:
        if rng.random() < 0.3:
            params[var.name] = rng.choice(var.choices)


def _get_focus_variables(
    search_space: SearchSpace,
    causal_graph: CausalGraph | None,
    objective_name: str,
) -> list[str]:
    """Identify variables to focus on, using causal graph if available."""
    if causal_graph is None:
        return search_space.variable_names

    ancestors = causal_graph.ancestors(objective_name)
    focus = [v for v in search_space.variable_names if v in ancestors]
    return focus if focus else search_space.variable_names


def _select_pomis_set(
    pomis_sets: list[frozenset[str]],
    experiment_log: ExperimentLog,
) -> frozenset[str] | None:
    """Select which POMIS set to explore next (round-robin strategy).

    Uses a deterministic round-robin based on experiment count to ensure
    each POMIS set gets equal exploration over time.
    """
    if not pomis_sets:
        return None

    # Count only optimization-phase experiments for balanced round-robin
    opt_count = sum(1 for r in experiment_log.results if r.metadata.get("phase") == "optimization")
    idx = opt_count % len(pomis_sets)
    return pomis_sets[idx]


def _suggest_causal_gp(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph,
    minimize: bool,
    objective_name: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Causal GP surrogate: separate GP per mechanism, interventional EI acquisition.

    Raises
    ------
    ImportError
        If botorch/gpytorch is not installed.
    """
    from causal_optimizer.optimizer.causal_gp import CausalGPSurrogate

    surrogate = CausalGPSurrogate(
        search_space=search_space,
        causal_graph=causal_graph,
        objective_name=objective_name,
        minimize=minimize,
        seed=seed,
    )
    surrogate.fit(experiment_log)
    return surrogate.suggest()


def _random_sample(search_space: SearchSpace, seed: int | None = None) -> dict[str, Any]:
    """Generate a random sample from the search space."""
    rng = np.random.default_rng(seed)
    params: dict[str, Any] = {}
    for var in search_space.variables:
        is_cont = var.variable_type == VariableType.CONTINUOUS
        is_int = var.variable_type == VariableType.INTEGER
        has_bounds = var.lower is not None and var.upper is not None
        if is_cont and has_bounds:
            assert var.lower is not None and var.upper is not None
            params[var.name] = float(rng.uniform(var.lower, var.upper))
        elif is_int and has_bounds:
            assert var.lower is not None and var.upper is not None
            params[var.name] = int(rng.integers(int(var.lower), int(var.upper) + 1))
        elif var.variable_type == VariableType.BOOLEAN:
            params[var.name] = bool(rng.choice([True, False]))
        elif var.variable_type == VariableType.CATEGORICAL and var.choices:
            params[var.name] = rng.choice(var.choices)
    return params


def _scalarize_log(
    experiment_log: ExperimentLog,
    objectives: list[ObjectiveSpec],
    target_name: str,
) -> None:
    """Add a scalarized objective to each result's metrics for surrogate training.

    Uses a weighted sum of (possibly sign-flipped) objectives so the surrogate
    always *minimizes* the scalar target.  Weights come from
    :attr:`ObjectiveSpec.weight`.  Objectives with ``minimize=False`` are negated
    before summing so that maximization objectives contribute correctly.

    The scalarized value is written to ``result.metrics[target_name]`` for every
    result in the log.  ``target_name`` should be a reserved key (e.g.
    :data:`_SCALARIZED_KEY`) that does not collide with any user-defined
    objective name, so that original metric values are never overwritten.
    """
    for result in experiment_log.results:
        scalar = 0.0
        for obj in objectives:
            # Missing metrics default to a large finite worst-case value for
            # the direction, consistent with ParetoResult.dominated_by semantics.
            # We use 1e10 instead of inf to keep surrogate training numerically
            # stable (RF trees handle finite extremes better than inf).
            worst = 1e10 if obj.minimize else -1e10
            val = result.metrics.get(obj.name, worst)
            # Negate maximize objectives so the surrogate always minimizes
            scalar += obj.weight * (val if obj.minimize else -val)
        result.metrics[target_name] = scalar
