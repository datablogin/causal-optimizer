"""Parameter suggestion strategies for the optimization loop.

Implements a progression of strategies:
- Exploration phase: DoE-based designs for screening (with optional causal weighting)
- Optimization phase: Bayesian optimization (with soft causal ranking)
- Exploitation phase: local search around best known configuration

Sprint 19 additions — softer causal influence:
- Causal-weighted exploration biases LHS toward graph ancestors without
  eliminating non-ancestor exploration (controlled by ``causal_exploration_weight``).
- Soft ranking during optimization trains the RF on ALL variables and adds a
  causal alignment bonus instead of hard focus-variable pinning
  (controlled by ``causal_softness``).
- Adaptive targeted candidate rebalancing adjusts LHS/targeted ratio based on
  experiment count within the optimization phase.

Sprint 22 — alignment-only re-ranking confirmed as production default:
- Sprint 20's balanced re-ranking (objective quality + causal alignment composite)
  was removed after Sprint 21's locked A/B comparison showed alignment-only
  re-ranking produces equal or better results on all benchmarks.
- The ``CAUSAL_OPT_RERANKING_MODE`` env-var toggle has been removed.
- ``_rerank_alignment_only`` is now the sole Ax candidate re-ranking function.

Backward compatibility:
- ``causal_exploration_weight=0.0`` + ``causal_softness=inf`` recovers Sprint 18 behavior.
- Without a causal graph, all behavior is identical to Sprint 18.
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

# --- Candidate generation constants ---
#: Total candidate budget for surrogate-guided search.
_TOTAL_CANDIDATES = 100
#: In surrogate-only mode, use 100 pure LHS candidates.
_SURROGATE_ONLY_CANDIDATES = 100
#: Number of LHS candidates to generate during causal-weighted exploration.
_EXPLORATION_LHS_CANDIDATES = 20
#: Parent vs non-parent weight ratio in causal exploitation (before normalization).
_PARENT_WEIGHT = 0.7
_NON_PARENT_WEIGHT = 0.3
#: Continuous perturbation scale for exploitation (fixed 10% of range).
_EXPLOITATION_SCALE = 0.1
#: Default period for categorical sweep during exploitation (every N-th step).
_CATEGORICAL_SWEEP_PERIOD = 5
#: Continuous perturbation scale range for targeted candidates (10-30% of range).
_TARGETED_SCALE_RANGE = (0.1, 0.3)
#: Seed offset (prime) for targeted candidate generation to avoid LHS seed collision.
_TARGETED_SEED_OFFSET = 7919
#: Backward-compatible aliases for the old fixed 50/50 candidate split constants.
#: Sprint 19 replaced these with adaptive ratios via :func:`_get_targeted_ratio`,
#: but they are retained for external test imports.
_CAUSAL_LHS_CANDIDATES = 50
_CAUSAL_TARGETED_CANDIDATES = 50
#: Threshold above which ``causal_softness`` is treated as hard focus (Sprint 18
#: behavior).  ``causal_softness >= _HARD_FOCUS_THRESHOLD`` pins non-focus vars.
_HARD_FOCUS_THRESHOLD = 1e5
#: Variable types eligible for normalized diversity scoring.
_NUMERIC_TYPES = frozenset({VariableType.CONTINUOUS, VariableType.INTEGER})


def _normalize_value(var: Variable, value: float) -> float:
    """Normalize *value* to [0, 1] using the variable's bounds.

    Returns 0.0 for variables without finite bounds or zero-width ranges.
    """
    if var.lower is None or var.upper is None:
        return 0.0
    rng_size = var.upper - var.lower
    if rng_size <= 0:
        return 0.0
    return (value - var.lower) / rng_size


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
    causal_exploration_weight: float = 0.0,
    causal_softness: float = 0.5,
    pomis_minimal_focus: bool = False,
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
        causal_exploration_weight: Strength of causal bias during exploration
            (0.0 = pure LHS, higher = more ancestor emphasis). Default 0.0
            (changed from 0.3 in Sprint 29 after ablation evidence).
        causal_softness: Strength of causal alignment bonus during optimization
            (0.0 = no bonus, large = approximates hard focus). Default 0.5.
        pomis_minimal_focus: Sprint 37 Option A1 flag. When ``True``, the
            focus helper may restrict optimization and exploitation to
            ``screened_variables ∩ ancestors`` whenever (a) a causal graph
            is supplied, (b) every search variable is an ancestor of
            ``objective_name``, and (c) the resulting intersection is a
            non-empty proper subset of the search space. When ``False``
            (the default), behavior is preserved exactly. The Open
            Bandit benchmark harness sets this to ``True`` only for the
            ``causal`` arm.
    """
    if phase == "exploration":
        step_seed = _derive_seed(seed, len(experiment_log.results))
        return _suggest_exploration(
            search_space,
            experiment_log,
            causal_graph=causal_graph,
            objective_name=objective_name,
            causal_exploration_weight=causal_exploration_weight,
            seed=step_seed,
        )

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
                causal_softness=causal_softness,
                pomis_minimal_focus=pomis_minimal_focus,
            )
        elif phase == "exploitation":
            # Apply A1 here so B80 (which crosses the >= 50 exploitation
            # boundary) does not silently revert to full-space perturbation.
            base_focus = _get_focus_variables(search_space, causal_graph, objective_name)
            focus_variables = _apply_minimal_focus_a1(
                base_focus=base_focus,
                search_space=search_space,
                causal_graph=causal_graph,
                objective_name=objective_name,
                screened_variables=screened_variables,
                enable=pomis_minimal_focus,
            )
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
            return _suggest_exploration(
                search_space,
                experiment_log,
                causal_graph=causal_graph,
                objective_name=objective_name,
                causal_exploration_weight=causal_exploration_weight,
                seed=step_seed,
            )
    finally:
        if needs_cleanup:
            for result in experiment_log.results:
                result.metrics.pop(_SCALARIZED_KEY, None)


def _suggest_exploration(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    causal_graph: CausalGraph | None = None,
    objective_name: str = "objective",
    causal_exploration_weight: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Exploration: Latin Hypercube sampling with optional causal weighting.

    When a causal graph is provided and ``causal_exploration_weight > 0``,
    generates multiple LHS candidates and selects the one that best
    emphasizes ancestor-variable diversity relative to existing experiments.

    With ``causal_exploration_weight=0.0`` or no graph, falls back to
    single-sample LHS (Sprint 18 behavior).

    Args:
        causal_graph: Optional causal graph for ancestor identification.
        objective_name: Name of the objective node in the graph.
        causal_exploration_weight: Strength of ancestor bias (alpha).
            0.0 = pure LHS, higher = more ancestor emphasis.
        seed: Random seed for LHS generation.
    """
    from causal_optimizer.designer.factorial import FactorialDesigner

    designer = FactorialDesigner(search_space)

    def _single_lhs() -> dict[str, Any]:
        designs = designer.latin_hypercube(n_samples=1, seed=seed)
        return designs[0] if designs else _random_sample(search_space, seed=seed)

    # No graph or weight=0: original behavior (single LHS sample)
    if causal_graph is None or causal_exploration_weight <= 0.0:
        return _single_lhs()

    # Causal-weighted exploration: generate N candidates, score, pick best
    ancestors = causal_graph.ancestors(objective_name)
    ancestor_names = {v for v in search_space.variable_names if v in ancestors}

    if not ancestor_names:
        return _single_lhs()

    candidates = designer.latin_hypercube(n_samples=_EXPLORATION_LHS_CANDIDATES, seed=seed)
    if not candidates:
        return _random_sample(search_space, seed=seed)

    # Gather existing experiment parameters for diversity scoring (read-only)
    existing_params = [r.parameters for r in experiment_log.results]
    # Build var_map once — reused across all candidate scoring calls.
    var_map = {v.name: v for v in search_space.variables}

    best_candidate = candidates[0]
    best_score = -float("inf")
    for candidate in candidates:
        score = _score_candidate_causal_exploration(
            candidate=candidate,
            existing_params=existing_params,
            ancestor_names=ancestor_names,
            search_space=search_space,
            alpha=causal_exploration_weight,
            var_map=var_map,
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _score_candidate_causal_exploration(
    candidate: dict[str, Any],
    existing_params: list[dict[str, Any]],
    ancestor_names: set[str],
    search_space: SearchSpace,
    alpha: float = 0.3,
    var_map: dict[str, Variable] | None = None,
) -> float:
    """Score an LHS candidate for causal-weighted exploration.

    Computes ``base_diversity + alpha * ancestor_diversity`` where:

    - **base_diversity** measures how far the candidate is from existing
      experiments across all dimensions (min-distance in normalized space).
    - **ancestor_diversity** measures the same but only over ancestor
      dimensions.

    Higher scores indicate candidates that fill unexplored space, with
    a bonus for exploring ancestor variables.

    Args:
        candidate: Proposed parameter dict.
        existing_params: Parameter dicts from prior experiments.
        ancestor_names: Names of variables that are causal ancestors.
        search_space: Search space for normalization bounds.
        alpha: Weight for ancestor diversity bonus.
        var_map: Pre-built ``{name: Variable}`` dict.  If *None*, built
            from *search_space* (slower when called in a loop).

    Returns:
        Combined diversity score (non-negative).
    """
    if var_map is None:
        var_map = {v.name: v for v in search_space.variables}
    # Only score numeric variables -- categorical/boolean can't be normalized.
    numeric_vars = [
        v
        for v in search_space.variables
        if v.variable_type in _NUMERIC_TYPES and v.lower is not None and v.upper is not None
    ]
    numeric_var_names = [v.name for v in numeric_vars]
    ancestor_numeric = [n for n in numeric_var_names if n in ancestor_names]

    if not numeric_var_names:
        return 1.0  # No numeric variables to score

    if not existing_params:
        ancestor_dims = [
            _normalize_value(var_map[v], float(candidate.get(v, 0.0))) for v in ancestor_numeric
        ]
        return 1.0 + alpha * float(np.std(ancestor_dims)) if ancestor_dims else 1.0

    min_base_dist = float("inf")
    min_ancestor_dist = float("inf")

    for existing in existing_params:
        base_diffs: list[float] = []
        ancestor_diffs: list[float] = []
        for v_name in numeric_var_names:
            var = var_map[v_name]
            c_val = _normalize_value(var, float(candidate.get(v_name, 0.0)))
            e_val = _normalize_value(var, float(existing.get(v_name, 0.0)))
            diff = abs(c_val - e_val)
            base_diffs.append(diff)
            if v_name in ancestor_names:
                ancestor_diffs.append(diff)

        base_dist = float(np.mean(base_diffs)) if base_diffs else 0.0
        ancestor_dist = float(np.mean(ancestor_diffs)) if ancestor_diffs else 0.0

        min_base_dist = min(min_base_dist, base_dist)
        min_ancestor_dist = min(min_ancestor_dist, ancestor_dist)

    if min_base_dist == float("inf"):
        min_base_dist = 0.0
    if min_ancestor_dist == float("inf"):
        min_ancestor_dist = 0.0

    return min_base_dist + alpha * min_ancestor_dist


def _get_targeted_ratio(experiment_count: int) -> float:
    """Compute adaptive targeted-candidate ratio based on experiment count.

    Returns the fraction of the candidate budget allocated to targeted
    (parent-perturbation) candidates during the optimization phase.

    The ratio ramps linearly from 0.3 (early optimization, experiment 10)
    to 0.7 (late optimization, experiment 50):

    - Experiment 10: 30% targeted / 70% LHS
    - Experiment 25: 50% targeted / 50% LHS (midpoint)
    - Experiment 50: 70% targeted / 30% LHS

    Args:
        experiment_count: Current total number of experiments.

    Returns:
        Targeted ratio in [0.3, 0.7].
    """
    # Clamp to optimization range [10, 50]
    lo, hi = 10, 50
    ratio_lo, ratio_hi = 0.3, 0.7
    clamped = max(lo, min(hi, experiment_count))
    t = (clamped - lo) / (hi - lo)  # 0.0 at experiment 10, 1.0 at experiment 50
    return ratio_lo + t * (ratio_hi - ratio_lo)


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
    causal_softness: float = 0.5,
    pomis_minimal_focus: bool = False,
) -> dict[str, Any]:
    """Optimization: Bayesian optimization with optional soft causal guidance.

    If a causal graph is available, uses it to identify which variables
    to prioritize (ancestors of the objective in the DAG). Screening results
    complement the graph-based focus.

    Sprint 19: When ``causal_softness`` is finite, the RF surrogate trains
    on ALL variables (not just focus vars) and applies a soft causal
    alignment bonus instead of hard focus-variable pinning.

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

    # When the A1 flag is on the helper may restrict ``focus_variables``
    # further; otherwise it returns the input unchanged.
    focus_variables = _apply_minimal_focus_a1(
        base_focus=focus_variables,
        search_space=search_space,
        causal_graph=causal_graph,
        objective_name=objective_name,
        screened_variables=screened_variables,
        enable=pomis_minimal_focus,
    )

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
            causal_softness=causal_softness,
            causal_graph=causal_graph,
            seed=step_seed,
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
            causal_softness=causal_softness,
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
    categorical_sweep_period: int = _CATEGORICAL_SWEEP_PERIOD,
) -> dict[str, Any]:
    """Exploitation: perturb the best known configuration or a provided base.

    If focus_variables is provided and non-empty, only perturb variables in
    that set, keeping others at their best-known values.
    If base_parameters is provided (e.g., from MAP-Elites), perturb those
    instead of the overall best.
    When *causal_graph* is provided, perturbation is biased toward direct
    parents of the objective (parents are weighted 7:3 over non-parents
    before normalization).

    Sprint 25 categorical sweep: every *categorical_sweep_period* exploitation
    steps, instead of the normal 1-2-variable perturbation, generate one
    candidate per categorical value (holding continuous parameters at the
    incumbent) and return the best by RF-predicted objective.  This gives
    bad seeds a guaranteed opportunity to escape categorical lock-in.
    """
    if base_parameters is not None:
        params = dict(base_parameters)
    else:
        best = experiment_log.best_result(objective_name, minimize)
        if best is None:
            return _random_sample(search_space, seed=seed)
        params = dict(best.parameters)

    rng = np.random.default_rng(seed)

    # Sprint 25: periodic categorical sweep.
    # step_count uses total log length (all phases), not exploitation-only steps.
    # With default phase boundaries (exploitation starts at step 50) and period=5,
    # the sweep fires on steps 50, 55, 60, 65, 70, 75 — giving 6 sweep
    # opportunities across 30 exploitation steps.  This coupling is acceptable
    # because the period is short enough that small shifts in phase boundaries
    # do not meaningfully change the sweep frequency.
    step_count = len(experiment_log.results)
    cat_vars = [
        v
        for v in search_space.variables
        if v.variable_type == VariableType.CATEGORICAL and v.choices
    ]
    if cat_vars and categorical_sweep_period > 0 and step_count % categorical_sweep_period == 0:
        sweep_result = _categorical_sweep(
            params, cat_vars, search_space, experiment_log, objective_name, minimize, rng
        )
        if sweep_result is not None:
            return sweep_result

    # --- Normal perturbation path (unchanged from Sprint 24) ---

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
        eligible_var_names = {v.name for _, v in eligible_vars}
        parent_focus = {name for name in parent_names if name in eligible_var_names}
        if parent_focus and len(parent_focus) < len(eligible_vars):
            weights = np.array(
                [
                    _PARENT_WEIGHT if v.name in parent_focus else _NON_PARENT_WEIGHT
                    for _, v in eligible_vars
                ]
            )
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
        _perturb_variable(params, var, rng, continuous_scale=_EXPLOITATION_SCALE)

    return params


def _categorical_sweep(
    incumbent: dict[str, Any],
    cat_vars: list[Variable],
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    objective_name: str,
    minimize: bool,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Generate one candidate per categorical value per variable and pick the best.

    Holds continuous parameters at the incumbent values.  For each categorical
    variable, creates one candidate per possible value (not the Cartesian
    product across variables — that would explode combinatorially).  Uses a
    lightweight RF surrogate to predict objective values and returns the
    candidate with the best predicted objective.

    Returns None if the RF cannot be fitted (too few results), falling through
    to the normal perturbation path.
    """
    # Build candidate set: one per categorical value per variable
    candidates: list[dict[str, Any]] = []
    for var in cat_vars:
        for value in var.choices or []:
            candidate = dict(incumbent)
            candidate[var.name] = value
            candidates.append(candidate)

    if not candidates:
        return None

    # Deduplicate (multiple cat vars may produce identical candidates)
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []
    for c in candidates:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    if len(candidates) == 1:
        return candidates[0]

    # Fit a lightweight RF on experiment history to predict objective.
    # Filter to rows with a valid (non-NaN) objective value so that CRASH
    # or partial results don't poison the RF fit.
    from sklearn.ensemble import RandomForestRegressor

    df = experiment_log.to_dataframe()
    var_names = [v.name for v in search_space.variables if v.name in df.columns]

    if len(var_names) == 0 or len(df) < 5:
        return None

    # Drop rows where objective is missing or NaN
    if objective_name not in df.columns:
        return None
    valid_mask = df[objective_name].notna()
    df = df.loc[valid_mask].reset_index(drop=True)

    if len(df) < 5:
        return None

    try:
        x_train = encode_dataframe_for_rf(df, var_names, search_space)
        y_train = df[objective_name].values
    except (ValueError, KeyError):
        return None

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=int(rng.integers(10**9)))
    rf.fit(x_train, y_train)

    # Predict objective for each candidate
    best_candidate = candidates[0]
    best_score: float = float("-inf") if not minimize else float("inf")

    for candidate in candidates:
        try:
            x = encode_params_for_rf(candidate, var_names, search_space)
            pred = float(rf.predict(x.reshape(1, -1))[0])
        except (ValueError, KeyError):
            continue

        if (not minimize and pred > best_score) or (minimize and pred < best_score):
            best_score = pred
            best_candidate = candidate

    return dict(best_candidate)


def _suggest_bayesian(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    minimize: bool,
    objective_name: str,
    focus_variables: list[str] | None = None,
    pomis_sets: list[frozenset[str]] | None = None,
    causal_softness: float = 0.5,
    causal_graph: CausalGraph | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bayesian optimization via Ax/BoTorch using :class:`AxBayesianOptimizer`.

    If focus_variables is provided and non-empty, only those variables are
    optimized; non-focus variables are fixed at their midpoint values.

    If pomis_sets is provided, the POMIS prior is forwarded to the optimizer
    so that candidates biased toward POMIS-only interventions are preferred.

    Sprint 19 soft-causal mode (``causal_softness < _HARD_FOCUS_THRESHOLD``):
    - Ax optimizes ALL variables (``focus_variables`` is not passed).
    - Multiple candidates are generated, then re-ranked by causal alignment
      score via :func:`_rerank_alignment_only` (weighted by ``causal_softness``).

    Hard-focus backward compatibility (``causal_softness >= _HARD_FOCUS_THRESHOLD``):
    - Only ``focus_variables`` are optimized by Ax; others fixed at midpoint.
    - Equivalent to Sprint 18 behavior.

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

    # Determine soft vs hard mode
    use_soft = causal_graph is not None and causal_softness < _HARD_FOCUS_THRESHOLD

    # In soft mode, let Ax optimize ALL variables (no focus restriction);
    # in hard mode, restrict to focus_variables (Sprint 18 behavior).
    ax_focus = None if use_soft else (focus_variables if focus_variables else None)

    optimizer = AxBayesianOptimizer(
        search_space=search_space,
        objective_name=objective_name,
        minimize=minimize,
        focus_variables=ax_focus,
        pomis_prior=pomis_sets,
        seed=seed,
    )

    # Feed historical data into the optimizer (O(N) replay — see Notes above)
    for result in experiment_log.results:
        if objective_name in result.metrics:
            optimizer.update(result.parameters, result.metrics[objective_name])

    # Hard mode or no graph: single candidate, no re-ranking needed
    if not use_soft:
        return optimizer.suggest()

    # Soft mode: generate multiple candidates, re-rank with causal alignment bonus.
    # Identify ancestor variables for alignment scoring.
    assert causal_graph is not None  # use_soft guarantees causal_graph is not None
    ancestors = causal_graph.ancestors(objective_name)
    all_var_names = [v.name for v in search_space.variables]
    ancestor_names = {v for v in all_var_names if v in ancestors}

    # Get best-known parameters for alignment scoring baseline
    best = experiment_log.best_result(objective_name, minimize)
    best_params = dict(best.parameters) if best else {}

    # Generate several candidates and pick the best adjusted score.
    # 5 candidates balances diversity with Ax overhead (each creates a GP trial).
    n_candidates = 5
    candidates: list[dict[str, Any]] = []
    for _ in range(n_candidates):
        candidates.append(optimizer.suggest())

    # If no ancestors or no best params, just return the first candidate
    if not ancestor_names or not best_params:
        return candidates[0]

    # Sprint 24: inject diversity candidates so every value of every
    # categorical variable appears in the batch.  This prevents the
    # GP model's categorical preference from locking out alternatives
    # (e.g., treat_day_filter="weekday" excluding "all").
    # Placed after the early-return guard because diversity candidates
    # are only useful when the alignment-only reranker has ancestors
    # to score by.
    candidates = inject_categorical_diversity(candidates, search_space)

    return _rerank_alignment_only(
        candidates,
        best_params,
        ancestor_names,
        search_space,
        causal_softness,
    )


def _rerank_alignment_only(
    candidates: list[dict[str, Any]],
    best_params: dict[str, Any],
    ancestor_names: set[str],
    search_space: SearchSpace,
    causal_softness: float,
) -> dict[str, Any]:
    """Re-rank Ax candidates by causal alignment score.

    Ranks candidates purely by ``causal_softness * alignment``, ignoring
    predicted objective quality.  This is the production re-ranking path
    for the Ax Bayesian optimizer in soft-causal mode.

    Sprint 21 A/B testing showed alignment-only re-ranking produces equal
    or better results than balanced (objective + alignment) re-ranking on
    all benchmarks.  Sprint 22 removed the balanced path and made this
    the sole production re-ranking function.
    """
    if len(candidates) <= 1:
        return candidates[0]
    best_candidate = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        alignment = _causal_alignment_score(candidate, best_params, ancestor_names, search_space)
        adjusted = causal_softness * alignment
        if adjusted > best_score:
            best_score = adjusted
            best_candidate = candidate
    return best_candidate


def _suggest_surrogate(
    search_space: SearchSpace,
    experiment_log: ExperimentLog,
    focus_variables: list[str],
    minimize: bool,
    objective_name: str,
    causal_graph: CausalGraph | None = None,
    seed: int | None = None,
    causal_softness: float = 0.5,
) -> dict[str, Any]:
    """Surrogate-guided sampling using random forest (fallback when Ax unavailable).

    Sprint 19 soft-ranking mode (``causal_softness`` finite):
    - Trains the RF on ALL variables (not just focus vars).
    - Generates candidates across the FULL search space.
    - Scores candidates as ``predicted_value + beta * causal_alignment``
      where ``causal_alignment`` measures ancestor-variable variation.
    - Non-focus variables are NOT pinned to best values (soft constraint).

    Hard-focus backward compatibility (``causal_softness >= _HARD_FOCUS_THRESHOLD``):
    - Trains RF only on focus variables.
    - Pins non-focus variables to best-known values.
    - Equivalent to Sprint 18 behavior.

    When *causal_graph* is provided, a portion of the candidate budget is
    replaced with "targeted intervention" candidates that perturb direct
    parents of the objective. The LHS/targeted split adapts based on
    experiment count via :func:`_get_targeted_ratio`.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = experiment_log.to_dataframe()
    all_var_names = [v.name for v in search_space.variables if v.name in df.columns]

    if len(all_var_names) == 0 or len(df) < 3:
        return _random_sample(search_space, seed=seed)

    # Determine if we use soft or hard mode
    use_soft = causal_graph is not None and causal_softness < _HARD_FOCUS_THRESHOLD

    # Filter to focus variables; fall back to all if empty
    focus_var_names = [v for v in all_var_names if v in focus_variables] if focus_variables else []
    if not focus_var_names:
        focus_var_names = all_var_names

    # Get best-known values for non-focus variables
    best = experiment_log.best_result(objective_name, minimize)
    best_params = dict(best.parameters) if best else {}

    # Sprint 19: in soft mode, train RF on ALL variables
    train_var_names = all_var_names if use_soft else focus_var_names

    features = encode_dataframe_for_rf(df, train_var_names, search_space)
    y = df[objective_name].values

    # Hardcoded random_state for RF training stability — the seed param
    # controls candidate generation (LHS) and fallback sampling, not the
    # surrogate model itself.
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(features, y)

    # Identify ancestor names for causal alignment scoring
    ancestor_names: set[str] = set()
    if causal_graph is not None:
        ancestors = causal_graph.ancestors(objective_name)
        ancestor_names = {v for v in all_var_names if v in ancestors}

    # Generate candidates
    from causal_optimizer.designer.factorial import FactorialDesigner

    designer = FactorialDesigner(search_space)
    n_experiments = len(experiment_log.results)

    if causal_graph is not None and best_params:
        # Adaptive LHS/targeted split based on experiment count
        targeted_ratio = _get_targeted_ratio(n_experiments)
        n_targeted = max(1, int(_TOTAL_CANDIDATES * targeted_ratio))
        n_lhs = _TOTAL_CANDIDATES - n_targeted

        candidates = designer.latin_hypercube(n_samples=n_lhs, seed=seed)
        targeted = _generate_targeted_candidates(
            search_space,
            best_params,
            causal_graph,
            objective_name,
            focus_var_names=focus_var_names,
            n_candidates=n_targeted,
            seed=seed,
        )
        candidates.extend(targeted)
    else:
        # Surrogate-only mode: pure LHS candidates (original behavior)
        candidates = designer.latin_hypercube(n_samples=_SURROGATE_ONLY_CANDIDATES, seed=seed)

    # Hard mode: pin non-focus variables to best-known values (Sprint 18 behavior)
    if not use_soft:
        non_focus_vars = set(all_var_names) - set(focus_var_names)
        if non_focus_vars and best_params:
            for candidate in candidates:
                for var_name in non_focus_vars:
                    if var_name in best_params:
                        candidate[var_name] = best_params[var_name]

    # Score candidates
    best_candidate = None
    best_score = float("inf") if minimize else float("-inf")

    for candidate in candidates:
        x = encode_params_for_rf(candidate, train_var_names, search_space)
        pred = float(rf.predict(x)[0])

        if use_soft and ancestor_names and best_params:
            # Compute causal alignment bonus: measures how much the candidate
            # varies ancestor variables relative to the current best
            alignment = _causal_alignment_score(
                candidate, best_params, ancestor_names, search_space
            )
            # For minimization: lower pred is better, higher alignment is better
            # adjusted = pred - beta * alignment (subtract bonus to make it "lower")
            # For maximization: higher pred is better, higher alignment is better
            # adjusted = pred + beta * alignment
            if minimize:
                adjusted = pred - causal_softness * alignment
            else:
                adjusted = pred + causal_softness * alignment
        else:
            adjusted = pred

        is_better = adjusted < best_score if minimize else adjusted > best_score
        if is_better:
            best_score = adjusted
            best_candidate = candidate

    return best_candidate or candidates[0]


def _causal_alignment_score(
    candidate: dict[str, Any],
    best_params: dict[str, Any],
    ancestor_names: set[str],
    search_space: SearchSpace,
) -> float:
    """Compute causal alignment score for a candidate during optimization.

    Measures how much the candidate explores ancestor dimensions relative to
    the current best configuration. Higher scores indicate candidates that
    vary ancestor variables more.

    The score is normalized to [0, 1] by dividing by the number of ancestor
    dimensions, so it represents the average normalized displacement along
    ancestor axes.

    Args:
        candidate: Proposed parameter dict.
        best_params: Best-known parameter dict.
        ancestor_names: Names of causal ancestor variables.
        search_space: Search space for normalization.

    Returns:
        Mean normalized displacement along ancestor dimensions.
    """
    var_map = {v.name: v for v in search_space.variables}
    diffs: list[float] = []

    for name in sorted(ancestor_names):
        var = var_map.get(name)
        if var is None or var.lower is None or var.upper is None:
            continue
        if (var.upper - var.lower) <= 0:
            continue
        mid = (var.lower + var.upper) / 2
        c_norm = _normalize_value(var, float(candidate.get(name, mid)))
        b_norm = _normalize_value(var, float(best_params.get(name, mid)))
        diffs.append(abs(c_norm - b_norm))

    return float(np.mean(diffs)) if diffs else 0.0


def _generate_targeted_candidates(
    search_space: SearchSpace,
    best_params: dict[str, Any],
    causal_graph: CausalGraph,
    objective_name: str,
    focus_var_names: list[str] | None = None,
    n_candidates: int = 50,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate targeted intervention candidates by perturbing direct parents.

    Each candidate starts from *best_params*, randomly selects 1 or 2 direct
    parents of the objective, and perturbs only those variables.  Uses a wider
    perturbation range (10-30%) than exploitation (fixed 10%) to ensure
    diversity among the targeted candidates.

    Only parents that are both in the search space *and* in *focus_var_names*
    are eligible for perturbation, so that non-focus variable pinning
    (applied downstream) does not silently revert perturbations.

    Falls back to random samples if no eligible parents remain.
    """
    rng = np.random.default_rng(seed if seed is None else seed + _TARGETED_SEED_OFFSET)

    parents = causal_graph.parents(objective_name)
    # Filter to parents that are in both the search space and focus variables
    var_map = {v.name: v for v in search_space.variables}
    focus_set = set(focus_var_names) if focus_var_names else set(var_map.keys())
    eligible_parents = sorted(name for name in parents if name in var_map and name in focus_set)

    if not eligible_parents:
        # No usable parents — fall back to LHS samples with offset seed so
        # these candidates are distinct from the first LHS batch generated
        # by _suggest_surrogate (which uses the raw seed).
        from causal_optimizer.designer.factorial import FactorialDesigner  # noqa: F811

        fallback_seed = (seed + _TARGETED_SEED_OFFSET) if seed is not None else None
        return FactorialDesigner(search_space).latin_hypercube(
            n_samples=n_candidates, seed=fallback_seed
        )

    candidates: list[dict[str, Any]] = []
    for _ in range(n_candidates):
        candidate = dict(best_params)
        n_perturb = rng.integers(1, min(2, len(eligible_parents)) + 1)
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
    continuous_scale: float | tuple[float, float] = _TARGETED_SCALE_RANGE,
) -> None:
    """Perturb a single variable in-place.

    Used by both exploitation (fixed scale) and targeted candidate generation
    (random scale range) to keep perturbation logic in one place.

    Args:
        continuous_scale: Fraction of the variable range used as the std-dev
            for Gaussian perturbation.  A single float gives a fixed scale
            (e.g. 0.1 = 10%).  A (lo, hi) tuple draws a uniform random
            scale per call to increase diversity among candidates.
            Integer: +/-1-2.  Boolean: flip with 30%.  Categorical: random 30%.
    """
    is_continuous = var.variable_type == VariableType.CONTINUOUS
    is_integer = var.variable_type == VariableType.INTEGER
    if is_continuous and var.lower is not None and var.upper is not None:
        current = params.get(var.name, (var.lower + var.upper) / 2)
        if isinstance(continuous_scale, tuple):
            pct = float(rng.uniform(continuous_scale[0], continuous_scale[1]))
        else:
            pct = continuous_scale
        scale = (var.upper - var.lower) * pct
        new_val = current + rng.normal(0, scale)
        params[var.name] = float(np.clip(new_val, var.lower, var.upper))
    elif is_integer and var.lower is not None and var.upper is not None:
        lower_i = int(var.lower)
        upper_i = int(var.upper)
        current = params.get(var.name, (lower_i + upper_i) // 2)
        new_val = current + rng.integers(-2, 3)
        params[var.name] = int(np.clip(new_val, lower_i, upper_i))
    elif var.variable_type == VariableType.BOOLEAN:
        if rng.random() < 0.3:
            params[var.name] = not params.get(var.name, False)
    elif var.variable_type == VariableType.CATEGORICAL and var.choices:
        if rng.random() < 0.3:
            params[var.name] = rng.choice(var.choices)


def _ancestors_in_space(
    search_space: SearchSpace,
    causal_graph: CausalGraph,
    objective_name: str,
) -> list[str]:
    """Return search-space variables that are graph ancestors of the objective.

    Order matches ``search_space.variable_names`` so callers can compare
    by length (the result is always a subset of the search space, so
    equal length implies equal membership).
    """
    ancestors = causal_graph.ancestors(objective_name)
    return [v for v in search_space.variable_names if v in ancestors]


def _get_focus_variables(
    search_space: SearchSpace,
    causal_graph: CausalGraph | None,
    objective_name: str,
) -> list[str]:
    """Identify variables to focus on, using causal graph if available."""
    if causal_graph is None:
        return search_space.variable_names

    focus = _ancestors_in_space(search_space, causal_graph, objective_name)
    return focus if focus else search_space.variable_names


def _apply_minimal_focus_a1(
    *,
    base_focus: list[str],
    search_space: SearchSpace,
    causal_graph: CausalGraph | None,
    objective_name: str,
    screened_variables: list[str] | None,
    enable: bool,
) -> list[str]:
    """Sprint 37 Option A1 minimal-focus heuristic.

    When *enable* is ``True`` and the supplied causal graph makes every
    search variable an ancestor of *objective_name*, return
    ``screened_variables ∩ ancestors`` -- but **only if** that
    intersection is a non-empty proper subset of the search space.
    Otherwise return *base_focus* unchanged.

    The two boundary conditions:

    1. Empty intersection -- screening agreed on nothing the graph
       considers ancestral. Falling back avoids collapsing onto an
       empty focus set.
    2. Intersection equals the full search space -- the "restriction"
       would not actually restrict anything; staying on *base_focus*
       keeps the engine path mechanically identical.
    """
    if not enable or causal_graph is None or not screened_variables:
        return base_focus

    all_var_names = search_space.variable_names
    ancestors_in_space = _ancestors_in_space(search_space, causal_graph, objective_name)

    # A1 binds only when graph ancestors cover the entire search space.
    # ``ancestors_in_space`` is built by filtering ``all_var_names``, so it
    # is always a subset; equal length implies equal membership.
    if len(ancestors_in_space) != len(all_var_names):
        return base_focus

    intersection = [v for v in ancestors_in_space if v in screened_variables]

    if not intersection or len(intersection) >= len(all_var_names):
        return base_focus

    return intersection


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


def inject_categorical_diversity(
    candidates: list[dict[str, Any]],
    search_space: SearchSpace,
) -> list[dict[str, Any]]:
    """Ensure every value of every categorical variable appears in the batch.

    For each categorical variable in the search space, checks which of its
    ``choices`` are missing from the candidate batch.  For each missing value,
    creates a diversity candidate by copying the first candidate and
    substituting that categorical value.  Appends diversity candidates to the
    end of the batch.

    Returns the original list unmodified when no categorical variables exist
    or all values are already represented.  When injection occurs, a new
    list is returned (the input list is not mutated).

    Note: the base for diversity candidates is ``candidates[0]`` — the first
    in Ax generation order, not necessarily the candidate with the highest
    acquisition value (Ax does not guarantee ordering).

    Args:
        candidates: List of candidate parameter dicts (from Ax or other source).
        search_space: The optimization search space (used to find categorical
            variables and their choices).

    Returns:
        A new list with diversity candidates appended, or the original list
        if no injection was needed.
    """
    if not candidates:
        return candidates

    # Find categorical variables with choices
    cat_vars = [
        v
        for v in search_space.variables
        if v.variable_type == VariableType.CATEGORICAL and v.choices
    ]

    if not cat_vars:
        return candidates

    # For each categorical variable, find which values are missing
    any_missing = False
    missing_per_var: list[tuple[Variable, list[Any]]] = []
    for var in cat_vars:
        present_values = {c.get(var.name) for c in candidates}
        choices = var.choices or []
        missing = [v for v in choices if v not in present_values]
        missing_per_var.append((var, missing))
        if missing:
            any_missing = True

    if not any_missing:
        return candidates

    # Inject diversity candidates: copy the first Ax candidate (the first
    # in generation order — Ax does not guarantee acquisition-value
    # ordering), substitute the missing categorical value.  One new
    # candidate per missing value per categorical variable.
    # Work on a copy to avoid mutating the caller's list.
    expanded = list(candidates)
    base = expanded[0]
    for var, missing_values in missing_per_var:
        for value in missing_values:
            diversity_candidate = dict(base)
            diversity_candidate[var.name] = value
            expanded.append(diversity_candidate)

    return expanded
