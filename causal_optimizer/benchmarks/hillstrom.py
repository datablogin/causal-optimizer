"""Sprint 31 Hillstrom benchmark harness.

Implements the launch contract locked in the Sprint 31 Hillstrom
benchmark contract doc. Reuses ``MarketingLogAdapter`` unchanged via a
narrow loader + wrapped runner pattern; no changes to
``causal_optimizer/domain_adapters/marketing_logs.py`` are required.

Contract anchors
----------------

- Primary slice: ``Womens E-Mail`` vs ``No E-Mail`` (binary).
- Pooled secondary slice: ``Any E-Mail`` (``Mens E-Mail`` +
  ``Womens E-Mail``) vs ``No E-Mail`` (binary).
- Per-slice propensity pinned as an exact constant:
  ``0.5`` for the primary slice (computed as ``(1/3) / (2/3)``) and
  ``2.0 / 3.0`` for the pooled slice (computed as ``2.0 / 3.0`` in
  code, not a rounded constant).
- Frozen parameter dimensions (pre-baked in every forwarded call):
  ``email_share = 1.0``, ``social_share_of_remainder = 0.0``,
  ``min_propensity_clip = 0.01``. Active tuned search space is therefore
  3 variables: ``eligibility_threshold``, ``regularization``,
  ``treatment_budget_pct``.
- Prior graph: projected to the 7-edge sub-DAG over the active nodes.
  The three frozen variables and the 7 edges incident to them are
  dropped.
- Null control: permuted-outcome null baseline is the raw scalar
  ``μ = mean(spend)`` computed on the unshuffled reshaped frame; this
  avoids the misleading "``policy_value`` under a uniform-treatment
  policy" framing, which on randomized data only recovers the
  treated-arm mean (see ``marketing_logs.py:322``).

Public API
----------
- :class:`HillstromSliceType`
- :func:`load_hillstrom_slice`
- :func:`hillstrom_active_search_space`
- :func:`hillstrom_projected_prior_graph`
- :class:`HillstromPolicyRunner`
- :func:`permute_hillstrom_spend`
- :func:`hillstrom_null_baseline`
- :class:`HillstromBenchmarkResult`
- :class:`HillstromScenario`
- :data:`HILLSTROM_FROZEN_PARAMS`
- :data:`HILLSTROM_PRIMARY_PROPENSITY`
- :data:`HILLSTROM_POOLED_PROPENSITY`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
    SearchSpace,
    Variable,
    VariableType,
)

# ── Contract constants ──────────────────────────────────────────────

HILLSTROM_PRIMARY_PROPENSITY: float = 0.5
"""Primary slice propensity: ``P(treated | row in primary slice) = 0.5``.

Derivation: each of Hillstrom's three arms (``Mens E-Mail``,
``Womens E-Mail``, ``No E-Mail``) is randomized at ``P = 1/3``. After
dropping the ``Mens E-Mail`` arm, the primary slice contains the
``Womens E-Mail`` and ``No E-Mail`` arms. Conditional on a row being in
the slice, ``P(treated) = P(Womens) / (P(Womens) + P(None)) =
(1/3) / (1/3 + 1/3) = 0.5``.
"""

HILLSTROM_POOLED_PROPENSITY: float = 2.0 / 3.0
"""Pooled slice propensity: ``P(treated | row in pooled slice) = 2/3`` exactly.

Computed as ``2.0 / 3.0`` rather than a rounded constant like ``0.667``
so the Sprint 31 smoke-test invariant ``propensity == 2.0 / 3.0`` is
bitwise-true on every pooled-slice row.
"""

HILLSTROM_FROZEN_PARAMS: dict[str, float] = {
    "email_share": 1.0,
    "social_share_of_remainder": 0.0,
    "min_propensity_clip": 0.01,
}
"""Frozen ``MarketingLogAdapter`` dimensions for Hillstrom.

All three are degenerate on Hillstrom:

- ``email_share`` and ``social_share_of_remainder``: Hillstrom treatment
  is single-channel e-mail, so there is no cross-channel allocation
  problem to solve.
- ``min_propensity_clip``: propensity is a per-slice constant
  (``0.5`` or ``2/3``) and both are ``>= 0.5`` (the upper bound of
  ``min_propensity_clip``'s range), so the clip never activates and
  the optimizer sees a flat response surface along this dimension.
"""

HILLSTROM_TREATED_ARMS: frozenset[str] = frozenset({"Mens E-Mail", "Womens E-Mail"})
HILLSTROM_CONTROL_ARM: str = "No E-Mail"
HILLSTROM_TREATED_COST: float = 0.10
HILLSTROM_CONTROL_COST: float = 0.0

# history_segment bucket map — matches the raw Hillstrom CSV strings,
# numeric prefix included, per Sprint 31 contract Section 3c.
_HISTORY_SEGMENT_TO_SEGMENT: dict[str, str] = {
    "1) $0 - $100": "low",
    "2) $100 - $200": "low",
    "3) $200 - $350": "medium",
    "4) $350 - $500": "medium",
    "5) $500 - $750": "high_value",
    "6) $750 - $1,000": "high_value",
    "7) $1,000 +": "high_value",
}


# ── Slice type ───────────────────────────────────────────────────────


class HillstromSliceType(str, Enum):
    """Which Hillstrom binary slice to load.

    ``PRIMARY``
        ``Womens E-Mail`` (``treatment=1``) vs ``No E-Mail``
        (``treatment=0``). Drops ``Mens E-Mail`` rows. Propensity is
        exactly ``0.5``.
    ``POOLED``
        ``Any E-Mail`` (``Mens`` or ``Womens``, ``treatment=1``) vs
        ``No E-Mail`` (``treatment=0``). Keeps every raw row.
        Propensity is exactly ``2/3``.
    """

    PRIMARY = "primary"
    POOLED = "pooled"


# ── Loader ───────────────────────────────────────────────────────────


def _treatment_from_segment(segment: pd.Series, slice_type: HillstromSliceType) -> pd.Series:
    """Map the raw ``segment`` column to binary ``treatment``."""
    if slice_type is HillstromSliceType.PRIMARY:
        return (segment == "Womens E-Mail").astype(int)
    return segment.isin(HILLSTROM_TREATED_ARMS).astype(int)


def _propensity_for_slice(slice_type: HillstromSliceType) -> float:
    if slice_type is HillstromSliceType.PRIMARY:
        return HILLSTROM_PRIMARY_PROPENSITY
    return HILLSTROM_POOLED_PROPENSITY


def load_hillstrom_slice(
    raw: pd.DataFrame,
    *,
    slice_type: HillstromSliceType,
    treated_cost: float = HILLSTROM_TREATED_COST,
    control_cost: float = HILLSTROM_CONTROL_COST,
) -> pd.DataFrame:
    """Reshape a raw Hillstrom frame into the ``MarketingLogAdapter`` schema.

    Filters to the requested slice, maps ``segment`` → ``treatment``,
    passes through ``spend`` → ``outcome``, assigns a constant per-slice
    propensity, a fixed per-send cost, the constant ``channel = "email"``,
    and a bucketed ``segment`` column derived from ``history_segment``.
    The ``visit`` and ``conversion`` columns are retained on the output
    frame as secondary reported outcomes (not as adapter descriptors),
    matching Sprint 31 contract Section 4c.

    Args:
        raw: DataFrame with the canonical Hillstrom column schema.
        slice_type: Which binary slice to produce.
        treated_cost: Per-treated-row fixed cost. Defaults to
            ``HILLSTROM_TREATED_COST``.
        control_cost: Per-control-row fixed cost. Defaults to
            ``HILLSTROM_CONTROL_COST``.

    Returns:
        A new DataFrame with the ``MarketingLogAdapter`` required
        columns (``treatment``, ``outcome``, ``cost``) plus the optional
        columns (``propensity``, ``channel``, ``segment``) and the
        retained Hillstrom ``visit`` and ``conversion`` columns.

    Raises:
        ValueError: If ``raw`` is missing any required Hillstrom
            column.
    """
    required = {"segment", "spend", "visit", "conversion", "history_segment"}
    missing = required - set(raw.columns)
    if missing:
        msg = f"Hillstrom frame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = raw.copy()

    # Slice filter
    if slice_type is HillstromSliceType.PRIMARY:
        keep = frame["segment"].isin({"Womens E-Mail", HILLSTROM_CONTROL_ARM})
        frame = frame.loc[keep].reset_index(drop=True)
    # Pooled slice keeps every row.

    treatment = _treatment_from_segment(frame["segment"], slice_type)
    propensity = _propensity_for_slice(slice_type)
    cost = np.where(treatment == 1, treated_cost, control_cost)

    # Bucket history_segment → {"low", "medium", "high_value"}. Any
    # unknown bucket is mapped to "low" to avoid silent NaNs; this
    # should not occur on canonical Hillstrom data but is defensive.
    segment_bucket = frame["history_segment"].map(_HISTORY_SEGMENT_TO_SEGMENT).fillna("low")

    reshaped = pd.DataFrame(
        {
            "treatment": treatment.astype(int),
            "outcome": frame["spend"].astype(float),
            "cost": cost.astype(float),
            "propensity": np.full(len(frame), propensity, dtype=float),
            "channel": "email",
            "segment": segment_bucket.astype(str),
            # Retained Hillstrom outcomes as secondary reported outcomes.
            "visit": frame["visit"].astype(int),
            "conversion": frame["conversion"].astype(int),
        }
    )

    return reshaped


# ── Active search space ──────────────────────────────────────────────


_ACTIVE_VAR_NAMES: tuple[str, ...] = (
    "eligibility_threshold",
    "regularization",
    "treatment_budget_pct",
)


@lru_cache(maxsize=1)
def hillstrom_active_search_space() -> SearchSpace:
    """Return the 3-variable active search space for Hillstrom.

    Bounds are inherited from ``MarketingLogAdapter.get_search_space()``
    so the active subspace is numerically identical to the adapter's
    native ranges for the three tuned dimensions. The three frozen
    dimensions (``email_share``, ``social_share_of_remainder``,
    ``min_propensity_clip``) are pre-baked by
    :class:`HillstromPolicyRunner` and do not appear in this space.

    The result is cached at module level (``lru_cache(maxsize=1)``)
    because the adapter's native bounds are constants — there is no
    per-call state to recompute. ``HillstromScenario.run_strategy``
    calls this on every ``(strategy, budget, seed)`` combination, so
    caching avoids rebuilding a throwaway adapter each time.

    **Do not mutate the returned object.** Every caller shares the
    same cached :class:`SearchSpace` instance. Appending or reassigning
    ``variables`` on the returned space would corrupt it for all
    subsequent callers in the process. Treat the returned search space
    as read-only; if you need a modified copy, construct a new
    :class:`SearchSpace` from its ``variables`` list.
    """
    # Build a throwaway adapter against a minimal placeholder frame to
    # extract the canonical bounds. Runs exactly once per process thanks
    # to the lru_cache decorator. The adapter's __init__ requires
    # non-empty data with treatment/outcome/cost columns, so we use a
    # 2-row stub. This is not a data dependency — it only reads the
    # adapter's static variable bounds.
    stub = pd.DataFrame(
        {
            "treatment": [0, 1],
            "outcome": [0.0, 1.0],
            "cost": [0.0, 0.0],
        }
    )
    full = MarketingLogAdapter(data=stub).get_search_space()
    by_name = {v.name: v for v in full.variables}
    variables = [
        Variable(
            name=name,
            variable_type=VariableType.CONTINUOUS,
            lower=by_name[name].lower,
            upper=by_name[name].upper,
        )
        for name in _ACTIVE_VAR_NAMES
    ]
    return SearchSpace(variables=variables)


# ── Projected prior graph ────────────────────────────────────────────


def hillstrom_projected_prior_graph() -> CausalGraph:
    """Return the 7-edge projected prior graph for Hillstrom.

    The projection drops the 3 frozen variables and the 7 edges
    incident to them from ``MarketingLogAdapter.get_prior_graph()``,
    leaving a sub-DAG over the active nodes. See Sprint 31 contract
    Section 4a.i for the enumeration and rationale.
    """
    return CausalGraph(
        edges=[
            ("eligibility_threshold", "treated_fraction"),
            ("treatment_budget_pct", "treated_fraction"),
            ("regularization", "treated_fraction"),
            ("regularization", "policy_value"),
            ("treated_fraction", "total_cost"),
            ("treated_fraction", "policy_value"),
            ("treated_fraction", "effective_sample_size"),
        ],
    )


# ── Wrapped runner ───────────────────────────────────────────────────


class HillstromPolicyRunner:
    """ExperimentRunner that pre-bakes frozen params before adapter call.

    Accepts an active-only parameter dict (with the three tuned
    variables) and forwards to ``MarketingLogAdapter.run_experiment``
    after injecting the three frozen constants in
    :data:`HILLSTROM_FROZEN_PARAMS`. Does not mutate the caller's dict
    and does not modify the adapter.

    The returned metrics include ``MarketingLogAdapter``'s full
    per-run dict (``policy_value``, ``total_cost``, ``treated_fraction``,
    ``effective_sample_size``, ``propensity_clip_fraction``,
    ``max_ips_weight``, ``weight_cv``, ``zero_support``) plus an
    ``objective`` alias equal to the negated ``policy_value`` — because
    ``policy_value`` is maximized but the engine minimizes ``objective``.
    """

    def __init__(self, adapter: MarketingLogAdapter) -> None:
        self._adapter = adapter

    def _forward_params(self, active_params: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with frozen params injected.

        Does not mutate the caller's ``active_params`` dict.
        """
        forwarded: dict[str, Any] = dict(active_params)
        for key, value in HILLSTROM_FROZEN_PARAMS.items():
            forwarded[key] = value
        return forwarded

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Evaluate one active-only parameter dict on the wrapped adapter."""
        forwarded = self._forward_params(parameters)
        metrics = self._adapter.run_experiment(forwarded)
        # Engine minimizes "objective"; Hillstrom maximizes policy_value.
        metrics["objective"] = -metrics["policy_value"]
        return metrics


# ── Null control helpers ─────────────────────────────────────────────


def permute_hillstrom_spend(
    reshaped: pd.DataFrame, *, seed: int, outcome_col: str = "outcome"
) -> pd.DataFrame:
    """Return a copy of ``reshaped`` with the ``outcome`` column permuted.

    The permutation is deterministic in ``seed`` and preserves:

    - the multiset of ``outcome`` values (permutation of the same column
      — so ``μ = mean(outcome)`` is identical on the original and the
      shuffled frame)
    - every other column, including ``treatment``, ``propensity``,
      ``cost``, ``channel``, ``segment``, ``visit``, and ``conversion``.

    Args:
        reshaped: A frame already reshaped by :func:`load_hillstrom_slice`.
        seed: RNG seed. Use one seed per null-control replicate.
        outcome_col: Name of the column to permute. Defaults to
            ``"outcome"``.

    Returns:
        A new frame with the outcome column shuffled; the original
        frame is not modified.

    Raises:
        ValueError: If ``outcome_col`` is not in the frame.
    """
    if outcome_col not in reshaped.columns:
        raise ValueError(f"column {outcome_col!r} not in reshaped frame")
    out = reshaped.copy()
    rng = np.random.default_rng(seed)
    values = out[outcome_col].to_numpy().copy()
    rng.shuffle(values)
    out[outcome_col] = values
    return out


def hillstrom_null_baseline(reshaped: pd.DataFrame, *, outcome_col: str = "outcome") -> float:
    """Return the Sprint 31 null-control baseline ``μ = mean(outcome)``.

    Called once on the original (unshuffled) reshaped frame; shuffling
    is a permutation and does not change the column mean, so the same
    scalar is used for every null-control seed.
    """
    return float(reshaped[outcome_col].astype(float).mean())


# ── Benchmark scenario ───────────────────────────────────────────────


VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})
"""Public set of strategy names accepted by :class:`HillstromScenario`.

Exported so that CLI wrappers and downstream callers import the same
source of truth rather than redefining their own frozenset — a local
copy would drift silently if a new strategy were added here.
"""


@dataclass
class HillstromBenchmarkResult:
    """Result of running one strategy on one Hillstrom slice.

    Attributes:
        strategy: ``"random"`` / ``"surrogate_only"`` / ``"causal"``.
        budget: Experiments run by the strategy.
        seed: RNG seed for the optimizer.
        slice_type: ``"primary"`` or ``"pooled"``.
        is_null_control: ``True`` when evaluated on a permuted-outcome
            frame (Sprint 31 null-control path).
        policy_value: Best ``policy_value`` found by the strategy on the
            given slice (higher is better).
        selected_parameters: Best active-only parameter dict, or
            ``None`` if no valid result was produced.
        runtime_seconds: Wall-clock time for the full strategy run.
    """

    strategy: str
    budget: int
    seed: int
    slice_type: str
    is_null_control: bool
    policy_value: float
    selected_parameters: dict[str, Any] | None = None
    runtime_seconds: float = 0.0
    null_baseline: float | None = None
    # Secondary reported outcomes — IPS-unweighted per-strategy aggregates
    # taken over the logged observations matched by the best policy.
    secondary_outcomes: dict[str, float] = field(default_factory=dict)


class HillstromScenario:
    """Top-level Hillstrom benchmark scenario.

    Wraps a reshaped Hillstrom frame in a :class:`MarketingLogAdapter`
    and a :class:`HillstromPolicyRunner`, then runs one strategy at a
    given ``(budget, seed)`` pair. Supports both the real slice and
    the permuted-outcome null-control path.

    Note: the scenario is intended as the unit of work for the Sprint 31
    benchmark runner; a separate CLI script composes scenarios over a
    grid of (strategy, budget, seed) combinations.
    """

    def __init__(
        self,
        raw: pd.DataFrame,
        *,
        slice_type: HillstromSliceType,
    ) -> None:
        self._raw = raw
        self._slice_type = slice_type
        self._real_slice = load_hillstrom_slice(raw, slice_type=slice_type)
        # The null-baseline μ is a scalar computed once from the real
        # (unshuffled) frame; shuffling preserves the column mean.
        self._null_baseline = hillstrom_null_baseline(self._real_slice)

    @property
    def slice_type(self) -> HillstromSliceType:
        return self._slice_type

    @property
    def null_baseline(self) -> float:
        return self._null_baseline

    @property
    def real_slice(self) -> pd.DataFrame:
        return self._real_slice

    def run_strategy(
        self,
        strategy: str,
        *,
        budget: int,
        seed: int,
        null_control: bool = False,
    ) -> HillstromBenchmarkResult:
        """Run one strategy on this scenario and return the result.

        Args:
            strategy: One of ``"random"``, ``"surrogate_only"``,
                ``"causal"``.
            budget: Number of experiments (adapter evaluations).
            seed: Optimizer RNG seed.
            null_control: When ``True``, the strategy runs against a
                permuted-outcome copy of the reshaped frame. Permutation
                seed is the same ``seed`` argument so null-control seeds
                are reproducible.

        Returns:
            A :class:`HillstromBenchmarkResult`.

        Raises:
            ValueError: If ``strategy`` is not in ``VALID_STRATEGIES``.
        """
        if strategy not in VALID_STRATEGIES:
            msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(VALID_STRATEGIES)}."
            raise ValueError(msg)

        t_start = time.perf_counter()

        frame = (
            permute_hillstrom_spend(self._real_slice, seed=seed)
            if null_control
            else self._real_slice
        )
        adapter = MarketingLogAdapter(data=frame, seed=seed)
        runner = HillstromPolicyRunner(adapter=adapter)
        space = hillstrom_active_search_space()

        best_policy_value = float("-inf")
        best_params: dict[str, Any] | None = None

        if strategy == "random":
            rng = np.random.default_rng(seed)
            for _ in range(budget):
                params = sample_random_params(space, rng)
                metrics = runner.run(params)
                pv = metrics["policy_value"]
                if pv > best_policy_value:
                    best_policy_value = pv
                    best_params = params
        else:
            graph = hillstrom_projected_prior_graph() if strategy == "causal" else None
            engine = ExperimentEngine(
                search_space=space,
                runner=runner,
                causal_graph=graph,
                objective_name="objective",
                minimize=True,
                seed=seed,
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result("objective", minimize=True)
            if best_result is not None:
                best_params = best_result.parameters
                best_policy_value = best_result.metrics.get("policy_value", float("-inf"))

        # Invariant: the engine-logged best parameters must contain only the
        # 3 active-only dimensions. Frozen dimensions are injected by
        # HillstromPolicyRunner before the adapter call and must never leak
        # back into the experiment log — if they did, downstream readers
        # would mis-identify the Hillstrom search scope. This guards the
        # "loader + wrapped runner" contract boundary against a future
        # engine change that could start forwarding runner-side metadata
        # back into the logged parameters.
        _check_active_params_invariant(best_params)

        runtime = time.perf_counter() - t_start

        secondary: dict[str, float] = {}
        if best_params is not None and not null_control:
            # Secondary outcomes: IPS-unweighted in-sample treated/control-arm
            # means of the retained Hillstrom columns on the reshaped frame.
            # These are report-time diagnostics, not policy-filtered —
            # policy-conditioned secondary outcomes are deferred to a later
            # sprint (Sprint 32+) when uplift/CATE scoring is in scope.
            secondary = _secondary_arm_aggregates(frame)

        return HillstromBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            slice_type=self._slice_type.value,
            is_null_control=null_control,
            policy_value=(
                best_policy_value if best_policy_value != float("-inf") else float("nan")
            ),
            selected_parameters=best_params,
            runtime_seconds=runtime,
            null_baseline=self._null_baseline,
            secondary_outcomes=secondary,
        )


def _check_active_params_invariant(best_params: dict[str, Any] | None) -> None:
    """Raise ``RuntimeError`` if ``best_params`` contains non-active keys.

    Guards the "loader + wrapped runner" contract boundary: the
    experiment log must only ever contain the 3 active Hillstrom
    dimensions. Frozen dimensions are injected by
    :class:`HillstromPolicyRunner` before the adapter call and must
    never leak back into the optimizer log.

    Raised as :class:`RuntimeError` (not :keyword:`assert`) so the
    check still fires under ``python -O`` / ``PYTHONOPTIMIZE=1``.

    Args:
        best_params: The best parameter dict returned by the engine
            log, or ``None`` if no valid result was produced.

    Raises:
        RuntimeError: If ``best_params`` is non-``None`` and contains
            any key that is not in :data:`_ACTIVE_VAR_NAMES`.
    """
    if best_params is None:
        return
    if set(best_params) != set(_ACTIVE_VAR_NAMES):
        msg = (
            f"HillstromScenario: best_params has unexpected keys "
            f"{sorted(best_params)!r}; expected exactly "
            f"{sorted(_ACTIVE_VAR_NAMES)!r}. Frozen Hillstrom dimensions "
            f"must not appear in the experiment log."
        )
        raise RuntimeError(msg)


def _secondary_arm_aggregates(frame: pd.DataFrame) -> dict[str, float]:
    """Compute in-sample treated/control-arm aggregates for secondary outcomes.

    The Sprint 31 contract (Section 4c) says ``visit`` and ``conversion``
    should be tracked as secondary reported outcomes. The Sprint 31
    harness reports simple in-sample treated-arm and control-arm means
    on the reshaped frame as a diagnostic — it does **not** filter to
    the policy's selected observations. Policy-conditioned secondary
    outcomes (uplift- or CATE-filtered subpopulations) are deferred to a
    later sprint when learned uplift scoring is in scope.

    These aggregates are adapter-independent and do not re-run the
    adapter; they are derived purely from the reshaped frame's
    ``treatment``, ``visit``, and ``conversion`` columns.
    """
    if "visit" not in frame.columns or "conversion" not in frame.columns:
        return {}
    # Treated-arm aggregates are adapter-independent diagnostics; they
    # do not re-run the adapter and do not enter the optimizer objective.
    treated_mask = frame["treatment"] == 1
    control_mask = frame["treatment"] == 0
    aggregates: dict[str, float] = {}
    if int(treated_mask.sum()) > 0:
        aggregates["treated_visit_rate"] = float(frame.loc[treated_mask, "visit"].mean())
        aggregates["treated_conversion_rate"] = float(frame.loc[treated_mask, "conversion"].mean())
    if int(control_mask.sum()) > 0:
        aggregates["control_visit_rate"] = float(frame.loc[control_mask, "visit"].mean())
        aggregates["control_conversion_rate"] = float(frame.loc[control_mask, "conversion"].mean())
    return aggregates
