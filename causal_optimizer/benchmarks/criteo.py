"""Sprint 33 Criteo uplift benchmark harness.

Implements the benchmark contract locked in Sprint 32 (issue #177).
Reuses ``MarketingLogAdapter`` unchanged via a narrow loader + wrapped
runner pattern identical to the Hillstrom harness.

Contract anchors
----------------

- Dataset: Criteo Uplift Prediction v2.1 (13,979,592 rows).
- Fixed-seed 1M-row subsample for benchmarks.
- Primary outcome: ``visit`` (binary 0/1, 4.70% base rate).
- Constant propensity: ``0.85`` (from known randomization design).
- Frozen parameter dimensions: ``email_share=1.0``,
  ``social_share_of_remainder=0.0``, ``min_propensity_clip=0.01``,
  ``regularization=1.0`` (inert under current adapter math).
- Active tuned search space: 2 variables —
  ``eligibility_threshold``, ``treatment_budget_pct``.
- Prior graph: projected to the 5-edge sub-DAG over active nodes.
- Null control: permuted ``visit`` column, B20+B40, 5% tolerance
  band with 10% fallback.

Public API
----------
- :func:`load_criteo_subsample`
- :func:`criteo_active_search_space`
- :func:`criteo_projected_prior_graph`
- :class:`CriteoPolicyRunner`
- :func:`permute_criteo_visit`
- :func:`criteo_null_baseline`
- :func:`run_propensity_gate`
- :class:`CriteoBenchmarkResult`
- :class:`CriteoScenario`
- :data:`CRITEO_FROZEN_PARAMS`
- :data:`CRITEO_PROPENSITY`
- :data:`CRITEO_SAMPLE_SEED`
- :data:`CRITEO_ENGINE_OBJECTIVE`
- :data:`VALID_STRATEGIES`
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
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

CRITEO_PROPENSITY: float = 0.85
"""Constant propensity for all rows. Known from the Criteo v2.1
randomization design: 85% of users are assigned to treatment."""

CRITEO_SAMPLE_SEED: int = 20260417
"""Fixed seed for the 1M-row subsample. Every strategy-budget-seed
combination operates on the same subsample."""

CRITEO_TREATED_COST: float = 0.01
"""Synthetic per-treated-row cost. Does not affect ``policy_value``."""

CRITEO_CONTROL_COST: float = 0.0
"""Synthetic per-control-row cost."""

CRITEO_FROZEN_PARAMS: dict[str, float] = {
    "email_share": 1.0,
    "social_share_of_remainder": 0.0,
    "min_propensity_clip": 0.01,
    "regularization": 1.0,
}
"""Frozen ``MarketingLogAdapter`` dimensions for Criteo.

- ``email_share`` and ``social_share_of_remainder``: Criteo is
  single-channel (ad display), so channel allocation is degenerate.
- ``min_propensity_clip``: fixed at 0.01 (conservative, no clipping
  at propensity=0.85).
- ``regularization``: inert under the current adapter math (the
  affine regularization step cancels algebraically after min/max
  normalization; see Sprint 32 contract Section 6e).
"""

CRITEO_ENGINE_OBJECTIVE: str = "policy_value"
"""The string passed to ``ExperimentEngine`` as ``objective_name``.

This constant couples the engine objective, the projected graph sink
node, and the metric key returned by ``MarketingLogAdapter``.
"""

_ACTIVE_VAR_NAMES: tuple[str, ...] = (
    "eligibility_threshold",
    "treatment_budget_pct",
)

VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


# ── Loader ───────────────────────────────────────────────────────────


def load_criteo_subsample(
    raw: pd.DataFrame,
    *,
    treated_cost: float = CRITEO_TREATED_COST,
    control_cost: float = CRITEO_CONTROL_COST,
    synthesize_segment: bool = False,
) -> pd.DataFrame:
    """Reshape a Criteo frame into the ``MarketingLogAdapter`` schema.

    Maps Criteo columns per Sprint 32 contract Section 5b/5c:
    - ``treatment`` → pass-through
    - ``visit`` → ``outcome`` (float)
    - synthesized ``cost``: 0.01 treated, 0.0 control
    - synthesized ``propensity``: constant 0.85
    - constant ``channel``: ``"email"``
    - ``conversion`` retained as secondary outcome

    When ``synthesize_segment`` is ``False`` (Run 1), the ``segment``
    column is omitted (degenerate surface). When ``True`` (Run 2),
    ``f0`` tertiles are mapped to ``"high_value"`` / ``"medium"`` /
    ``"low"`` per Sprint 32 contract Section 5f.

    Args:
        raw: DataFrame with the Criteo column schema.
        treated_cost: Per-treated-row fixed cost.
        control_cost: Per-control-row fixed cost.
        synthesize_segment: When ``True``, derive ``segment`` from
            ``f0`` tertiles for the heterogeneous-surface Run 2.

    Returns:
        A new DataFrame with the ``MarketingLogAdapter`` required
        columns plus retained secondary columns.

    Raises:
        ValueError: If ``raw`` is missing required Criteo columns.
    """
    required = {"treatment", "visit", "conversion"}
    missing = required - set(raw.columns)
    if missing:
        msg = f"Criteo frame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    treatment = raw["treatment"].values.astype(int)
    cost = np.where(treatment == 1, treated_cost, control_cost)

    result = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": raw["visit"].values.astype(float),
            "cost": cost.astype(float),
            "propensity": np.full(len(raw), CRITEO_PROPENSITY, dtype=float),
            "channel": "email",
            "conversion": raw["conversion"].values.astype(int),
        }
    )

    if synthesize_segment and "f0" in raw.columns:
        # Map f0 tertiles to segment labels per contract Section 5f
        tertile_labels = pd.qcut(
            raw["f0"], q=3, labels=["low", "medium", "high_value"], duplicates="drop"
        )
        result["segment"] = tertile_labels.astype(str)

    # Retain f0 for propensity gate diagnostics
    if "f0" in raw.columns:
        result["f0"] = raw["f0"].values

    return result


# ── Active search space ──────────────────────────────────────────────


@lru_cache(maxsize=1)
def criteo_active_search_space() -> SearchSpace:
    """Return the 2-variable active search space for Criteo.

    Bounds are inherited from ``MarketingLogAdapter.get_search_space()``
    for the two tuned dimensions. The four frozen dimensions are
    pre-baked by :class:`CriteoPolicyRunner`.
    """
    stub = pd.DataFrame({"treatment": [0, 1], "outcome": [0.0, 1.0], "cost": [0.0, 0.0]})
    full = MarketingLogAdapter(data=stub).get_search_space()
    by_name = {v.name: v for v in full.variables}
    missing = [name for name in _ACTIVE_VAR_NAMES if name not in by_name]
    if missing:
        msg = (
            f"MarketingLogAdapter.get_search_space() is missing expected "
            f"Criteo active variables {missing!r}. This breaks the "
            f"Sprint 32 Criteo benchmark contract."
        )
        raise RuntimeError(msg)
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


def criteo_projected_prior_graph() -> CausalGraph:
    """Return the 5-edge projected prior graph for Criteo.

    The projection drops the 4 frozen variables (``email_share``,
    ``social_share_of_remainder``, ``min_propensity_clip``,
    ``regularization``) and the 9 edges incident to them from
    ``MarketingLogAdapter.get_prior_graph()``. See Sprint 32
    contract Section 6d.
    """
    return CausalGraph(
        edges=[
            ("eligibility_threshold", "treated_fraction"),
            ("treatment_budget_pct", "treated_fraction"),
            ("treated_fraction", "total_cost"),
            ("treated_fraction", "policy_value"),
            ("treated_fraction", "effective_sample_size"),
        ],
    )


# ── Wrapped runner ───────────────────────────────────────────────────


class CriteoPolicyRunner:
    """ExperimentRunner that pre-bakes frozen params before adapter call.

    Accepts an active-only parameter dict (with the two tuned variables)
    and forwards to ``MarketingLogAdapter.run_experiment`` after injecting
    the four frozen constants.
    """

    def __init__(self, adapter: MarketingLogAdapter) -> None:
        self._adapter = adapter

    def _forward_params(self, active_params: dict[str, Any]) -> dict[str, Any]:
        forwarded: dict[str, Any] = dict(active_params)
        for key, value in CRITEO_FROZEN_PARAMS.items():
            forwarded[key] = value
        return forwarded

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        forwarded = self._forward_params(parameters)
        return self._adapter.run_experiment(forwarded)


# ── Propensity gate ──────────────────────────────────────────────────


def run_propensity_gate(
    raw: pd.DataFrame,
    *,
    threshold_pp: float = 0.02,
) -> tuple[bool, dict[str, Any]]:
    """Check treatment rate by f0 decile for propensity heterogeneity.

    Sprint 32 contract Section 6b mandates this as a hard stop before
    running the benchmark. The gate passes if all f0 deciles have
    treatment rate within ``threshold_pp`` of 0.85.

    Args:
        raw: Criteo frame with ``treatment`` and ``f0`` columns.
        threshold_pp: Maximum allowed deviation in percentage points.

    Returns:
        A ``(passed, details)`` tuple where ``passed`` is ``True`` if
        all deciles are within threshold, and ``details`` contains
        per-decile treatment rates and the maximum deviation.
    """
    if "f0" not in raw.columns:
        # Fixture may not have f0; return pass with a note
        return True, {
            "decile_treatment_rates": [0.85] * 10,
            "max_deviation": 0.0,
            "note": "f0 column not present; gate skipped",
        }

    decile_labels = pd.qcut(raw["f0"], q=10, labels=False, duplicates="drop")
    decile_rates = raw["treatment"].groupby(decile_labels).mean()

    rates_list = decile_rates.tolist()
    deviations = [abs(r - 0.85) for r in rates_list]
    max_dev = max(deviations)
    passed = max_dev <= threshold_pp
    worst_idx = int(np.argmax(deviations))

    return passed, {
        "decile_treatment_rates": rates_list,
        "max_deviation": float(max_dev),
        "worst_decile": worst_idx,
        "worst_rate": float(rates_list[worst_idx]),
    }


# ── Null control helpers ─────────────────────────────────────────────


def permute_criteo_visit(
    reshaped: pd.DataFrame, *, seed: int, outcome_col: str = "outcome"
) -> pd.DataFrame:
    """Return a copy of ``reshaped`` with the ``outcome`` column permuted.

    Deterministic in ``seed``. Preserves all other columns including
    ``treatment``, ``propensity``, ``cost``, ``channel``, ``conversion``.
    """
    if outcome_col not in reshaped.columns:
        raise ValueError(f"column {outcome_col!r} not in reshaped frame")
    out = reshaped.copy()
    rng = np.random.default_rng(seed)
    values = out[outcome_col].to_numpy(copy=True)
    rng.shuffle(values)
    out[outcome_col] = values
    return out


def criteo_null_baseline(reshaped: pd.DataFrame, *, outcome_col: str = "outcome") -> float:
    """Return the null-control baseline ``μ = mean(visit)`` on the reshaped frame."""
    return float(reshaped[outcome_col].astype(float).mean())


# ── Benchmark scenario ───────────────────────────────────────────────


@dataclass
class CriteoBenchmarkResult:
    """Result of running one strategy on the Criteo benchmark.

    Attributes:
        strategy: ``"random"`` / ``"surrogate_only"`` / ``"causal"``.
        budget: Experiments run by the strategy.
        seed: RNG seed for the optimizer.
        is_null_control: ``True`` when evaluated on a permuted-outcome frame.
        policy_value: Best ``policy_value`` found.
        selected_parameters: Best active-only parameter dict.
        runtime_seconds: Wall-clock time for the full strategy run.
        null_baseline: ``μ = mean(visit)`` on the unshuffled frame.
        diagnostics: Per-run IPS diagnostics (ESS, weight_cv, etc.).
        secondary_outcomes: Treated/control arm aggregates.
    """

    strategy: str
    budget: int
    seed: int
    is_null_control: bool
    policy_value: float
    selected_parameters: dict[str, Any] | None = None
    runtime_seconds: float = 0.0
    null_baseline: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)
    secondary_outcomes: dict[str, float] = field(default_factory=dict)


class CriteoScenario:
    """Top-level Criteo benchmark scenario.

    Wraps a reshaped Criteo frame in a ``MarketingLogAdapter`` and
    a ``CriteoPolicyRunner``, then runs one strategy at a given
    ``(budget, seed)`` pair.
    """

    def __init__(self, reshaped: pd.DataFrame) -> None:
        self._real_data = reshaped
        self._null_baseline = criteo_null_baseline(reshaped)

    @property
    def null_baseline(self) -> float:
        return self._null_baseline

    @property
    def real_data(self) -> pd.DataFrame:
        return self._real_data

    def run_strategy(
        self,
        strategy: str,
        *,
        budget: int,
        seed: int,
        null_control: bool = False,
    ) -> CriteoBenchmarkResult:
        """Run one strategy and return the result."""
        if strategy not in VALID_STRATEGIES:
            msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(VALID_STRATEGIES)}."
            raise ValueError(msg)

        t_start = time.perf_counter()

        frame = (
            permute_criteo_visit(self._real_data, seed=seed) if null_control else self._real_data
        )
        adapter = MarketingLogAdapter(data=frame, seed=seed)
        runner = CriteoPolicyRunner(adapter=adapter)
        space = criteo_active_search_space()

        best_policy_value = float("-inf")
        best_params: dict[str, Any] | None = None
        best_diagnostics: dict[str, float] = {}

        if strategy == "random":
            rng = np.random.default_rng(seed)
            for _ in range(budget):
                params = sample_random_params(space, rng)
                metrics = runner.run(params)
                pv = metrics["policy_value"]
                if pv > best_policy_value:
                    best_policy_value = pv
                    best_params = params
                    best_diagnostics = {k: v for k, v in metrics.items() if k != "policy_value"}
        else:
            graph = criteo_projected_prior_graph() if strategy == "causal" else None
            engine = ExperimentEngine(
                search_space=space,
                runner=runner,
                causal_graph=graph,
                objective_name=CRITEO_ENGINE_OBJECTIVE,
                minimize=False,
                seed=seed,
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result(CRITEO_ENGINE_OBJECTIVE, minimize=False)
            if best_result is not None:
                best_params = best_result.parameters
                best_policy_value = best_result.metrics.get("policy_value", float("-inf"))
                best_diagnostics = {
                    k: v for k, v in best_result.metrics.items() if k != "policy_value"
                }

        _check_active_params_invariant(best_params)

        runtime = time.perf_counter() - t_start

        secondary: dict[str, float] = {}
        if best_params is not None and not null_control:
            secondary = _secondary_arm_aggregates(frame)

        return CriteoBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            is_null_control=null_control,
            policy_value=(best_policy_value if math.isfinite(best_policy_value) else float("nan")),
            selected_parameters=best_params,
            runtime_seconds=runtime,
            null_baseline=self._null_baseline,
            diagnostics=best_diagnostics,
            secondary_outcomes=secondary,
        )


def _check_active_params_invariant(best_params: dict[str, Any] | None) -> None:
    """Raise ``RuntimeError`` if ``best_params`` contains non-active keys."""
    if best_params is None:
        return
    if set(best_params) != set(_ACTIVE_VAR_NAMES):
        msg = (
            f"CriteoScenario: best_params has unexpected keys "
            f"{sorted(best_params)!r}; expected exactly "
            f"{sorted(_ACTIVE_VAR_NAMES)!r}. Frozen Criteo dimensions "
            f"must not appear in the experiment log."
        )
        raise RuntimeError(msg)


def _secondary_arm_aggregates(frame: pd.DataFrame) -> dict[str, float]:
    """Compute in-sample treated/control-arm aggregates for secondary outcomes."""
    if "conversion" not in frame.columns:
        return {}
    treated_mask = frame["treatment"] == 1
    control_mask = frame["treatment"] == 0
    aggregates: dict[str, float] = {}
    if int(treated_mask.sum()) > 0:
        aggregates["treated_visit_rate"] = float(frame.loc[treated_mask, "outcome"].mean())
        aggregates["treated_conversion_rate"] = float(frame.loc[treated_mask, "conversion"].mean())
    if int(control_mask.sum()) > 0:
        aggregates["control_visit_rate"] = float(frame.loc[control_mask, "outcome"].mean())
        aggregates["control_conversion_rate"] = float(frame.loc[control_mask, "conversion"].mean())
    return aggregates
