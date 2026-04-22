"""Sprint 35.C Open Bandit benchmark runner.

Wires together:

1. the ``BanditLogAdapter`` (Sprint 35.A) which exposes the six-variable
   item-scoring search space over a logged bandit-feedback dict;
2. the OPE stack in :mod:`causal_optimizer.benchmarks.open_bandit`
   (Sprint 35.B) which provides SNIPW/DM/DR estimators and the
   Section 7 support gates;
3. the core :class:`ExperimentEngine` (for ``surrogate_only`` /
   ``causal`` strategies) and a uniform-random sampler (for the
   ``random`` baseline).

Public entry points in this module are adapter-agnostic — they accept a
pre-materialized OBP-style ``bandit_feedback`` dict.  The thin
:mod:`scripts.open_bandit_benchmark` CLI translates a ``--data-path`` to
a dict via :func:`load_men_random_slice`.

Scope — Sprint 34 contract Section 3
-----------------------------------

- Men / uniform-Random campaign slice only.
- SNIPW primary; DM and DR secondary.
- Section 7 gate evaluation happens in the caller
  (``scripts/open_bandit_benchmark.py``) — this module ships the
  per-strategy loop and surfaces per-seed diagnostics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from causal_optimizer.benchmarks.open_bandit import (
    compute_dm,
    compute_dr,
    compute_min_propensity_clip,
    compute_snipw,
    permute_rewards_stratified,
)
from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter
from causal_optimizer.engine.loop import ExperimentEngine

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ── Public constants ────────────────────────────────────────────────

VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})
"""Strategies supported by :class:`OpenBanditScenario`.  The same three
strategies as the Criteo and Hillstrom harnesses."""

OBD_ENGINE_OBJECTIVE: str = "policy_value"
"""Primary objective fed to :class:`ExperimentEngine`."""

OBD_N_POSITIONS: int = 3
"""Men/Random ships three positions (left, center, right) per
Saito et al. 2021 Table 1."""

_DEFAULT_POSITION_HANDLING_FLAG: str = "position_1_only"
"""Default Section 4c position-handling flag, applied to SNIPW/DM/DR
evaluation of best-of-seed policies when the optimizer picked an
invalid literal.  Matches the Sprint 34 contract Section 4c default."""


# ── Loader: raw CSV → OBP-shaped bandit_feedback ──────────────────


def build_bandit_feedback_from_raw(
    *,
    data: pd.DataFrame,
    item_context: pd.DataFrame,
) -> dict[str, Any]:
    """Translate raw Men/Random CSVs into an OBP-shaped bandit_feedback dict.

    OBP 0.4.1's default ``pre_process`` calls
    ``DataFrame.drop(..., 1)`` with a positional ``axis`` which modern
    pandas rejects; this function bypasses that path and reads the raw
    columns directly, matching the Sprint 35.A smoke-test approach.

    Parameters
    ----------
    data:
        The raw ``men.csv`` frame. Must carry ``timestamp``, ``item_id``,
        ``position``, ``click``, ``propensity_score``, and the 34
        ``user-item_affinity_<k>`` columns.
    item_context:
        The raw ``item_context.csv`` frame. Must carry ``item_id`` and
        ``item_feature_0``.

    Returns
    -------
    A dict with OBP keys ``n_rounds``, ``n_actions``, ``action``,
    ``position``, ``reward``, ``pscore``, ``context``, ``action_context``.
    Positions are re-ranked to 0-indexed contiguous integers via
    ``scipy.stats.rankdata(..., "dense") - 1`` to match OBP's convention.
    """
    from scipy.stats import rankdata

    required_data_cols = {"item_id", "position", "click", "propensity_score"}
    missing = required_data_cols - set(data.columns)
    if missing:
        raise ValueError(
            f"raw data frame is missing required columns: {sorted(missing)}. "
            "Expected the Men/Random men.csv schema from "
            "https://research.zozo.com/data_release/open_bandit_dataset.zip"
        )
    required_ic_cols = {"item_feature_0"}
    if not required_ic_cols.issubset(set(item_context.columns)):
        missing_ic = required_ic_cols - set(item_context.columns)
        raise ValueError(f"item_context frame is missing required columns: {sorted(missing_ic)}")

    if "timestamp" in data.columns:
        data = data.sort_values("timestamp").reset_index(drop=True)

    action = data["item_id"].to_numpy().astype(int)
    position = (rankdata(data["position"].to_numpy(), "dense") - 1).astype(int)
    reward = data["click"].to_numpy().astype(float)
    pscore = data["propensity_score"].to_numpy().astype(float)

    affinity_cols = sorted(
        (c for c in data.columns if c.startswith("user-item_affinity_")),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    context = data[affinity_cols].to_numpy().astype(float)

    # Sort item_context by item_id so column 0 of ``action_context``
    # matches action index 0 (BanditLogAdapter indexes into this array
    # by raw item id).
    if "item_id" in item_context.columns:
        item_context = item_context.sort_values("item_id")
    action_context = item_context[["item_feature_0"]].to_numpy().astype(float)

    # Defensive sanity: action indices must lie in [0, n_actions).
    n_actions = int(action_context.shape[0])
    if int(action.min()) < 0 or int(action.max()) >= n_actions:
        raise ValueError(
            f"raw action ids range [{int(action.min())}, {int(action.max())}] but "
            f"item_context provides {n_actions} items. The loader cannot align the "
            "action column to the item_context rows."
        )

    # Widen context to at least n_actions columns so the adapter's
    # per-candidate affinity lookup (context[:, :n_actions]) remains well
    # defined even under small fixtures. The Men/Random slice already
    # has 34 affinity columns, so this pad is a no-op on real data.
    if context.shape[1] < n_actions:
        pad = np.zeros((context.shape[0], n_actions - context.shape[1]), dtype=float)
        context = np.concatenate([context, pad], axis=1)

    return {
        "n_rounds": int(len(data)),
        "n_actions": n_actions,
        "action": action,
        "position": position,
        "reward": reward,
        "pscore": pscore,
        "context": context,
        "action_context": action_context,
    }


def load_men_random_slice(*, data_path: Path) -> dict[str, Any]:
    """Load the full Men/Random slice from a local Open Bandit Dataset root.

    Expects ``data_path`` to point at a directory laid out like::

        data_path/
          random/
            men/
              men.csv
              item_context.csv

    This matches the layout inside
    ``open_bandit_dataset.zip`` from
    ``https://research.zozo.com/data_release/``.

    Raises
    ------
    FileNotFoundError:
        When either CSV is missing under the expected layout.
    """
    import pandas as pd

    root = Path(data_path)
    men_csv = root / "random" / "men" / "men.csv"
    item_csv = root / "random" / "men" / "item_context.csv"
    if not men_csv.is_file():
        raise FileNotFoundError(
            f"Men/Random men.csv not found at {men_csv}. Expected the "
            "open_bandit_dataset.zip layout with a ``random/men/`` "
            "subdirectory under --data-path."
        )
    if not item_csv.is_file():
        raise FileNotFoundError(f"Men/Random item_context.csv not found at {item_csv}.")
    data = pd.read_csv(men_csv, index_col=0)
    item_context = pd.read_csv(item_csv, index_col=0)
    return build_bandit_feedback_from_raw(data=data, item_context=item_context)


# ── Policy action_dist builder ────────────────────────────────────


def build_policy_action_dist(
    *, adapter: BanditLogAdapter, parameters: dict[str, Any]
) -> np.ndarray:
    """Return the ``[n_rounds, n_actions]`` policy distribution for ``parameters``.

    Mirrors the adapter's softmax-over-linear-scores + epsilon-uniform
    math in :meth:`BanditLogAdapter.run_experiment`. The two paths share
    an identical closed-form definition of the policy, so feeding the
    dist returned here into :func:`compute_snipw` reproduces the adapter's
    own ``policy_value`` (up to numerical noise from the pscore clip).

    For ``position_handling_flag == "position_1_only"``, rows that fall
    outside position 0 have their row set to the uniform distribution,
    which yields a neutral SNIPW contribution once the caller subsets to
    position 0 for the verdict. Callers that want strict subsetting can
    filter the bandit_feedback and the action_dist by
    ``position == 0`` in lockstep.
    """
    tau = float(parameters.get("tau", 1.0))
    eps = float(parameters.get("eps", 0.0))
    w_item = float(parameters.get("w_item_feature_0", 0.0))
    w_affinity = float(parameters.get("w_user_item_affinity", 0.0))
    w_popularity = float(parameters.get("w_item_popularity", 0.0))
    position_flag = str(parameters.get("position_handling_flag", _DEFAULT_POSITION_HANDLING_FLAG))

    n_rounds = adapter._n_rounds  # noqa: SLF001 — adapter is our own private surface
    n_actions = adapter._n_actions  # noqa: SLF001

    # Scores for every (row, action) candidate. Under
    # ``position_1_only`` the adapter instead scores the position-0
    # subset only (see :meth:`BanditLogAdapter.run_experiment`). Here
    # we score every row up front and then, below, overwrite the
    # non-position-0 rows with the uniform distribution; the overwrite
    # discards the score values on those rows, so the two paths agree
    # element-wise on position-0 rows and the benchmark SNIPW matches
    # the adapter's own ``policy_value``.
    item_term = w_item * adapter._item_feature_0[None, :]  # noqa: SLF001
    pop_term = w_popularity * adapter._item_popularity[None, :]  # noqa: SLF001
    affinity_term = w_affinity * adapter._affinity  # noqa: SLF001
    scores = item_term + pop_term + affinity_term  # shape (n_rounds, n_actions)

    safe_tau = max(tau, 1e-6)
    scaled = scores / safe_tau
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_scores = np.exp(scaled)
    softmax = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    uniform = np.full_like(softmax, 1.0 / n_actions)
    policy = (1.0 - eps) * softmax + eps * uniform

    if position_flag == "position_1_only":
        # Rows outside position 0 get the uniform distribution so the
        # row-sums remain valid probabilities and SNIPW on the full
        # feedback degrades to a uniform-policy contribution on the
        # masked rows. The per-verdict quote subsets to position 0.
        mask = adapter._position == 0  # noqa: SLF001
        if not mask.all():
            policy = policy.copy()
            policy[~mask] = 1.0 / n_actions
    elif position_flag != "marginalize":
        # The Sprint 34 contract Section 4c only allows two values;
        # reject unknown strings loudly rather than silently falling
        # through to the marginalize branch. The adapter does the same
        # validation on its own path.
        raise ValueError(
            f"position_handling_flag must be 'marginalize' or 'position_1_only', "
            f"got {position_flag!r}"
        )

    # Shape check on computed data (not a test assertion) — use an
    # explicit raise so the invariant survives `python -O`.
    if policy.shape != (n_rounds, n_actions):
        raise ValueError(
            f"build_policy_action_dist produced shape {policy.shape}, "
            f"expected {(n_rounds, n_actions)}"
        )
    return np.asarray(policy, dtype=float)


# ── Reward model for DM / DR ──────────────────────────────────────


def compute_reward_model(bandit_feedback: dict[str, Any], *, seed: int) -> np.ndarray:
    """Return an ``[n_rounds, n_actions]`` reward-model estimate.

    Used by DM and DR as the ``reward_hat`` surface.  The estimator is
    deliberately simple: for each action ``a`` we estimate
    ``E[reward | action=a]`` as the mean of the logged rewards for rows
    that selected that action, and broadcast that scalar per row.  This
    keeps the reward model honest enough to catch blatant DM bias while
    staying well under the per-budget runtime budget.

    ``seed`` is accepted for API consistency; the estimator is
    deterministic given ``bandit_feedback``.

    On Men/Random the per-action empirical CTR ranges from roughly
    ``0.003`` to ``0.010`` (overall CTR ~0.005), so the scalar-per-action
    reward hat is a reasonable zero-context baseline.
    """
    del seed  # reserved
    action = np.asarray(bandit_feedback["action"], dtype=int)
    reward = np.asarray(bandit_feedback["reward"], dtype=float)
    n_actions = int(bandit_feedback["n_actions"])
    n_rounds = int(bandit_feedback["n_rounds"])

    # Per-action mean reward; fall back to the global mean for actions
    # that never appear (should not happen on Men/Random but keeps the
    # estimator well defined on edge cases).
    action_sum = np.bincount(action, weights=reward, minlength=n_actions).astype(float)
    action_count = np.bincount(action, minlength=n_actions).astype(float)
    global_mean = float(reward.mean()) if reward.size > 0 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_counts = np.maximum(action_count, 1)
        action_mean = np.where(action_count > 0, action_sum / safe_counts, global_mean)
    # Clamp to [0, 1] — per-action means are already in that range on
    # binary rewards, the clamp is a defence against rare numerical drift.
    action_mean = np.clip(action_mean, 0.0, 1.0)

    # Broadcast to (n_rounds, n_actions): every row sees the same
    # per-action scalar. Richer reward models are deferred to Sprint 36+.
    reward_hat = np.broadcast_to(action_mean[None, :], (n_rounds, n_actions)).copy()
    return np.asarray(reward_hat, dtype=float)


# ── Result dataclass ──────────────────────────────────────────────


@dataclass
class OpenBanditBenchmarkResult:
    """Result of running one strategy on the Open Bandit benchmark.

    Attributes
    ----------
    strategy:
        One of ``"random"`` / ``"surrogate_only"`` / ``"causal"``.
    budget:
        Number of experiments evaluated by the strategy.
    seed:
        RNG seed for the strategy's optimizer.
    is_null_control:
        ``True`` when the reward column was permuted before evaluation.
    permutation_seed:
        The permutation seed used for the null control, or ``None``.
    policy_value_snipw:
        Primary verdict estimator. Reported on the best-of-budget
        parameters for every strategy.
    policy_value_dm:
        Secondary DM estimate, ``None`` when the reward model is disabled.
    policy_value_dr:
        Secondary DR estimate, ``None`` when the reward model is disabled.
    selected_parameters:
        Best-of-budget parameter dict returned by the strategy.
    runtime_seconds:
        Wall-clock runtime of the strategy run.
    diagnostics:
        The Section 4d diagnostic dict
        (``ess``, ``weight_cv``, ``max_weight``,
        ``zero_support_fraction``, ``n_effective_actions``) plus
        ``n_clipped_rows``.
    """

    strategy: str
    budget: int
    seed: int
    is_null_control: bool
    permutation_seed: int | None
    policy_value_snipw: float
    policy_value_dm: float | None
    policy_value_dr: float | None
    selected_parameters: dict[str, Any] | None = None
    runtime_seconds: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


# ── Strategy scenario ─────────────────────────────────────────────


class OpenBanditScenario:
    """Top-level Open Bandit benchmark scenario.

    Wraps a pre-materialized OBP-shaped bandit-feedback dict in a
    :class:`BanditLogAdapter` and runs one strategy at a given
    ``(budget, seed)`` pair.

    Parameters
    ----------
    bandit_feedback:
        OBP-shaped dict. Not mutated.
    use_reward_model:
        When ``True`` (default), compute a per-action reward-model
        baseline and report DM and DR estimates on the best-of-budget
        policy. Set to ``False`` to skip reward-model fitting.
    min_propensity_clip:
        Section 5c floor for SNIPW/DR. Defaults to
        ``1 / (2 * n_actions * n_positions)`` per contract Section 5c.
    """

    def __init__(
        self,
        *,
        bandit_feedback: dict[str, Any],
        use_reward_model: bool = True,
        min_propensity_clip: float | None = None,
    ) -> None:
        self._bandit_feedback = bandit_feedback
        self._use_reward_model = use_reward_model

        n_actions = int(bandit_feedback["n_actions"])
        # Infer n_positions from the data; callers should already have a
        # well-formed bandit_feedback with 0-indexed contiguous positions.
        position = np.asarray(bandit_feedback["position"], dtype=int)
        n_positions = int(position.max() + 1) if position.size > 0 else OBD_N_POSITIONS

        self._n_actions = n_actions
        self._n_positions = n_positions

        if min_propensity_clip is None:
            min_propensity_clip = compute_min_propensity_clip(
                n_actions=n_actions, n_positions=n_positions
            )
        self._min_propensity_clip = float(min_propensity_clip)

        self._reward_hat: np.ndarray | None
        if use_reward_model:
            self._reward_hat = compute_reward_model(bandit_feedback, seed=0)
        else:
            self._reward_hat = None

    # ── Accessors ────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def n_positions(self) -> int:
        return self._n_positions

    @property
    def min_propensity_clip(self) -> float:
        return self._min_propensity_clip

    @property
    def bandit_feedback(self) -> dict[str, Any]:
        return self._bandit_feedback

    # ── Core loop ────────────────────────────────────────────────

    def run_strategy(
        self,
        strategy: str,
        *,
        budget: int,
        seed: int,
        null_control: bool = False,
        permutation_seed: int | None = None,
    ) -> OpenBanditBenchmarkResult:
        """Run one strategy and return an :class:`OpenBanditBenchmarkResult`.

        Parameters
        ----------
        strategy:
            Must be one of :data:`VALID_STRATEGIES`.
        budget:
            Positive integer. Number of experiments the strategy runs.
        seed:
            RNG seed for the strategy.
        null_control:
            When ``True``, the scenario permutes the reward column
            stratified by position under ``permutation_seed`` and
            evaluates the strategy on the permuted data. Matches the
            Sprint 34 contract Section 7a.
        permutation_seed:
            Required when ``null_control`` is ``True``.
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Must be one of {sorted(VALID_STRATEGIES)}."
            )
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget!r}")
        if null_control and permutation_seed is None:
            raise ValueError("permutation_seed is required when null_control=True")

        if null_control:
            # ``permutation_seed`` is already validated above; narrow to
            # int without a runtime `assert` so the guard survives
            # `python -O`.
            perm_seed = int(permutation_seed) if permutation_seed is not None else 0
            bf = permute_rewards_stratified(self._bandit_feedback, seed=perm_seed)
            reward_hat = compute_reward_model(bf, seed=0) if self._use_reward_model else None
        else:
            bf = self._bandit_feedback
            reward_hat = self._reward_hat

        t_start = time.perf_counter()
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=seed)
        space = adapter.get_search_space()

        best_value = -math.inf
        best_params: dict[str, Any] | None = None
        best_metrics: dict[str, float] = {}

        if strategy == "random":
            rng = np.random.default_rng(seed)
            for _ in range(budget):
                params = sample_random_params(space, rng)
                metrics = adapter.run_experiment(params)
                pv = float(metrics.get("policy_value", -math.inf))
                if pv > best_value:
                    best_value = pv
                    best_params = params
                    # Cast to float to match the typed `diagnostics` contract
                    # and the engine-path branch below.
                    best_metrics = {k: float(v) for k, v in metrics.items() if k != "policy_value"}
        else:
            graph = adapter.get_prior_graph() if strategy == "causal" else None
            # Sprint 37 Option A1 (issue #197): only the ``causal`` arm
            # opts in to the minimal-focus heuristic; ``surrogate_only``
            # remains mechanically identical to its Sprint 35 behavior so
            # the comparison row is unchanged on the surrogate side.
            engine = ExperimentEngine(
                search_space=space,
                runner=adapter,
                causal_graph=graph,
                objective_name=OBD_ENGINE_OBJECTIVE,
                minimize=False,
                seed=seed,
                pomis_minimal_focus=(strategy == "causal"),
            )
            engine.run_loop(budget)
            best_result = engine.log.best_result(OBD_ENGINE_OBJECTIVE, minimize=False)
            if best_result is not None:
                best_params = dict(best_result.parameters)
                best_value = float(best_result.metrics.get("policy_value", -math.inf))
                best_metrics = {
                    k: float(v) for k, v in best_result.metrics.items() if k != "policy_value"
                }

        runtime = time.perf_counter() - t_start

        # ── Re-evaluate best-of-budget under SNIPW / DM / DR ──
        snipw_value = best_value if math.isfinite(best_value) else float("nan")
        dm_value: float | None = None
        dr_value: float | None = None
        diagnostics: dict[str, float] = dict(best_metrics)

        if best_params is not None:
            action_dist = build_policy_action_dist(adapter=adapter, parameters=best_params)
            # SNIPW primary. Re-computed from the action_dist so the
            # estimator's own clip is applied consistently (the adapter's
            # internal SNIPW does not clip). This is the value quoted in
            # the report.
            snipw_value = compute_snipw(
                bf, action_dist, min_propensity_clip=self._min_propensity_clip
            )
            if reward_hat is not None:
                dm_value = compute_dm(action_dist, reward_hat)
                dr_value = compute_dr(
                    bf, action_dist, reward_hat, min_propensity_clip=self._min_propensity_clip
                )

        resolved_permutation_seed: int | None = (
            int(permutation_seed) if null_control and permutation_seed is not None else None
        )
        return OpenBanditBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            is_null_control=bool(null_control),
            permutation_seed=resolved_permutation_seed,
            policy_value_snipw=float(snipw_value),
            policy_value_dm=None if dm_value is None else float(dm_value),
            policy_value_dr=None if dr_value is None else float(dr_value),
            selected_parameters=best_params,
            runtime_seconds=float(runtime),
            diagnostics=diagnostics,
        )


# ── Summaries ──────────────────────────────────────────────────────


def summarize_strategy_budget(
    results: list[OpenBanditBenchmarkResult],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Aggregate results into per-(strategy, budget) mean / std cells.

    Null-control results are ignored by this summary — reports should
    table real and permuted runs separately.
    """
    groups: dict[tuple[str, int], list[OpenBanditBenchmarkResult]] = {}
    for r in results:
        if r.is_null_control:
            continue
        key = (r.strategy, r.budget)
        groups.setdefault(key, []).append(r)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, bucket in groups.items():
        snipw = [r.policy_value_snipw for r in bucket if math.isfinite(r.policy_value_snipw)]
        dm = [r.policy_value_dm for r in bucket if r.policy_value_dm is not None]
        dr = [r.policy_value_dr for r in bucket if r.policy_value_dr is not None]
        out[key] = {
            "n_seeds": len(bucket),
            "mean_policy_value_snipw": float(np.mean(snipw)) if snipw else float("nan"),
            "std_policy_value_snipw": float(np.std(snipw, ddof=0)) if snipw else float("nan"),
            "mean_policy_value_dm": float(np.mean(dm)) if dm else None,
            "mean_policy_value_dr": float(np.mean(dr)) if dr else None,
            "seeds": [r.seed for r in bucket],
        }
    return out


__all__ = [
    "OBD_ENGINE_OBJECTIVE",
    "OBD_N_POSITIONS",
    "VALID_STRATEGIES",
    "OpenBanditBenchmarkResult",
    "OpenBanditScenario",
    "build_bandit_feedback_from_raw",
    "build_policy_action_dist",
    "compute_reward_model",
    "load_men_random_slice",
    "summarize_strategy_budget",
]
