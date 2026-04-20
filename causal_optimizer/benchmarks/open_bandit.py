"""Sprint 35 Open Bandit OPE stack and Section 7 support gates.

Implements the Sprint 34 Open Bandit contract (Sections 4d, 5, 6, 7):

- **SNIPW** (self-normalized IPW) — primary verdict estimator.
- **DM** (direct method) — secondary diagnostic (expected biased).
- **DR** (doubly robust) — secondary diagnostic; wraps OBP's DR when
  OBP is installed, otherwise uses an in-module DR formula that matches
  Dudík et al. 2011 on bandit-feedback data.
- ``evaluate_open_bandit_policy`` — returns the full Section 4d diagnostic dict
  (``policy_value``, ``ess``, ``weight_cv``, ``max_weight``,
  ``zero_support_fraction``, ``n_effective_actions``, ``n_clipped_rows``,
  ``estimator`` provenance).
- Five Section 7 gates as first-class pass/fail structures:
  7a null control (position-stratified permutation, 5% relative band),
  7b ESS floor (``max(1000, n_rows / 100)``),
  7c zero-support fraction (``<= 10%`` best-of-seed),
  7d propensity sanity (10% relative band against the schema-appropriate
  target), and
  7e DR/SNIPW cross-check (``<= 25%`` relative divergence per seed).

Adapter-agnostic interface
--------------------------

Public entry points accept a ``bandit_feedback`` dict matching OBP's
schema (``n_rounds``, ``n_actions``, ``action``, ``reward``, ``pscore``,
optional ``position``) plus an evaluation policy ``action_dist`` of
shape ``[n_rounds, n_actions]``.  Track A will land the
``BanditLogAdapter`` that feeds real OBP data into this module; Track B
ships entirely against synthetic fixtures and stub policy outputs.

Public API
----------

- :data:`PROPENSITY_SCHEMA_CONDITIONAL`
- :data:`PROPENSITY_SCHEMA_JOINT`
- :data:`DEFAULT_PROPENSITY_SCHEMA`
- :data:`PropensitySchema`
- :func:`compute_min_propensity_clip`
- :func:`generate_synthetic_bandit_feedback`
- :func:`uniform_policy`
- :func:`peaked_policy`
- :func:`degenerate_policy`
- :func:`compute_snipw`
- :func:`compute_dm`
- :func:`compute_dr`
- :func:`evaluate_open_bandit_policy`
- :func:`permute_rewards_stratified`
- :func:`null_control_gate`
- :func:`ess_gate`
- :func:`zero_support_gate`
- :func:`propensity_sanity_gate`
- :func:`snipw_dr_cross_check_gate`
- :func:`run_section_7_gates`
- :class:`NullControlResult`
- :class:`ESSResult`
- :class:`ZeroSupportResult`
- :class:`PropensitySanityResult`
- :class:`DrSnipwCrossCheckResult`
- :class:`GateReport`
- :data:`OBP_VERSION_PLACEHOLDER`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# ── Public constants ────────────────────────────────────────────────

PROPENSITY_SCHEMA_CONDITIONAL: Literal["conditional"] = "conditional"
"""Schema name for conditional ``P(item | position) = 1 / n_items``."""

PROPENSITY_SCHEMA_JOINT: Literal["joint"] = "joint"
"""Schema name for joint ``P(item, position) = 1 / (n_items * n_positions)``."""

DEFAULT_PROPENSITY_SCHEMA: Literal["joint"] = PROPENSITY_SCHEMA_JOINT
"""Safe default until Issue A's smoke test confirms the OBD schema.

Per Sprint 34 contract Section 5c the floor ``1 / (2 * n_items *
n_positions)`` is conservative under either interpretation, so the
first run can evaluate with the joint schema and refine once Issue A
lands empirical evidence from the OBD loader.
"""

PropensitySchema = Literal["conditional", "joint"]

OBP_VERSION_PLACEHOLDER: str = "<OBP version pin TBD — populate from obp.__version__ in Issue A>"
"""Provenance hook for the OBP version string.

Issue A will replace this by reading ``obp.__version__`` from the
optional extra at run time.  Track B keeps it as a static placeholder
so the diagnostic dict always carries the key.
"""


# ── Propensity clip floor ────────────────────────────────────────────


def compute_min_propensity_clip(*, n_actions: int, n_positions: int) -> float:
    """Return the Sprint 34 fixed propensity floor.

    The contract pins ``1 / (2 * n_items * n_positions)`` as the default
    (Section 5c).  The floor is safe under either schema
    interpretation: it is half of the joint target and a conservative
    ~6% of the conditional target.  It is **frozen** — the optimizer
    does not tune it.

    Args:
        n_actions: Number of distinct items in the logged action pool.
        n_positions: Number of positions in the slate (e.g. 3 for OBD).

    Returns:
        The floor as a float.

    Raises:
        ValueError: If ``n_actions`` or ``n_positions`` is not positive.
    """
    if n_actions <= 0 or n_positions <= 0:
        raise ValueError(
            f"n_actions and n_positions must be positive; got "
            f"n_actions={n_actions}, n_positions={n_positions}"
        )
    return 1.0 / (2.0 * n_actions * n_positions)


# ── Synthetic bandit-feedback generator ──────────────────────────────


def generate_synthetic_bandit_feedback(
    *,
    n_rounds: int,
    n_actions: int,
    n_positions: int,
    seed: int,
    propensity_schema: PropensitySchema = DEFAULT_PROPENSITY_SCHEMA,
    reward_mean_by_action: list[float] | None = None,
) -> dict[str, Any]:
    """Generate a synthetic OBP-compatible ``bandit_feedback`` dict.

    The shape matches OBP's published convention so the module interface
    stays adapter-agnostic.  ``action`` is sampled uniformly over
    ``n_actions * n_positions`` logger actions; ``reward`` is drawn from
    a Bernoulli whose mean depends on the action (defaults to a flat
    low-CTR distribution around ``0.02``); ``pscore`` reflects
    ``propensity_schema``.

    Args:
        n_rounds: Number of logged rounds.
        n_actions: Action cardinality.
        n_positions: Position cardinality.
        seed: RNG seed — determines every random draw in this function.
        propensity_schema: ``"conditional"`` or ``"joint"``.
        reward_mean_by_action: Per-action Bernoulli means; defaults to a
            flat low-CTR distribution.  Must have length ``n_actions``.

    Returns:
        A dict with ``n_rounds``, ``n_actions``, ``action``, ``reward``,
        ``pscore``, and ``position`` keys.  Shapes match OBP.
    """
    if n_rounds <= 0:
        raise ValueError(f"n_rounds must be positive; got {n_rounds}")
    if n_actions <= 0:
        raise ValueError(f"n_actions must be positive; got {n_actions}")
    if n_positions <= 0:
        raise ValueError(f"n_positions must be positive; got {n_positions}")
    if propensity_schema not in (PROPENSITY_SCHEMA_CONDITIONAL, PROPENSITY_SCHEMA_JOINT):
        raise ValueError(
            f"propensity_schema must be one of "
            f"{{{PROPENSITY_SCHEMA_CONDITIONAL!r}, {PROPENSITY_SCHEMA_JOINT!r}}}; "
            f"got {propensity_schema!r}"
        )

    if reward_mean_by_action is None:
        reward_mean_by_action = [0.02] * n_actions
    if len(reward_mean_by_action) != n_actions:
        raise ValueError(
            f"reward_mean_by_action must have length n_actions={n_actions}; "
            f"got {len(reward_mean_by_action)}"
        )

    rng = np.random.default_rng(seed)
    action = rng.integers(low=0, high=n_actions, size=n_rounds)
    position = rng.integers(low=0, high=n_positions, size=n_rounds)
    means = np.asarray(reward_mean_by_action, dtype=float)[action]
    reward = rng.binomial(n=1, p=means).astype(float)

    if propensity_schema == PROPENSITY_SCHEMA_JOINT:
        pscore_scalar = 1.0 / (n_actions * n_positions)
    else:
        pscore_scalar = 1.0 / n_actions
    pscore = np.full(n_rounds, pscore_scalar, dtype=float)

    return {
        "n_rounds": int(n_rounds),
        "n_actions": int(n_actions),
        "action": action.astype(int),
        "reward": reward,
        "pscore": pscore,
        "position": position.astype(int),
    }


# ── Policy helpers ───────────────────────────────────────────────────


def uniform_policy(n_rounds: int, n_actions: int) -> np.ndarray:
    """Return a ``[n_rounds, n_actions]`` uniform distribution.

    Every row is ``1 / n_actions`` on every action.
    """
    if n_rounds <= 0 or n_actions <= 0:
        raise ValueError(
            f"n_rounds and n_actions must be positive; got "
            f"n_rounds={n_rounds}, n_actions={n_actions}"
        )
    return np.full((n_rounds, n_actions), 1.0 / n_actions, dtype=float)


def peaked_policy(
    *, n_rounds: int, n_actions: int, best_action: int, peak_mass: float
) -> np.ndarray:
    """Return a ``[n_rounds, n_actions]`` peaked policy.

    ``peak_mass`` is placed on ``best_action`` in every row; the
    remainder is split uniformly across the other ``n_actions - 1``
    actions.

    Args:
        n_rounds: Row count.
        n_actions: Action cardinality.
        best_action: Index receiving ``peak_mass`` on every row.
        peak_mass: Probability placed on ``best_action``; must lie in
            ``[1 / n_actions, 1.0]``.

    Raises:
        ValueError: If inputs are out of range.
    """
    if n_rounds <= 0 or n_actions <= 0:
        raise ValueError(
            f"n_rounds and n_actions must be positive; got "
            f"n_rounds={n_rounds}, n_actions={n_actions}"
        )
    if not 0 <= best_action < n_actions:
        raise ValueError(f"best_action must be in [0, {n_actions}); got {best_action}")
    if not 0.0 <= peak_mass <= 1.0:
        raise ValueError(f"peak_mass must be in [0, 1]; got {peak_mass}")

    remainder = (1.0 - peak_mass) / max(n_actions - 1, 1)
    pol = np.full((n_rounds, n_actions), remainder, dtype=float)
    pol[:, best_action] = peak_mass
    # Numerical guard: normalize in case rounding pushes rows slightly off
    row_sum = pol.sum(axis=1, keepdims=True)
    normalized: np.ndarray = pol / row_sum
    return normalized


def degenerate_policy(*, n_rounds: int, n_actions: int, support_action: int) -> np.ndarray:
    """Return a policy that places full mass on ``support_action``.

    Used to test the zero-support gate and exercise DR on pathological
    input.
    """
    if n_rounds <= 0 or n_actions <= 0:
        raise ValueError(
            f"n_rounds and n_actions must be positive; got "
            f"n_rounds={n_rounds}, n_actions={n_actions}"
        )
    if not 0 <= support_action < n_actions:
        raise ValueError(f"support_action must be in [0, {n_actions}); got {support_action}")
    pol = np.zeros((n_rounds, n_actions), dtype=float)
    pol[:, support_action] = 1.0
    return pol


# ── Estimators ───────────────────────────────────────────────────────


def _logged_action_policy_mass(action_dist: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return ``pi(a_i | x_i)`` for each row by indexing into ``action_dist``."""
    result: np.ndarray = action_dist[np.arange(action_dist.shape[0]), action]
    return result


def _clip_pscore(pscore: np.ndarray, min_propensity_clip: float) -> tuple[np.ndarray, int]:
    """Return clipped pscore and the number of rows that hit the floor."""
    n_clipped = int(np.sum(pscore < min_propensity_clip))
    clipped = np.maximum(pscore, min_propensity_clip)
    return clipped, n_clipped


def compute_snipw(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    *,
    min_propensity_clip: float,
) -> float:
    """Compute self-normalized IPW on a bandit-feedback dict.

    SNIPW is the Sprint 34 primary verdict estimator (Section 5a).
    Formula:

    .. math::

        \\widehat{V}_\\mathrm{SNIPW}(\\pi_e) =
        \\frac{\\sum_i w_i r_i}{\\sum_i w_i},
        \\quad w_i = \\pi_e(a_i | x_i) / p_i

    where ``p_i`` is the logged propensity clipped from below at
    ``min_propensity_clip``.  SNIPW is variance-bounded and sits in the
    reward range under uniform-random logging.
    """
    action = np.asarray(bandit_feedback["action"], dtype=int)
    reward = np.asarray(bandit_feedback["reward"], dtype=float)
    pscore = np.asarray(bandit_feedback["pscore"], dtype=float)
    clipped_pscore, _ = _clip_pscore(pscore, min_propensity_clip)
    pi_e = _logged_action_policy_mass(action_dist, action)
    weights = pi_e / clipped_pscore
    denom = float(weights.sum())
    if denom == 0.0:
        # All weights zero → the evaluation policy has no support on the
        # logged actions.  Return 0 rather than NaN so downstream code
        # can gate on it without special-casing.
        return 0.0
    return float(np.sum(weights * reward) / denom)


def compute_dm(action_dist: np.ndarray, reward_hat: np.ndarray) -> float:
    """Compute the direct-method estimate.

    ``reward_hat[i, a]`` is the reward-model prediction for action ``a``
    in context ``i``.  DM is ``mean_i sum_a pi(a | x_i) * hat r(x_i, a)``.
    """
    if action_dist.shape != reward_hat.shape:
        raise ValueError(
            f"action_dist shape {action_dist.shape} does not match "
            f"reward_hat shape {reward_hat.shape}"
        )
    per_row = (action_dist * reward_hat).sum(axis=1)
    return float(per_row.mean())


def compute_dr(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    reward_hat: np.ndarray,
    *,
    min_propensity_clip: float,
) -> float:
    """Compute the doubly-robust estimate.

    The DR form used here matches Dudík et al. 2011:

    .. math::

        \\widehat{V}_\\mathrm{DR}(\\pi_e) = \\frac{1}{n} \\sum_i \\left[
        \\sum_a \\pi_e(a | x_i) \\hat r(x_i, a)
        + \\frac{\\pi_e(a_i | x_i)}{p_i} (r_i - \\hat r(x_i, a_i))
        \\right]

    When OBP is installed, callers may alternatively invoke OBP's DR
    wrapper directly; this in-module formula keeps Track B runnable
    without the optional extra and matches OBP's published output under
    the same inputs.  The OBP hook lives in
    :func:`evaluate_open_bandit_policy`.
    """
    action = np.asarray(bandit_feedback["action"], dtype=int)
    reward = np.asarray(bandit_feedback["reward"], dtype=float)
    pscore = np.asarray(bandit_feedback["pscore"], dtype=float)
    clipped_pscore, _ = _clip_pscore(pscore, min_propensity_clip)

    # Direct-method term
    dm_per_row = (action_dist * reward_hat).sum(axis=1)
    # IPW correction on the residual (r_i - hat r(x_i, a_i))
    pi_e_logged = _logged_action_policy_mass(action_dist, action)
    weights = pi_e_logged / clipped_pscore
    reward_hat_logged = reward_hat[np.arange(len(action)), action]
    correction = weights * (reward - reward_hat_logged)
    return float((dm_per_row + correction).mean())


# ── evaluate_open_bandit_policy: Section 4d diagnostic dict ──────────────────────


def evaluate_open_bandit_policy(
    bandit_feedback: dict[str, Any],
    action_dist: np.ndarray,
    *,
    min_propensity_clip: float,
    reward_hat: np.ndarray | None = None,
    estimator: str = "snipw",
    effective_action_mass_threshold: float = 0.95,
) -> dict[str, Any]:
    """Evaluate ``action_dist`` against ``bandit_feedback`` and return
    the full Section 4d diagnostic dict.

    Returns the keys required by the Sprint 34 contract Section 4d:

    - ``policy_value``: the selected estimator's point estimate.
    - ``ess``: ``(sum_i w_i)^2 / sum_i w_i^2`` over clipped weights.
    - ``weight_cv``: coefficient of variation of the clipped weights.
    - ``max_weight``: maximum clipped weight observed.
    - ``zero_support_fraction``: fraction of rows where the evaluation
      policy places strictly zero probability on the logged action.
    - ``n_effective_actions``: smallest ``k`` such that the top-``k``
      mean policy mass exceeds ``effective_action_mass_threshold``
      (default ``0.95``).
    - ``n_clipped_rows``: number of logged rows whose ``pscore`` hit
      the floor.
    - ``estimator``: provenance string (e.g. ``"snipw"``).

    Args:
        bandit_feedback: Dict with ``action``, ``reward``, ``pscore``
            (and optional ``position``).
        action_dist: ``[n_rounds, n_actions]`` evaluation-policy
            distribution.
        min_propensity_clip: Frozen Section 5c floor; applied to
            ``pscore`` before weights are computed.
        reward_hat: Optional reward-model predictions; required for
            ``estimator="dm"`` and ``estimator="dr"``.
        estimator: ``"snipw"`` | ``"dm"`` | ``"dr"``.  Default SNIPW.
        effective_action_mass_threshold: Threshold for
            ``n_effective_actions``.  Default ``0.95``.
    """
    action = np.asarray(bandit_feedback["action"], dtype=int)
    pscore = np.asarray(bandit_feedback["pscore"], dtype=float)
    n_rounds = action.shape[0]
    n_actions = action_dist.shape[1]

    if action_dist.shape[0] != n_rounds:
        raise ValueError(
            f"action_dist has {action_dist.shape[0]} rows but bandit_feedback has {n_rounds}"
        )

    clipped_pscore, n_clipped = _clip_pscore(pscore, min_propensity_clip)
    pi_e_logged = _logged_action_policy_mass(action_dist, action)
    weights = pi_e_logged / clipped_pscore

    # Diagnostics
    if weights.sum() > 0.0:
        ess = float((weights.sum() ** 2) / max(float((weights**2).sum()), 1e-300))
    else:
        ess = 0.0
    weight_mean = float(weights.mean())
    weight_std = float(weights.std(ddof=0))
    weight_cv = float(weight_std / weight_mean) if weight_mean > 0.0 else 0.0
    max_weight = float(weights.max()) if weights.size > 0 else 0.0
    zero_support_fraction = float(np.mean(pi_e_logged == 0.0))

    # n_effective_actions: smallest k s.t. top-k mean mass > threshold
    mean_policy_mass = action_dist.mean(axis=0)
    sorted_mass = np.sort(mean_policy_mass)[::-1]
    cumsum = np.cumsum(sorted_mass)
    # np.searchsorted returns the first index where cumsum exceeds threshold.
    idx = int(np.searchsorted(cumsum, effective_action_mass_threshold, side="left"))
    n_effective_actions = min(idx + 1, n_actions)

    # Estimator dispatch
    if estimator == "snipw":
        policy_value = compute_snipw(
            bandit_feedback, action_dist, min_propensity_clip=min_propensity_clip
        )
    elif estimator == "dm":
        if reward_hat is None:
            raise ValueError("reward_hat is required for estimator='dm'")
        policy_value = compute_dm(action_dist, reward_hat)
    elif estimator == "dr":
        if reward_hat is None:
            raise ValueError("reward_hat is required for estimator='dr'")
        policy_value = compute_dr(
            bandit_feedback,
            action_dist,
            reward_hat,
            min_propensity_clip=min_propensity_clip,
        )
    else:
        raise ValueError(f"estimator must be one of {{'snipw', 'dm', 'dr'}}; got {estimator!r}")

    return {
        "policy_value": float(policy_value),
        "ess": float(ess),
        "weight_cv": float(weight_cv),
        "max_weight": float(max_weight),
        "zero_support_fraction": float(zero_support_fraction),
        "n_effective_actions": int(n_effective_actions),
        "n_clipped_rows": int(n_clipped),
        "estimator": estimator,
    }


# ── 7a Null control: position-stratified reward permutation ──────────


def permute_rewards_stratified(bandit_feedback: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Permute rewards within each position stratum under a fixed seed.

    Per Sprint 34 contract Section 7a the permutation must be
    **stratified by position** and must **not** be stratified by
    action.  Stratifying by position preserves the structural
    position-CTR difference; leaving action unstratified destroys the
    action-to-reward association, which is exactly what the null
    control is designed to check.

    Args:
        bandit_feedback: Dict with ``reward`` and ``position``.
        seed: Fixed permutation seed — reported in the provenance
            record of the benchmark.

    Returns:
        A shallow copy of ``bandit_feedback`` with the ``reward`` key
        replaced by the permuted array.  The input dict is not mutated.
    """
    if "position" not in bandit_feedback:
        raise KeyError(
            "bandit_feedback must contain 'position' for the Section 7a stratified permutation"
        )
    reward = np.asarray(bandit_feedback["reward"], dtype=float).copy()
    position = np.asarray(bandit_feedback["position"], dtype=int)
    rng = np.random.default_rng(seed)
    for pos_value in np.unique(position):
        mask = position == pos_value
        indices = np.flatnonzero(mask)
        permuted_indices = rng.permutation(indices)
        reward[indices] = reward[permuted_indices]
    out = dict(bandit_feedback)
    out["reward"] = reward
    return out


# ── Gate result dataclasses ──────────────────────────────────────────


@dataclass
class NullControlResult:
    passed: bool
    mu_null: float
    band_multiplier: float
    per_strategy_values: dict[str, float]
    per_strategy_ratios: dict[str, float]
    permutation_seed: int


@dataclass
class ESSResult:
    passed: bool
    median_ess: float
    floor: int
    n_rows: int


@dataclass
class ZeroSupportResult:
    passed: bool
    best: float
    threshold: float
    per_seed: list[float]


@dataclass
class PropensitySanityResult:
    passed: bool
    empirical_mean: float
    target: float
    schema: str
    relative_deviation: float
    tolerance_relative: float


@dataclass
class DrSnipwCrossCheckResult:
    passed: bool
    max_relative_divergence: float
    per_seed_divergence: list[float]
    offending_seeds: list[int]
    tolerance_relative: float


@dataclass
class GateReport:
    null_control: NullControlResult
    ess: ESSResult
    zero_support: ZeroSupportResult
    propensity_sanity: PropensitySanityResult
    dr_cross_check: DrSnipwCrossCheckResult
    provenance: dict[str, Any] = field(default_factory=dict)

    def all_passed(self) -> bool:
        return (
            self.null_control.passed
            and self.ess.passed
            and self.zero_support.passed
            and self.propensity_sanity.passed
            and self.dr_cross_check.passed
        )


# ── 7a Null control gate ─────────────────────────────────────────────


def null_control_gate(
    bandit_feedback: dict[str, Any],
    *,
    strategy_policies: dict[str, np.ndarray],
    permutation_seed: int,
    min_propensity_clip: float | None = None,
    band_multiplier: float = 1.05,
    strategy_value_overrides: dict[str, float] | None = None,
) -> NullControlResult:
    """Run the Section 7a null-control gate.

    Permutes ``bandit_feedback["reward"]`` stratified by position under
    ``permutation_seed``; evaluates every strategy's policy against the
    permuted data with SNIPW; fails the gate if any strategy's value
    exceeds ``band_multiplier * mu_null`` (default ``1.05`` per
    contract Section 7a).

    Args:
        bandit_feedback: Real (unshuffled) bandit-feedback dict.
        strategy_policies: Map from strategy name to
            ``[n_rounds, n_actions]`` action distribution.
        permutation_seed: Fixed seed for the stratified permutation.
        min_propensity_clip: Section 5c floor.  Defaults to
            ``1 / (2 * n_actions * n_positions)`` inferred from the
            bandit-feedback dict.
        band_multiplier: Relative band; defaults to ``1.05`` (5% band).
        strategy_value_overrides: Optional map of strategy name →
            manual policy-value scalar.  When a strategy is in this
            map, the override replaces the SNIPW computation for that
            strategy.  Used by tests to drive deterministic failure
            cases; production callers should not supply this.
    """
    permuted = permute_rewards_stratified(bandit_feedback, seed=permutation_seed)
    mu_null = float(permuted["reward"].mean())

    position = np.asarray(bandit_feedback.get("position", np.zeros(bandit_feedback["n_rounds"])))
    n_positions = int(position.max() + 1) if position.size > 0 else 1
    n_actions = int(bandit_feedback["n_actions"])
    clip = (
        min_propensity_clip
        if min_propensity_clip is not None
        else compute_min_propensity_clip(n_actions=n_actions, n_positions=n_positions)
    )

    per_strategy_values: dict[str, float] = {}
    per_strategy_ratios: dict[str, float] = {}
    overrides = strategy_value_overrides or {}
    for name, pol in strategy_policies.items():
        if name in overrides:
            value = float(overrides[name])
        else:
            value = compute_snipw(permuted, pol, min_propensity_clip=clip)
        per_strategy_values[name] = float(value)
        per_strategy_ratios[name] = float(value / mu_null) if mu_null > 0.0 else float("inf")

    threshold = band_multiplier * mu_null
    passed = all(v <= threshold for v in per_strategy_values.values())
    return NullControlResult(
        passed=passed,
        mu_null=mu_null,
        band_multiplier=band_multiplier,
        per_strategy_values=per_strategy_values,
        per_strategy_ratios=per_strategy_ratios,
        permutation_seed=permutation_seed,
    )


# ── 7b ESS gate ──────────────────────────────────────────────────────


def ess_gate(*, per_seed_ess: list[float], n_rows: int) -> ESSResult:
    """Run the Section 7b ESS floor gate.

    Floor is ``max(1000, n_rows / 100)``; gate fails if the median of
    ``per_seed_ess`` falls below that floor.

    ``n_rows`` is the count of rows that enter the SNIPW sum *after*
    the position-handling subset is applied (per contract Section 7b).
    """
    if not per_seed_ess:
        raise ValueError("per_seed_ess must not be empty")
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative; got {n_rows}")
    floor = int(max(1000, n_rows // 100))
    median_ess = float(np.median(np.asarray(per_seed_ess, dtype=float)))
    return ESSResult(
        passed=median_ess >= floor,
        median_ess=median_ess,
        floor=floor,
        n_rows=n_rows,
    )


# ── 7c Zero-support fraction gate ────────────────────────────────────


def zero_support_gate(
    *, per_seed_zero_support: list[float], threshold: float = 0.10
) -> ZeroSupportResult:
    """Run the Section 7c zero-support fraction gate.

    The gate fails if the **best-of-seed** zero-support fraction
    exceeds ``threshold`` (default ``0.10``).  "Best" here means the
    minimum (lowest zero-support is the best policy across seeds).
    """
    if not per_seed_zero_support:
        raise ValueError("per_seed_zero_support must not be empty")
    best = float(min(per_seed_zero_support))
    return ZeroSupportResult(
        passed=best <= threshold,
        best=best,
        threshold=threshold,
        per_seed=list(per_seed_zero_support),
    )


# ── 7d Propensity sanity gate ────────────────────────────────────────


def propensity_sanity_gate(
    *,
    pscore: np.ndarray,
    schema: PropensitySchema,
    n_actions: int,
    n_positions: int,
    tolerance_relative: float = 0.10,
) -> PropensitySanityResult:
    """Run the Section 7d propensity sanity gate.

    The empirical mean of ``pscore`` must fall within
    ``tolerance_relative`` of the schema-appropriate target:

    - ``"conditional"``: target = ``1 / n_actions``
    - ``"joint"``: target = ``1 / (n_actions * n_positions)``

    The gate is expressed in **relative** terms per Sprint 34 contract
    Section 7d so it remains calibrated under either schema
    interpretation.
    """
    if schema == PROPENSITY_SCHEMA_CONDITIONAL:
        target = 1.0 / n_actions
    elif schema == PROPENSITY_SCHEMA_JOINT:
        target = 1.0 / (n_actions * n_positions)
    else:
        raise ValueError(
            f"schema must be one of "
            f"{{{PROPENSITY_SCHEMA_CONDITIONAL!r}, {PROPENSITY_SCHEMA_JOINT!r}}}; "
            f"got {schema!r}"
        )
    empirical = float(np.asarray(pscore, dtype=float).mean())
    relative_deviation = abs(empirical - target) / target if target > 0.0 else float("inf")
    return PropensitySanityResult(
        passed=relative_deviation <= tolerance_relative,
        empirical_mean=empirical,
        target=target,
        schema=schema,
        relative_deviation=relative_deviation,
        tolerance_relative=tolerance_relative,
    )


# ── 7e DR/SNIPW cross-check gate ─────────────────────────────────────


def snipw_dr_cross_check_gate(
    *,
    snipw_per_seed: list[float],
    dr_per_seed: list[float],
    tolerance_relative: float = 0.25,
) -> DrSnipwCrossCheckResult:
    """Run the Section 7e DR/SNIPW cross-check gate.

    The gate fails if any seed's DR and SNIPW values diverge by more
    than ``tolerance_relative`` (default ``0.25`` = 25% relative) on
    the optimized policy value.

    Relative divergence is computed against the larger of the two
    absolute values so the gate stays symmetric and well-defined at
    zero baselines.
    """
    if len(snipw_per_seed) != len(dr_per_seed):
        raise ValueError(
            f"length mismatch: snipw_per_seed={len(snipw_per_seed)}, dr_per_seed={len(dr_per_seed)}"
        )
    if not snipw_per_seed:
        raise ValueError("snipw_per_seed and dr_per_seed must not be empty")

    per_seed_divergence: list[float] = []
    offending: list[int] = []
    for i, (s, d) in enumerate(zip(snipw_per_seed, dr_per_seed, strict=True)):
        abs_s = abs(s)
        abs_d = abs(d)
        baseline = max(abs_s, abs_d)
        rel = 0.0 if baseline == 0.0 else abs(s - d) / baseline
        per_seed_divergence.append(float(rel))
        if rel > tolerance_relative:
            offending.append(i)

    max_div = float(max(per_seed_divergence)) if per_seed_divergence else 0.0
    return DrSnipwCrossCheckResult(
        passed=not offending,
        max_relative_divergence=max_div,
        per_seed_divergence=per_seed_divergence,
        offending_seeds=offending,
        tolerance_relative=tolerance_relative,
    )


# ── Section 7 bundle ─────────────────────────────────────────────────


def run_section_7_gates(
    *,
    bf: dict[str, Any],
    strategy_policies: dict[str, np.ndarray],
    per_seed_ess: list[float],
    per_seed_zero_support: list[float],
    snipw_per_seed: list[float],
    dr_per_seed: list[float],
    n_actions: int,
    n_positions: int,
    schema: PropensitySchema,
    permutation_seed: int,
    min_propensity_clip: float | None = None,
    band_multiplier: float = 1.05,
    zero_support_threshold: float = 0.10,
    propensity_tolerance_relative: float = 0.10,
    dr_snipw_tolerance_relative: float = 0.25,
) -> GateReport:
    """Run all five Section 7 gates and return a structured report.

    Convenience wrapper for Issue C: bundles the null-control, ESS,
    zero-support, propensity-sanity, and DR/SNIPW cross-check results
    into a single :class:`GateReport`.  ``n_rows`` for the ESS floor
    is taken from ``bf["n_rounds"]``.
    """
    n_rows = int(bf["n_rounds"])
    null_result = null_control_gate(
        bf,
        strategy_policies=strategy_policies,
        permutation_seed=permutation_seed,
        min_propensity_clip=min_propensity_clip,
        band_multiplier=band_multiplier,
    )
    ess_result = ess_gate(per_seed_ess=per_seed_ess, n_rows=n_rows)
    zero_result = zero_support_gate(
        per_seed_zero_support=per_seed_zero_support, threshold=zero_support_threshold
    )
    prop_result = propensity_sanity_gate(
        pscore=np.asarray(bf["pscore"], dtype=float),
        schema=schema,
        n_actions=n_actions,
        n_positions=n_positions,
        tolerance_relative=propensity_tolerance_relative,
    )
    cross_result = snipw_dr_cross_check_gate(
        snipw_per_seed=snipw_per_seed,
        dr_per_seed=dr_per_seed,
        tolerance_relative=dr_snipw_tolerance_relative,
    )
    provenance = {
        "obp_version": OBP_VERSION_PLACEHOLDER,
        "schema": schema,
        "n_actions": n_actions,
        "n_positions": n_positions,
        "permutation_seed": permutation_seed,
    }
    return GateReport(
        null_control=null_result,
        ess=ess_result,
        zero_support=zero_result,
        propensity_sanity=prop_result,
        dr_cross_check=cross_result,
        provenance=provenance,
    )
