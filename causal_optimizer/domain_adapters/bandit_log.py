"""Open Bandit ``DomainAdapter`` for logged multi-action policy data.

Implements the Sprint 34 Open Bandit contract (Sections 4 and 8) for the
ZOZOTOWN Men / uniform-random slice. Parameterizes a contextual
item-scoring policy in a six-variable search space (``tau`` softmax
temperature, ``eps`` exploration epsilon, three context-feature weights,
a ``position_handling_flag``) and evaluates it against a logged
bandit-feedback dict with a minimal in-house SNIPW-style estimator.

**Track A scope.** This adapter ships the Section 4 interface surface
and a narrow first-pass SNIPW-style ``policy_value`` calculation so the
adapter is reviewable end-to-end without blocking on the OPE stack.
Track B (Sprint 35.B, issue #186) owns the production OPE stack:
the OBP-backed SNIPW / DM / DR estimators, the Section 7 support gates,
and the cross-estimator cross-check.

Sprint 35.A smoke-test findings (Sprint 34 contract Section 10.A)
-----------------------------------------------------------------

1. **Row count.** The ``obp==0.4.1`` wheel bundles a 10,000-row sample
   of the Men/Random slice at
   ``site-packages/obp/dataset/obd/random/men/men.csv``. The full
   ~452,949-row slice from Saito et al. 2021 Table 1 must be loaded
   separately by passing ``data_path=`` to :meth:`from_obp`; Issue C
   will exercise that path, Issue A's smoke test pins only the bundled
   sample.

2. **``action_prob`` / ``pscore`` schema.** Confirmed as **conditional
   ``P(item | position)``**. The empirical mean of
   ``propensity_score`` on the bundled slice is ``0.0294117...``, which
   matches ``1/n_items = 1/34`` to floating-point precision. It does
   not match joint ``P(item, position) = 1/(n_items * n_positions) =
   1/102 ≈ 0.0098``. All three positions report the same constant
   propensity, consistent with the conditional interpretation under
   uniform-random logging. Section 7d's propensity-mean sanity gate
   should therefore use the ``1/n_items`` target on Men/Random.

3. **Chosen context features (three).** The adapter's search space
   pins three context-feature weights:

   - ``w_item_feature_0`` — multiplies the per-item continuous
     ``item_feature_0`` column from ``item_context.csv``
     (range ≈ ``[-0.73, 0.75]`` on Men/Random).
   - ``w_user_item_affinity`` — multiplies the per-row, per-candidate
     ``user-item_affinity_<k>`` lookup (34 columns, one per item id).
   - ``w_item_popularity`` — multiplies the per-item log-normalized
     in-log appearance count, pre-computed at adapter construction
     from ``bandit_feedback["action"]`` so the score is deterministic
     across policy evaluations.

   Rationale: these three features are the minimum honest subset that
   (i) exposes per-item continuous signal, (ii) exposes per-(row, item)
   heterogeneity, and (iii) anchors to an empirical popularity prior
   without blowing up the search dimensionality. Each weight has a
   continuous bound of ``[-3.0, 3.0]``.

Public interface
----------------

:class:`BanditLogAdapter` takes a pre-materialized OBP-style
bandit-feedback dict directly (the common case for unit tests and
downstream OPE harnesses). Use :meth:`BanditLogAdapter.from_obp` to
load the Men/Random slice through OBP; this path requires the optional
``bandit`` extra (``uv sync --extra bandit``) and fails fast with an
actionable error if ``obp`` is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import SearchSpace, Variable, VariableType

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph

__all__ = ["BanditLogAdapter"]


# ── Constants ────────────────────────────────────────────────────────

_REQUIRED_FEEDBACK_KEYS: frozenset[str] = frozenset(
    {
        "n_rounds",
        "n_actions",
        "action",
        "position",
        "reward",
        "pscore",
        "context",
        "action_context",
    }
)

_CONTEXT_WEIGHT_NAMES: tuple[str, ...] = (
    "w_item_feature_0",
    "w_user_item_affinity",
    "w_item_popularity",
)

_POSITION_HANDLING_CHOICES: tuple[str, ...] = ("marginalize", "position_1_only")

# Search-space bounds — see Sprint 34 contract Section 4c.
_TAU_LOWER: float = 0.1
_TAU_UPPER: float = 10.0
_EPS_LOWER: float = 0.0
_EPS_UPPER: float = 0.5
_WEIGHT_LOWER: float = -3.0
_WEIGHT_UPPER: float = 3.0

# Effective-action mass threshold: n_effective_actions is the smallest
# k such that the top-k actions' combined policy mass exceeds this
# threshold. Pinned at 0.95 per Sprint 34 contract Section 4d.
_EFFECTIVE_ACTIONS_MASS_THRESHOLD: float = 0.95

# Minimum tau used inside the softmax. If a caller passes ``tau=0`` (or
# numerically close) the adapter floors the temperature to keep the
# softmax numerically stable; the search space bounds already forbid
# zero, this is a defence against numerical edge cases inside the
# optimizer.
_TAU_FLOOR: float = 1e-6


# ── Adapter ──────────────────────────────────────────────────────────


class BanditLogAdapter(DomainAdapter):
    """``DomainAdapter`` for logged multi-action bandit-feedback data.

    Parameters
    ----------
    bandit_feedback:
        OBP-shaped ``dict`` with the keys ``n_rounds``, ``n_actions``,
        ``action``, ``position``, ``reward``, ``pscore``, ``context``,
        and ``action_context``. Use :meth:`from_obp` to build this
        dict from OBP's ``OpenBanditDataset``; pass one directly in
        unit tests and when consuming a pre-cached slice.
    seed:
        Reproducibility seed. Accepted for API consistency with
        other adapters — evaluation is fully deterministic given
        the bandit-feedback dict and parameters, so this value is
        forwarded to :class:`numpy.random.Generator` only if future
        stochastic diagnostics are added.

    Notes
    -----
    Sprint 34 contract Section 4a forbids subclassing
    ``MarketingLogAdapter``. This class inherits directly from
    :class:`DomainAdapter` and exposes the Section 4d diagnostic dict
    (``policy_value``, ``ess``, ``weight_cv``, ``max_weight``,
    ``zero_support_fraction``, ``n_effective_actions``).
    """

    def __init__(
        self,
        *,
        bandit_feedback: dict[str, Any],
        seed: int | None = None,
    ) -> None:
        self._validate_feedback(bandit_feedback)
        self._seed = seed

        self._n_rounds: int = int(bandit_feedback["n_rounds"])
        self._n_actions: int = int(bandit_feedback["n_actions"])
        self._action: np.ndarray = np.asarray(bandit_feedback["action"], dtype=np.int64)
        self._position: np.ndarray = np.asarray(bandit_feedback["position"], dtype=np.int64)
        self._reward: np.ndarray = np.asarray(bandit_feedback["reward"], dtype=np.int64)
        self._pscore: np.ndarray = np.asarray(bandit_feedback["pscore"], dtype=float)
        self._context: np.ndarray = np.asarray(bandit_feedback["context"], dtype=float)
        self._action_context: np.ndarray = np.asarray(
            bandit_feedback["action_context"], dtype=float
        )

        # Pre-compute per-item score components once. These are
        # independent of the policy parameters, so hoisting them out of
        # run_experiment keeps the optimizer loop cheap.
        self._item_feature_0: np.ndarray = self._action_context[:, 0].astype(float)

        # Per-row, per-candidate affinity matrix. The raw log's
        # ``user-item_affinity_<k>`` columns are per-row lookups for
        # candidate item k; ``context`` is therefore expected to have
        # one column per action. When ``context`` has fewer columns the
        # adapter falls back to a zero-affinity surface (still valid,
        # the search space's ``w_user_item_affinity`` weight then has
        # no effect — useful for small synthetic fixtures).
        if self._context.shape[1] >= self._n_actions:
            self._affinity: np.ndarray = self._context[:, : self._n_actions]
        else:
            self._affinity = np.zeros((self._n_rounds, self._n_actions), dtype=float)

        # Per-item popularity prior (log-normalized appearance count).
        # Computed once at construction from ``action`` so the prior is
        # deterministic across policy evaluations.
        counts = np.bincount(self._action, minlength=self._n_actions).astype(float)
        raw_max = float(counts.max())
        max_count = raw_max if raw_max > 0.0 else 1.0
        self._item_popularity: np.ndarray = np.log1p(counts) / np.log1p(max_count)

    # ── Constructor from OBP ──────────────────────────────────────────

    @classmethod
    def from_obp(
        cls,
        *,
        campaign: str = "men",
        behavior_policy: str = "random",
        data_path: Any | None = None,
        seed: int | None = None,
    ) -> BanditLogAdapter:
        """Build an adapter from an OBP ``OpenBanditDataset``.

        Loads the requested campaign / logger slice via OBP, extracts
        the bandit-feedback dict, and returns a ready-to-use adapter.
        Requires the optional ``bandit`` extra (``uv sync --extra
        bandit``); raises a clear :class:`ImportError` if the extra is
        not installed.

        Parameters
        ----------
        campaign:
            ZOZOTOWN campaign name. Must be one of ``"all"``, ``"men"``,
            or ``"women"``. Defaults to ``"men"`` per the Sprint 34
            first-slice decision.
        behavior_policy:
            Logging policy. Must be ``"random"`` or ``"bts"``. Defaults
            to ``"random"`` per the Sprint 34 first-slice decision.
        data_path:
            Optional ``pathlib.Path`` to the full released dataset.
            When ``None`` the OBP bundled small-sized sample is used
            (10,000 rows on Men/Random as of obp 0.4.1).
        seed:
            Reproducibility seed; forwarded to the adapter constructor.
        """
        try:
            # Inside-function import is deliberate: the ``obp`` package
            # is the optional ``bandit`` extra, and keeping this import
            # lazy is what lets the core adapter module stay importable
            # without the extra installed.
            from obp.dataset import OpenBanditDataset
        except ImportError as exc:  # pragma: no cover - exercised by monkeypatched test
            msg = (
                "BanditLogAdapter.from_obp requires the 'bandit' extra. "
                "Install it with: uv sync --extra bandit  "
                "(or: pip install 'causal-optimizer[bandit]')."
            )
            raise ImportError(msg) from exc

        dataset = OpenBanditDataset(
            behavior_policy=behavior_policy,
            campaign=campaign,
            data_path=data_path,
        )
        bandit_feedback = dataset.obtain_batch_bandit_feedback()
        # OBP's ``obtain_batch_bandit_feedback`` returns a dict under the
        # default call path, but it can return a ``(train, test)`` tuple
        # when ``is_timeseries_split=True`` — we never pass that flag,
        # but guard with a real TypeError (not ``assert``) so the check
        # survives ``python -O`` / ``PYTHONOPTIMIZE=1``.
        if not isinstance(bandit_feedback, dict):
            raise TypeError(
                "OpenBanditDataset.obtain_batch_bandit_feedback did not return a "
                f"dict; got {type(bandit_feedback).__name__!r}. This likely means "
                "the OBP API changed or timeseries splitting was accidentally "
                "enabled."
            )
        return cls(bandit_feedback=bandit_feedback, seed=seed)

    # ── DomainAdapter interface ────────────────────────────────────────

    def get_search_space(self) -> SearchSpace:
        variables: list[Variable] = [
            Variable(
                name="tau",
                variable_type=VariableType.CONTINUOUS,
                lower=_TAU_LOWER,
                upper=_TAU_UPPER,
            ),
            Variable(
                name="eps",
                variable_type=VariableType.CONTINUOUS,
                lower=_EPS_LOWER,
                upper=_EPS_UPPER,
            ),
        ]
        for name in _CONTEXT_WEIGHT_NAMES:
            variables.append(
                Variable(
                    name=name,
                    variable_type=VariableType.CONTINUOUS,
                    lower=_WEIGHT_LOWER,
                    upper=_WEIGHT_UPPER,
                )
            )
        variables.append(
            Variable(
                name="position_handling_flag",
                variable_type=VariableType.CATEGORICAL,
                choices=list(_POSITION_HANDLING_CHOICES),
            )
        )
        return SearchSpace(variables=variables)

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        tau = float(parameters.get("tau", 1.0))
        eps = float(parameters.get("eps", 0.0))
        w_item = float(parameters.get("w_item_feature_0", 0.0))
        w_affinity = float(parameters.get("w_user_item_affinity", 0.0))
        w_popularity = float(parameters.get("w_item_popularity", 0.0))
        position_flag = str(parameters.get("position_handling_flag", "position_1_only"))

        if position_flag not in _POSITION_HANDLING_CHOICES:
            msg = (
                f"position_handling_flag must be one of "
                f"{list(_POSITION_HANDLING_CHOICES)!r}, got {position_flag!r}"
            )
            raise ValueError(msg)

        # Row mask: restrict to position index 0 when requested. The
        # incoming ``position`` array is 0-indexed after ``rankdata``
        # re-ranking in the raw OBP loader, so position_1 ↔ index 0.
        if position_flag == "position_1_only":
            mask = self._position == 0
        else:
            mask = np.ones(self._n_rounds, dtype=bool)

        n_active = int(mask.sum())
        if n_active == 0:
            # No rows survive the position subset — return a
            # pessimistic but well-defined result.
            return {
                "policy_value": 0.0,
                "ess": 0.0,
                "weight_cv": 0.0,
                "max_weight": 0.0,
                "zero_support_fraction": 1.0,
                "n_effective_actions": 1.0,
            }

        # ── Score each (row, candidate action) ────────────────────────
        # Shape: (n_active, n_actions)
        item_term = w_item * self._item_feature_0[None, :]
        pop_term = w_popularity * self._item_popularity[None, :]
        affinity_term = w_affinity * self._affinity[mask]
        scores = item_term + pop_term + affinity_term

        # ── Softmax with temperature and epsilon mixing ───────────────
        safe_tau = max(tau, _TAU_FLOOR)
        scaled = scores / safe_tau
        scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
        exp_scores = np.exp(scaled)
        softmax = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        uniform = np.full_like(softmax, 1.0 / self._n_actions)
        policy = (1.0 - eps) * softmax + eps * uniform
        # policy is shape (n_active, n_actions) and sums to 1 along axis 1.

        # ── SNIPW-style policy value ──────────────────────────────────
        active_action = self._action[mask]
        active_reward = self._reward[mask].astype(float)
        active_pscore = self._pscore[mask]

        # Evaluation-policy probability of the logged action.
        row_idx = np.arange(n_active)
        pi_logged = policy[row_idx, active_action]

        # Importance weights w_i = pi_e(a_i | x_i) / pi_b(a_i | x_i).
        weights = pi_logged / active_pscore

        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            # Self-normalized IPW: V_hat = sum(w_i r_i) / sum(w_i).
            policy_value = float((weights * active_reward).sum() / weight_sum)
            # Kish ESS: (sum w)^2 / sum(w^2).
            ess = float(weight_sum * weight_sum / float((weights * weights).sum()))
        else:
            policy_value = 0.0
            ess = 0.0

        # ── Weight statistics ─────────────────────────────────────────
        max_weight = float(weights.max()) if weights.size > 0 else 0.0
        positive = weights[weights > 0.0]
        # Use population std (ddof=0) so the CV is well defined even at
        # tiny n_active; fall back to 0.0 when fewer than two positive
        # weights survive (no variability to report).
        weight_cv = float(positive.std(ddof=0) / positive.mean()) if positive.size > 1 else 0.0

        # ── Zero-support fraction ─────────────────────────────────────
        # A logged row has "structurally zero support" under the
        # evaluation policy when pi_e(a_logged | x) == 0 exactly.
        # Under eps > 0 this is impossible; we still compute the
        # fraction defensively.
        zero_support_fraction = float((pi_logged <= 0.0).mean())

        # ── n_effective_actions ───────────────────────────────────────
        # Count actions that carry non-negligible mass: the smallest k
        # such that the top-k sorted policy-mass items on average
        # exceed 0.95. Equivalently, count the per-row top-k until the
        # threshold is crossed, averaged over rows. We compute the
        # average directly via cumulative-sum-of-sorted-mass.
        sorted_mass = -np.sort(-policy, axis=1)
        cum_mass = np.cumsum(sorted_mass, axis=1)
        # Per-row: smallest k with cum_mass[k-1] >= threshold.
        # np.argmax returns the first True along axis; if all False
        # (shouldn't happen because cum_mass ends at ~1), fall back to
        # n_actions.
        crosses = cum_mass >= _EFFECTIVE_ACTIONS_MASS_THRESHOLD
        any_cross = crosses.any(axis=1)
        first_cross = np.where(
            any_cross,
            crosses.argmax(axis=1) + 1,
            self._n_actions,
        )
        n_effective_actions = float(first_cross.mean())

        return {
            "policy_value": policy_value,
            "ess": ess,
            "weight_cv": weight_cv,
            "max_weight": max_weight,
            "zero_support_fraction": zero_support_fraction,
            "n_effective_actions": n_effective_actions,
        }

    def get_prior_graph(self) -> CausalGraph | None:
        """Sprint 34 contract Section 4e: prior graph is optional, minimal,
        or deferred. Returns ``None`` for the first implementation.

        Authoring a multi-action prior graph is a Sprint 36+ decision —
        see contract Section 4e. The engine will run without a graph;
        ``focus_variables`` degrades to the full search space.
        """
        return None

    def get_objective_name(self) -> str:
        return "policy_value"

    def get_minimize(self) -> bool:
        return False  # maximize SNIPW-estimated CTR

    def get_strategy(self) -> str:
        return "bayesian"

    def get_descriptor_names(self) -> list[str]:
        # Descriptors for MAP-Elites diversity: n_effective_actions
        # (policy concentration) and zero_support_fraction
        # (coverage) are the two most behavior-meaningful axes for
        # multi-action policies. They are already returned by
        # run_experiment so MAP-Elites can read them without
        # re-evaluating.
        return ["n_effective_actions", "zero_support_fraction"]

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_feedback(bandit_feedback: dict[str, Any]) -> None:
        """Fail fast with actionable errors on malformed bandit-feedback.

        Checks keys, array lengths, reward binary-ness, and non-zero
        propensity. Raises :class:`ValueError` on any violation so
        downstream callers see a clear failure site instead of an
        opaque numpy error deep inside :meth:`run_experiment`.
        """
        missing = _REQUIRED_FEEDBACK_KEYS - set(bandit_feedback)
        if missing:
            msg = (
                "bandit_feedback is missing required key(s): "
                f"{sorted(missing)!r}. Required keys: "
                f"{sorted(_REQUIRED_FEEDBACK_KEYS)!r}."
            )
            raise ValueError(msg)

        n_rounds = int(bandit_feedback["n_rounds"])
        if n_rounds <= 0:
            raise ValueError(f"n_rounds must be positive, got {n_rounds}")

        n_actions = int(bandit_feedback["n_actions"])
        if n_actions < 2:
            raise ValueError(f"n_actions must be >= 2 for a multi-action bandit, got {n_actions}")

        for key in ("action", "position", "reward", "pscore"):
            arr = np.asarray(bandit_feedback[key])
            if arr.shape != (n_rounds,):
                raise ValueError(
                    f"bandit_feedback[{key!r}] has shape {arr.shape}, expected ({n_rounds},)"
                )

        context = np.asarray(bandit_feedback["context"])
        if context.ndim != 2 or context.shape[0] != n_rounds:
            raise ValueError(
                f"bandit_feedback['context'] has shape {context.shape}, "
                f"expected ({n_rounds}, d_context)"
            )

        action_context = np.asarray(bandit_feedback["action_context"])
        if action_context.ndim != 2 or action_context.shape[0] != n_actions:
            raise ValueError(
                f"bandit_feedback['action_context'] has shape {action_context.shape}, "
                f"expected ({n_actions}, d_action). ``action_context`` must contain "
                "at least one column; the adapter uses column 0 as ``item_feature_0``."
            )
        if action_context.shape[1] < 1:
            raise ValueError(
                "bandit_feedback['action_context'] must have >= 1 column; the adapter "
                "reads column 0 as ``item_feature_0``."
            )

        reward = np.asarray(bandit_feedback["reward"])
        unique = np.unique(reward)
        if not set(unique.astype(float).tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                "bandit_feedback['reward'] must be binary (0/1); "
                f"found unique values {unique.tolist()}"
            )

        pscore = np.asarray(bandit_feedback["pscore"], dtype=float)
        if (pscore <= 0.0).any():
            raise ValueError(
                "bandit_feedback['pscore'] must be strictly positive for SNIPW; "
                f"found min={float(pscore.min())}. The Sprint 34 contract "
                "Section 5c clip is applied downstream by the OPE stack, but the "
                "adapter rejects a zero-propensity row at construction."
            )

        action = np.asarray(bandit_feedback["action"])
        if action.min() < 0 or action.max() >= n_actions:
            raise ValueError(
                f"bandit_feedback['action'] values must be in [0, {n_actions}); "
                f"found range [{int(action.min())}, {int(action.max())}]"
            )
