"""Unit tests for ``BanditLogAdapter`` (Sprint 35.A).

Uses a synthetic OBP-shaped ``bandit_feedback`` dict so these tests are
fast and do not require the real Men/Random slice or the ``obp`` extra.
The slow smoke test against the bundled OBP data lives in
``tests/integration/test_bandit_log_adapter_smoke.py``.

Covers the Sprint 34 Open Bandit contract, Sections 4b and 4d:
the minimum ``DomainAdapter`` interface, the 3-to-5 context-feature
weight search space, the ``policy_value`` objective and ``maximize``
direction, and the full Section 4d diagnostic dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter
from causal_optimizer.types import VariableType


def _synthetic_bandit_feedback(
    *,
    n_rounds: int = 600,
    n_actions: int = 5,
    len_list: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a small OBP-shaped ``bandit_feedback`` dict for unit tests.

    Matches the OBP schema exactly:
    - ``action``: shape (n_rounds,), int in [0, n_actions)
    - ``position``: shape (n_rounds,), int in [0, len_list)
    - ``reward``: shape (n_rounds,), binary {0, 1}
    - ``pscore``: shape (n_rounds,), conditional ``P(item | position)``
      fixed to ``1/n_actions`` (mirrors the Men/Random logger)
    - ``context``: shape (n_rounds, d_context), user features
    - ``action_context``: shape (n_actions, d_action), item features
    """
    rng = np.random.default_rng(seed)
    action = rng.integers(0, n_actions, size=n_rounds)
    position = rng.integers(0, len_list, size=n_rounds)
    reward = (rng.random(n_rounds) < 0.01).astype(int)
    pscore = np.full(n_rounds, 1.0 / n_actions, dtype=float)
    context = rng.normal(size=(n_rounds, 4))
    action_context = rng.normal(size=(n_actions, 3))
    return {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "action": action,
        "position": position,
        "reward": reward,
        "pscore": pscore,
        "context": context,
        "action_context": action_context,
    }


@pytest.fixture
def feedback() -> dict[str, Any]:
    return _synthetic_bandit_feedback()


@pytest.fixture
def adapter(feedback: dict[str, Any]) -> BanditLogAdapter:
    return BanditLogAdapter(bandit_feedback=feedback, seed=7)


@pytest.fixture
def default_params() -> dict[str, Any]:
    return {
        "tau": 1.0,
        "eps": 0.1,
        "w_item_feature_0": 0.5,
        "w_user_item_affinity": 0.5,
        "w_item_popularity": 0.0,
        "position_handling_flag": "position_1_only",
    }


class TestAdapterContract:
    """Section 4b required interface methods."""

    def test_adapter_subclasses_domain_adapter(self, adapter: BanditLogAdapter) -> None:
        from causal_optimizer.domain_adapters.base import DomainAdapter

        assert isinstance(adapter, DomainAdapter)

    def test_adapter_does_not_subclass_marketing_log_adapter(
        self, adapter: BanditLogAdapter
    ) -> None:
        # Sprint 34 contract Section 4a forbids subclassing
        # ``MarketingLogAdapter``; the binary treatment path there is
        # load-bearing and reshaping it to multi-action would hide the
        # structural break.
        from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter

        assert not isinstance(adapter, MarketingLogAdapter)

    def test_objective_name_is_policy_value(self, adapter: BanditLogAdapter) -> None:
        assert adapter.get_objective_name() == "policy_value"

    def test_minimize_is_false(self, adapter: BanditLogAdapter) -> None:
        assert adapter.get_minimize() is False

    def test_strategy_is_bayesian(self, adapter: BanditLogAdapter) -> None:
        assert adapter.get_strategy() == "bayesian"

    def test_prior_graph_is_none_by_default(self, adapter: BanditLogAdapter) -> None:
        assert adapter.get_prior_graph() is None


class TestSearchSpace:
    """Section 4c item-scoring policy parameterization."""

    def test_search_space_has_6_to_8_variables(self, adapter: BanditLogAdapter) -> None:
        space = adapter.get_search_space()
        assert 6 <= len(space.variables) <= 8

    def test_search_space_contains_tau(self, adapter: BanditLogAdapter) -> None:
        space = adapter.get_search_space()
        tau = next((v for v in space.variables if v.name == "tau"), None)
        assert tau is not None
        assert tau.variable_type == VariableType.CONTINUOUS
        assert tau.lower is not None and tau.upper is not None
        assert tau.lower >= 0.1 and tau.upper <= 10.0

    def test_search_space_contains_eps(self, adapter: BanditLogAdapter) -> None:
        space = adapter.get_search_space()
        eps = next((v for v in space.variables if v.name == "eps"), None)
        assert eps is not None
        assert eps.variable_type == VariableType.CONTINUOUS
        assert eps.lower == 0.0
        assert eps.upper == 0.5

    def test_search_space_contains_position_handling_flag(self, adapter: BanditLogAdapter) -> None:
        space = adapter.get_search_space()
        flag = next(
            (v for v in space.variables if v.name == "position_handling_flag"),
            None,
        )
        assert flag is not None
        assert flag.variable_type == VariableType.CATEGORICAL
        assert set(flag.choices or []) == {"marginalize", "position_1_only"}

    def test_search_space_contains_3_to_5_context_weights(self, adapter: BanditLogAdapter) -> None:
        space = adapter.get_search_space()
        weight_vars = [
            v
            for v in space.variables
            if v.name.startswith("w_") and v.variable_type == VariableType.CONTINUOUS
        ]
        assert 3 <= len(weight_vars) <= 5


class TestRunExperimentDiagnosticKeys:
    """Section 4d: required keys in the ``run_experiment`` return dict."""

    _REQUIRED = (
        "policy_value",
        "ess",
        "weight_cv",
        "max_weight",
        "zero_support_fraction",
        "n_effective_actions",
    )

    def test_run_returns_all_required_keys(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        missing = [k for k in self._REQUIRED if k not in metrics]
        assert missing == [], f"Adapter omitted required keys: {missing}"

    def test_run_returns_only_floats(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        for key in self._REQUIRED:
            assert isinstance(metrics[key], float), f"{key}={metrics[key]!r} is not a float"


class TestRunExperimentSemantics:
    """Semantic invariants on the computed diagnostics."""

    def test_policy_value_is_in_unit_interval(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        # SNIPW estimates expected CTR; on binary rewards it must be in [0, 1].
        metrics = adapter.run_experiment(default_params)
        pv = metrics["policy_value"]
        assert 0.0 <= pv <= 1.0, f"policy_value={pv} out of [0, 1]"

    def test_ess_is_positive_when_support_exists(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert metrics["ess"] > 0.0

    def test_zero_support_fraction_is_zero_when_eps_positive(
        self, adapter: BanditLogAdapter
    ) -> None:
        # With eps > 0 the evaluation policy gives every action >= eps/n mass,
        # so structurally zero support on the logged action is impossible.
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.1,
                "w_item_feature_0": 0.0,
                "w_user_item_affinity": 0.0,
                "w_item_popularity": 0.0,
                "position_handling_flag": "marginalize",
            }
        )
        assert metrics["zero_support_fraction"] == 0.0

    def test_uniform_policy_has_policy_value_near_logged_ctr(
        self, feedback: dict[str, Any]
    ) -> None:
        # Under eps=0.5 with zero feature weights, the evaluation policy is
        # uniform. SNIPW should then return a policy value close to the
        # marginal logged reward mean (because every action has equal
        # propensity under both policies).
        adapter = BanditLogAdapter(bandit_feedback=feedback, seed=0)
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.5,
                "w_item_feature_0": 0.0,
                "w_user_item_affinity": 0.0,
                "w_item_popularity": 0.0,
                "position_handling_flag": "marginalize",
            }
        )
        logged_ctr = float(feedback["reward"].mean())
        # Wide tolerance — this is a sanity check on the sign/magnitude.
        assert abs(metrics["policy_value"] - logged_ctr) < 0.01

    def test_position_1_only_restricts_rows(self, feedback: dict[str, Any]) -> None:
        # position_1_only must subset to position == 0 (0-indexed after rank).
        # ESS under position_1_only is computed over ~n_rows / len_list rows,
        # so it should be materially smaller than the marginalize ESS.
        adapter = BanditLogAdapter(bandit_feedback=feedback, seed=0)
        base_params: dict[str, Any] = {
            "tau": 1.0,
            "eps": 0.1,
            "w_item_feature_0": 0.5,
            "w_user_item_affinity": 0.0,
            "w_item_popularity": 0.0,
        }
        m_marg = adapter.run_experiment({**base_params, "position_handling_flag": "marginalize"})
        m_pos1 = adapter.run_experiment(
            {**base_params, "position_handling_flag": "position_1_only"}
        )
        # Strict inequality would be flaky; check they differ by a
        # meaningful margin. The marginalize ESS must be materially larger
        # than the position_1_only ESS because it uses ~3x more rows.
        assert m_marg["ess"] > m_pos1["ess"] * 1.5

    def test_weight_cv_nonnegative(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert metrics["weight_cv"] >= 0.0

    def test_max_weight_nonnegative(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert metrics["max_weight"] >= 0.0

    def test_n_effective_actions_in_valid_range(
        self, adapter: BanditLogAdapter, feedback: dict[str, Any], default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        n = metrics["n_effective_actions"]
        assert 1.0 <= n <= float(feedback["n_actions"])

    def test_position_1_only_with_no_matching_rows_returns_pessimistic(self) -> None:
        # If every row sits at position > 0, position_1_only leaves no
        # rows to estimate on; the adapter returns a deterministic
        # pessimistic dict rather than crashing.
        fb = _synthetic_bandit_feedback(n_rounds=200, n_actions=4, seed=1)
        fb["position"] = np.full_like(fb["position"], 1)  # all at position 1 (index 1)
        adapter = BanditLogAdapter(bandit_feedback=fb)
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.1,
                "w_item_feature_0": 0.0,
                "w_user_item_affinity": 0.0,
                "w_item_popularity": 0.0,
                "position_handling_flag": "position_1_only",
            }
        )
        assert metrics["policy_value"] == 0.0
        assert metrics["ess"] == 0.0
        assert metrics["zero_support_fraction"] == 1.0

    def test_invalid_position_handling_flag_raises(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        bad = {**default_params, "position_handling_flag": "all_positions_please"}
        with pytest.raises(ValueError, match="position_handling_flag"):
            adapter.run_experiment(bad)

    def test_zero_or_small_context_gracefully_handled(self) -> None:
        # When ``context`` has fewer columns than n_actions (small
        # synthetic fixtures), the affinity weight silently contributes
        # zero. The adapter still returns a valid diagnostic dict.
        fb = _synthetic_bandit_feedback(n_rounds=200, n_actions=4, seed=2)
        # Drop context to a single column.
        fb["context"] = fb["context"][:, :1]
        adapter = BanditLogAdapter(bandit_feedback=fb)
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.1,
                "w_item_feature_0": 0.2,
                "w_user_item_affinity": 1.0,  # would normally dominate
                "w_item_popularity": 0.0,
                "position_handling_flag": "marginalize",
            }
        )
        assert 0.0 <= metrics["policy_value"] <= 1.0

    def test_eps_zero_uniform_scores_produces_uniform_policy(
        self, feedback: dict[str, Any]
    ) -> None:
        # With every weight zero and ``eps=0``, the softmax over a row
        # of identical scores must be uniform (no numerical blow-up).
        # n_effective_actions should equal the smallest k with
        # cum_mass >= 0.95 under a uniform distribution: for n=5,
        # 4/5=0.80 < 0.95 but 5/5=1.00, so k=5.
        adapter = BanditLogAdapter(bandit_feedback=feedback, seed=0)
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.0,
                "w_item_feature_0": 0.0,
                "w_user_item_affinity": 0.0,
                "w_item_popularity": 0.0,
                "position_handling_flag": "marginalize",
            }
        )
        assert metrics["n_effective_actions"] == float(feedback["n_actions"])
        assert metrics["zero_support_fraction"] == 0.0

    def test_negative_feature_weights_produce_valid_policy(self, adapter: BanditLogAdapter) -> None:
        # Negative weights penalize rather than reward items; the
        # adapter must still produce valid diagnostics (no NaN, no
        # probability outside [0, 1], finite ESS).
        metrics = adapter.run_experiment(
            {
                "tau": 1.0,
                "eps": 0.05,
                "w_item_feature_0": -2.0,
                "w_user_item_affinity": -1.5,
                "w_item_popularity": -0.5,
                "position_handling_flag": "marginalize",
            }
        )
        assert 0.0 <= metrics["policy_value"] <= 1.0
        assert np.isfinite(metrics["ess"])
        assert np.isfinite(metrics["weight_cv"])

    def test_sharp_softmax_reduces_n_effective_actions(self, adapter: BanditLogAdapter) -> None:
        # Small tau => sharp softmax => most mass on one action under
        # non-zero weights => n_effective_actions drops toward 1.
        params_sharp = {
            "tau": 0.1,
            "eps": 0.0,
            "w_item_feature_0": 2.0,
            "w_user_item_affinity": 0.0,
            "w_item_popularity": 0.0,
            "position_handling_flag": "marginalize",
        }
        params_flat = {
            "tau": 10.0,
            "eps": 0.0,
            "w_item_feature_0": 0.1,
            "w_user_item_affinity": 0.0,
            "w_item_popularity": 0.0,
            "position_handling_flag": "marginalize",
        }
        m_sharp = adapter.run_experiment(params_sharp)
        m_flat = adapter.run_experiment(params_flat)
        assert m_sharp["n_effective_actions"] < m_flat["n_effective_actions"]


class TestConstructorValidation:
    """Clear errors when the input is malformed."""

    def test_missing_required_key_raises(self) -> None:
        # ``reward`` is required; dropping it must fail fast with a clear
        # message, not an opaque KeyError deep inside run_experiment.
        fb = _synthetic_bandit_feedback()
        del fb["reward"]
        with pytest.raises(ValueError, match="reward"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_mismatched_array_length_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["reward"] = fb["reward"][:-1]  # off-by-one
        with pytest.raises(ValueError):
            BanditLogAdapter(bandit_feedback=fb)

    def test_non_binary_reward_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["reward"] = fb["reward"].astype(float)
        fb["reward"][0] = 2.5  # not binary
        with pytest.raises(ValueError, match="binary"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_pscore_with_zero_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["pscore"] = fb["pscore"].copy()
        fb["pscore"][0] = 0.0
        with pytest.raises(ValueError, match="pscore"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_action_out_of_range_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["action"] = fb["action"].copy()
        fb["action"][0] = fb["n_actions"]  # one past the last valid id
        with pytest.raises(ValueError, match="action"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_negative_action_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["action"] = fb["action"].copy()
        fb["action"][0] = -1
        with pytest.raises(ValueError, match="action"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_negative_position_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["position"] = fb["position"].copy()
        fb["position"][0] = -1
        with pytest.raises(ValueError, match="position"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_non_positive_n_rounds_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["n_rounds"] = 0
        with pytest.raises(ValueError, match="n_rounds"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_too_few_actions_raises(self) -> None:
        fb = _synthetic_bandit_feedback(n_actions=2)
        fb["n_actions"] = 1  # a bandit with one arm is not multi-action
        with pytest.raises(ValueError, match="n_actions"):
            BanditLogAdapter(bandit_feedback=fb)

    def test_action_context_with_zero_columns_raises(self) -> None:
        fb = _synthetic_bandit_feedback()
        fb["action_context"] = np.zeros((fb["n_actions"], 0))
        with pytest.raises(ValueError, match="action_context"):
            BanditLogAdapter(bandit_feedback=fb)


class TestFromObpConstructor:
    """Section 8c: adapter must fail fast with a clear error if OBP is
    missing, and must delegate cleanly when OBP is present."""

    def test_from_obp_raises_clear_error_when_obp_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the adapter's import of obp to fail. The error must be
        # actionable (name the extra), not a bare ImportError.
        import sys

        monkeypatch.setitem(sys.modules, "obp", None)
        monkeypatch.setitem(sys.modules, "obp.dataset", None)
        with pytest.raises(ImportError, match="bandit"):
            BanditLogAdapter.from_obp(campaign="men", behavior_policy="random")

    def test_from_obp_rejects_invalid_campaign(self) -> None:
        with pytest.raises(ValueError, match="campaign"):
            BanditLogAdapter.from_obp(campaign="bogus", behavior_policy="random")

    def test_from_obp_rejects_invalid_behavior_policy(self) -> None:
        with pytest.raises(ValueError, match="behavior_policy"):
            BanditLogAdapter.from_obp(campaign="men", behavior_policy="epsilon_greedy")

    def test_from_obp_delegates_to_open_bandit_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub out OpenBanditDataset so the classmethod can be exercised
        # end-to-end without depending on the (broken on modern pandas)
        # pre_process step inside OBP 0.4.1. This covers the happy-path
        # loader delegation without hitting real OBP code.
        import sys
        import types

        fb = _synthetic_bandit_feedback(n_rounds=400, n_actions=4, len_list=3)

        class _StubDataset:
            def __init__(self, *, behavior_policy: str, campaign: str, data_path: Any) -> None:
                self.behavior_policy = behavior_policy
                self.campaign = campaign
                self.data_path = data_path

            def obtain_batch_bandit_feedback(self) -> dict[str, Any]:
                return fb

        fake_obp_module = types.ModuleType("obp")
        fake_dataset_module = types.ModuleType("obp.dataset")
        fake_dataset_module.OpenBanditDataset = _StubDataset  # type: ignore[attr-defined]
        fake_obp_module.dataset = fake_dataset_module  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "obp", fake_obp_module)
        monkeypatch.setitem(sys.modules, "obp.dataset", fake_dataset_module)

        adapter = BanditLogAdapter.from_obp(campaign="men", behavior_policy="random", seed=3)
        assert isinstance(adapter, BanditLogAdapter)
        # Adapter picked up n_actions from the stubbed feedback dict.
        assert adapter.get_search_space().dimensionality >= 6


class TestPublicInterfaceDoesNotExposeObpTypes:
    """Section 8c: the adapter must not expose OBP types at the public
    interface."""

    def test_run_experiment_returns_plain_dict(
        self, adapter: BanditLogAdapter, default_params: dict[str, Any]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        # Plain dict of floats, no OBP classes.
        assert type(metrics) is dict  # noqa: E721 - exact-type check intentional
        for k, v in metrics.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_get_search_space_returns_causal_optimizer_type(
        self, adapter: BanditLogAdapter
    ) -> None:
        from causal_optimizer.types import SearchSpace

        assert isinstance(adapter.get_search_space(), SearchSpace)
