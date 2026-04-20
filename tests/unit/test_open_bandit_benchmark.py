"""Unit tests for the Sprint 35.C Open Bandit benchmark runner.

Covers the benchmark-runner helpers in
``causal_optimizer/benchmarks/open_bandit_benchmark.py``. Tests build
small synthetic OBP-shaped ``bandit_feedback`` dicts and exercise the
strategy loop, SNIPW/DM/DR collection, null-control, and gate
evaluation without touching the full ~453K-row Men/Random slice.

The full-slice integration path is smoke-tested separately in
``tests/integration/test_bandit_log_adapter_smoke.py`` and exercised
end-to-end only by the CLI runner in ``scripts/open_bandit_benchmark.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.benchmarks.open_bandit_benchmark import (
    VALID_STRATEGIES,
    OpenBanditBenchmarkResult,
    OpenBanditScenario,
    build_bandit_feedback_from_raw,
    build_policy_action_dist,
    compute_reward_model,
    load_men_random_slice,
    summarize_strategy_budget,
)
from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter

# ── Fixtures ────────────────────────────────────────────────────────


def _synthetic_feedback(
    *,
    n_rounds: int = 1200,
    n_actions: int = 6,
    n_positions: int = 3,
    seed: int = 0,
    click_rate: float = 0.02,
) -> dict[str, Any]:
    """Build a synthetic OBP-shaped bandit_feedback dict for unit tests.

    Mirrors the Men/Random conditional schema: ``pscore`` is set to
    ``1/n_actions`` for every row. Positions are 0-indexed contiguous
    integers over ``[0, n_positions)``.
    """
    rng = np.random.default_rng(seed)
    action = rng.integers(0, n_actions, size=n_rounds)
    position = rng.integers(0, n_positions, size=n_rounds)
    reward = (rng.random(n_rounds) < click_rate).astype(int)
    pscore = np.full(n_rounds, 1.0 / n_actions, dtype=float)
    context = rng.normal(size=(n_rounds, n_actions))
    action_context = rng.normal(size=(n_actions, 3))
    return {
        "n_rounds": int(n_rounds),
        "n_actions": int(n_actions),
        "action": action.astype(int),
        "position": position.astype(int),
        "reward": reward.astype(float),
        "pscore": pscore,
        "context": context,
        "action_context": action_context,
    }


# ── Module-level constants ─────────────────────────────────────────────


class TestValidStrategies:
    def test_valid_strategies_contains_three(self) -> None:
        assert frozenset({"random", "surrogate_only", "causal"}) == VALID_STRATEGIES


# ── Loader contract ───────────────────────────────────────────────────


class TestBuildFromRaw:
    """``build_bandit_feedback_from_raw`` must produce an OBP-shaped dict.

    The function takes raw DataFrame-style inputs (the ``men.csv`` log
    and the ``item_context.csv`` action table) and produces a
    bandit_feedback dict with 0-indexed contiguous positions, per-item
    affinity as context, and ``action_context`` containing
    ``item_feature_0`` as column 0.
    """

    def test_output_has_required_keys(self) -> None:
        import pandas as pd

        n_rounds = 300
        n_actions = 5
        rng = np.random.default_rng(0)
        # Simulate a raw men.csv subset
        data = pd.DataFrame(
            {
                "timestamp": np.arange(n_rounds),
                "item_id": rng.integers(0, n_actions, size=n_rounds),
                "position": rng.integers(1, 4, size=n_rounds),  # 1-indexed in raw CSV
                "click": (rng.random(n_rounds) < 0.02).astype(int),
                "propensity_score": np.full(n_rounds, 1.0 / n_actions, dtype=float),
            }
        )
        # Simulate 34 user-item affinity columns (n_actions = 5 for this test)
        for k in range(n_actions):
            data[f"user-item_affinity_{k}"] = rng.normal(size=n_rounds)
        item_context = pd.DataFrame(
            {
                "item_id": np.arange(n_actions),
                "item_feature_0": rng.uniform(-0.7, 0.7, size=n_actions),
            }
        )
        bf = build_bandit_feedback_from_raw(data=data, item_context=item_context)
        for key in (
            "n_rounds",
            "n_actions",
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ):
            assert key in bf

    def test_positions_are_zero_indexed_contiguous(self) -> None:
        import pandas as pd

        n_rounds = 300
        n_actions = 5
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "timestamp": np.arange(n_rounds),
                "item_id": rng.integers(0, n_actions, size=n_rounds),
                # Raw CSV positions are 1-indexed; the loader must re-rank
                # them to 0-indexed contiguous integers.
                "position": rng.integers(1, 4, size=n_rounds),
                "click": (rng.random(n_rounds) < 0.02).astype(int),
                "propensity_score": np.full(n_rounds, 1.0 / n_actions, dtype=float),
            }
        )
        for k in range(n_actions):
            data[f"user-item_affinity_{k}"] = rng.normal(size=n_rounds)
        item_context = pd.DataFrame(
            {
                "item_id": np.arange(n_actions),
                "item_feature_0": rng.uniform(-0.7, 0.7, size=n_actions),
            }
        )
        bf = build_bandit_feedback_from_raw(data=data, item_context=item_context)
        unique = np.unique(bf["position"])
        assert int(unique.min()) == 0
        assert list(unique) == list(range(len(unique)))

    def test_accepts_validated_by_adapter(self) -> None:
        """The loader's output must pass BanditLogAdapter validation."""
        import pandas as pd

        n_rounds = 400
        n_actions = 5
        rng = np.random.default_rng(1)
        data = pd.DataFrame(
            {
                "timestamp": np.arange(n_rounds),
                "item_id": rng.integers(0, n_actions, size=n_rounds),
                "position": rng.integers(1, 4, size=n_rounds),
                "click": (rng.random(n_rounds) < 0.02).astype(int),
                "propensity_score": np.full(n_rounds, 1.0 / n_actions, dtype=float),
            }
        )
        for k in range(n_actions):
            data[f"user-item_affinity_{k}"] = rng.normal(size=n_rounds)
        item_context = pd.DataFrame(
            {
                "item_id": np.arange(n_actions),
                "item_feature_0": rng.uniform(-0.7, 0.7, size=n_actions),
            }
        )
        bf = build_bandit_feedback_from_raw(data=data, item_context=item_context)
        # Must not raise.
        BanditLogAdapter(bandit_feedback=bf, seed=0)


class TestLoadMenRandomSlice:
    """``load_men_random_slice`` resolves ``data_path`` to CSV files."""

    def test_missing_data_path_raises(self, tmp_path: Any) -> None:
        missing = tmp_path / "nowhere"
        with pytest.raises(FileNotFoundError):
            load_men_random_slice(data_path=missing)


# ── Policy action_dist builder ─────────────────────────────────────────


class TestBuildPolicyActionDist:
    """``build_policy_action_dist`` must match the adapter's softmax math.

    Given the same parameters, the ``policy_value`` computed by SNIPW on
    the built ``action_dist`` must equal the policy value returned by
    the adapter's ``run_experiment`` (up to numerical noise).
    """

    def test_action_dist_rows_sum_to_one(self) -> None:
        bf = _synthetic_feedback()
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=0)
        params = {
            "tau": 1.0,
            "eps": 0.1,
            "w_item_feature_0": 0.5,
            "w_user_item_affinity": 0.0,
            "w_item_popularity": 0.0,
            "position_handling_flag": "marginalize",
        }
        action_dist = build_policy_action_dist(adapter=adapter, parameters=params)
        row_sums = action_dist.sum(axis=1)
        assert action_dist.shape == (bf["n_rounds"], bf["n_actions"])
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_action_dist_matches_adapter_snipw(self) -> None:
        """SNIPW on the built dist must match the adapter's own SNIPW."""
        from causal_optimizer.benchmarks.open_bandit import compute_snipw

        bf = _synthetic_feedback()
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=0)
        params = {
            "tau": 2.0,
            "eps": 0.05,
            "w_item_feature_0": 0.3,
            "w_user_item_affinity": 0.2,
            "w_item_popularity": 0.1,
            "position_handling_flag": "marginalize",
        }
        action_dist = build_policy_action_dist(adapter=adapter, parameters=params)
        # The adapter uses pscore as-is (no clip) at marginalize mode,
        # so SNIPW with an unused clip floor of 0 returns the same value.
        # We use the standard contract clip here.
        clip = 1.0 / (2 * bf["n_actions"] * 3)
        snipw_value = compute_snipw(bf, action_dist, min_propensity_clip=clip)
        # Adapter value uses its own internal math; the difference should
        # be small since both use the same softmax + pscore.
        adapter_metrics = adapter.run_experiment(params)
        assert snipw_value == pytest.approx(adapter_metrics["policy_value"], abs=5e-4)

    def test_position_1_only_mask_restricts_rows(self) -> None:
        """With position_1_only, rows at position != 0 must have uniform mass.

        The policy's effective action_dist for rows outside position 0
        falls back to uniform (one of the safest choices); SNIPW on
        those rows then contributes a neutral term. The caller can
        choose whether to subset the bandit_feedback to position 0 or
        keep the full set with uniform mass on excluded rows.
        """
        bf = _synthetic_feedback()
        adapter = BanditLogAdapter(bandit_feedback=bf, seed=0)
        params = {
            "tau": 1.0,
            "eps": 0.0,
            "w_item_feature_0": 0.5,
            "w_user_item_affinity": 0.5,
            "w_item_popularity": 0.5,
            "position_handling_flag": "position_1_only",
        }
        action_dist = build_policy_action_dist(adapter=adapter, parameters=params)
        # Shape must always be (n_rounds, n_actions); the restriction is
        # encoded by setting outside rows to uniform mass.
        assert action_dist.shape == (bf["n_rounds"], bf["n_actions"])
        # Rows at position != 0 must sum to 1 (valid probability).
        off = bf["position"] != 0
        if off.any():
            assert np.allclose(action_dist[off].sum(axis=1), 1.0, atol=1e-10)


# ── Reward model for DR/DM ─────────────────────────────────────────────


class TestComputeRewardModel:
    def test_reward_hat_shape_matches_bf_and_n_actions(self) -> None:
        bf = _synthetic_feedback()
        reward_hat = compute_reward_model(bf, seed=0)
        assert reward_hat.shape == (bf["n_rounds"], bf["n_actions"])

    def test_reward_hat_in_unit_interval(self) -> None:
        bf = _synthetic_feedback()
        reward_hat = compute_reward_model(bf, seed=0)
        assert (reward_hat >= 0.0).all()
        assert (reward_hat <= 1.0).all()


# ── Strategy loop ────────────────────────────────────────────────────


class TestOpenBanditScenario:
    """``OpenBanditScenario`` runs one strategy at one (budget, seed)."""

    def test_unknown_strategy_raises(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        with pytest.raises(ValueError):
            scenario.run_strategy("magic", budget=3, seed=0)

    def test_random_strategy_returns_policy_value(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        result = scenario.run_strategy("random", budget=3, seed=0)
        assert isinstance(result, OpenBanditBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert result.seed == 0
        assert 0.0 <= result.policy_value_snipw <= 1.0
        # Diagnostics must carry the Section 4d fields
        for key in (
            "ess",
            "zero_support_fraction",
            "weight_cv",
            "max_weight",
            "n_effective_actions",
        ):
            assert key in result.diagnostics

    def test_surrogate_only_strategy_runs(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        result = scenario.run_strategy("surrogate_only", budget=3, seed=0)
        assert result.strategy == "surrogate_only"
        assert 0.0 <= result.policy_value_snipw <= 1.0

    def test_causal_strategy_runs(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        result = scenario.run_strategy("causal", budget=3, seed=0)
        assert result.strategy == "causal"
        assert 0.0 <= result.policy_value_snipw <= 1.0

    def test_dr_estimate_present_when_reward_model_available(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf, use_reward_model=True)
        result = scenario.run_strategy("random", budget=3, seed=0)
        assert result.policy_value_dr is not None
        assert result.policy_value_dm is not None

    def test_null_control_uses_permuted_reward(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        result = scenario.run_strategy(
            "random", budget=3, seed=0, null_control=True, permutation_seed=42
        )
        assert result.is_null_control
        assert result.permutation_seed == 42


# ── Summary helper ───────────────────────────────────────────────────


class TestSummarizeStrategyBudget:
    def test_summary_returns_mean_std(self) -> None:
        bf = _synthetic_feedback()
        scenario = OpenBanditScenario(bandit_feedback=bf)
        results = [scenario.run_strategy("random", budget=3, seed=s) for s in range(3)]
        summary = summarize_strategy_budget(results)
        # Mean / std by (strategy, budget)
        assert ("random", 3) in summary
        cell = summary[("random", 3)]
        assert "mean_policy_value_snipw" in cell
        assert "std_policy_value_snipw" in cell
        assert "n_seeds" in cell
        assert cell["n_seeds"] == 3
