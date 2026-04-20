"""Unit tests for the Sprint 35 Open Bandit OPE stack and Section 7 gates.

Covers the invariants pinned by the Sprint 34 Open Bandit contract:

1. SNIPW / DM / DR estimators compute expected values on synthetic
   bandit-feedback dicts.
2. Propensity clipping is applied at ``min_propensity_clip`` floor with
   the contract default ``1 / (2 * n_items * n_positions)``.
3. Section 4d diagnostic dict is returned by the OPE evaluation
   function (``policy_value``, ``ess``, ``weight_cv``, ``max_weight``,
   ``zero_support_fraction``, ``n_effective_actions``).
4. Section 7a null-control gate uses position-stratified reward
   permutation (deterministic under a fixed seed) and fails strategies
   that exceed ``1.05 * mu_null``.
5. Section 7b ESS floor is ``max(1000, n_rows / 100)`` for post-subset
   row counts.
6. Section 7c zero-support fraction gate fires at > 10% best-of-seed.
7. Section 7d propensity sanity gate checks the empirical mean of
   ``action_prob`` within 10% relative of the configured target
   (conditional or joint schema).
8. Section 7e DR/SNIPW cross-check gate fires on > 25% relative
   divergence.
9. Synthetic fixtures (bandit-feedback dict generator, uniform /
   peaked / degenerate policies) are deterministic under a fixed seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.benchmarks.open_bandit import (
    DEFAULT_PROPENSITY_SCHEMA,
    PROPENSITY_SCHEMA_CONDITIONAL,
    PROPENSITY_SCHEMA_JOINT,
    compute_dm,
    compute_dr,
    compute_min_propensity_clip,
    compute_snipw,
    degenerate_policy,
    ess_gate,
    evaluate_open_bandit_policy,
    generate_synthetic_bandit_feedback,
    null_control_gate,
    peaked_policy,
    permute_rewards_stratified,
    propensity_sanity_gate,
    snipw_dr_cross_check_gate,
    uniform_policy,
    zero_support_gate,
)

# ── Synthetic fixture generator ──────────────────────────────────────


class TestSyntheticFixtures:
    """Synthetic bandit-feedback generator and policy helpers."""

    def test_bandit_feedback_has_required_keys(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=100, n_actions=4, n_positions=2, seed=0)
        for key in ("n_rounds", "n_actions", "action", "reward", "pscore"):
            assert key in bf

    def test_bandit_feedback_shapes(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=6, n_positions=3, seed=0)
        assert bf["n_rounds"] == 50
        assert bf["n_actions"] == 6
        assert bf["action"].shape == (50,)
        assert bf["reward"].shape == (50,)
        assert bf["pscore"].shape == (50,)
        assert bf["position"].shape == (50,)

    def test_bandit_feedback_is_deterministic(self) -> None:
        bf1 = generate_synthetic_bandit_feedback(n_rounds=100, n_actions=4, n_positions=2, seed=42)
        bf2 = generate_synthetic_bandit_feedback(n_rounds=100, n_actions=4, n_positions=2, seed=42)
        np.testing.assert_array_equal(bf1["action"], bf2["action"])
        np.testing.assert_array_equal(bf1["reward"], bf2["reward"])
        np.testing.assert_array_equal(bf1["pscore"], bf2["pscore"])

    def test_uniform_logger_pscore_matches_joint(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=200,
            n_actions=5,
            n_positions=3,
            seed=0,
            propensity_schema=PROPENSITY_SCHEMA_JOINT,
        )
        expected = 1.0 / (5 * 3)
        np.testing.assert_allclose(bf["pscore"], expected)

    def test_uniform_logger_pscore_matches_conditional(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=200,
            n_actions=5,
            n_positions=3,
            seed=0,
            propensity_schema=PROPENSITY_SCHEMA_CONDITIONAL,
        )
        expected = 1.0 / 5
        np.testing.assert_allclose(bf["pscore"], expected)

    def test_uniform_policy_is_uniform(self) -> None:
        pol = uniform_policy(n_rounds=10, n_actions=4)
        assert pol.shape == (10, 4)
        np.testing.assert_allclose(pol, 0.25)

    def test_peaked_policy_sums_to_one(self) -> None:
        pol = peaked_policy(n_rounds=20, n_actions=5, best_action=2, peak_mass=0.9)
        assert pol.shape == (20, 5)
        row_sums = pol.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)
        assert pol[0, 2] == pytest.approx(0.9)

    def test_degenerate_policy_has_zero_support_rows(self) -> None:
        pol = degenerate_policy(n_rounds=10, n_actions=4, support_action=0)
        assert pol.shape == (10, 4)
        # All mass on action 0 → action 1/2/3 have zero support
        assert (pol[:, 1] == 0.0).all()
        assert (pol[:, 0] == 1.0).all()


# ── Propensity clip floor ────────────────────────────────────────────


class TestPropensityClip:
    def test_default_clip_formula(self) -> None:
        # Contract: 1 / (2 * n_items * n_positions)
        clip = compute_min_propensity_clip(n_actions=34, n_positions=3)
        assert clip == pytest.approx(1.0 / (2 * 34 * 3))

    def test_clip_floor_prevents_divide_by_zero(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=20, n_actions=4, n_positions=1, seed=0)
        # Inject a pathologically small propensity
        bf["pscore"] = bf["pscore"].copy()
        bf["pscore"][0] = 1e-12
        pol = uniform_policy(n_rounds=20, n_actions=4)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert np.isfinite(out["policy_value"])
        assert np.isfinite(out["max_weight"])

    def test_clipped_row_count_reported(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=20, n_actions=4, n_positions=1, seed=0)
        bf["pscore"] = bf["pscore"].copy()
        bf["pscore"][:5] = 1e-6  # 5 rows below the 0.01 floor
        pol = uniform_policy(n_rounds=20, n_actions=4)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert out["n_clipped_rows"] == 5


# ── SNIPW / DM / DR estimators ───────────────────────────────────────


class TestEstimators:
    """Self-normalized IPW, Direct Method, Doubly Robust."""

    def test_snipw_matches_hand_computed(self) -> None:
        # 4 rounds, 2 actions. Logger = uniform. Eval policy = peaked on action 1.
        bf: dict[str, Any] = {
            "n_rounds": 4,
            "n_actions": 2,
            "action": np.array([0, 1, 0, 1]),
            "reward": np.array([0.0, 1.0, 0.0, 1.0]),
            "pscore": np.array([0.5, 0.5, 0.5, 0.5]),
            "position": np.zeros(4, dtype=int),
        }
        # eval policy puts 0.8 on logged action for each row
        pol = np.array(
            [
                [0.2, 0.8],  # logged action 0, weight = 0.2/0.5 = 0.4
                [0.2, 0.8],  # logged action 1, weight = 0.8/0.5 = 1.6
                [0.2, 0.8],  # logged action 0, weight = 0.4
                [0.2, 0.8],  # logged action 1, weight = 1.6
            ]
        )
        out = compute_snipw(bf, pol, min_propensity_clip=0.01)
        # numerator = sum(w_i * r_i) = 0.4*0 + 1.6*1 + 0.4*0 + 1.6*1 = 3.2
        # denominator = sum(w_i) = 0.4 + 1.6 + 0.4 + 1.6 = 4.0
        # snipw = 3.2 / 4.0 = 0.8
        assert out == pytest.approx(0.8)

    def test_snipw_is_bounded_in_reward_range(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=500, n_actions=4, n_positions=1, seed=1)
        pol = peaked_policy(n_rounds=500, n_actions=4, best_action=1, peak_mass=0.9)
        val = compute_snipw(bf, pol, min_propensity_clip=0.01)
        assert 0.0 <= val <= 1.0

    def test_snipw_with_heterogeneous_propensities(self) -> None:
        # Real loggers rarely have uniform pscore. Verify SNIPW correctly
        # reweights when pscore varies row-by-row: two rows with pscore 0.2
        # and two rows with pscore 0.8, same eval policy mass → weights
        # should scale inversely with pscore.
        bf: dict[str, Any] = {
            "n_rounds": 4,
            "n_actions": 2,
            "action": np.array([0, 0, 1, 1]),
            "reward": np.array([1.0, 0.0, 1.0, 0.0]),
            "pscore": np.array([0.2, 0.2, 0.8, 0.8]),
            "position": np.zeros(4, dtype=int),
        }
        # Eval policy: uniform (pi_e = 0.5 on each logged action).
        pol = np.full((4, 2), 0.5)
        # Weights: 0.5/0.2 = 2.5 (rows 0,1), 0.5/0.8 = 0.625 (rows 2,3).
        # Numerator = 2.5*1 + 2.5*0 + 0.625*1 + 0.625*0 = 3.125
        # Denominator = 2.5 + 2.5 + 0.625 + 0.625 = 6.25
        # SNIPW = 3.125 / 6.25 = 0.5
        val = compute_snipw(bf, pol, min_propensity_clip=0.01)
        assert val == pytest.approx(0.5)

    def test_dm_with_constant_reward_model(self) -> None:
        # reward_hat constant 0.3 for every (action, position) → DM = 0.3
        pol = uniform_policy(n_rounds=10, n_actions=3)
        reward_hat = np.full((10, 3), 0.3)
        val = compute_dm(pol, reward_hat)
        assert val == pytest.approx(0.3)

    def test_dm_weighted_by_eval_policy(self) -> None:
        pol = np.array([[0.0, 1.0]])  # round 1, all mass on action 1
        reward_hat = np.array([[0.0, 0.5]])  # action 1 estimate = 0.5
        val = compute_dm(pol, reward_hat)
        assert val == pytest.approx(0.5)

    def test_dr_falls_back_to_dm_when_pscore_is_exact(self) -> None:
        # If pscore == eval policy mass on logged action, IPW correction is
        # weighted by reward residual. With perfect reward_hat → DR ≈ DM.
        bf: dict[str, Any] = {
            "n_rounds": 5,
            "n_actions": 2,
            "action": np.array([0, 1, 0, 1, 0]),
            "reward": np.array([0.1, 0.2, 0.1, 0.2, 0.1]),
            "pscore": np.full(5, 0.5),
            "position": np.zeros(5, dtype=int),
        }
        pol = np.full((5, 2), 0.5)
        # reward_hat is exactly the row's reward → residual is zero → DR = DM
        reward_hat = np.zeros((5, 2))
        reward_hat[np.arange(5), bf["action"]] = bf["reward"]
        dm_val = compute_dm(pol, reward_hat)
        dr_val = compute_dr(bf, pol, reward_hat, min_propensity_clip=0.01)
        # With residual zero the DR IPW correction vanishes; DR collapses to DM.
        assert dr_val == pytest.approx(dm_val, abs=1e-9)

    def test_dr_with_nonzero_residuals_matches_hand_computation(self) -> None:
        # Hand-computed DR on 2 rows:
        # - row 0: action 0, reward 1.0, pscore 0.5, pi_e(0)=0.5, reward_hat=[0.2, 0.3]
        #   DM term = 0.5*0.2 + 0.5*0.3 = 0.25
        #   IPW correction = (0.5/0.5) * (1.0 - 0.2) = 0.8
        #   row 0 contribution = 0.25 + 0.8 = 1.05
        # - row 1: action 1, reward 0.0, pscore 0.5, pi_e(1)=0.5, reward_hat=[0.2, 0.3]
        #   DM term = 0.5*0.2 + 0.5*0.3 = 0.25
        #   IPW correction = (0.5/0.5) * (0.0 - 0.3) = -0.3
        #   row 1 contribution = 0.25 - 0.3 = -0.05
        # DR = mean([1.05, -0.05]) = 0.5
        bf: dict[str, Any] = {
            "n_rounds": 2,
            "n_actions": 2,
            "action": np.array([0, 1]),
            "reward": np.array([1.0, 0.0]),
            "pscore": np.array([0.5, 0.5]),
            "position": np.zeros(2, dtype=int),
        }
        pol = np.full((2, 2), 0.5)
        reward_hat = np.array([[0.2, 0.3], [0.2, 0.3]])
        dr_val = compute_dr(bf, pol, reward_hat, min_propensity_clip=0.01)
        assert dr_val == pytest.approx(0.5)


# ── evaluate_open_bandit_policy diagnostic dict ──────────────────────────────────


class TestEvaluatePolicyDiagnostics:
    """Section 4d diagnostics returned by ``evaluate_open_bandit_policy``."""

    def test_diagnostic_keys_present(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=100, n_actions=4, n_positions=2, seed=0)
        pol = uniform_policy(n_rounds=100, n_actions=4)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        for key in (
            "policy_value",
            "ess",
            "weight_cv",
            "max_weight",
            "zero_support_fraction",
            "n_effective_actions",
            "n_clipped_rows",
            "estimator",
        ):
            assert key in out

    def test_ess_is_bounded_by_n_rows(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=200, n_actions=4, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=200, n_actions=4)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert 0.0 <= out["ess"] <= 200.0

    def test_zero_support_fraction_is_one_for_degenerate_on_non_logged(self) -> None:
        # logger took action 1 for every row; eval policy puts everything on 0
        bf: dict[str, Any] = {
            "n_rounds": 10,
            "n_actions": 2,
            "action": np.ones(10, dtype=int),
            "reward": np.zeros(10),
            "pscore": np.full(10, 0.5),
            "position": np.zeros(10, dtype=int),
        }
        pol = degenerate_policy(n_rounds=10, n_actions=2, support_action=0)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert out["zero_support_fraction"] == pytest.approx(1.0)

    def test_n_effective_actions_for_uniform_is_n_actions(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=5, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=50, n_actions=5)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert out["n_effective_actions"] == 5

    def test_n_effective_actions_for_peaked_is_small(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=10, n_positions=1, seed=0)
        pol = peaked_policy(n_rounds=50, n_actions=10, best_action=3, peak_mass=0.99)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        # 0.99 already exceeds 0.95 → only 1 effective action
        assert out["n_effective_actions"] == 1

    def test_n_effective_actions_custom_threshold(self) -> None:
        # Exercise the effective_action_mass_threshold keyword. With a
        # peaked policy at 0.5 mass on the best action, a 0.4 threshold
        # resolves to 1 (the peak alone exceeds 0.4), while the default
        # 0.95 requires multiple actions.
        bf = generate_synthetic_bandit_feedback(n_rounds=20, n_actions=5, n_positions=1, seed=0)
        pol = peaked_policy(n_rounds=20, n_actions=5, best_action=0, peak_mass=0.5)
        out_low = evaluate_open_bandit_policy(
            bf, pol, min_propensity_clip=0.01, effective_action_mass_threshold=0.4
        )
        out_high = evaluate_open_bandit_policy(
            bf, pol, min_propensity_clip=0.01, effective_action_mass_threshold=0.95
        )
        assert out_low["n_effective_actions"] == 1
        assert out_high["n_effective_actions"] > 1

    def test_provenance_records_estimator_name(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=20, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=20, n_actions=3)
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        assert out["estimator"] == "snipw"


# ── 7a Null control gate ─────────────────────────────────────────────


class TestNullControlGate:
    def test_permutation_is_stratified_by_position(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=100, n_actions=4, n_positions=3, seed=0)
        # Build a reward vector whose values cluster by position so that an
        # unstratified permutation would scramble that structure.
        reward = np.zeros(100)
        reward[bf["position"] == 0] = 1.0
        reward[bf["position"] == 1] = 2.0
        reward[bf["position"] == 2] = 3.0
        bf["reward"] = reward
        permuted = permute_rewards_stratified(bf, seed=42)
        # Per-position mean must be preserved (permutation within strata)
        for pos in range(3):
            mask = bf["position"] == pos
            assert permuted["reward"][mask].sum() == pytest.approx(reward[mask].sum())
            # Means by position remain identical after within-stratum permutation
            assert permuted["reward"][mask].mean() == pytest.approx(reward[mask].mean())

    def test_permutation_is_deterministic_under_seed(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=4, n_positions=2, seed=0)
        p1 = permute_rewards_stratified(bf, seed=7)
        p2 = permute_rewards_stratified(bf, seed=7)
        np.testing.assert_array_equal(p1["reward"], p2["reward"])

    def test_permutation_does_not_mutate_input(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=30, n_actions=3, n_positions=2, seed=0)
        original = bf["reward"].copy()
        permute_rewards_stratified(bf, seed=1)
        np.testing.assert_array_equal(bf["reward"], original)

    def test_null_control_gate_passes_on_permuted_data(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=500, n_actions=4, n_positions=2, seed=0)
        # Uniform eval policy on permuted data should sit at the permuted mean
        pol = uniform_policy(n_rounds=500, n_actions=4)
        result = null_control_gate(bf, strategy_policies={"uniform": pol}, permutation_seed=13)
        assert result.passed is True
        assert "uniform" in result.per_strategy_values
        # ratio under uniform is exactly 1.0 (ignoring floating point)
        assert result.per_strategy_ratios["uniform"] <= 1.05

    def test_null_control_gate_fails_when_strategy_exceeds_band(self) -> None:
        # A permuted-outcome strategy should sit at mu_null. The gate
        # must fail when any strategy reports a value above 1.05 * mu_null,
        # so we drive failure via the test-only override hook.
        bf = generate_synthetic_bandit_feedback(n_rounds=500, n_actions=4, n_positions=2, seed=0)
        pol = uniform_policy(n_rounds=500, n_actions=4)
        permuted = permute_rewards_stratified(bf, seed=3)
        mu_null = float(permuted["reward"].mean())
        inflated = null_control_gate(
            bf,
            strategy_policies={"adversarial": pol},
            permutation_seed=3,
            strategy_value_overrides={"adversarial": 2.0 * mu_null + 1e-3},
        )
        assert inflated.passed is False
        assert np.isfinite(inflated.per_strategy_values["adversarial"])

    def test_null_control_rejects_one_indexed_positions(self) -> None:
        # OBD uses 0-indexed {0, 1, 2} positions; a 1-indexed loader
        # would silently over-count n_positions and mis-scale the clip
        # floor. The gate must refuse 1-indexed input.
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=4, n_positions=2, seed=0)
        bf["position"] = bf["position"] + 1  # shift to 1-indexed
        pol = uniform_policy(n_rounds=50, n_actions=4)
        with pytest.raises(ValueError, match="0-indexed"):
            null_control_gate(bf, strategy_policies={"s": pol}, permutation_seed=1)

    def test_permute_rejects_one_indexed_positions(self) -> None:
        # Validation must also fire from permute_rewards_stratified so
        # independent callers cannot smuggle in a shifted position array.
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=3, n_positions=2, seed=0)
        bf["position"] = bf["position"] + 1
        with pytest.raises(ValueError, match="0-indexed"):
            permute_rewards_stratified(bf, seed=1)

    def test_permute_rejects_gapped_positions(self) -> None:
        # Positions {0, 2} (missing 1) would make n_positions look like 3
        # to downstream math and mis-scale the clip floor.
        bf = generate_synthetic_bandit_feedback(n_rounds=50, n_actions=3, n_positions=2, seed=0)
        # Replace all position=1 rows with position=2 so uniques become {0, 2}.
        pos = bf["position"].copy()
        pos[pos == 1] = 2
        bf["position"] = pos
        with pytest.raises(ValueError, match="contiguous"):
            permute_rewards_stratified(bf, seed=1)

    def test_permute_accepts_empty_positions(self) -> None:
        # Defensive empty-input branch in _validate_positions: an empty
        # position array is a no-op and must not raise.
        bf: dict[str, Any] = {
            "n_rounds": 0,
            "n_actions": 3,
            "action": np.array([], dtype=int),
            "reward": np.array([], dtype=float),
            "pscore": np.array([], dtype=float),
            "position": np.array([], dtype=int),
        }
        out = permute_rewards_stratified(bf, seed=1)
        assert out["reward"].size == 0


# ── 7b ESS floor ─────────────────────────────────────────────────────


class TestESSGate:
    def test_ess_floor_formula(self) -> None:
        # floor = max(1000, n_rows/100)
        result = ess_gate(per_seed_ess=[5000] * 10, n_rows=200000)
        assert result.floor == 2000
        assert result.passed is True

    def test_ess_floor_lower_bound_1000(self) -> None:
        # small n_rows → floor is 1000
        result = ess_gate(per_seed_ess=[1500] * 10, n_rows=1000)
        assert result.floor == 1000
        assert result.passed is True

    def test_ess_gate_fails_when_median_below_floor(self) -> None:
        result = ess_gate(per_seed_ess=[100, 200, 300, 400, 500], n_rows=200000)
        assert result.floor == 2000
        assert result.median_ess == 300
        assert result.passed is False


# ── 7c Zero-support fraction ─────────────────────────────────────────


class TestZeroSupportGate:
    def test_zero_support_gate_passes_below_threshold(self) -> None:
        result = zero_support_gate(per_seed_zero_support=[0.01, 0.02, 0.0, 0.05])
        assert result.passed is True
        assert result.best == 0.0

    def test_zero_support_gate_fails_when_best_exceeds_10pct(self) -> None:
        # best = min (lowest zero-support); but gate should fail if
        # *best-of-seed policy's* zero-support exceeds 10%.  Using "best"
        # from per-seed as the minimum here: every seed is above 10% → fail.
        result = zero_support_gate(per_seed_zero_support=[0.15, 0.20, 0.12])
        assert result.passed is False
        assert result.best == pytest.approx(0.12)


# ── 7d Propensity sanity ─────────────────────────────────────────────


class TestPropensitySanityGate:
    def test_joint_schema_target(self) -> None:
        pscore = np.full(1000, 1.0 / (34 * 3))
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_JOINT,
            n_actions=34,
            n_positions=3,
        )
        assert result.passed is True
        assert result.target == pytest.approx(1.0 / (34 * 3))

    def test_conditional_schema_target(self) -> None:
        pscore = np.full(1000, 1.0 / 34)
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_CONDITIONAL,
            n_actions=34,
            n_positions=3,
        )
        assert result.passed is True
        assert result.target == pytest.approx(1.0 / 34)

    def test_outside_10pct_relative_band_fails(self) -> None:
        # target is 1/102 ≈ 0.0098; set empirical = 0.012 (22% high)
        pscore = np.full(1000, 0.012)
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_JOINT,
            n_actions=34,
            n_positions=3,
        )
        assert result.passed is False
        assert result.relative_deviation > 0.10

    def test_within_band_passes(self) -> None:
        target = 1.0 / (34 * 3)
        pscore = np.full(1000, target * 1.05)  # 5% high
        result = propensity_sanity_gate(
            pscore=pscore,
            schema=PROPENSITY_SCHEMA_JOINT,
            n_actions=34,
            n_positions=3,
        )
        assert result.passed is True

    def test_unknown_schema_raises(self) -> None:
        pscore = np.full(100, 0.01)
        with pytest.raises(ValueError, match="schema"):
            propensity_sanity_gate(
                pscore=pscore,
                schema="bogus",  # type: ignore[arg-type]
                n_actions=34,
                n_positions=3,
            )


# ── 7e DR/SNIPW cross-check ──────────────────────────────────────────


class TestSnipwDrCrossCheck:
    def test_matching_values_pass(self) -> None:
        snipw_per_seed = [0.10, 0.11, 0.09]
        dr_per_seed = [0.105, 0.112, 0.088]
        result = snipw_dr_cross_check_gate(snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed)
        assert result.passed is True
        assert result.max_relative_divergence < 0.25

    def test_large_divergence_fails(self) -> None:
        snipw_per_seed = [0.10, 0.11, 0.09]
        dr_per_seed = [0.20, 0.30, 0.50]  # all > 25% divergent
        result = snipw_dr_cross_check_gate(snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed)
        assert result.passed is False
        assert result.max_relative_divergence > 0.25

    def test_single_seed_over_threshold_fails(self) -> None:
        snipw_per_seed = [0.10, 0.10, 0.10]
        dr_per_seed = [0.10, 0.10, 0.50]
        result = snipw_dr_cross_check_gate(snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed)
        assert result.passed is False
        # Flag the offending seed index
        assert 2 in result.offending_seeds

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            snipw_dr_cross_check_gate(snipw_per_seed=[0.1, 0.2], dr_per_seed=[0.1])

    def test_zero_baseline_handled_gracefully(self) -> None:
        # When SNIPW is zero we can't divide; treat divergence as relative to
        # DR magnitude
        result = snipw_dr_cross_check_gate(snipw_per_seed=[0.0], dr_per_seed=[0.05])
        # Very large relative divergence
        assert result.passed is False


# ── Input validation (covers ValueError branches) ──────────────────


class TestInputValidation:
    def test_compute_min_propensity_clip_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_min_propensity_clip(n_actions=0, n_positions=3)
        with pytest.raises(ValueError, match="positive"):
            compute_min_propensity_clip(n_actions=3, n_positions=0)

    def test_generate_synthetic_bandit_feedback_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="n_rounds"):
            generate_synthetic_bandit_feedback(n_rounds=0, n_actions=3, n_positions=1, seed=0)
        with pytest.raises(ValueError, match="n_actions"):
            generate_synthetic_bandit_feedback(n_rounds=10, n_actions=0, n_positions=1, seed=0)
        with pytest.raises(ValueError, match="n_positions"):
            generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=0, seed=0)
        with pytest.raises(ValueError, match="propensity_schema"):
            generate_synthetic_bandit_feedback(
                n_rounds=10,
                n_actions=3,
                n_positions=1,
                seed=0,
                propensity_schema="bogus",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="reward_mean_by_action"):
            generate_synthetic_bandit_feedback(
                n_rounds=10,
                n_actions=3,
                n_positions=1,
                seed=0,
                reward_mean_by_action=[0.1, 0.2],  # wrong length
            )
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            generate_synthetic_bandit_feedback(
                n_rounds=10,
                n_actions=3,
                n_positions=1,
                seed=0,
                reward_mean_by_action=[0.1, 0.2, 1.5],  # out of [0, 1]
            )

    def test_uniform_policy_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            uniform_policy(n_rounds=0, n_actions=3)

    def test_peaked_policy_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            peaked_policy(n_rounds=0, n_actions=3, best_action=0, peak_mass=0.5)
        with pytest.raises(ValueError, match="best_action"):
            peaked_policy(n_rounds=2, n_actions=3, best_action=5, peak_mass=0.5)
        # peak_mass above 1.0 is rejected
        with pytest.raises(ValueError, match="peak_mass"):
            peaked_policy(n_rounds=2, n_actions=3, best_action=0, peak_mass=1.5)
        # peak_mass below 1/n_actions would be "less peaked than uniform"
        with pytest.raises(ValueError, match="peak_mass"):
            peaked_policy(n_rounds=2, n_actions=4, best_action=0, peak_mass=0.1)

    def test_degenerate_policy_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            degenerate_policy(n_rounds=0, n_actions=3, support_action=0)
        with pytest.raises(ValueError, match="support_action"):
            degenerate_policy(n_rounds=2, n_actions=3, support_action=5)

    def test_compute_dm_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_dm(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_compute_dr_rejects_shape_mismatch(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=5, n_actions=3, n_positions=1, seed=0)
        with pytest.raises(ValueError, match="shape"):
            compute_dr(bf, np.zeros((5, 3)), np.zeros((5, 2)), min_propensity_clip=0.01)

    def test_compute_dr_rejects_row_mismatch(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=5, n_actions=3, n_positions=1, seed=0)
        with pytest.raises(ValueError, match="rows"):
            compute_dr(bf, np.zeros((6, 3)), np.zeros((6, 3)), min_propensity_clip=0.01)

    def test_propensity_sanity_gate_rejects_non_positive_dims(self) -> None:
        pscore = np.full(10, 0.01)
        with pytest.raises(ValueError, match="positive"):
            propensity_sanity_gate(
                pscore=pscore,
                schema=PROPENSITY_SCHEMA_JOINT,
                n_actions=0,
                n_positions=3,
            )
        with pytest.raises(ValueError, match="positive"):
            propensity_sanity_gate(
                pscore=pscore,
                schema=PROPENSITY_SCHEMA_JOINT,
                n_actions=3,
                n_positions=0,
            )

    def test_evaluate_policy_rejects_row_mismatch(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=20, n_actions=3)  # wrong row count
        with pytest.raises(ValueError, match="rows"):
            evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)

    def test_evaluate_policy_requires_reward_hat_for_dm(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=10, n_actions=3)
        with pytest.raises(ValueError, match="reward_hat"):
            evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01, estimator="dm")

    def test_evaluate_policy_requires_reward_hat_for_dr(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=10, n_actions=3)
        with pytest.raises(ValueError, match="reward_hat"):
            evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01, estimator="dr")

    def test_evaluate_policy_rejects_unknown_estimator(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=10, n_actions=3)
        with pytest.raises(ValueError, match="estimator"):
            evaluate_open_bandit_policy(
                bf,
                pol,
                min_propensity_clip=0.01,
                estimator="switch_dr",  # type: ignore[arg-type]
            )

    def test_evaluate_policy_with_dm_estimator_returns_value(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=10, n_actions=3)
        reward_hat = np.full((10, 3), 0.05)
        out = evaluate_open_bandit_policy(
            bf, pol, min_propensity_clip=0.01, reward_hat=reward_hat, estimator="dm"
        )
        assert out["estimator"] == "dm"
        assert out["policy_value"] == pytest.approx(0.05)

    def test_evaluate_policy_with_dr_estimator_returns_value(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        pol = uniform_policy(n_rounds=10, n_actions=3)
        reward_hat = np.zeros((10, 3))
        out = evaluate_open_bandit_policy(
            bf, pol, min_propensity_clip=0.01, reward_hat=reward_hat, estimator="dr"
        )
        assert out["estimator"] == "dr"
        assert np.isfinite(out["policy_value"])

    def test_permute_rewards_requires_position(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=10, n_actions=3, n_positions=1, seed=0)
        del bf["position"]
        with pytest.raises(KeyError, match="position"):
            permute_rewards_stratified(bf, seed=0)

    def test_ess_gate_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ess_gate(per_seed_ess=[], n_rows=100)

    def test_ess_gate_rejects_negative_n_rows(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ess_gate(per_seed_ess=[1000], n_rows=-1)

    def test_zero_support_gate_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            zero_support_gate(per_seed_zero_support=[])

    def test_snipw_dr_cross_check_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            snipw_dr_cross_check_gate(snipw_per_seed=[], dr_per_seed=[])

    def test_snipw_returns_zero_when_all_weights_zero(self) -> None:
        # Evaluation policy places zero mass on every logged action → denom=0
        bf = {
            "n_rounds": 4,
            "n_actions": 2,
            "action": np.array([0, 0, 0, 0]),
            "reward": np.array([1.0, 1.0, 1.0, 1.0]),
            "pscore": np.full(4, 0.5),
            "position": np.zeros(4, dtype=int),
        }
        # Put all mass on action 1 → pi_e on logged action 0 is always 0
        pol = degenerate_policy(n_rounds=4, n_actions=2, support_action=1)
        assert compute_snipw(bf, pol, min_propensity_clip=0.01) == 0.0


# ── Provenance / schema constants ────────────────────────────────────


class TestProvenance:
    def test_schema_constants_stable(self) -> None:
        assert PROPENSITY_SCHEMA_CONDITIONAL == "conditional"
        assert PROPENSITY_SCHEMA_JOINT == "joint"
        assert DEFAULT_PROPENSITY_SCHEMA in {
            PROPENSITY_SCHEMA_CONDITIONAL,
            PROPENSITY_SCHEMA_JOINT,
        }
