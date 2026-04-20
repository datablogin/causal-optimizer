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
    PropensitySchema,
    compute_dm,
    compute_dr,
    compute_min_propensity_clip,
    compute_snipw,
    degenerate_policy,
    ess_gate,
    evaluate_policy,
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
        bf = generate_synthetic_bandit_feedback(
            n_rounds=100, n_actions=4, n_positions=2, seed=0
        )
        for key in ("n_rounds", "n_actions", "action", "reward", "pscore"):
            assert key in bf

    def test_bandit_feedback_shapes(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=50, n_actions=6, n_positions=3, seed=0
        )
        assert bf["n_rounds"] == 50
        assert bf["n_actions"] == 6
        assert bf["action"].shape == (50,)
        assert bf["reward"].shape == (50,)
        assert bf["pscore"].shape == (50,)
        assert bf["position"].shape == (50,)

    def test_bandit_feedback_is_deterministic(self) -> None:
        bf1 = generate_synthetic_bandit_feedback(
            n_rounds=100, n_actions=4, n_positions=2, seed=42
        )
        bf2 = generate_synthetic_bandit_feedback(
            n_rounds=100, n_actions=4, n_positions=2, seed=42
        )
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
        pol = peaked_policy(
            n_rounds=20, n_actions=5, best_action=2, peak_mass=0.9
        )
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
        bf = generate_synthetic_bandit_feedback(
            n_rounds=20, n_actions=4, n_positions=1, seed=0
        )
        # Inject a pathologically small propensity
        bf["pscore"] = bf["pscore"].copy()
        bf["pscore"][0] = 1e-12
        pol = uniform_policy(n_rounds=20, n_actions=4)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
        assert np.isfinite(out["policy_value"])
        assert np.isfinite(out["max_weight"])

    def test_clipped_row_count_reported(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=20, n_actions=4, n_positions=1, seed=0
        )
        bf["pscore"] = bf["pscore"].copy()
        bf["pscore"][:5] = 1e-6  # 5 rows below the 0.01 floor
        pol = uniform_policy(n_rounds=20, n_actions=4)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
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
        bf = generate_synthetic_bandit_feedback(
            n_rounds=500, n_actions=4, n_positions=1, seed=1
        )
        pol = peaked_policy(n_rounds=500, n_actions=4, best_action=1, peak_mass=0.9)
        val = compute_snipw(bf, pol, min_propensity_clip=0.01)
        assert 0.0 <= val <= 1.0

    def test_dm_with_constant_reward_model(self) -> None:
        # reward_hat constant 0.3 for every (action, position) → DM = 0.3
        bf: dict[str, Any] = {
            "n_rounds": 10,
            "n_actions": 3,
            "action": np.zeros(10, dtype=int),
            "reward": np.zeros(10),
            "pscore": np.full(10, 1.0 / 3),
            "position": np.zeros(10, dtype=int),
        }
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


# ── evaluate_policy diagnostic dict ──────────────────────────────────


class TestEvaluatePolicyDiagnostics:
    """Section 4d diagnostics returned by ``evaluate_policy``."""

    def test_diagnostic_keys_present(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=100, n_actions=4, n_positions=2, seed=0
        )
        pol = uniform_policy(n_rounds=100, n_actions=4)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
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
        bf = generate_synthetic_bandit_feedback(
            n_rounds=200, n_actions=4, n_positions=1, seed=0
        )
        pol = uniform_policy(n_rounds=200, n_actions=4)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
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
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
        assert out["zero_support_fraction"] == pytest.approx(1.0)

    def test_n_effective_actions_for_uniform_is_n_actions(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=50, n_actions=5, n_positions=1, seed=0
        )
        pol = uniform_policy(n_rounds=50, n_actions=5)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
        assert out["n_effective_actions"] == 5

    def test_n_effective_actions_for_peaked_is_small(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=50, n_actions=10, n_positions=1, seed=0
        )
        pol = peaked_policy(n_rounds=50, n_actions=10, best_action=3, peak_mass=0.99)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
        # 0.99 already exceeds 0.95 → only 1 effective action
        assert out["n_effective_actions"] == 1

    def test_provenance_records_estimator_name(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=20, n_actions=3, n_positions=1, seed=0
        )
        pol = uniform_policy(n_rounds=20, n_actions=3)
        out = evaluate_policy(bf, pol, min_propensity_clip=0.01)
        assert out["estimator"] == "snipw"


# ── 7a Null control gate ─────────────────────────────────────────────


class TestNullControlGate:
    def test_permutation_is_stratified_by_position(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=100, n_actions=4, n_positions=3, seed=0
        )
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
            assert permuted["reward"][mask].mean() == pytest.approx(
                reward[mask].mean()
            )

    def test_permutation_is_deterministic_under_seed(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=50, n_actions=4, n_positions=2, seed=0
        )
        p1 = permute_rewards_stratified(bf, seed=7)
        p2 = permute_rewards_stratified(bf, seed=7)
        np.testing.assert_array_equal(p1["reward"], p2["reward"])

    def test_permutation_does_not_mutate_input(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=30, n_actions=3, n_positions=2, seed=0
        )
        original = bf["reward"].copy()
        permute_rewards_stratified(bf, seed=1)
        np.testing.assert_array_equal(bf["reward"], original)

    def test_null_control_gate_passes_on_permuted_data(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=500, n_actions=4, n_positions=2, seed=0
        )
        # Uniform eval policy on permuted data should sit at the permuted mean
        pol = uniform_policy(n_rounds=500, n_actions=4)
        result = null_control_gate(
            bf, strategy_policies={"uniform": pol}, permutation_seed=13
        )
        assert result.passed is True
        assert "uniform" in result.per_strategy_values
        # ratio under uniform is exactly 1.0 (ignoring floating point)
        assert result.per_strategy_ratios["uniform"] <= 1.05

    def test_null_control_gate_fails_when_strategy_exceeds_band(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=500, n_actions=4, n_positions=2, seed=0
        )
        # Craft a "bad" policy that correlates with reward: force a policy that
        # heavily weights the logged action where reward is high.  We do this
        # by supplying ground-truth reward as the policy's input.
        permuted = permute_rewards_stratified(bf, seed=3)
        action = bf["action"]
        n = bf["n_rounds"]
        n_a = bf["n_actions"]
        pol = np.full((n, n_a), 1e-8)
        # place dominant mass on the logged action so the estimate matches
        # the per-row reward — this is the pathological case
        pol[np.arange(n), action] = 1.0
        pol = pol / pol.sum(axis=1, keepdims=True)
        result = null_control_gate(
            bf,
            strategy_policies={"adversarial": pol},
            permutation_seed=3,
        )
        # The permuted reward is random, but matching the logged action makes
        # SNIPW return the per-row permuted reward mean — still within the band
        # unless we rig it. Directly rig: pretend the policy achieved 2x the
        # null mean via a manual override.
        mu_null = float(permuted["reward"].mean())
        inflated_value = 2.0 * mu_null + 1e-3
        inflated = null_control_gate(
            bf,
            strategy_policies={"adversarial": pol},
            permutation_seed=3,
            strategy_value_overrides={"adversarial": inflated_value},
        )
        # With manual override above 1.05 * mu_null → gate fails
        assert inflated.passed is False
        # And the unrigged value should also be finite
        assert np.isfinite(result.per_strategy_values["adversarial"])


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
        result = snipw_dr_cross_check_gate(
            snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed
        )
        assert result.passed is True
        assert result.max_relative_divergence < 0.25

    def test_large_divergence_fails(self) -> None:
        snipw_per_seed = [0.10, 0.11, 0.09]
        dr_per_seed = [0.20, 0.30, 0.50]  # all > 25% divergent
        result = snipw_dr_cross_check_gate(
            snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed
        )
        assert result.passed is False
        assert result.max_relative_divergence > 0.25

    def test_single_seed_over_threshold_fails(self) -> None:
        snipw_per_seed = [0.10, 0.10, 0.10]
        dr_per_seed = [0.10, 0.10, 0.50]
        result = snipw_dr_cross_check_gate(
            snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed
        )
        assert result.passed is False
        # Flag the offending seed index
        assert 2 in result.offending_seeds

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            snipw_dr_cross_check_gate(
                snipw_per_seed=[0.1, 0.2], dr_per_seed=[0.1]
            )

    def test_zero_baseline_handled_gracefully(self) -> None:
        # When SNIPW is zero we can't divide; treat divergence as relative to
        # DR magnitude
        result = snipw_dr_cross_check_gate(
            snipw_per_seed=[0.0], dr_per_seed=[0.05]
        )
        # Very large relative divergence
        assert result.passed is False


# ── Provenance / schema constants ────────────────────────────────────


class TestProvenance:
    def test_schema_constants_stable(self) -> None:
        assert PROPENSITY_SCHEMA_CONDITIONAL == "conditional"
        assert PROPENSITY_SCHEMA_JOINT == "joint"
        assert DEFAULT_PROPENSITY_SCHEMA in {
            PROPENSITY_SCHEMA_CONDITIONAL,
            PROPENSITY_SCHEMA_JOINT,
        }

    def test_propensity_schema_literal_type(self) -> None:
        # Runtime import smoke
        assert PropensitySchema is not None
