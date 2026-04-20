"""Integration tests for the Sprint 35 Open Bandit OPE gates.

End-to-end gate runs against synthetic bandit-feedback dicts.  Verifies
the five Section 7 gates interact cleanly with ``evaluate_open_bandit_policy`` and
each gate has at least one passing-and-one-failing scenario that fires
deterministically.
"""

from __future__ import annotations

import numpy as np

from causal_optimizer.benchmarks.open_bandit import (
    PROPENSITY_SCHEMA_JOINT,
    GateReport,
    compute_dr,
    compute_snipw,
    ess_gate,
    evaluate_open_bandit_policy,
    generate_synthetic_bandit_feedback,
    null_control_gate,
    peaked_policy,
    propensity_sanity_gate,
    run_section_7_gates,
    snipw_dr_cross_check_gate,
    uniform_policy,
    zero_support_gate,
)


class TestEndToEndPassingRun:
    """All five gates pass on a clean synthetic setup."""

    def test_all_gates_pass_on_uniform_logger_uniform_policy(self) -> None:
        n_rounds = 5000
        n_actions = 10
        n_positions = 3
        # Use the first seed's feedback dict as the slice fed into
        # run_section_7_gates (the null control and propensity gates
        # operate on the actual data), while each seed evaluates on its
        # own independently-sampled feedback so the per-seed lists
        # contain distinct values.
        bf = generate_synthetic_bandit_feedback(
            n_rounds=n_rounds,
            n_actions=n_actions,
            n_positions=n_positions,
            seed=0,
            propensity_schema=PROPENSITY_SCHEMA_JOINT,
        )
        policies = {
            "random": uniform_policy(n_rounds=n_rounds, n_actions=n_actions),
            "surrogate_only": uniform_policy(n_rounds=n_rounds, n_actions=n_actions),
            "causal": uniform_policy(n_rounds=n_rounds, n_actions=n_actions),
        }
        per_seed_ess = []
        per_seed_zero_support = []
        snipw_per_seed = []
        dr_per_seed = []
        clip = 1.0 / (2 * n_actions * n_positions)
        reward_hat = np.full((n_rounds, n_actions), 0.02)
        for seed in range(3):
            bf_seed = generate_synthetic_bandit_feedback(
                n_rounds=n_rounds,
                n_actions=n_actions,
                n_positions=n_positions,
                seed=seed,
                propensity_schema=PROPENSITY_SCHEMA_JOINT,
            )
            pol = uniform_policy(n_rounds=n_rounds, n_actions=n_actions)
            out = evaluate_open_bandit_policy(bf_seed, pol, min_propensity_clip=clip)
            per_seed_ess.append(out["ess"])
            per_seed_zero_support.append(out["zero_support_fraction"])
            snipw_per_seed.append(out["policy_value"])
            dr_per_seed.append(compute_dr(bf_seed, pol, reward_hat, min_propensity_clip=clip))

        report = run_section_7_gates(
            bandit_feedback=bf,
            strategy_policies=policies,
            per_seed_ess=per_seed_ess,
            per_seed_zero_support=per_seed_zero_support,
            snipw_per_seed=snipw_per_seed,
            dr_per_seed=dr_per_seed,
            n_actions=n_actions,
            n_positions=n_positions,
            schema=PROPENSITY_SCHEMA_JOINT,
            permutation_seed=7,
        )
        assert isinstance(report, GateReport)
        assert report.null_control.passed
        assert report.ess.passed
        assert report.zero_support.passed
        assert report.propensity_sanity.passed
        assert report.dr_cross_check.passed
        assert report.all_passed() is True


class TestIndividualGateFailures:
    """Each gate must have a failing scenario too."""

    def test_ess_gate_fails_on_small_median(self) -> None:
        result = ess_gate(per_seed_ess=[10] * 10, n_rows=200_000)
        assert result.passed is False

    def test_zero_support_fails_on_degenerate(self) -> None:
        # degenerate policy on actions the logger never took → 100% zero support
        # (matches the unit test) — worst seed is still over 10%
        result = zero_support_gate(per_seed_zero_support=[0.4, 0.5, 0.6])
        assert result.passed is False

    def test_propensity_sanity_fails_outside_band(self) -> None:
        bad_pscore = np.full(1000, 1.0 / 34)
        result = propensity_sanity_gate(
            pscore=bad_pscore,
            schema=PROPENSITY_SCHEMA_JOINT,
            n_actions=34,
            n_positions=3,
        )
        assert result.passed is False  # joint target is 1/102, not 1/34

    def test_dr_snipw_fails_on_divergence(self) -> None:
        result = snipw_dr_cross_check_gate(snipw_per_seed=[0.01], dr_per_seed=[0.50])
        assert result.passed is False

    def test_null_control_fails_with_override(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=500, n_actions=4, n_positions=2, seed=0)
        pol = uniform_policy(n_rounds=500, n_actions=4)
        # Use an override to drive the value well above the 5% band
        permuted_mean_upper_bound = 10.0  # synthetic reward mean guaranteed < this
        result = null_control_gate(
            bf,
            strategy_policies={"rigged": pol},
            permutation_seed=1,
            strategy_value_overrides={"rigged": permuted_mean_upper_bound},
        )
        assert result.passed is False


class TestSnipwSanity:
    """SNIPW is the primary verdict estimator; cross-check vs a known result."""

    def test_snipw_on_degenerate_eval_policy_matches_logged_action_reward_mean(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=2000, n_actions=5, n_positions=1, seed=3)
        # Eval policy that perfectly mirrors the logger → SNIPW ≈ mean(reward)
        n_rounds = bf["n_rounds"]
        n_actions = bf["n_actions"]
        pol = np.full((n_rounds, n_actions), 1.0 / n_actions)
        expected = float(bf["reward"].mean())
        snipw_val = compute_snipw(bf, pol, min_propensity_clip=0.01)
        # With a uniform eval policy matching the uniform logger, every
        # SNIPW weight is 1 and the estimate collapses to the sample
        # reward mean.
        assert np.isclose(snipw_val, expected, atol=1e-9)

    def test_snipw_peaked_policy_moves_estimate(self) -> None:
        bf = generate_synthetic_bandit_feedback(
            n_rounds=2000,
            n_actions=5,
            n_positions=1,
            seed=3,
            reward_mean_by_action=[0.01, 0.02, 0.10, 0.04, 0.01],
        )
        # Peak on the best action → SNIPW should rise toward 0.10
        pol = peaked_policy(n_rounds=2000, n_actions=5, best_action=2, peak_mass=0.95)
        snipw_val = compute_snipw(bf, pol, min_propensity_clip=0.01)
        assert snipw_val > 0.05
        # Peak on the worst action → SNIPW drops to ~0.01
        pol_bad = peaked_policy(n_rounds=2000, n_actions=5, best_action=0, peak_mass=0.95)
        snipw_val_bad = compute_snipw(bf, pol_bad, min_propensity_clip=0.01)
        assert snipw_val_bad < snipw_val


class TestDegeneratePolicyIsZeroSupportHeavy:
    """Degenerate policies on actions the logger never chose saturate the gate."""

    def test_degenerate_policy_flags_zero_support_gate(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=500, n_actions=4, n_positions=1, seed=0)
        # Force every action to be NOT taken → exists some action the logger
        # never chose, but here the logger is uniform so every action gets hits.
        # Build a policy that places *negligible* probability on action 0
        # everywhere; effectively zero support.
        n_rounds = bf["n_rounds"]
        n_actions = bf["n_actions"]
        pol = np.full((n_rounds, n_actions), 0.0)
        pol[:, 1] = 1.0  # all mass on action 1
        out = evaluate_open_bandit_policy(bf, pol, min_propensity_clip=0.01)
        # zero_support_fraction = fraction of rows where pol[i, logged_action_i] == 0
        # Logged actions uniform across {0,1,2,3}; pol puts mass only on 1 → ~75% zero support
        assert out["zero_support_fraction"] > 0.5


class TestGateReportStructure:
    def test_gate_report_individual_results_are_structured(self) -> None:
        bf = generate_synthetic_bandit_feedback(n_rounds=200, n_actions=3, n_positions=1, seed=0)
        policies = {"s": uniform_policy(n_rounds=200, n_actions=3)}
        report = run_section_7_gates(
            bandit_feedback=bf,
            strategy_policies=policies,
            per_seed_ess=[150] * 5,
            per_seed_zero_support=[0.0] * 5,
            snipw_per_seed=[0.05] * 5,
            dr_per_seed=[0.05] * 5,
            n_actions=3,
            n_positions=1,
            schema=PROPENSITY_SCHEMA_JOINT,
            permutation_seed=1,
        )
        # Each gate result exposes a ``passed`` bool and a ``value`` / detail
        assert hasattr(report.null_control, "passed")
        assert hasattr(report.ess, "passed")
        assert hasattr(report.zero_support, "passed")
        assert hasattr(report.propensity_sanity, "passed")
        assert hasattr(report.dr_cross_check, "passed")

    def test_all_passed_returns_false_when_any_gate_fails(self) -> None:
        # ESS below the floor but every other gate green → all_passed() must
        # return False. Protects against a GateReport.all_passed() that
        # silently ignores one of the five fields.
        bf = generate_synthetic_bandit_feedback(n_rounds=200, n_actions=3, n_positions=1, seed=0)
        policies = {"s": uniform_policy(n_rounds=200, n_actions=3)}
        report = run_section_7_gates(
            bandit_feedback=bf,
            strategy_policies=policies,
            per_seed_ess=[10] * 5,  # median 10, floor 1000 → fails
            per_seed_zero_support=[0.0] * 5,
            snipw_per_seed=[0.05] * 5,
            dr_per_seed=[0.05] * 5,
            n_actions=3,
            n_positions=1,
            schema=PROPENSITY_SCHEMA_JOINT,
            permutation_seed=1,
        )
        assert report.ess.passed is False
        assert report.all_passed() is False
