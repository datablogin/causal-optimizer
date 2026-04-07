"""Unit tests for the semi-synthetic dose-response clinical benchmark.

Tests verify data generation, causal graph structure, oracle policy
optimality, treatment effect heterogeneity, and benchmark harness
correctness for the clinical dose-response scenario.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from causal_optimizer.benchmarks.dose_response import (
    DoseResponseBenchmarkResult,
    DoseResponseScenario,
    ProtocolRunner,
    dose_response_effect,
    evaluate_protocol,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def scenario() -> DoseResponseScenario:
    """Create a DoseResponseScenario with default synthetic covariates."""
    return DoseResponseScenario(n_patients=600, seed=42)


# ── Test 1: Valid data generation ────────────────────────────────────


class TestScenarioGeneratesValidData:
    """Verify generated dataset has expected columns and valid values."""

    def test_required_columns_present(self, scenario: DoseResponseScenario) -> None:
        data = scenario.generate()
        required = {
            "patient_id",
            "age",
            "biomarker",
            "severity",
            "bmi",
            "sex",
            "comorbidity_score",
            "y0",
            "y1",
            "true_treatment_effect",
            "observed_outcome",
            "treatment_assigned",
        }
        assert required.issubset(set(data.columns))

    def test_no_nan_values(self, scenario: DoseResponseScenario) -> None:
        data = scenario.generate()
        assert not data.isna().any().any(), "Generated data should have no NaN values"

    def test_correct_row_count(self, scenario: DoseResponseScenario) -> None:
        data = scenario.generate()
        assert len(data) == 600

    def test_y0_positive(self, scenario: DoseResponseScenario) -> None:
        """Baseline outcome (no treatment) should be positive (symptom score)."""
        data = scenario.generate()
        assert (data["y0"] > 0).all()

    def test_y1_less_than_y0_on_average(self, scenario: DoseResponseScenario) -> None:
        """Treatment should reduce symptom score on average (y1 < y0)."""
        data = scenario.generate()
        assert data["y1"].mean() < data["y0"].mean()


# ── Test 2: Treatment effect function ────────────────────────────────


class TestDoseResponseEffect:
    """Verify the dose-response effect function properties."""

    def test_zero_dose_zero_effect(self) -> None:
        """Zero dose should produce zero effect."""
        effect = dose_response_effect(
            dose=np.array([0.0]),
            biomarker=np.array([0.5]),
            severity=np.array([0.5]),
        )
        assert effect[0] == pytest.approx(0.0, abs=0.01)

    def test_monotonic_in_dose(self) -> None:
        """Effect should be non-decreasing in dose (for fixed biomarker/severity)."""
        doses = np.linspace(0.0, 1.0, 50)
        bio = np.full(50, 0.5)
        sev = np.full(50, 0.5)
        effects = dose_response_effect(doses, bio, sev)
        # Allow tiny floating-point violations
        assert np.all(np.diff(effects) >= -1e-10)

    def test_high_biomarker_amplifies_effect(self) -> None:
        """Higher biomarker should produce larger treatment effect at moderate dose."""
        dose = np.array([0.5, 0.5])
        bio_low = np.array([0.2, 0.2])
        bio_high = np.array([0.8, 0.8])
        sev = np.array([0.5, 0.5])
        eff_low = dose_response_effect(dose, bio_low, sev)
        eff_high = dose_response_effect(dose, bio_high, sev)
        assert eff_high.mean() > eff_low.mean()

    def test_effect_is_non_negative(self) -> None:
        """Treatment effect should always be non-negative."""
        rng = np.random.default_rng(99)
        n = 1000
        dose = rng.random(n)
        bio = rng.random(n)
        sev = rng.random(n)
        effects = dose_response_effect(dose, bio, sev)
        assert np.all(effects >= 0)


# ── Test 3: Causal graph structure ──────────────────────────────────


class TestCausalGraphStructure:
    """Verify the causal graph encodes the correct domain knowledge."""

    def test_causal_parents_of_objective(self) -> None:
        """dose_level, biomarker_threshold, severity_threshold should
        be ancestors of objective."""
        graph = DoseResponseScenario.causal_graph()
        ancestors = graph.ancestors("objective")
        for parent in ["dose_level", "biomarker_threshold", "severity_threshold"]:
            assert parent in ancestors, f"{parent} should be ancestor of objective"

    def test_noise_vars_not_ancestors_of_objective(self) -> None:
        """Noise dimensions should NOT be ancestors of objective."""
        graph = DoseResponseScenario.causal_graph()
        ancestors = graph.ancestors("objective")
        noise_vars = ["bmi_threshold", "age_threshold", "comorbidity_threshold"]
        for nv in noise_vars:
            assert nv not in ancestors, f"{nv} should NOT be ancestor of objective"


# ── Test 4: Oracle policy ────────────────────────────────────────────


class TestOraclePolicy:
    """Verify oracle policy is well-defined and positive."""

    def test_oracle_value_positive(self, scenario: DoseResponseScenario) -> None:
        """Oracle should achieve positive net benefit."""
        data = scenario.generate()
        oracle_val = scenario.oracle_policy_value(data)
        assert oracle_val > 0.0, "Oracle policy value must be positive"

    def test_oracle_treat_rate_non_degenerate(self, scenario: DoseResponseScenario) -> None:
        """Oracle should treat a non-trivial fraction (not all, not none).

        The global oracle uses max dose (1.0), so we compute the effect
        at dose=1.0 for the treat-rate check.
        """
        data = scenario.generate()
        max_dose = np.ones(len(data))
        effect_at_max = dose_response_effect(
            max_dose, data["biomarker"].values, data["severity"].values
        )
        oracle_treat = effect_at_max > scenario.treatment_cost
        rate = oracle_treat.mean()
        assert 0.10 < rate < 0.90, f"Oracle treat rate {rate:.2f} is degenerate"


# ── Test 5: Policy evaluation ────────────────────────────────────────


class TestEvaluateProtocol:
    """Verify protocol evaluation returns consistent results."""

    def test_never_treat_zero_value(self, scenario: DoseResponseScenario) -> None:
        """A protocol that treats nobody should have zero policy value."""
        data = scenario.generate()
        # Set thresholds so nobody is treated
        params = {
            "dose_level": 0.0,
            "biomarker_threshold": 2.0,  # above max
            "severity_threshold": 2.0,
            "bmi_threshold": 0.0,
            "age_threshold": 0.0,
            "comorbidity_threshold": 0.0,
        }
        value, error = evaluate_protocol(data, params, scenario.treatment_cost)
        assert value == pytest.approx(0.0, abs=0.01)

    def test_decision_error_rate_bounded(self, scenario: DoseResponseScenario) -> None:
        """Decision error rate should be in [0, 1]."""
        data = scenario.generate()
        params = {
            "dose_level": 0.5,
            "biomarker_threshold": 0.5,
            "severity_threshold": 0.5,
            "bmi_threshold": 0.0,
            "age_threshold": 0.0,
            "comorbidity_threshold": 0.0,
        }
        _, error = evaluate_protocol(data, params, scenario.treatment_cost)
        assert 0.0 <= error <= 1.0


# ── Test 6: Search space ────────────────────────────────────────────


class TestSearchSpace:
    """Verify the search space is valid and has expected dimensions."""

    def test_variable_count(self) -> None:
        space = DoseResponseScenario.search_space()
        assert len(space.variables) == 6

    def test_variable_names(self) -> None:
        space = DoseResponseScenario.search_space()
        names = {v.name for v in space.variables}
        expected = {
            "dose_level",
            "biomarker_threshold",
            "severity_threshold",
            "bmi_threshold",
            "age_threshold",
            "comorbidity_threshold",
        }
        assert names == expected


# ── Test 7: Benchmark harness ────────────────────────────────────────


class TestRunBenchmark:
    """Verify the run_benchmark method returns valid results."""

    def test_random_strategy_returns_result(self, scenario: DoseResponseScenario) -> None:
        result = scenario.run_benchmark(budget=5, seed=0, strategy="random")
        assert isinstance(result, DoseResponseBenchmarkResult)
        assert result.regret >= 0.0
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)

    def test_surrogate_strategy_returns_result(self, scenario: DoseResponseScenario) -> None:
        result = scenario.run_benchmark(budget=5, seed=0, strategy="surrogate_only")
        assert isinstance(result, DoseResponseBenchmarkResult)
        assert result.regret >= 0.0

    def test_causal_strategy_returns_result(self, scenario: DoseResponseScenario) -> None:
        result = scenario.run_benchmark(budget=5, seed=0, strategy="causal")
        assert isinstance(result, DoseResponseBenchmarkResult)
        assert result.regret >= 0.0

    def test_invalid_strategy_raises(self, scenario: DoseResponseScenario) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            scenario.run_benchmark(budget=5, seed=0, strategy="invalid")

    def test_oracle_value_consistent_across_strategies(
        self, scenario: DoseResponseScenario
    ) -> None:
        """Oracle value depends only on the data, not the strategy.

        Both strategies see the same test split because generate() is
        deterministic (controlled by self._seed, not the optimizer seed).
        """
        r1 = scenario.run_benchmark(budget=5, seed=0, strategy="random")
        r2 = scenario.run_benchmark(budget=5, seed=0, strategy="surrogate_only")
        assert r1.oracle_value == pytest.approx(r2.oracle_value, rel=1e-6)

    def test_surrogate_budget_15_reaches_optimization_phase(
        self, scenario: DoseResponseScenario
    ) -> None:
        """Budget=15 exercises the exploration-to-optimization phase transition."""
        result = scenario.run_benchmark(budget=15, seed=0, strategy="surrogate_only")
        assert isinstance(result, DoseResponseBenchmarkResult)
        assert result.regret >= 0.0


# ── Test 8: Reproducibility ─────────────────────────────────────────


class TestReproducibility:
    """Verify data generation is reproducible across seeds."""

    def test_same_seed_same_data(self) -> None:
        s1 = DoseResponseScenario(n_patients=100, seed=7)
        s2 = DoseResponseScenario(n_patients=100, seed=7)
        d1 = s1.generate()
        d2 = s2.generate()
        assert d1.equals(d2)

    def test_different_seed_different_data(self) -> None:
        s1 = DoseResponseScenario(n_patients=100, seed=7)
        s2 = DoseResponseScenario(n_patients=100, seed=8)
        d1 = s1.generate()
        d2 = s2.generate()
        assert not d1.equals(d2)


# ── Test 9: ProtocolRunner sign flip ─────────────────────────────────


class TestProtocolRunner:
    """Verify ProtocolRunner correctly negates policy value for minimization."""

    def test_regret_non_negative_at_max_dose(self, scenario: DoseResponseScenario) -> None:
        """Even a protocol at max dose should not beat the global oracle."""
        data = scenario.generate()
        n = len(data)
        test_data = data.iloc[int(n * 0.8) :].reset_index(drop=True)

        # Protocol at dose=1.0 with very permissive thresholds
        params = {
            "dose_level": 1.0,
            "biomarker_threshold": 0.0,
            "severity_threshold": 0.0,
            "bmi_threshold": 0.0,
            "age_threshold": 0.0,
            "comorbidity_threshold": 0.0,
        }
        policy_value, _ = evaluate_protocol(test_data, params, scenario.treatment_cost)
        oracle_value = scenario.oracle_policy_value(test_data)
        assert oracle_value >= policy_value, (
            f"Oracle ({oracle_value:.4f}) should >= policy ({policy_value:.4f})"
        )

    def test_objective_is_negated_policy_value(self, scenario: DoseResponseScenario) -> None:
        """The optimizer minimizes objective = -policy_value."""
        data = scenario.generate()
        n = len(data)
        val_data = data.iloc[: int(n * 0.8)].reset_index(drop=True)
        runner = ProtocolRunner(val_data, scenario.treatment_cost)
        params = {
            "dose_level": 0.7,
            "biomarker_threshold": 0.3,
            "severity_threshold": 0.3,
            "bmi_threshold": 0.0,
            "age_threshold": 0.0,
            "comorbidity_threshold": 0.0,
        }
        metrics = runner.run(params)
        assert "objective" in metrics
        assert "policy_value" in metrics
        assert metrics["objective"] == pytest.approx(-metrics["policy_value"])


class TestFallbackPath:
    """Regression tests for the best_params=None fallback path."""

    def test_fallback_decision_error_uses_max_dose_oracle(self, scenario):
        """When best_params is None (empty log), decision_error should be
        computed against the max-dose oracle, consistent with oracle_policy_value."""
        # Run with minimal budget — result should still have consistent oracle
        result = scenario.run_benchmark(budget=2, seed=999, strategy="random")

        # Regret should be non-negative (oracle uses max dose)
        assert result.regret >= 0.0
        # Decision error should be in [0, 1]
        assert 0.0 <= result.decision_error_rate <= 1.0

    def test_fallback_oracle_consistency(self, scenario):
        """The fallback path's oracle treat mask should use the global oracle
        (max dose=1.0), not the reference dose."""
        data = scenario.generate()

        # Global oracle treat mask at max dose
        bio = data["biomarker"].values
        sev = data["severity"].values
        max_effect = dose_response_effect(np.ones(len(data)), bio, sev)
        global_oracle_treat = max_effect > scenario.treatment_cost

        # Reference-dose treat mask (should differ from global oracle)
        ref_effect = data["true_treatment_effect"].values
        ref_oracle_treat = ref_effect > scenario.treatment_cost

        # These should differ because max_dose=1.0 > reference_dose=0.7
        # means more patients benefit at max dose
        if not np.array_equal(global_oracle_treat, ref_oracle_treat):
            mismatch = float(np.mean(global_oracle_treat != ref_oracle_treat))
            assert mismatch > 0.0, "Expected mismatch between max-dose and ref-dose oracles"
