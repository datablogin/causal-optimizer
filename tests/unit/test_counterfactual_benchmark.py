"""Unit tests for the semi-synthetic counterfactual energy benchmark.

Tests verify the data generation scenario, causal graph structure,
oracle policy optimality, treatment effect heterogeneity, and
benchmark harness correctness.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
    _treatment_effect,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def scenario() -> DemandResponseScenario:
    """Create a scenario with synthetic covariates for unit tests."""
    # Build a small synthetic covariate frame that mimics ERCOT structure
    rng = np.random.default_rng(42)
    n = 480  # 20 days of hourly data — enough for propensity correlation tests
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    days = np.repeat(np.arange(n // 24 + 1), 24)[:n]
    temps = 10.0 + 31.0 * rng.random(n)  # 10-41C range (wide for hot/cold contrast)
    humidity = 30.0 + 50.0 * rng.random(n)  # 30-80%
    base_load = 1000.0 + 200.0 * rng.random(n)

    # Create timestamps so split_time_frame can work
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temps,
            "humidity": humidity,
            "hour_of_day": hours,
            "day_of_week": days % 7,
            "is_holiday": np.zeros(n, dtype=int),
            "target_load": base_load,
            "load_lag_1h": np.roll(base_load, 1),
            "load_lag_24h": np.roll(base_load, 24),
        }
    )
    return DemandResponseScenario(covariates=df, seed=123)


# ── Test 1: Valid data generation ────────────────────────────────────


class TestScenarioGeneratesValidData:
    """Verify generated dataset has expected columns and valid values."""

    def test_has_required_columns(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        required = {
            "timestamp",
            "temperature",
            "humidity",
            "hour_of_day",
            "day_of_week",
            "is_holiday",
            "target_load",
            "demand_response_event",
            "y0",
            "y1",
            "observed_outcome",
            "true_treatment_effect",
        }
        assert required.issubset(set(data.columns))

    def test_no_nan_in_outcomes(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        assert data["y0"].notna().all()
        assert data["y1"].notna().all()
        assert data["observed_outcome"].notna().all()
        assert data["true_treatment_effect"].notna().all()

    def test_treatment_binary(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        assert set(data["demand_response_event"].unique()).issubset({0, 1})

    def test_treatment_correlated_with_temperature(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        # Compare at the SAME hour range to isolate temperature effect.
        # Use afternoon hours (14-18) where propensity is highest.
        afternoon = data[data["hour_of_day"].between(14, 18)]
        high_temp = afternoon[afternoon["temperature"] > 32]  # >32C (~90F)
        low_temp = afternoon[afternoon["temperature"] < 21]  # <21C (~70F)
        if len(high_temp) > 3 and len(low_temp) > 3:
            high_rate = high_temp["demand_response_event"].mean()
            low_rate = low_temp["demand_response_event"].mean()
            assert high_rate > low_rate, (
                f"Treatment rate at high temp ({high_rate:.2f}) should exceed "
                f"low temp ({low_rate:.2f}) during afternoon hours"
            )

    def test_treatment_correlated_with_hour(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        # Treatment should be more likely during afternoon hours (14-18)
        afternoon = data[data["hour_of_day"].between(14, 18)]
        night = data[data["hour_of_day"].isin([0, 1, 2, 3, 4, 5])]
        if len(afternoon) > 5 and len(night) > 5:
            afternoon_rate = afternoon["demand_response_event"].mean()
            night_rate = night["demand_response_event"].mean()
            assert afternoon_rate > night_rate, (
                f"Treatment rate in afternoon ({afternoon_rate:.2f}) should exceed "
                f"night ({night_rate:.2f})"
            )

    def test_observed_outcome_consistency(self, scenario: DemandResponseScenario) -> None:
        """observed_outcome should equal y0 when untreated, y1 when treated."""
        data = scenario.generate()
        treated = data["demand_response_event"] == 1
        untreated = data["demand_response_event"] == 0
        # Untreated: observed = y0
        np.testing.assert_array_almost_equal(
            data.loc[untreated, "observed_outcome"].values,
            data.loc[untreated, "y0"].values,
        )
        # Treated: observed = y1
        np.testing.assert_array_almost_equal(
            data.loc[treated, "observed_outcome"].values,
            data.loc[treated, "y1"].values,
        )


# ── Test 2: Oracle policy is optimal ────────────────────────────────


class TestOraclePolicyIsOptimal:
    """Verify oracle achieves the best possible policy value."""

    def test_oracle_beats_always_treat(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        cost = scenario.treatment_cost
        oracle_value = scenario.oracle_policy_value(data)
        # Always-treat: net benefit = mean(effect - cost) for all units
        always_treat_value = float((data["true_treatment_effect"] - cost).mean())
        assert oracle_value >= always_treat_value - 1e-9

    def test_oracle_beats_never_treat(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        oracle_value = scenario.oracle_policy_value(data)
        # Never-treat: net benefit = 0 (baseline by definition)
        never_treat_value = 0.0
        assert oracle_value >= never_treat_value - 1e-9

    def test_oracle_beats_random_policy(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        cost = scenario.treatment_cost
        oracle_value = scenario.oracle_policy_value(data)
        rng = np.random.default_rng(123)
        random_decisions = rng.integers(0, 2, size=len(data)).astype(bool)
        # Random policy net benefit: treat randomly
        effect = data["true_treatment_effect"].values
        random_value = float(np.where(random_decisions, effect - cost, 0.0).mean())
        assert oracle_value >= random_value - 1e-9


# ── Test 3: Treatment effect varies by context ───────────────────────


class TestTreatmentEffectVariesByContext:
    """Verify the structural rule: hot-afternoon effect > mild-night effect."""

    def test_hot_afternoon_effect_larger(self, scenario: DemandResponseScenario) -> None:
        data = scenario.generate()
        # Hot afternoons: temp > 30C (~86F), hour in [14, 18]
        hot_afternoon = data[(data["temperature"] > 30) & (data["hour_of_day"].between(14, 18))]
        # Mild nights: temp < 21C (~70F), hour in [0, 6]
        mild_night = data[(data["temperature"] < 21) & (data["hour_of_day"].isin(range(7)))]
        if len(hot_afternoon) > 0 and len(mild_night) > 0:
            hot_effect = hot_afternoon["true_treatment_effect"].mean()
            mild_effect = mild_night["true_treatment_effect"].mean()
            assert hot_effect > mild_effect, (
                f"Hot afternoon effect ({hot_effect:.2f}) should exceed "
                f"mild night effect ({mild_effect:.2f})"
            )

    def test_treatment_effect_is_non_negative(self, scenario: DemandResponseScenario) -> None:
        """Treatment effect (load reduction) should be >= 0."""
        data = scenario.generate()
        assert (data["true_treatment_effect"] >= -1e-9).all()


# ── Test 4: Causal graph has non-parents ─────────────────────────────


class TestCausalGraphHasNonParents:
    """Verify the graph structure: non-parents excluded from objective parents.

    Graph nodes use search space variable names so the engine's
    focus_variables logic correctly identifies ancestors of "objective".
    """

    def test_objective_parents_exclude_humidity_threshold(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("objective")
        assert "treat_humidity_threshold" not in parents

    def test_objective_parents_exclude_day_filter(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("objective")
        assert "treat_day_filter" not in parents

    def test_objective_parents_include_temp_threshold(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("objective")
        assert "treat_temp_threshold" in parents

    def test_objective_parents_include_hour_start(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("objective")
        assert "treat_hour_start" in parents

    def test_objective_parents_include_hour_end(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("objective")
        assert "treat_hour_end" in parents

    def test_graph_has_non_parent_nodes(self) -> None:
        """Noise dimensions are in the graph but not parents of objective."""
        graph = DemandResponseScenario.causal_graph()
        assert "treat_humidity_threshold" in graph.nodes
        assert "treat_day_filter" in graph.nodes
        parents = graph.parents("objective")
        non_parents = {"treat_humidity_threshold", "treat_day_filter"}
        assert non_parents.isdisjoint(parents)


# ── Test 5: Benchmark smoke ──────────────────────────────────────────


class TestBenchmarkSmoke:
    """Run with budget=3, seed=0, verify result has expected fields."""

    def test_smoke_result_has_expected_fields(self, scenario: DemandResponseScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert result.seed == 0
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)
        assert math.isfinite(result.regret)
        assert math.isfinite(result.decision_error_rate)
        assert math.isfinite(result.runtime_seconds)
        assert result.runtime_seconds > 0

    def test_smoke_regret_is_non_negative(self, scenario: DemandResponseScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert result.regret >= -1e-9, f"Regret should be non-negative, got {result.regret}"

    def test_smoke_surrogate_only(self, scenario: DemandResponseScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="surrogate_only")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "surrogate_only"
        assert math.isfinite(result.policy_value)

    def test_smoke_causal(self, scenario: DemandResponseScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="causal")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "causal"
        assert math.isfinite(result.policy_value)

    def test_invalid_strategy_raises(self, scenario: DemandResponseScenario) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            scenario.run_benchmark(budget=3, seed=0, strategy="invalid")


# ── Test 6: Reproducibility ─────────────────────────────────────────


class TestReproducibility:
    """Same seed produces same results."""

    def test_same_seed_same_data(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": 10 + 25 * rng.random(n),  # 10-35C
                "humidity": 40 + 30 * rng.random(n),
                "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
                "day_of_week": np.zeros(n, dtype=int),
                "is_holiday": np.zeros(n, dtype=int),
                "target_load": 1000 + 100 * rng.random(n),
                "load_lag_1h": 1000 + 100 * rng.random(n),
                "load_lag_24h": 1000 + 100 * rng.random(n),
            }
        )
        s1 = DemandResponseScenario(covariates=df, seed=7)
        s2 = DemandResponseScenario(covariates=df, seed=7)
        d1 = s1.generate()
        d2 = s2.generate()
        pd.testing.assert_frame_equal(d1, d2)

    def test_different_seed_different_treatment(self) -> None:
        rng = np.random.default_rng(99)
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": 10 + 25 * rng.random(n),  # 10-35C
                "humidity": 40 + 30 * rng.random(n),
                "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
                "day_of_week": np.zeros(n, dtype=int),
                "is_holiday": np.zeros(n, dtype=int),
                "target_load": 1000 + 100 * rng.random(n),
                "load_lag_1h": 1000 + 100 * rng.random(n),
                "load_lag_24h": 1000 + 100 * rng.random(n),
            }
        )
        s1 = DemandResponseScenario(covariates=df, seed=7)
        s2 = DemandResponseScenario(covariates=df, seed=42)
        d1 = s1.generate()
        d2 = s2.generate()
        # Treatment assignments should differ (same covariates, different seed)
        assert not (d1["demand_response_event"] == d2["demand_response_event"]).all()


# ── Test 7: Non-degenerate oracle (Sprint 18 repair) ─────────────


class TestOracleTreatsMeaningfulMinority:
    """Oracle should selectively treat 10-40% of rows, not 0% or 100%."""

    def test_oracle_treat_rate_in_range(self, scenario: DemandResponseScenario) -> None:
        """Oracle treats between 10% and 40% of rows."""
        data = scenario.generate()
        cost = scenario.treatment_cost
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > cost
        treat_rate = float(oracle_treat.mean())
        assert 0.10 <= treat_rate <= 0.40, f"Oracle treat rate {treat_rate:.3f} not in [0.10, 0.40]"

    def test_oracle_value_positive(self, scenario: DemandResponseScenario) -> None:
        """Oracle policy value must be strictly positive (not zero)."""
        data = scenario.generate()
        oracle_value = scenario.oracle_policy_value(data)
        assert oracle_value > 0.0, f"Oracle value should be > 0, got {oracle_value:.4f}"

    def test_never_treat_has_regret(self, scenario: DemandResponseScenario) -> None:
        """'Never treat' should have measurable regret vs oracle."""
        data = scenario.generate()
        oracle_value = scenario.oracle_policy_value(data)
        never_treat_value = 0.0  # by definition
        regret = oracle_value - never_treat_value
        assert regret > 1.0, f"Never-treat regret should be > 1.0, got {regret:.4f}"

    def test_always_treat_has_regret(self, scenario: DemandResponseScenario) -> None:
        """'Always treat' should have measurable regret vs oracle.

        Treating mild nights costs more than it saves, so always-treat
        should underperform the selective oracle policy.
        """
        data = scenario.generate()
        cost = scenario.treatment_cost
        oracle_value = scenario.oracle_policy_value(data)
        # Always-treat: net benefit = mean(effect - cost) for all rows
        always_treat_value = float((data["true_treatment_effect"] - cost).mean())
        regret = oracle_value - always_treat_value
        assert regret > 0.5, f"Always-treat regret should be > 0.5, got {regret:.4f}"


# ── Test 8: Treatment effect heterogeneity (quantitative) ────────


class TestTreatmentEffectHeterogeneity:
    """Treatment effect must vary significantly by covariate context."""

    def test_hot_afternoon_effect_much_larger_than_cool_night(self) -> None:
        """Peak hot-afternoon effect should be at least 10x cool-night effect."""
        temps_hot = np.array([35.0, 38.0, 41.0])  # 35-41C (~95-106F)
        hours_hot = np.array([15.0, 16.0, 17.0])
        hot_effects = _treatment_effect(temps_hot, hours_hot)

        temps_cool = np.array([13.0, 16.0, 18.0])  # 13-18C (~55-64F)
        hours_cool = np.array([2.0, 3.0, 4.0])
        cool_effects = _treatment_effect(temps_cool, hours_cool)

        assert hot_effects.mean() > 10.0 * cool_effects.mean(), (
            f"Hot afternoon mean effect ({hot_effects.mean():.1f}) should be "
            f">10x cool night mean effect ({cool_effects.mean():.1f})"
        )

    def test_warm_afternoon_effect_moderate(self) -> None:
        """Warm (27-32C) afternoon effect should be moderate -- above cost but below hot."""
        temps = np.array([28.0, 29.0, 31.0])  # 28-31C (~82-88F)
        hours = np.array([15.0, 16.0, 17.0])
        effects = _treatment_effect(temps, hours)
        # Moderate: should be above treatment cost (60.0) but well below hot afternoon
        assert effects.mean() > 60.0, (
            f"Warm afternoon effect ({effects.mean():.1f}) should be > treatment_cost (60.0)"
        )
        hot_effects = _treatment_effect(np.array([38.0]), np.array([16.0]))  # 38C (~100F)
        assert effects.mean() < hot_effects[0], (
            f"Warm afternoon effect ({effects.mean():.1f}) should be < "
            f"hot afternoon effect ({hot_effects[0]:.1f})"
        )


# ── Test 9: Smoke benchmark non-zero regret ──────────────────────


class TestSmokeBenchmarkNonDegenerate:
    """Budget=3 benchmark should produce non-trivial oracle and some regret."""

    def test_smoke_oracle_positive(self, scenario: DemandResponseScenario) -> None:
        """Oracle value should be positive even in the smoke test."""
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert result.oracle_value > 0.0, (
            f"Oracle value should be > 0, got {result.oracle_value:.4f}"
        )

    def test_smoke_nonzero_regret_for_random(self, scenario: DemandResponseScenario) -> None:
        """Random strategy with budget=3 should have measurable regret."""
        # With only 3 random policies, it is extremely unlikely to match oracle
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert result.regret > 0.0, (
            f"Random budget=3 should have regret > 0, got {result.regret:.4f}"
        )


# ── Test 10: Benchmark reproducibility ────────────────────────────


class TestBenchmarkReproducibility:
    """Same scenario + seed + strategy should produce identical results."""

    def test_same_seed_same_result(self, scenario: DemandResponseScenario) -> None:
        r1 = scenario.run_benchmark(budget=3, seed=42, strategy="random")
        r2 = scenario.run_benchmark(budget=3, seed=42, strategy="random")
        assert r1.policy_value == pytest.approx(r2.policy_value, abs=1e-9)
        assert r1.oracle_value == pytest.approx(r2.oracle_value, abs=1e-9)
        assert r1.regret == pytest.approx(r2.regret, abs=1e-9)
        assert r1.decision_error_rate == pytest.approx(r2.decision_error_rate, abs=1e-9)

    def test_different_seed_different_result(self, scenario: DemandResponseScenario) -> None:
        """Different seeds should generally produce different policies."""
        r1 = scenario.run_benchmark(budget=5, seed=0, strategy="random")
        r2 = scenario.run_benchmark(budget=5, seed=99, strategy="random")
        # At least the policy value or decision error should differ
        differs = (
            abs(r1.policy_value - r2.policy_value) > 1e-9
            or abs(r1.decision_error_rate - r2.decision_error_rate) > 1e-9
        )
        assert differs, "Different seeds should produce different random policies"
