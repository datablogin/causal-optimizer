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
    temps = 50.0 + 55.0 * rng.random(n)  # 50-105F range (wide for hot/cold contrast)
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

    def test_treatment_correlated_with_temperature(
        self, scenario: DemandResponseScenario
    ) -> None:
        data = scenario.generate()
        # Compare at the SAME hour range to isolate temperature effect.
        # Use afternoon hours (14-18) where propensity is highest.
        afternoon = data[data["hour_of_day"].between(14, 18)]
        high_temp = afternoon[afternoon["temperature"] > 90]
        low_temp = afternoon[afternoon["temperature"] < 70]
        if len(high_temp) > 3 and len(low_temp) > 3:
            high_rate = high_temp["demand_response_event"].mean()
            low_rate = low_temp["demand_response_event"].mean()
            assert high_rate > low_rate, (
                f"Treatment rate at high temp ({high_rate:.2f}) should exceed "
                f"low temp ({low_rate:.2f}) during afternoon hours"
            )

    def test_treatment_correlated_with_hour(
        self, scenario: DemandResponseScenario
    ) -> None:
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
        # Hot afternoons: temp > 85, hour in [14, 18]
        hot_afternoon = data[
            (data["temperature"] > 85) & (data["hour_of_day"].between(14, 18))
        ]
        # Mild nights: temp < 70, hour in [0, 6]
        mild_night = data[
            (data["temperature"] < 70) & (data["hour_of_day"].isin(range(7)))
        ]
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
    """Verify the graph structure: non-parents excluded from load_reduction parents."""

    def test_load_reduction_parents_exclude_humidity(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("load_reduction")
        assert "humidity" not in parents

    def test_load_reduction_parents_exclude_day_of_week(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("load_reduction")
        assert "day_of_week" not in parents

    def test_load_reduction_parents_include_temperature(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("load_reduction")
        assert "temperature" in parents

    def test_load_reduction_parents_include_demand_response_event(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("load_reduction")
        assert "demand_response_event" in parents

    def test_load_reduction_parents_include_hour_of_day(self) -> None:
        graph = DemandResponseScenario.causal_graph()
        parents = graph.parents("load_reduction")
        assert "hour_of_day" in parents

    def test_graph_has_non_parent_nodes(self) -> None:
        """humidity and day_of_week are in the graph but not parents of load_reduction."""
        graph = DemandResponseScenario.causal_graph()
        assert "humidity" in graph.nodes
        assert "day_of_week" in graph.nodes
        parents = graph.parents("load_reduction")
        non_parents = {"humidity", "day_of_week"}
        assert non_parents.isdisjoint(parents)


# ── Test 5: Benchmark smoke ──────────────────────────────────────────


class TestBenchmarkSmoke:
    """Run with budget=3, seed=0, verify result has expected fields."""

    def test_smoke_result_has_expected_fields(
        self, scenario: DemandResponseScenario
    ) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert result.seed == 0
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)
        assert math.isfinite(result.regret)
        assert math.isfinite(result.treatment_effect_mae)
        assert math.isfinite(result.runtime_seconds)
        assert result.runtime_seconds > 0

    def test_smoke_regret_is_non_negative(
        self, scenario: DemandResponseScenario
    ) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert result.regret >= -1e-9, f"Regret should be non-negative, got {result.regret}"


# ── Test 6: Reproducibility ─────────────────────────────────────────


class TestReproducibility:
    """Same seed produces same results."""

    def test_same_seed_same_data(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": 60 + 30 * rng.random(n),
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
                "temperature": 60 + 30 * rng.random(n),
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
