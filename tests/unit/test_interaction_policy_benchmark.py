"""Unit tests for the multi-threshold interaction policy benchmark.

Tests verify the data generation scenario, causal graph structure,
oracle policy optimality, interaction-driven treatment effects, and
benchmark harness correctness.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.interaction_policy import (
    InteractionPolicyScenario,
    interaction_propensity,
    interaction_treatment_effect,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def scenario() -> InteractionPolicyScenario:
    """Create a scenario with synthetic covariates for unit tests."""
    rng = np.random.default_rng(42)
    n = 480  # 20 days of hourly data
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    days = np.repeat(np.arange(n // 24 + 1), 24)[:n]
    temps = 10.0 + 31.0 * rng.random(n)  # 10-41C
    humidity = 20.0 + 70.0 * rng.random(n)  # 20-90%
    base_load = 1000.0 + 200.0 * rng.random(n)
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
    return InteractionPolicyScenario(covariates=df, seed=123)


# ── Test 1: Valid data generation ────────────────────────────────────


class TestScenarioGeneratesValidData:
    """Verify generated dataset has expected columns and valid values."""

    def test_has_required_columns(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        required = {
            "timestamp",
            "temperature",
            "humidity",
            "hour_of_day",
            "day_of_week",
            "target_load",
            "y0",
            "y1",
            "observed_outcome",
            "true_treatment_effect",
        }
        assert required.issubset(set(data.columns))

    def test_no_nan_in_outcomes(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        assert data["y0"].notna().all()
        assert data["y1"].notna().all()
        assert data["observed_outcome"].notna().all()
        assert data["true_treatment_effect"].notna().all()

    def test_treatment_binary(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        assert set(data["treatment_event"].unique()).issubset({0, 1})

    def test_treatment_effect_is_non_negative(self, scenario: InteractionPolicyScenario) -> None:
        """Treatment effect (load reduction) should be >= 0."""
        data = scenario.generate()
        assert (data["true_treatment_effect"] >= -1e-9).all()

    def test_nan_covariates_filled(self) -> None:
        """NaN humidity/temperature rows should be filled, not propagated."""
        rng = np.random.default_rng(99)
        n = 48
        temps = 20.0 + 15.0 * rng.random(n)
        temps[0] = np.nan  # inject NaN temperature
        humidity = 40.0 + 30.0 * rng.random(n)
        humidity[1] = np.nan  # inject NaN humidity
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": temps,
                "humidity": humidity,
                "hour_of_day": np.tile(np.arange(24), 2),
                "day_of_week": np.zeros(n, dtype=int),
                "is_holiday": np.zeros(n, dtype=int),
                "target_load": 1000.0 + 100 * rng.random(n),
                "load_lag_1h": 1000.0 + 100 * rng.random(n),
                "load_lag_24h": 1000.0 + 100 * rng.random(n),
            }
        )
        scenario = InteractionPolicyScenario(covariates=df, seed=0)
        data = scenario.generate()
        # No NaN in any generated column
        assert data["true_treatment_effect"].notna().all()
        assert data["y0"].notna().all()
        assert data["y1"].notna().all()
        # Cleaned covariates should not have NaN either
        assert data["temperature"].notna().all()
        assert data["humidity"].notna().all()


# ── Test 2: Interaction-driven effects ────────────────────────────────


class TestInteractionDrivenEffects:
    """The treatment effect must depend on INTERACTIONS between covariates."""

    def test_hot_humid_afternoon_stronger_than_sum_of_marginals(self) -> None:
        """Super-additive: joint condition effect > sum of individual conditions.

        This is the key structural property: the interaction term makes the
        joint effect larger than what you would get from adding up effects
        of each condition independently.
        """
        # Joint: hot + humid + afternoon
        joint_effect = interaction_treatment_effect(
            temperature=np.array([38.0]),
            humidity=np.array([85.0]),
            hour_of_day=np.array([15.0]),
        )
        # Marginals: only one condition 'active', others at baseline
        temp_only = interaction_treatment_effect(
            temperature=np.array([38.0]),
            humidity=np.array([40.0]),  # low humidity
            hour_of_day=np.array([3.0]),  # night
        )
        humid_only = interaction_treatment_effect(
            temperature=np.array([18.0]),  # cool
            humidity=np.array([85.0]),
            hour_of_day=np.array([3.0]),  # night
        )
        hour_only = interaction_treatment_effect(
            temperature=np.array([18.0]),  # cool
            humidity=np.array([40.0]),  # low humidity
            hour_of_day=np.array([15.0]),
        )
        marginal_sum = temp_only[0] + humid_only[0] + hour_only[0]
        assert joint_effect[0] > marginal_sum, (
            f"Joint effect ({joint_effect[0]:.1f}) should exceed sum of "
            f"marginals ({marginal_sum:.1f}) -- interaction is super-additive"
        )

    def test_effect_varies_with_humidity(self) -> None:
        """At fixed temperature and hour, effect should vary with humidity."""
        temps = np.full(3, 35.0)
        hours = np.full(3, 15.0)
        humidities = np.array([25.0, 55.0, 85.0])
        effects = interaction_treatment_effect(temps, humidities, hours)
        assert effects[2] > effects[0], (
            f"Effect at high humidity ({effects[2]:.1f}) should exceed "
            f"low humidity ({effects[0]:.1f})"
        )

    def test_cool_dry_night_effect_near_zero(self) -> None:
        """Cool + dry + night should produce near-zero effect."""
        effect = interaction_treatment_effect(
            temperature=np.array([12.0]),
            humidity=np.array([25.0]),
            hour_of_day=np.array([3.0]),
        )
        assert effect[0] < 10.0, f"Cool dry night effect ({effect[0]:.1f}) should be near zero"


# ── Test 3: Oracle policy is optimal ────────────────────────────────


class TestOraclePolicyIsOptimal:
    """Verify oracle achieves the best possible policy value.

    Tests compare oracle value against policies that exercise real
    evaluation logic, not just mathematical identities.
    """

    def test_oracle_beats_good_parametric_policy(
        self, scenario: InteractionPolicyScenario
    ) -> None:
        """Oracle should beat a hand-tuned 'good' policy targeting hot-humid afternoons."""
        from causal_optimizer.benchmarks.interaction_policy import evaluate_interaction_policy

        data = scenario.generate()
        oracle_value = scenario.oracle_policy_value(data)
        good_params = {
            "policy_temp_threshold": 28.0,
            "policy_humidity_threshold": 50.0,
            "policy_hour_start": 12,
            "policy_hour_end": 20,
            "noise_wind_speed": 5.0,
            "noise_pressure": 1013.0,
            "noise_cloud_cover": 50.0,
        }
        good_value, _ = evaluate_interaction_policy(
            data, good_params, scenario.treatment_cost
        )
        assert oracle_value > good_value, (
            f"Oracle ({oracle_value:.2f}) should beat good parametric "
            f"policy ({good_value:.2f})"
        )

    def test_oracle_beats_random_policy(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        cost = scenario.treatment_cost
        oracle_value = scenario.oracle_policy_value(data)
        rng = np.random.default_rng(123)
        random_decisions = rng.integers(0, 2, size=len(data)).astype(bool)
        effect = data["true_treatment_effect"].values
        random_value = float(np.where(random_decisions, effect - cost, 0.0).mean())
        assert oracle_value >= random_value - 1e-9


# ── Test 4: Non-degenerate oracle ──────────────────────────────────


class TestOracleTreatsMeaningfulMinority:
    """Oracle should selectively treat 20-40% of rows, not 0% or 100%."""

    def test_oracle_treat_rate_in_range(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        cost = scenario.treatment_cost
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > cost
        treat_rate = float(oracle_treat.mean())
        assert 0.15 <= treat_rate <= 0.45, f"Oracle treat rate {treat_rate:.3f} not in [0.15, 0.45]"

    def test_oracle_value_positive(self, scenario: InteractionPolicyScenario) -> None:
        data = scenario.generate()
        oracle_value = scenario.oracle_policy_value(data)
        assert oracle_value > 0.0, f"Oracle value should be > 0, got {oracle_value:.4f}"


# ── Test 5: Causal graph structure ─────────────────────────────────


class TestCausalGraphStructure:
    """Verify causal graph: real parents connected, noise dims disconnected."""

    def test_objective_parents_include_real_vars(self) -> None:
        graph = InteractionPolicyScenario.causal_graph()
        parents = graph.parents("objective")
        assert "policy_temp_threshold" in parents
        assert "policy_humidity_threshold" in parents
        assert "policy_hour_start" in parents
        assert "policy_hour_end" in parents

    def test_objective_parents_exclude_noise(self) -> None:
        graph = InteractionPolicyScenario.causal_graph()
        parents = graph.parents("objective")
        assert "noise_wind_speed" not in parents
        assert "noise_pressure" not in parents
        assert "noise_cloud_cover" not in parents

    def test_noise_vars_in_graph(self) -> None:
        """Noise dimensions exist in graph but are not parents of objective."""
        graph = InteractionPolicyScenario.causal_graph()
        assert "noise_wind_speed" in graph.nodes
        assert "noise_pressure" in graph.nodes
        assert "noise_cloud_cover" in graph.nodes


# ── Test 6: Search space ──────────────────────────────────────────


class TestSearchSpace:
    """Verify search space has the right variables."""

    def test_search_space_has_seven_vars(self) -> None:
        space = InteractionPolicyScenario.search_space()
        assert len(space.variables) == 7

    def test_all_vars_are_continuous_or_integer(self) -> None:
        """No categorical variables -- different challenge from DemandResponse."""
        from causal_optimizer.types import VariableType

        space = InteractionPolicyScenario.search_space()
        for var in space.variables:
            assert var.variable_type in (
                VariableType.CONTINUOUS,
                VariableType.INTEGER,
            ), f"Var {var.name} is {var.variable_type}, expected CONTINUOUS or INTEGER"


# ── Test 7: Benchmark smoke ────────────────────────────────────────


class TestBenchmarkSmoke:
    """Run with budget=3, seed=0, verify result has expected fields."""

    def test_smoke_random(self, scenario: InteractionPolicyScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert result.strategy == "random"
        assert result.budget == 3
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)
        assert math.isfinite(result.regret)
        assert result.regret >= -1e-9

    def test_smoke_surrogate_only(self, scenario: InteractionPolicyScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="surrogate_only")
        assert result.strategy == "surrogate_only"
        assert math.isfinite(result.policy_value)

    def test_smoke_causal(self, scenario: InteractionPolicyScenario) -> None:
        result = scenario.run_benchmark(budget=3, seed=0, strategy="causal")
        assert result.strategy == "causal"
        assert math.isfinite(result.policy_value)


# ── Test 8: Reproducibility ────────────────────────────────────────


class TestReproducibility:
    """Same seed produces same results."""

    def test_same_seed_same_data(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": 10 + 25 * rng.random(n),
                "humidity": 20 + 60 * rng.random(n),
                "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
                "day_of_week": np.zeros(n, dtype=int),
                "is_holiday": np.zeros(n, dtype=int),
                "target_load": 1000 + 100 * rng.random(n),
                "load_lag_1h": 1000 + 100 * rng.random(n),
                "load_lag_24h": 1000 + 100 * rng.random(n),
            }
        )
        s1 = InteractionPolicyScenario(covariates=df, seed=7)
        s2 = InteractionPolicyScenario(covariates=df, seed=7)
        d1 = s1.generate()
        d2 = s2.generate()
        pd.testing.assert_frame_equal(d1, d2)

    def test_different_seed_different_treatment(self) -> None:
        rng = np.random.default_rng(99)
        n = 100
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
                "temperature": 10 + 25 * rng.random(n),
                "humidity": 20 + 60 * rng.random(n),
                "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
                "day_of_week": np.zeros(n, dtype=int),
                "is_holiday": np.zeros(n, dtype=int),
                "target_load": 1000 + 100 * rng.random(n),
                "load_lag_1h": 1000 + 100 * rng.random(n),
                "load_lag_24h": 1000 + 100 * rng.random(n),
            }
        )
        s1 = InteractionPolicyScenario(covariates=df, seed=7)
        s2 = InteractionPolicyScenario(covariates=df, seed=42)
        d1 = s1.generate()
        d2 = s2.generate()
        assert not (d1["treatment_event"] == d2["treatment_event"]).all()


# ── Test 9: Policy evaluation uses interactions ────────────────────


class TestPolicyEvaluationUsesInteractions:
    """Good policy parameters should produce better value than bad ones."""

    def test_good_params_beat_bad_params(self, scenario: InteractionPolicyScenario) -> None:
        """Policy targeting hot-humid afternoons should beat one targeting cool-dry nights."""
        from causal_optimizer.benchmarks.interaction_policy import evaluate_interaction_policy

        data = scenario.generate()
        n = len(data)
        opt_end = int(n * 0.8)
        val_data = data.iloc[:opt_end].reset_index(drop=True)

        # Good policy: target hot, humid afternoons
        good_params = {
            "policy_temp_threshold": 28.0,
            "policy_humidity_threshold": 50.0,
            "policy_hour_start": 12,
            "policy_hour_end": 20,
            "noise_wind_speed": 5.0,
            "noise_pressure": 1013.0,
            "noise_cloud_cover": 50.0,
        }
        # Bad policy: target cool, dry nights
        bad_params = {
            "policy_temp_threshold": 5.0,
            "policy_humidity_threshold": 10.0,
            "policy_hour_start": 0,
            "policy_hour_end": 5,
            "noise_wind_speed": 5.0,
            "noise_pressure": 1013.0,
            "noise_cloud_cover": 50.0,
        }

        good_value, _ = evaluate_interaction_policy(val_data, good_params, scenario.treatment_cost)
        bad_value, _ = evaluate_interaction_policy(val_data, bad_params, scenario.treatment_cost)
        assert good_value > bad_value, (
            f"Good policy value ({good_value:.2f}) should beat bad policy value ({bad_value:.2f})"
        )


# ── Test 10: Propensity function ────────────────────────────────────


class TestInteractionPropensity:
    """Verify interaction propensity bounds and directional behavior."""

    def test_propensity_within_bounds(self) -> None:
        """Propensity should be clipped to [0.05, 0.85]."""
        temps = np.array([-10.0, 0.0, 20.0, 35.0, 45.0])
        humidities = np.array([10.0, 30.0, 50.0, 80.0, 100.0])
        hours = np.array([0.0, 6.0, 12.0, 18.0, 23.0])
        prop = interaction_propensity(temps, humidities, hours)
        assert prop.min() >= 0.05, f"Propensity min ({prop.min():.4f}) should be >= 0.05"
        assert prop.max() <= 0.85, f"Propensity max ({prop.max():.4f}) should be <= 0.85"

    def test_propensity_increases_with_temperature(self) -> None:
        """Higher temperature -> higher propensity, holding hour and humidity constant."""
        hours = np.full(3, 15.0)
        humidities = np.full(3, 60.0)
        temps = np.array([10.0, 25.0, 40.0])
        prop = interaction_propensity(temps, humidities, hours)
        assert prop[2] > prop[0], (
            f"Propensity at 40C ({prop[2]:.3f}) should exceed 10C ({prop[0]:.3f})"
        )

    def test_propensity_afternoon_higher_than_night(self) -> None:
        """Afternoon propensity should exceed night propensity."""
        temps = np.full(2, 30.0)
        humidities = np.full(2, 60.0)
        hours = np.array([15.0, 3.0])
        prop = interaction_propensity(temps, humidities, hours)
        assert prop[0] > prop[1], (
            f"Afternoon propensity ({prop[0]:.3f}) should exceed night ({prop[1]:.3f})"
        )
