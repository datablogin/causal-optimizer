"""Unit tests for harder counterfactual benchmark variants.

Tests cover two new variants:
1. HighNoiseDemandResponse — 10+ nuisance dimensions with zero causal effect
2. ConfoundedDemandResponse — hidden confounder creating Simpson's paradox

Each variant must produce a non-degenerate oracle (10-40% treat rate),
maintain the same treatment-effect function for the 3 true causal parents,
and expose the structural advantage that causal graph knowledge provides.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
)
from causal_optimizer.benchmarks.counterfactual_variants import (
    ConfoundedDemandResponse,
    HighNoiseDemandResponse,
)

# ── Shared fixture: synthetic covariates ────────────────────────────


@pytest.fixture()
def covariates() -> pd.DataFrame:
    """Small synthetic covariate frame mimicking ERCOT structure."""
    rng = np.random.default_rng(42)
    n = 480  # 20 days of hourly data
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    days = np.repeat(np.arange(n // 24 + 1), 24)[:n]
    temps = 10.0 + 31.0 * rng.random(n)  # 10-41C
    humidity = 30.0 + 50.0 * rng.random(n)
    base_load = 1000.0 + 200.0 * rng.random(n)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")

    return pd.DataFrame(
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


@pytest.fixture()
def high_noise_scenario(covariates: pd.DataFrame) -> HighNoiseDemandResponse:
    """High-noise variant with synthetic covariates."""
    return HighNoiseDemandResponse(covariates=covariates, seed=123)


@pytest.fixture()
def confounded_scenario(covariates: pd.DataFrame) -> ConfoundedDemandResponse:
    """Confounded variant with synthetic covariates."""
    return ConfoundedDemandResponse(covariates=covariates, seed=123)


# ── Test 1: High-noise oracle is non-trivial (10-40% treat rate) ────


class TestHighNoiseOracleNontrivial:
    """Oracle treat rate must be in [10%, 40%] for the high-noise variant."""

    def test_high_noise_oracle_nontrivial(
        self, high_noise_scenario: HighNoiseDemandResponse
    ) -> None:
        data = high_noise_scenario.generate()
        cost = high_noise_scenario.treatment_cost
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > cost
        treat_rate = float(oracle_treat.mean())
        assert 0.10 <= treat_rate <= 0.40, (
            f"High-noise oracle treat rate {treat_rate:.3f} not in [0.10, 0.40]"
        )


# ── Test 2: Nuisance variables have zero effect ────────────────────


class TestHighNoiseNuisanceVarsNoEffect:
    """Permuting nuisance variables must not change oracle policy value."""

    def test_high_noise_nuisance_vars_no_effect(
        self, high_noise_scenario: HighNoiseDemandResponse
    ) -> None:
        data = high_noise_scenario.generate()
        oracle_value_original = high_noise_scenario.oracle_policy_value(data)

        # Permute all nuisance columns
        rng = np.random.default_rng(999)
        data_permuted = data.copy()
        nuisance_cols = [c for c in data.columns if c.startswith("noise_var_")]
        assert len(nuisance_cols) >= 10, (
            f"Expected at least 10 nuisance columns, found {len(nuisance_cols)}"
        )
        for col in nuisance_cols:
            data_permuted[col] = rng.permutation(data_permuted[col].values)

        oracle_value_permuted = high_noise_scenario.oracle_policy_value(data_permuted)

        assert oracle_value_original == pytest.approx(oracle_value_permuted, abs=1e-9), (
            f"Oracle changed after nuisance permutation: "
            f"{oracle_value_original:.4f} vs {oracle_value_permuted:.4f}"
        )


# ── Test 3: Search space has 13+ dimensions ────────────────────────


class TestHighNoiseSearchSpaceDimensions:
    """High-noise search space must have at least 13 dimensions."""

    def test_high_noise_search_space_dimensions(self) -> None:
        space = HighNoiseDemandResponse.search_space()
        assert space.dimensionality >= 13, (
            f"High-noise search space has {space.dimensionality} dimensions, expected >= 13"
        )


# ── Test 4: Causal graph excludes nuisance variables ──────────────


class TestHighNoiseGraphExcludesNuisance:
    """Nuisance vars must NOT be ancestors of objective in the causal graph."""

    def test_high_noise_graph_excludes_nuisance(self) -> None:
        graph = HighNoiseDemandResponse.causal_graph()
        ancestors = graph.ancestors("objective")

        # The 3 true causal parents must be ancestors
        assert "treat_temp_threshold" in ancestors
        assert "treat_hour_start" in ancestors
        assert "treat_hour_end" in ancestors

        # No nuisance variable should be an ancestor of objective
        nuisance_vars = [n for n in graph.nodes if n.startswith("noise_var_")]
        for nv in nuisance_vars:
            assert nv not in ancestors, (
                f"Nuisance variable {nv!r} is an ancestor of objective in the causal graph"
            )


# ── Test 5: Confounded oracle differs from naive best-predictor ───


class TestConfoundedOracleDiffersFromNaive:
    """Oracle policy (true causal effect) must differ from naive predictor.

    The naive predictor observes the biased outcome surface (where the
    confounder inflates the apparent treatment benefit) and recommends
    treating more often than the oracle.
    """

    def test_confounded_oracle_differs_from_naive(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        data = confounded_scenario.generate()
        cost = confounded_scenario.treatment_cost

        # Oracle: treat where true treatment effect > cost
        true_effect = data["true_treatment_effect"].values
        oracle_treat = true_effect > cost

        # Naive: treat where observed outcome improvement suggests benefit.
        # The naive estimator estimates effect from observed data without
        # adjusting for confounding. Use confounded_scenario's naive policy.
        naive_treat = confounded_scenario.naive_policy(data, cost)

        # The two policies must disagree on a meaningful fraction of rows
        disagreement = float(np.mean(oracle_treat != naive_treat))
        assert disagreement > 0.05, (
            f"Oracle and naive policies agree too closely "
            f"(disagreement rate = {disagreement:.3f}, need > 0.05)"
        )


# ── Test 6: Confounder creates bias ───────────────────────────────


class TestConfoundedConfounderCreatesBias:
    """Naive treatment effect estimate must be biased relative to true effect."""

    def test_confounded_confounder_creates_bias(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        data = confounded_scenario.generate()

        # True average treatment effect
        true_ate = float(data["true_treatment_effect"].mean())

        # Naive ATE: difference in means between treated and untreated
        # observed outcomes (ignoring confounding)
        treated = data[data["demand_response_event"] == 1]
        untreated = data[data["demand_response_event"] == 0]
        assert len(treated) > 10, "Too few treated units for bias test"
        assert len(untreated) > 10, "Too few untreated units for bias test"
        naive_ate = float(untreated["observed_outcome"].mean() - treated["observed_outcome"].mean())

        # The naive estimate should be biased (significantly differ from true ATE)
        # The confounder inflates the naive estimate because treated units
        # have higher load (due to grid stress), making the apparent reduction larger.
        bias = abs(naive_ate - true_ate)
        relative_bias = bias / max(abs(true_ate), 1.0)
        assert relative_bias > 0.10, (
            f"Naive ATE ({naive_ate:.2f}) too close to true ATE ({true_ate:.2f}); "
            f"relative bias = {relative_bias:.3f}, need > 0.10"
        )


# ── Test 7: Confounded oracle treat rate 10-40% ──────────────────


class TestConfoundedOracleNontrivial:
    """Confounded variant must have oracle treat rate in [10%, 40%]."""

    def test_confounded_oracle_nontrivial(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        data = confounded_scenario.generate()
        cost = confounded_scenario.treatment_cost
        effect = data["true_treatment_effect"].values
        oracle_treat = effect > cost
        treat_rate = float(oracle_treat.mean())
        assert 0.10 <= treat_rate <= 0.40, (
            f"Confounded oracle treat rate {treat_rate:.3f} not in [0.10, 0.40]"
        )


# ── Test 8: Confounded graph has bidirected edge ──────────────────


class TestConfoundedGraphHasBidirected:
    """Causal graph must contain at least one bidirected edge for the confounder."""

    def test_confounded_graph_has_bidirected(self) -> None:
        graph = ConfoundedDemandResponse.causal_graph()
        assert len(graph.bidirected_edges) >= 1, (
            f"Confounded graph has {len(graph.bidirected_edges)} bidirected edges, "
            "expected at least 1"
        )
        assert graph.has_confounders, "Graph should report has_confounders=True"


# ── Test 9: Smoke high-noise (budget=3, no crash) ────────────────


class TestSmokeHighNoise:
    """Run high-noise variant with budget=3, seed=0 — no crashes."""

    def test_smoke_high_noise(self, high_noise_scenario: HighNoiseDemandResponse) -> None:
        result = high_noise_scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)
        assert math.isfinite(result.regret)
        assert result.regret >= -1e-9

    def test_smoke_high_noise_surrogate(self, high_noise_scenario: HighNoiseDemandResponse) -> None:
        result = high_noise_scenario.run_benchmark(budget=3, seed=0, strategy="surrogate_only")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert math.isfinite(result.policy_value)

    def test_smoke_high_noise_causal(self, high_noise_scenario: HighNoiseDemandResponse) -> None:
        result = high_noise_scenario.run_benchmark(budget=3, seed=0, strategy="causal")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert math.isfinite(result.policy_value)


# ── Test 10: Smoke confounded (budget=3, no crash) ───────────────


class TestSmokeConfounded:
    """Run confounded variant with budget=3, seed=0 — no crashes."""

    def test_smoke_confounded(self, confounded_scenario: ConfoundedDemandResponse) -> None:
        result = confounded_scenario.run_benchmark(budget=3, seed=0, strategy="random")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert math.isfinite(result.policy_value)
        assert math.isfinite(result.oracle_value)
        assert math.isfinite(result.regret)
        assert result.regret >= -1e-9

    def test_smoke_confounded_surrogate(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        result = confounded_scenario.run_benchmark(budget=3, seed=0, strategy="surrogate_only")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert math.isfinite(result.policy_value)

    def test_smoke_confounded_causal(self, confounded_scenario: ConfoundedDemandResponse) -> None:
        result = confounded_scenario.run_benchmark(budget=3, seed=0, strategy="causal")
        assert isinstance(result, CounterfactualBenchmarkResult)
        assert math.isfinite(result.policy_value)


# ── Test 11: Reproducibility for each variant ────────────────────


class TestReproducibilityVariants:
    """Same seed produces identical results for each variant."""

    def test_reproducibility_high_noise(self, covariates: pd.DataFrame) -> None:
        s1 = HighNoiseDemandResponse(covariates=covariates, seed=123)
        s2 = HighNoiseDemandResponse(covariates=covariates, seed=123)
        r1 = s1.run_benchmark(budget=3, seed=42, strategy="random")
        r2 = s2.run_benchmark(budget=3, seed=42, strategy="random")
        assert r1.policy_value == pytest.approx(r2.policy_value, abs=1e-9)
        assert r1.oracle_value == pytest.approx(r2.oracle_value, abs=1e-9)
        assert r1.regret == pytest.approx(r2.regret, abs=1e-9)
        assert r1.decision_error_rate == pytest.approx(r2.decision_error_rate, abs=1e-9)

    def test_reproducibility_confounded(self, covariates: pd.DataFrame) -> None:
        s1 = ConfoundedDemandResponse(covariates=covariates, seed=123)
        s2 = ConfoundedDemandResponse(covariates=covariates, seed=123)
        r1 = s1.run_benchmark(budget=3, seed=42, strategy="random")
        r2 = s2.run_benchmark(budget=3, seed=42, strategy="random")
        assert r1.policy_value == pytest.approx(r2.policy_value, abs=1e-9)
        assert r1.oracle_value == pytest.approx(r2.oracle_value, abs=1e-9)
        assert r1.regret == pytest.approx(r2.regret, abs=1e-9)
        assert r1.decision_error_rate == pytest.approx(r2.decision_error_rate, abs=1e-9)


# ── Test 12: Deconfounding swap produces y0 - y1 = effect ────────


class TestDeconfoundingSwap:
    """Verify that swapping y0 to _deconfounded_y0 recovers the true causal effect."""

    def test_confounded_y0_minus_y1_differs_from_effect(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        data = confounded_scenario.generate()
        y0_confounded = data["y0"].values
        y1 = data["y1"].values
        effect = data["true_treatment_effect"].values

        # Confounded: y0 - y1 != effect (inflated by grid stress)
        apparent_benefit = y0_confounded - y1
        assert not np.allclose(apparent_benefit, effect, atol=1.0), (
            "Confounded y0 - y1 should differ from true_treatment_effect"
        )

    def test_deconfounded_y0_minus_y1_equals_effect(
        self, confounded_scenario: ConfoundedDemandResponse
    ) -> None:
        data = confounded_scenario.generate()
        y0_deconfounded = data["_deconfounded_y0"].values
        y1 = data["y1"].values
        effect = data["true_treatment_effect"].values

        # Deconfounded: y0 - y1 = y0_base - (y0_base - effect) = effect
        deconfounded_benefit = y0_deconfounded - y1
        assert np.allclose(deconfounded_benefit, effect, atol=1e-9), (
            "Deconfounded y0 - y1 should equal true_treatment_effect"
        )
