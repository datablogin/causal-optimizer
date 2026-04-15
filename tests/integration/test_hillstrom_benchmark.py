"""Integration tests for the Sprint 31 Hillstrom benchmark scenario.

These exercise the full ``HillstromScenario.run_strategy`` path end to
end on the committed fixture, including the engine loop and the
adapter. They are intentionally small (3 experiments per strategy) so
they run quickly inside CI and do not depend on the optional
Ax/BoTorch stack — each strategy is exercised with a budget that can
complete under either the RF fallback or the Ax primary backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.hillstrom import (
    HillstromBenchmarkResult,
    HillstromScenario,
    HillstromSliceType,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "hillstrom_fixture.csv"


@pytest.fixture
def raw_hillstrom() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


class TestHillstromScenarioPrimary:
    """Primary slice ``Womens E-Mail vs No E-Mail``."""

    def test_scenario_reshapes_into_primary_slice(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        reshaped = scenario.real_slice
        # Treatment is binary and both arms are present
        assert set(reshaped["treatment"].unique()) == {0, 1}
        # Propensity is a scalar 0.5 on every row
        assert (reshaped["propensity"] == 0.5).all()
        # Null baseline equals raw mean spend of the reshaped frame
        expected_mu = float(reshaped["outcome"].mean())
        assert scenario.null_baseline == expected_mu

    def test_random_strategy_returns_finite_policy_value(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        result = scenario.run_strategy("random", budget=8, seed=0)
        assert isinstance(result, HillstromBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 8
        assert result.seed == 0
        assert result.slice_type == "primary"
        assert result.is_null_control is False
        assert np.isfinite(result.policy_value)
        assert result.selected_parameters is not None
        assert set(result.selected_parameters.keys()) == {
            "eligibility_threshold",
            "regularization",
            "treatment_budget_pct",
        }

    def test_surrogate_only_strategy_runs_on_fixture(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        result = scenario.run_strategy("surrogate_only", budget=8, seed=0)
        assert result.strategy == "surrogate_only"
        assert np.isfinite(result.policy_value)

    def test_causal_strategy_runs_on_fixture(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        result = scenario.run_strategy("causal", budget=8, seed=0)
        assert result.strategy == "causal"
        assert np.isfinite(result.policy_value)

    def test_unknown_strategy_raises(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        with pytest.raises(ValueError, match="Unknown strategy"):
            scenario.run_strategy("magic", budget=1, seed=0)

    def test_secondary_outcomes_reported(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        result = scenario.run_strategy("random", budget=8, seed=1)
        # Secondary-outcome aggregates should be populated on a real (non-null)
        # run; the contract treats these as post-hoc diagnostics, not
        # optimization objectives.
        assert "treated_visit_rate" in result.secondary_outcomes
        assert "control_visit_rate" in result.secondary_outcomes
        for val in result.secondary_outcomes.values():
            assert 0.0 <= val <= 1.0


class TestHillstromScenarioPooled:
    """Pooled secondary slice ``Any E-Mail vs No E-Mail``."""

    def test_pooled_propensity_is_two_thirds(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        reshaped = scenario.real_slice
        expected = 2.0 / 3.0
        assert (reshaped["propensity"] == expected).all()

    def test_pooled_scenario_runs_random_strategy(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.POOLED)
        result = scenario.run_strategy("random", budget=8, seed=0)
        assert result.slice_type == "pooled"
        assert np.isfinite(result.policy_value)


class TestHillstromNullControl:
    """Permuted-outcome null-control path on the primary slice."""

    def test_null_control_preserves_mu(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        # Running under null_control must leave the scenario's internal
        # baseline μ unchanged — it was computed once from the real frame.
        mu_before = scenario.null_baseline
        _ = scenario.run_strategy("random", budget=8, seed=0, null_control=True)
        assert scenario.null_baseline == mu_before

    def test_null_control_result_is_flagged(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        result = scenario.run_strategy("random", budget=8, seed=0, null_control=True)
        assert result.is_null_control is True
        assert result.null_baseline == scenario.null_baseline

    def test_null_control_determinism_under_fixed_seed(self, raw_hillstrom: pd.DataFrame) -> None:
        scenario = HillstromScenario(raw_hillstrom, slice_type=HillstromSliceType.PRIMARY)
        a = scenario.run_strategy("random", budget=8, seed=42, null_control=True)
        b = scenario.run_strategy("random", budget=8, seed=42, null_control=True)
        # Random strategy is seed-deterministic → same best policy_value
        assert a.policy_value == b.policy_value
        assert a.selected_parameters == b.selected_parameters
