"""Unit tests for MarketingLogAdapter.

Tests the IPS/IPW-weighted policy evaluation, search space, causal graph,
dataset validation, and metric outputs of the marketing log adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.types import VariableType

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "marketing_log_fixture.csv"


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    """Load the fixture CSV."""
    return pd.read_csv(FIXTURE_PATH)


@pytest.fixture
def adapter(fixture_df: pd.DataFrame) -> MarketingLogAdapter:
    """Create a MarketingLogAdapter with the fixture data."""
    return MarketingLogAdapter(data=fixture_df, seed=42)


@pytest.fixture
def default_params() -> dict[str, float]:
    """Reasonable default parameters for testing."""
    return {
        "eligibility_threshold": 0.3,
        "email_share": 0.4,
        "social_share_of_remainder": 0.3,
        "min_propensity_clip": 0.05,
        "regularization": 1.0,
        "treatment_budget_pct": 0.5,
    }


class TestAdapterContract:
    """Adapter contract: search space, objective, minimize."""

    def test_search_space_has_6_variables(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        assert len(space.variables) == 6

    def test_search_space_variable_names(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        names = space.variable_names
        expected = [
            "eligibility_threshold",
            "email_share",
            "social_share_of_remainder",
            "min_propensity_clip",
            "regularization",
            "treatment_budget_pct",
        ]
        assert set(names) == set(expected)

    def test_objective_is_policy_value(self, adapter: MarketingLogAdapter) -> None:
        assert adapter.get_objective_name() == "policy_value"

    def test_minimize_is_false(self, adapter: MarketingLogAdapter) -> None:
        assert adapter.get_minimize() is False


class TestPriorGraph:
    """Prior graph: exists, has edges connecting policy variables to outcomes."""

    def test_prior_graph_exists(self, adapter: MarketingLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None

    def test_prior_graph_has_edges(self, adapter: MarketingLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert len(graph.edges) >= 6

    def test_prior_graph_connects_policy_to_outcomes(self, adapter: MarketingLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        edge_set = set(graph.edges)
        # Check key causal paths
        assert ("eligibility_threshold", "treated_fraction") in edge_set
        assert ("treatment_budget_pct", "treated_fraction") in edge_set
        assert ("min_propensity_clip", "policy_value") in edge_set
        assert ("regularization", "policy_value") in edge_set

    def test_prior_graph_nodes_include_policy_vars(self, adapter: MarketingLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        policy_vars = {"eligibility_threshold", "min_propensity_clip", "regularization"}
        assert policy_vars.issubset(set(graph.nodes))


class TestDescriptorNames:
    """Descriptor names for MAP-Elites diversity."""

    def test_descriptor_names(self, adapter: MarketingLogAdapter) -> None:
        assert adapter.get_descriptor_names() == ["total_cost", "treated_fraction"]


class TestDatasetValidation:
    """Dataset validation: raises ValueError on invalid data."""

    def test_missing_treatment_column(self) -> None:
        df = pd.DataFrame({"outcome": [1.0], "cost": [1.0]})
        with pytest.raises(ValueError, match="treatment"):
            MarketingLogAdapter(data=df, seed=42)

    def test_missing_outcome_column(self) -> None:
        df = pd.DataFrame({"treatment": [1], "cost": [1.0]})
        with pytest.raises(ValueError, match="outcome"):
            MarketingLogAdapter(data=df, seed=42)

    def test_missing_cost_column(self) -> None:
        df = pd.DataFrame({"treatment": [1], "outcome": [1.0]})
        with pytest.raises(ValueError, match="cost"):
            MarketingLogAdapter(data=df, seed=42)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"treatment": [], "outcome": [], "cost": []})
        with pytest.raises(ValueError, match="[Ee]mpty"):
            MarketingLogAdapter(data=df, seed=42)

    def test_custom_column_names(self) -> None:
        df = pd.DataFrame(
            {
                "t": [0, 1, 0, 1, 0],
                "y": [10.0, 20.0, 15.0, 25.0, 12.0],
                "c": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        adapter = MarketingLogAdapter(
            data=df,
            seed=42,
            treatment_col="t",
            outcome_col="y",
            cost_col="c",
        )
        space = adapter.get_search_space()
        assert len(space.variables) == 6


class TestDeterminism:
    """Same seed + same parameters -> identical metrics."""

    def test_same_seed_same_result(
        self, fixture_df: pd.DataFrame, default_params: dict[str, float]
    ) -> None:
        adapter1 = MarketingLogAdapter(data=fixture_df, seed=42)
        adapter2 = MarketingLogAdapter(data=fixture_df, seed=42)
        m1 = adapter1.run_experiment(default_params)
        m2 = adapter2.run_experiment(default_params)
        assert m1 == m2

    def test_different_params_different_result(self, fixture_df: pd.DataFrame) -> None:
        adapter1 = MarketingLogAdapter(data=fixture_df, seed=42)
        adapter2 = MarketingLogAdapter(data=fixture_df, seed=42)
        params1 = {
            "eligibility_threshold": 0.1,
            "email_share": 0.3,
            "social_share_of_remainder": 0.3,
            "min_propensity_clip": 0.05,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        params2 = {
            "eligibility_threshold": 0.9,
            "email_share": 0.3,
            "social_share_of_remainder": 0.3,
            "min_propensity_clip": 0.05,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        m1 = adapter1.run_experiment(params1)
        m2 = adapter2.run_experiment(params2)
        assert m1["policy_value"] != pytest.approx(m2["policy_value"], abs=1e-10)


class TestMetricCompleteness:
    """Returns all expected metrics, all numeric."""

    def test_returns_all_metrics(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        expected_keys = {
            "policy_value",
            "total_cost",
            "treated_fraction",
            "effective_sample_size",
            "zero_support",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_all_metrics_are_numeric(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"

    def test_all_metrics_are_finite(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} = {value} is not finite"


class TestPropensityHandling:
    """Works with and without propensity column."""

    def test_with_propensity_column(
        self, fixture_df: pd.DataFrame, default_params: dict[str, float]
    ) -> None:
        assert "propensity" in fixture_df.columns
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert "policy_value" in metrics
        assert np.isfinite(metrics["policy_value"])

    def test_without_propensity_column(
        self, fixture_df: pd.DataFrame, default_params: dict[str, float]
    ) -> None:
        df_no_prop = fixture_df.drop(columns=["propensity"])
        adapter = MarketingLogAdapter(data=df_no_prop, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert "policy_value" in metrics
        assert np.isfinite(metrics["policy_value"])

    def test_custom_propensity_col(self, default_params: dict[str, float]) -> None:
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0, 22.0, 11.0, 21.0, 13.0, 23.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.1, 2.1, 1.3, 2.3],
                "my_propensity": [0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.32, 0.68, 0.38, 0.62],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42, propensity_col="my_propensity")
        metrics = adapter.run_experiment(default_params)
        assert np.isfinite(metrics["policy_value"])


class TestSearchSpaceVariables:
    """Search space variable types and bounds."""

    def test_eligibility_threshold_is_continuous(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        var = next(v for v in space.variables if v.name == "eligibility_threshold")
        assert var.variable_type == VariableType.CONTINUOUS
        assert var.lower == 0.0
        assert var.upper == 1.0

    def test_min_propensity_clip_is_continuous(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        var = next(v for v in space.variables if v.name == "min_propensity_clip")
        assert var.variable_type == VariableType.CONTINUOUS
        assert var.lower == 0.01
        assert var.upper == 0.5

    def test_regularization_bounds(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        var = next(v for v in space.variables if v.name == "regularization")
        assert var.variable_type == VariableType.CONTINUOUS
        assert var.lower == 0.001
        assert var.upper == 10.0

    def test_treatment_budget_pct_bounds(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        var = next(v for v in space.variables if v.name == "treatment_budget_pct")
        assert var.variable_type == VariableType.CONTINUOUS
        assert var.lower == 0.1
        assert var.upper == 1.0

    def test_all_variables_continuous(self, adapter: MarketingLogAdapter) -> None:
        space = adapter.get_search_space()
        for var in space.variables:
            assert var.variable_type == VariableType.CONTINUOUS, f"{var.name} should be CONTINUOUS"


class TestEdgeCases:
    """All-treated or all-control data handled gracefully."""

    def test_all_treated(self, default_params: dict[str, float]) -> None:
        df = pd.DataFrame(
            {
                "treatment": [1] * 20,
                "outcome": np.random.default_rng(42).uniform(10, 50, 20).tolist(),
                "cost": np.random.default_rng(42).uniform(1, 10, 20).tolist(),
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert np.isfinite(metrics["policy_value"])

    def test_all_control(self, default_params: dict[str, float]) -> None:
        df = pd.DataFrame(
            {
                "treatment": [0] * 20,
                "outcome": np.random.default_rng(42).uniform(10, 50, 20).tolist(),
                "cost": np.random.default_rng(42).uniform(1, 10, 20).tolist(),
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert np.isfinite(metrics["policy_value"])

    def test_single_row(self, default_params: dict[str, float]) -> None:
        df = pd.DataFrame(
            {
                "treatment": [1],
                "outcome": [25.0],
                "cost": [5.0],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert np.isfinite(metrics["policy_value"])


class TestPositivity:
    """Adapter handles extreme propensities without crashing."""

    def test_extreme_propensities(self, default_params: dict[str, float]) -> None:
        rng = np.random.default_rng(42)
        n = 50
        # Very extreme propensities
        propensity = np.concatenate(
            [
                np.full(25, 0.001),  # near-zero
                np.full(25, 0.999),  # near-one
            ]
        )
        df = pd.DataFrame(
            {
                "treatment": rng.binomial(1, propensity),
                "outcome": rng.uniform(10, 50, n),
                "cost": rng.uniform(1, 10, n),
                "propensity": propensity,
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        metrics = adapter.run_experiment(default_params)
        assert np.isfinite(metrics["policy_value"])
        assert np.isfinite(metrics["effective_sample_size"])

    def test_min_propensity_clip_effect(self, fixture_df: pd.DataFrame) -> None:
        """Clipping propensities should affect ESS."""
        params_tight = {
            "eligibility_threshold": 0.3,
            "email_share": 0.4,
            "social_share_of_remainder": 0.3,
            "min_propensity_clip": 0.01,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        params_wide = {
            "eligibility_threshold": 0.3,
            "email_share": 0.4,
            "social_share_of_remainder": 0.3,
            "min_propensity_clip": 0.4,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        adapter1 = MarketingLogAdapter(data=fixture_df, seed=42)
        adapter2 = MarketingLogAdapter(data=fixture_df, seed=42)
        m1 = adapter1.run_experiment(params_tight)
        m2 = adapter2.run_experiment(params_wide)
        # More clipping should increase ESS (more uniform weights)
        assert m2["effective_sample_size"] >= m1["effective_sample_size"]


class TestRunProtocol:
    """Test the ExperimentRunner protocol (run method)."""

    def test_run_delegates_to_run_experiment(
        self, fixture_df: pd.DataFrame, default_params: dict[str, float]
    ) -> None:
        adapter1 = MarketingLogAdapter(data=fixture_df, seed=42)
        adapter2 = MarketingLogAdapter(data=fixture_df, seed=42)
        run_result = adapter1.run(default_params)
        experiment_result = adapter2.run_experiment(default_params)
        assert run_result == experiment_result


class TestLoadFromPath:
    """Test loading data from file path."""

    def test_load_from_path(self, default_params: dict[str, float]) -> None:
        adapter = MarketingLogAdapter(data_path=str(FIXTURE_PATH), seed=42)
        metrics = adapter.run_experiment(default_params)
        assert "policy_value" in metrics
        assert np.isfinite(metrics["policy_value"])

    def test_path_and_data_mutually_exclusive(self, fixture_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="[Ee]xactly one"):
            MarketingLogAdapter(data=fixture_df, data_path=str(FIXTURE_PATH), seed=42)

    def test_neither_path_nor_data(self) -> None:
        with pytest.raises(ValueError, match="[Ee]xactly one"):
            MarketingLogAdapter(seed=42)


# ---------------------------------------------------------------------------
# Warning metrics (Task 5)
# ---------------------------------------------------------------------------


class TestWarningMetrics:
    """Warning/diagnostic metrics are present and valid."""

    def test_warning_metrics_present(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert "propensity_clip_fraction" in metrics
        assert "max_ips_weight" in metrics
        assert "weight_cv" in metrics

    def test_all_warning_metrics_finite(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        for key in ["propensity_clip_fraction", "max_ips_weight", "weight_cv"]:
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    def test_propensity_clip_fraction_bounded(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        metrics = adapter.run_experiment(default_params)
        assert 0.0 <= metrics["propensity_clip_fraction"] <= 1.0

    def test_aggressive_clip_increases_clip_fraction(self, fixture_df: pd.DataFrame) -> None:
        adapter = MarketingLogAdapter(data=fixture_df, seed=42)
        default = {
            "eligibility_threshold": 0.3,
            "email_share": 0.4,
            "social_share_of_remainder": 0.3,
            "regularization": 1.0,
            "treatment_budget_pct": 0.5,
        }
        params_tight = {**default, "min_propensity_clip": 0.01}
        params_wide = {**default, "min_propensity_clip": 0.45}
        m_tight = adapter.run_experiment(params_tight)
        m_wide = adapter.run_experiment(params_wide)
        assert m_wide["propensity_clip_fraction"] >= m_tight["propensity_clip_fraction"]


# ---------------------------------------------------------------------------
# Parquet loading (Task 3)
# ---------------------------------------------------------------------------


class TestParquetLoading:
    """Test that Parquet files can be loaded."""

    @pytest.fixture(autouse=True)
    def _require_pyarrow(self) -> None:
        pytest.importorskip("pyarrow")

    def test_load_from_parquet(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        parquet_path = tmp_path / "marketing.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        adapter = MarketingLogAdapter(data_path=str(parquet_path), seed=42)
        space = adapter.get_search_space()
        assert len(space.variables) == 6

    def test_parquet_produces_same_metrics_as_csv(
        self, fixture_df: pd.DataFrame, tmp_path: Path, default_params: dict[str, float]
    ) -> None:
        parquet_path = tmp_path / "marketing.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        csv_adapter = MarketingLogAdapter(data=fixture_df, seed=42)
        parquet_adapter = MarketingLogAdapter(data_path=str(parquet_path), seed=42)
        m_csv = csv_adapter.run_experiment(default_params)
        m_parquet = parquet_adapter.run_experiment(default_params)
        for key in m_csv:
            assert m_csv[key] == pytest.approx(m_parquet[key], abs=1e-10), (
                f"Metric '{key}' differs between CSV and Parquet adapters"
            )


# ---------------------------------------------------------------------------
# Input validation: treatment binary & propensity bounds (Issue #47)
# ---------------------------------------------------------------------------


class TestTreatmentBinaryValidation:
    """Treatment column must contain only 0/1 values."""

    def test_non_binary_treatment_raises(self) -> None:
        """Treatment with values outside {0, 1} should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 2, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with pytest.raises(ValueError, match="Treatment column.*must be binary"):
            MarketingLogAdapter(data=df, seed=42)

    def test_float_non_binary_treatment_raises(self) -> None:
        """Treatment with float values like 0.5 should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0.0, 0.5, 1.0, 0.0, 1.0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with pytest.raises(ValueError, match="Treatment column.*must be binary"):
            MarketingLogAdapter(data=df, seed=42)

    def test_negative_treatment_raises(self) -> None:
        """Treatment with negative values should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0, -1, 1, 0, 1],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with pytest.raises(ValueError, match="Treatment column.*must be binary"):
            MarketingLogAdapter(data=df, seed=42)

    def test_custom_treatment_col_non_binary_raises(self) -> None:
        """Non-binary treatment raises even with a custom column name."""
        df = pd.DataFrame(
            {
                "t": [0, 1, 3, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with pytest.raises(ValueError, match="Treatment column 't' must be binary"):
            MarketingLogAdapter(data=df, seed=42, treatment_col="t")

    def test_string_treatment_raises(self) -> None:
        """Treatment with string values should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": ["yes", "no", "yes", "no", "yes"],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with pytest.raises(ValueError, match="Treatment column.*must be binary"):
            MarketingLogAdapter(data=df, seed=42)

    def test_valid_binary_treatment_passes(self) -> None:
        """Valid binary {0, 1} treatment should not raise."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None

    def test_valid_binary_float_treatment_passes(self) -> None:
        """Valid binary {0.0, 1.0} as floats should not raise."""
        df = pd.DataFrame(
            {
                "treatment": [0.0, 1.0, 0.0, 1.0, 0.0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None

    def test_boolean_treatment_passes(self) -> None:
        """Boolean treatment column should be accepted (True==1, False==0)."""
        df = pd.DataFrame(
            {
                "treatment": [True, False, True, False, True],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None


class TestPropensityBoundsValidation:
    """Propensity column values must be in [0, 1]."""

    def test_propensity_above_one_raises(self) -> None:
        """Propensity > 1 should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [0.3, 0.7, 0.4, 1.5, 0.35],
            }
        )
        with pytest.raises(ValueError, match="Propensity column.*values must be in"):
            MarketingLogAdapter(data=df, seed=42)

    def test_propensity_below_zero_raises(self) -> None:
        """Propensity < 0 should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [-0.1, 0.7, 0.4, 0.6, 0.35],
            }
        )
        with pytest.raises(ValueError, match="Propensity column.*values must be in"):
            MarketingLogAdapter(data=df, seed=42)

    def test_custom_propensity_col_out_of_range_raises(self) -> None:
        """Out-of-range propensity raises even with a custom column name."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "ps": [0.3, 0.7, 0.4, 2.0, 0.35],
            }
        )
        with pytest.raises(ValueError, match="Propensity column 'ps' values must be in"):
            MarketingLogAdapter(data=df, seed=42, propensity_col="ps")

    def test_valid_propensity_passes(self) -> None:
        """Propensity in [0, 1] should not raise."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [0.3, 0.7, 0.4, 0.6, 0.35],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None

    def test_propensity_at_boundaries_passes(self) -> None:
        """Propensity at exact 0.0 and 1.0 boundaries should be valid."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [0.0, 1.0, 0.5, 0.5, 0.5],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None

    def test_non_numeric_propensity_raises(self) -> None:
        """Propensity with non-numeric dtype should raise ValueError."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": ["low", "high", "low", "high", "low"],
            }
        )
        with pytest.raises(ValueError, match="Propensity column.*must be numeric"):
            MarketingLogAdapter(data=df, seed=42)

    def test_missing_propensity_col_skips_validation(self) -> None:
        """When propensity column is absent, no propensity validation occurs."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        adapter = MarketingLogAdapter(data=df, seed=42)
        assert adapter is not None


class TestBoundaryPropensityWarning:
    """Warn when propensity values are at exact 0.0 or 1.0 boundaries."""

    _LOGGER = "causal_optimizer.domain_adapters.marketing_logs"

    def test_boundary_propensity_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Propensity at 0.0 or 1.0 should log a warning about clipping."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [0.0, 1.0, 0.5, 0.5, 0.5],
            }
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            MarketingLogAdapter(data=df, seed=42)
        assert any("boundary values" in r.message.lower() for r in caplog.records)

    def test_interior_propensity_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Propensity strictly inside (0, 1) should not log a boundary warning."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
                "propensity": [0.3, 0.7, 0.4, 0.6, 0.35],
            }
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            MarketingLogAdapter(data=df, seed=42)
        assert not any("boundary values" in r.message.lower() for r in caplog.records)


class TestSingleArmWarning:
    """Warn when one treatment arm is entirely absent."""

    _LOGGER = "causal_optimizer.domain_adapters.marketing_logs"

    def test_all_treated_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """All-treated data (no control arm) should log a warning."""
        df = pd.DataFrame(
            {
                "treatment": [1, 1, 1, 1, 1],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            MarketingLogAdapter(data=df, seed=42)
        assert any("single treatment arm" in r.message.lower() for r in caplog.records)

    def test_all_control_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """All-control data (no treated arm) should log a warning."""
        df = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 0, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            MarketingLogAdapter(data=df, seed=42)
        assert any("single treatment arm" in r.message.lower() for r in caplog.records)

    def test_both_arms_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Data with both arms should not log a single-arm warning."""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0],
                "outcome": [10.0, 20.0, 15.0, 25.0, 12.0],
                "cost": [1.0, 2.0, 1.5, 2.5, 1.2],
            }
        )
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            MarketingLogAdapter(data=df, seed=42)
        assert not any("single treatment arm" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Zero-support guard (Issue #46)
# ---------------------------------------------------------------------------


class TestZeroSupportGuard:
    """When no logged observations match the proposed policy, the adapter
    should use a pessimistic fallback instead of the unweighted population mean.
    """

    @pytest.fixture
    def zero_support_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """All-treated DataFrame where a no-treat policy has zero support."""
        rng = np.random.default_rng(99)
        n = 50
        outcomes = rng.uniform(10, 50, n)
        df = pd.DataFrame(
            {
                "treatment": np.ones(n, dtype=int),
                "outcome": outcomes,
                "cost": rng.uniform(1, 5, n),
                "propensity": np.full(n, 0.8),
            }
        )
        return df, outcomes

    @pytest.fixture
    def zero_support_metrics(
        self, zero_support_data: tuple[pd.DataFrame, np.ndarray]
    ) -> dict[str, float]:
        """Run a no-treat policy against all-treated data (zero support)."""
        df, _outcomes = zero_support_data
        adapter = MarketingLogAdapter(data=df, seed=42)
        params = {
            "eligibility_threshold": 0.99,
            "email_share": 0.5,
            "social_share_of_remainder": 0.5,
            "min_propensity_clip": 0.05,
            "regularization": 0.001,
            "treatment_budget_pct": 0.1,
        }
        return adapter.run_experiment(params)

    def test_zero_support_returns_pessimistic_policy_value(
        self,
        zero_support_data: tuple[pd.DataFrame, np.ndarray],
        zero_support_metrics: dict[str, float],
    ) -> None:
        """policy_value should equal outcome.min(), below the population mean."""
        _df, outcomes = zero_support_data
        population_mean = float(outcomes.mean())
        assert zero_support_metrics["policy_value"] < population_mean
        assert zero_support_metrics["policy_value"] == pytest.approx(
            float(outcomes.min()), abs=1e-10
        )

    def test_zero_support_effective_sample_size_is_zero(
        self, zero_support_metrics: dict[str, float]
    ) -> None:
        """ESS must be 0.0 when no observations match the policy."""
        assert zero_support_metrics["effective_sample_size"] == 0.0

    def test_zero_support_metric_is_one(self, zero_support_metrics: dict[str, float]) -> None:
        """zero_support metric must be 1.0 when weight_sum == 0."""
        assert zero_support_metrics["zero_support"] == 1.0

    def test_normal_case_zero_support_metric_is_zero(
        self, adapter: MarketingLogAdapter, default_params: dict[str, float]
    ) -> None:
        """zero_support metric must be 0.0 when observations match the policy."""
        metrics = adapter.run_experiment(default_params)
        assert metrics["zero_support"] == 0.0
