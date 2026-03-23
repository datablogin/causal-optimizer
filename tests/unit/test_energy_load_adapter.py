"""Unit tests for EnergyLoadAdapter.

Tests the adapter contract, prior graph, search space, determinism,
metric completeness, no-leakage split, validation, and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter
from causal_optimizer.types import VariableType

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


@pytest.fixture()
def adapter(fixture_df: pd.DataFrame) -> EnergyLoadAdapter:
    return EnergyLoadAdapter(data=fixture_df, seed=42)


# ---------------------------------------------------------------------------
# 1. Adapter contract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    """Search space has 6-8 variables, objective is 'mae', minimize is True."""

    def test_search_space_variable_count(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        assert 6 <= len(space.variables) <= 8

    def test_search_space_has_mixed_types(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        types_present = {v.variable_type for v in space.variables}
        assert VariableType.CONTINUOUS in types_present
        assert VariableType.INTEGER in types_present
        assert VariableType.CATEGORICAL in types_present
        assert VariableType.BOOLEAN in types_present

    def test_objective_name_is_mae(self, adapter: EnergyLoadAdapter) -> None:
        assert adapter.get_objective_name() == "mae"

    def test_minimize_is_true(self, adapter: EnergyLoadAdapter) -> None:
        assert adapter.get_minimize() is True


# ---------------------------------------------------------------------------
# 2. Prior graph
# ---------------------------------------------------------------------------


class TestPriorGraph:
    """Prior graph exists, has edges to 'mae' and 'runtime_seconds', no isolated nodes."""

    def test_prior_graph_exists(self, adapter: EnergyLoadAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None

    def test_prior_graph_has_edges_to_mae(self, adapter: EnergyLoadAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        targets = {v for _, v in graph.edges}
        assert "mae" in targets

    def test_prior_graph_has_edges_to_runtime(self, adapter: EnergyLoadAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        targets = {v for _, v in graph.edges}
        assert "runtime_seconds" in targets

    def test_prior_graph_no_isolated_nodes(self, adapter: EnergyLoadAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        connected_nodes: set[str] = set()
        for u, v in graph.edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        for u, v in graph.bidirected_edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        for node in graph.nodes:
            assert node in connected_nodes, f"Node {node!r} is isolated"


# ---------------------------------------------------------------------------
# 3. Descriptor names
# ---------------------------------------------------------------------------


class TestDescriptorNames:
    def test_descriptor_names(self, adapter: EnergyLoadAdapter) -> None:
        assert adapter.get_descriptor_names() == ["runtime_seconds", "feature_count"]


# ---------------------------------------------------------------------------
# 4. Dataset validation
# ---------------------------------------------------------------------------


class TestDatasetValidation:
    def test_missing_required_columns_raises(self, fixture_df: pd.DataFrame) -> None:
        bad_df = fixture_df.drop(columns=["target_load"])
        with pytest.raises(ValueError, match="target_load"):
            EnergyLoadAdapter(data=bad_df, seed=42)

    def test_missing_timestamp_raises(self, fixture_df: pd.DataFrame) -> None:
        bad_df = fixture_df.drop(columns=["timestamp"])
        with pytest.raises(ValueError, match="timestamp"):
            EnergyLoadAdapter(data=bad_df, seed=42)

    def test_empty_dataframe_raises(self) -> None:
        empty_df = pd.DataFrame(columns=["timestamp", "target_load", "temperature"])
        with pytest.raises(ValueError, match="[Ee]mpty"):
            EnergyLoadAdapter(data=empty_df, seed=42)

    def test_non_numeric_target_load_raises(self, fixture_df: pd.DataFrame) -> None:
        bad_df = fixture_df.copy()
        bad_df["target_load"] = "not_a_number"
        with pytest.raises(ValueError, match="target_load"):
            EnergyLoadAdapter(data=bad_df, seed=42)

    def test_no_covariates_raises(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h").strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "target_load": np.random.default_rng(0).normal(500, 50, 50),
            }
        )
        with pytest.raises(ValueError, match="[Cc]ovariate"):
            EnergyLoadAdapter(data=df, seed=42)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self, fixture_df: pd.DataFrame) -> None:
        params = {
            "model_type": "rf",
            "lookback_window": 6,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 0.1,
            "n_estimators": 50,
        }
        a1 = EnergyLoadAdapter(data=fixture_df, seed=42)
        a2 = EnergyLoadAdapter(data=fixture_df, seed=42)
        m1 = a1.run_experiment(params)
        m2 = a2.run_experiment(params)
        assert m1["mae"] == pytest.approx(m2["mae"], abs=1e-10)
        assert m1["rmse"] == pytest.approx(m2["rmse"], abs=1e-10)

    def test_different_seed_different_result(self, fixture_df: pd.DataFrame) -> None:
        params = {
            "model_type": "rf",
            "lookback_window": 6,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 0.1,
            "n_estimators": 50,
        }
        a1 = EnergyLoadAdapter(data=fixture_df, seed=42)
        a2 = EnergyLoadAdapter(data=fixture_df, seed=99)
        m1 = a1.run_experiment(params)
        m2 = a2.run_experiment(params)
        # RF with different seeds should give different results
        assert m1["mae"] != pytest.approx(m2["mae"], abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Metric completeness
# ---------------------------------------------------------------------------


class TestMetricCompleteness:
    def test_returns_required_metrics(self, adapter: EnergyLoadAdapter) -> None:
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        metrics = adapter.run_experiment(params)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "runtime_seconds" in metrics

    def test_metrics_are_numeric_floats(self, adapter: EnergyLoadAdapter) -> None:
        params = {
            "model_type": "gbm",
            "lookback_window": 12,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 0.5,
            "n_estimators": 100,
        }
        metrics = adapter.run_experiment(params)
        for key, val in metrics.items():
            assert isinstance(val, float), f"Metric {key!r} is {type(val)}, expected float"

    def test_mape_metric_returned(self, adapter: EnergyLoadAdapter) -> None:
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        metrics = adapter.run_experiment(params)
        assert "mape" in metrics

    def test_feature_count_metric_returned(self, adapter: EnergyLoadAdapter) -> None:
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        metrics = adapter.run_experiment(params)
        assert "feature_count" in metrics
        assert metrics["feature_count"] >= 1


# ---------------------------------------------------------------------------
# 7. No leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_validation_from_separate_time_window(self, fixture_df: pd.DataFrame) -> None:
        """Train/validation split must be blocked by time, not shuffled."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42, train_ratio=0.7)
        n_total = len(fixture_df)
        n_train = int(n_total * 0.7)

        # Access internal split indices
        assert adapter._train_end == n_train
        assert adapter._train_end < n_total


# ---------------------------------------------------------------------------
# 8. Search space variables
# ---------------------------------------------------------------------------


class TestSearchSpaceVariables:
    def test_model_type_is_categorical(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        model_var = next(v for v in space.variables if v.name == "model_type")
        assert model_var.variable_type == VariableType.CATEGORICAL
        assert "ridge" in (model_var.choices or [])
        assert "rf" in (model_var.choices or [])
        assert "gbm" in (model_var.choices or [])

    def test_lookback_window_is_integer(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        lb_var = next(v for v in space.variables if v.name == "lookback_window")
        assert lb_var.variable_type == VariableType.INTEGER
        assert lb_var.lower is not None and lb_var.lower >= 1
        assert lb_var.upper is not None and lb_var.upper <= 48

    def test_boolean_variables(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        bool_vars = [v for v in space.variables if v.variable_type == VariableType.BOOLEAN]
        bool_names = {v.name for v in bool_vars}
        assert "use_temperature" in bool_names
        assert "use_humidity" in bool_names
        assert "use_calendar" in bool_names

    def test_regularization_is_continuous(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        reg_var = next(v for v in space.variables if v.name == "regularization")
        assert reg_var.variable_type == VariableType.CONTINUOUS
        assert reg_var.lower is not None and reg_var.lower > 0
        assert reg_var.upper is not None and reg_var.upper <= 10.0

    def test_n_estimators_is_integer(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        ne_var = next(v for v in space.variables if v.name == "n_estimators")
        assert ne_var.variable_type == VariableType.INTEGER
        assert ne_var.lower is not None and ne_var.lower >= 10
        assert ne_var.upper is not None and ne_var.upper <= 200

    def test_all_variables_have_valid_bounds_or_choices(self, adapter: EnergyLoadAdapter) -> None:
        space = adapter.get_search_space()
        for var in space.variables:
            if var.variable_type in (VariableType.CONTINUOUS, VariableType.INTEGER):
                assert var.lower is not None
                assert var.upper is not None
                assert var.lower < var.upper
            elif var.variable_type == VariableType.CATEGORICAL:
                assert var.choices is not None
                assert len(var.choices) >= 2


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_ridge_ignores_n_estimators(self, adapter: EnergyLoadAdapter) -> None:
        """Ridge model should not crash even with n_estimators set."""
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 100,
        }
        metrics = adapter.run_experiment(params)
        assert "mae" in metrics

    def test_all_features_disabled_still_works(self, adapter: EnergyLoadAdapter) -> None:
        """Even with all optional features off, lagged load should provide features."""
        params = {
            "model_type": "ridge",
            "lookback_window": 1,
            "use_temperature": False,
            "use_humidity": False,
            "use_calendar": False,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        metrics = adapter.run_experiment(params)
        assert "mae" in metrics
        assert metrics["feature_count"] >= 1

    def test_max_lookback_window(self, adapter: EnergyLoadAdapter) -> None:
        """Lookback window of 48 should not crash."""
        params = {
            "model_type": "ridge",
            "lookback_window": 48,
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        metrics = adapter.run_experiment(params)
        assert "mae" in metrics

    def test_from_data_path(self) -> None:
        """Constructor with data_path should work."""
        adapter = EnergyLoadAdapter(data_path=str(FIXTURE_PATH), seed=42)
        space = adapter.get_search_space()
        assert len(space.variables) >= 6

    def test_run_delegates_to_run_experiment(self, fixture_df: pd.DataFrame) -> None:
        """run() (ExperimentRunner protocol) should delegate to run_experiment()."""
        params = {
            "model_type": "rf",
            "lookback_window": 6,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 0.1,
            "n_estimators": 50,
        }
        a1 = EnergyLoadAdapter(data=fixture_df, seed=42)
        a2 = EnergyLoadAdapter(data=fixture_df, seed=42)
        run_result = a1.run(params)
        exp_result = a2.run_experiment(params)
        # Compare deterministic metrics (runtime_seconds is wall-clock, so skip it)
        assert run_result["mae"] == pytest.approx(exp_result["mae"], abs=1e-10)
        assert run_result["rmse"] == pytest.approx(exp_result["rmse"], abs=1e-10)
        assert run_result["feature_count"] == exp_result["feature_count"]


# ---------------------------------------------------------------------------
# 10. Feature effect direction
# ---------------------------------------------------------------------------


class TestFeatureEffects:
    def test_more_features_lower_mae(self, fixture_df: pd.DataFrame) -> None:
        """Using more features (temperature, calendar) should reduce MAE."""
        base = {
            "model_type": "rf",
            "lookback_window": 6,
            "use_humidity": False,
            "regularization": 0.1,
            "n_estimators": 100,
        }
        # Minimal features
        a1 = EnergyLoadAdapter(data=fixture_df, seed=42)
        m_minimal = a1.run_experiment({**base, "use_temperature": False, "use_calendar": False})
        # Rich features
        a2 = EnergyLoadAdapter(data=fixture_df, seed=42)
        m_rich = a2.run_experiment({**base, "use_temperature": True, "use_calendar": True})
        assert m_rich["mae"] < m_minimal["mae"], (
            "Adding temperature + calendar features should reduce MAE"
        )

    def test_larger_lookback_helps(self, fixture_df: pd.DataFrame) -> None:
        """Larger lookback window should capture more patterns."""
        base = {
            "model_type": "rf",
            "use_temperature": True,
            "use_humidity": True,
            "use_calendar": True,
            "regularization": 0.1,
            "n_estimators": 100,
        }
        a1 = EnergyLoadAdapter(data=fixture_df, seed=42)
        m_short = a1.run_experiment({**base, "lookback_window": 1})
        a2 = EnergyLoadAdapter(data=fixture_df, seed=42)
        m_long = a2.run_experiment({**base, "lookback_window": 24})
        # Longer lookback should do at least as well
        assert m_long["mae"] <= m_short["mae"] * 1.2, (
            "Longer lookback should not drastically worsen MAE"
        )


# ---------------------------------------------------------------------------
# 11. Timestamp handling (Task 4)
# ---------------------------------------------------------------------------


class TestTimestampHandling:
    """Timestamp sorting, validation, and gap handling."""

    def test_unsorted_data_is_sorted(self, fixture_df: pd.DataFrame) -> None:
        """Passing shuffled data should produce same results as sorted."""
        shuffled = fixture_df.sample(frac=1, random_state=99).reset_index(drop=True)
        adapter_sorted = EnergyLoadAdapter(data=fixture_df, seed=42)
        adapter_shuffled = EnergyLoadAdapter(data=shuffled, seed=42)
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        m1 = adapter_sorted.run_experiment(params)
        m2 = adapter_shuffled.run_experiment(params)
        assert m1["mae"] == pytest.approx(m2["mae"], rel=1e-6)

    def test_duplicate_timestamps_raises(self, fixture_df: pd.DataFrame) -> None:
        """Duplicate timestamps should raise ValueError instead of silently dropping."""
        duped = pd.concat([fixture_df, fixture_df.iloc[:5]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate timestamps"):
            EnergyLoadAdapter(data=duped, seed=42)

    def test_timestamp_parsed_as_datetime(self, fixture_df: pd.DataFrame) -> None:
        """Timestamp column should be parsed to datetime."""
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        assert pd.api.types.is_datetime64_any_dtype(adapter._data["timestamp"])

    def test_duplicate_timestamps_error_message(self, fixture_df: pd.DataFrame) -> None:
        """ValueError message should mention count and multi-area guidance."""
        duped = pd.concat([fixture_df, fixture_df.iloc[:5]], ignore_index=True)
        with pytest.raises(ValueError, match=r"Found 5 duplicate timestamps") as exc_info:
            EnergyLoadAdapter(data=duped, seed=42)
        assert "filter to one area" in str(exc_info.value)

    def test_cadence_metrics_present(self, adapter: EnergyLoadAdapter) -> None:
        """run_experiment output must include cadence_gaps and cadence_regularity."""
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert "cadence_gaps" in metrics
        assert "cadence_regularity" in metrics

    def test_regular_cadence_near_one(self, adapter: EnergyLoadAdapter) -> None:
        """Fixture data is hourly; cadence_regularity should be > 0.95."""
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert metrics["cadence_regularity"] > 0.95

    def test_irregular_cadence_detected(self, fixture_df: pd.DataFrame) -> None:
        """Dropping every 3rd row should lower regularity and produce gaps."""
        irregular = fixture_df.drop(fixture_df.index[::3]).reset_index(drop=True)
        adapter = EnergyLoadAdapter(data=irregular, seed=42)
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert metrics["cadence_regularity"] < 0.8
        assert metrics["cadence_gaps"] > 0


# ---------------------------------------------------------------------------
# 12. Warning metrics (Task 5)
# ---------------------------------------------------------------------------


DEFAULT_PARAMS = {
    "model_type": "ridge",
    "lookback_window": 3,
    "use_temperature": True,
    "use_humidity": False,
    "use_calendar": True,
    "regularization": 1.0,
    "n_estimators": 50,
}


class TestWarningMetrics:
    """Warning/diagnostic metrics are present and valid."""

    def test_warning_metrics_present(self, adapter: EnergyLoadAdapter) -> None:
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert "validation_set_size" in metrics
        assert "nan_rows_dropped" in metrics
        assert "train_fraction_actual" in metrics

    def test_validation_set_size_is_positive(self, adapter: EnergyLoadAdapter) -> None:
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert metrics["validation_set_size"] > 0

    def test_nan_rows_dropped_nonnegative(self, adapter: EnergyLoadAdapter) -> None:
        metrics = adapter.run_experiment(DEFAULT_PARAMS)
        assert metrics["nan_rows_dropped"] >= 0

    def test_large_lookback_increases_nan_drops(self, fixture_df: pd.DataFrame) -> None:
        adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        params_small = {**DEFAULT_PARAMS, "lookback_window": 1}
        params_large = {**DEFAULT_PARAMS, "lookback_window": 24}
        m_small = adapter.run_experiment(params_small)
        m_large = adapter.run_experiment(params_large)
        assert m_large["nan_rows_dropped"] > m_small["nan_rows_dropped"]


# ---------------------------------------------------------------------------
# 13. Parquet loading (Task 3)
# ---------------------------------------------------------------------------


class TestParquetLoading:
    """Test that Parquet files can be loaded."""

    @pytest.fixture(autouse=True)
    def _require_pyarrow(self) -> None:
        pytest.importorskip("pyarrow")

    def test_load_from_parquet(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        parquet_path = tmp_path / "energy.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        adapter = EnergyLoadAdapter(data_path=str(parquet_path), seed=42)
        space = adapter.get_search_space()
        assert len(space.variables) == 7

    def test_parquet_produces_same_metrics_as_csv(
        self, fixture_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        parquet_path = tmp_path / "energy.parquet"
        fixture_df.to_parquet(parquet_path, index=False)
        csv_adapter = EnergyLoadAdapter(data=fixture_df, seed=42)
        parquet_adapter = EnergyLoadAdapter(data_path=str(parquet_path), seed=42)
        params = {
            "model_type": "ridge",
            "lookback_window": 3,
            "use_temperature": True,
            "use_humidity": False,
            "use_calendar": True,
            "regularization": 1.0,
            "n_estimators": 50,
        }
        m_csv = csv_adapter.run_experiment(params)
        m_parquet = parquet_adapter.run_experiment(params)
        # Compare deterministic metrics (runtime_seconds is wall-clock)
        for key in ("mae", "rmse", "mape", "feature_count"):
            assert m_csv[key] == pytest.approx(m_parquet[key], abs=1e-10)
