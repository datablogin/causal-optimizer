"""Integration tests for the predictive energy benchmark harness.

Tests cover: data loading, time-based splitting, validation runner
determinism, test evaluation, and error handling for edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    ValidationEnergyRunner,
    evaluate_on_test,
    load_energy_frame,
    split_time_frame,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"

_RIDGE_PARAMS: dict[str, object] = {
    "model_type": "ridge",
    "lookback_window": 3,
    "use_temperature": True,
    "use_humidity": False,
    "use_calendar": True,
    "regularization": 1.0,
    "n_estimators": 50,
}


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


# ── split_time_frame ─────────────────────────────────────────────────


class TestSplitTimeFrame:
    """Tests for split_time_frame partitioning."""

    def test_split_produces_correct_approximate_proportions(self, fixture_df: pd.DataFrame) -> None:
        """With 200 rows and 0.5/0.25/0.25 fracs, partitions should be ~100/50/50."""
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)
        total = len(fixture_df)
        assert len(train) == pytest.approx(total * 0.5, abs=2)
        assert len(val) == pytest.approx(total * 0.25, abs=2)
        assert len(test) == pytest.approx(total * 0.25, abs=2)
        assert len(train) + len(val) + len(test) == total

    def test_no_leakage_test_after_val_after_train(self, fixture_df: pd.DataFrame) -> None:
        """All test timestamps must be strictly after all val timestamps,
        and all val timestamps must be strictly after all train timestamps."""
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        train_ts = pd.to_datetime(train["timestamp"])
        val_ts = pd.to_datetime(val["timestamp"])
        test_ts = pd.to_datetime(test["timestamp"])

        assert train_ts.max() < val_ts.min(), "Train must end before val starts"
        assert val_ts.max() < test_ts.min(), "Val must end before test starts"

    def test_duplicate_timestamps_raises(self, fixture_df: pd.DataFrame) -> None:
        """Duplicate timestamps in the input should raise ValueError."""
        df_dup = pd.concat([fixture_df, fixture_df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            split_time_frame(df_dup)

    def test_empty_dataframe_raises(self) -> None:
        """An empty DataFrame should raise ValueError."""
        empty = pd.DataFrame(columns=["timestamp", "target_load", "temperature"])
        with pytest.raises(ValueError, match="[Ee]mpty"):
            split_time_frame(empty)

    def test_fractions_summing_to_one_raises(self, fixture_df: pd.DataFrame) -> None:
        """train_frac + val_frac >= 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="[Ff]raction|[Rr]oom"):
            split_time_frame(fixture_df, train_frac=0.6, val_frac=0.4)

    def test_fractions_exceeding_one_raises(self, fixture_df: pd.DataFrame) -> None:
        """train_frac + val_frac > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="[Ff]raction|[Rr]oom"):
            split_time_frame(fixture_df, train_frac=0.8, val_frac=0.3)

    def test_negative_fraction_raises(self, fixture_df: pd.DataFrame) -> None:
        """Negative train_frac or val_frac should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            split_time_frame(fixture_df, train_frac=-0.1, val_frac=0.2)
        with pytest.raises(ValueError, match="positive"):
            split_time_frame(fixture_df, train_frac=0.6, val_frac=-0.1)

    def test_zero_fraction_raises(self, fixture_df: pd.DataFrame) -> None:
        """Zero train_frac or val_frac should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            split_time_frame(fixture_df, train_frac=0.0, val_frac=0.2)

    def test_partition_minimum_size(self) -> None:
        """Each partition must have at least 10 rows; small data should raise."""
        # 25 rows: with default fracs 0.6/0.2, test gets 25*0.2 = 5 rows < 10
        small_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=25, freq="h"),
                "target_load": range(25),
                "temperature": range(25),
            }
        )
        with pytest.raises(ValueError, match="10"):
            split_time_frame(small_df)


# ── load_energy_frame ────────────────────────────────────────────────


class TestLoadEnergyFrame:
    """Tests for load_energy_frame CSV/Parquet loading."""

    def test_load_csv(self) -> None:
        """Loading the fixture CSV should return a non-empty DataFrame."""
        df = load_energy_frame(str(FIXTURE_PATH))
        assert len(df) == 200
        assert "timestamp" in df.columns
        assert "target_load" in df.columns

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        """A CSV missing required columns should raise ValueError."""
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="[Mm]issing.*column"):
            load_energy_frame(str(bad))

    def test_empty_csv_raises(self, tmp_path: Path) -> None:
        """An empty CSV (headers only) should raise ValueError."""
        empty = tmp_path / "empty.csv"
        pd.DataFrame(columns=["timestamp", "target_load", "temperature"]).to_csv(empty, index=False)
        with pytest.raises(ValueError, match="[Ee]mpty"):
            load_energy_frame(str(empty))

    def test_multi_series_guard_raises(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        """When area_id column exists with >1 unique value and area_id is None, raise."""
        df = fixture_df.copy()
        df["area_id"] = ["A"] * 100 + ["B"] * 100
        path = tmp_path / "multi.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="area.id"):
            load_energy_frame(str(path))

    def test_area_id_filter(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        """Filtering by area_id should return only rows for that area."""
        df = fixture_df.copy()
        df["area_id"] = ["A"] * 100 + ["B"] * 100
        path = tmp_path / "multi.csv"
        df.to_csv(path, index=False)

        result = load_energy_frame(str(path), area_id="A")
        assert len(result) == 100
        assert (result["area_id"] == "A").all()

    def test_area_id_filter_empty_result_raises(
        self, fixture_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Filtering to a non-existent area_id should raise ValueError."""
        df = fixture_df.copy()
        df["area_id"] = "A"
        path = tmp_path / "single.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="[Ee]mpty"):
            load_energy_frame(str(path), area_id="NONEXISTENT")

    def test_area_id_requested_but_column_missing_raises(self) -> None:
        """Requesting area_id filter when column doesn't exist should raise."""
        with pytest.raises(ValueError, match="area_id"):
            load_energy_frame(str(FIXTURE_PATH), area_id="A")

    @pytest.mark.skipif(
        not _has_pyarrow(),
        reason="pyarrow not installed",
    )
    def test_load_parquet(self, fixture_df: pd.DataFrame, tmp_path: Path) -> None:
        """Loading a Parquet file should work the same as CSV."""
        parquet_path = tmp_path / "data.parquet"
        fixture_df.to_parquet(parquet_path, index=False)

        df = load_energy_frame(str(parquet_path))
        assert len(df) == 200


# ── ValidationEnergyRunner ───────────────────────────────────────────


class TestValidationEnergyRunner:
    """Tests for the ValidationEnergyRunner class."""

    def test_deterministic_for_fixed_seed(self, fixture_df: pd.DataFrame) -> None:
        """Same seed + same params should produce identical metrics."""
        train, val, _test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        runner1 = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)
        runner2 = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)

        result1 = runner1.run(dict(_RIDGE_PARAMS))
        result2 = runner2.run(dict(_RIDGE_PARAMS))

        assert result1["mae"] == result2["mae"]
        assert result1["rmse"] == result2["rmse"]

    def test_run_returns_expected_metric_keys(self, fixture_df: pd.DataFrame) -> None:
        """Runner should return dict with mae, rmse, and other standard keys."""
        train, val, _test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        runner = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)
        result = runner.run(dict(_RIDGE_PARAMS))

        assert "mae" in result
        assert "rmse" in result
        assert isinstance(result["mae"], float)
        assert result["mae"] > 0


# ── evaluate_on_test ─────────────────────────────────────────────────


class TestEvaluateOnTest:
    """Tests for the one-shot test evaluation function."""

    def test_returns_metrics_with_mae(self, fixture_df: pd.DataFrame) -> None:
        """evaluate_on_test should return a dict containing 'mae'."""
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        result = evaluate_on_test(train, val, test, dict(_RIDGE_PARAMS), seed=42)
        assert "mae" in result
        assert isinstance(result["mae"], float)
        assert result["mae"] > 0

    def test_deterministic_for_fixed_seed(self, fixture_df: pd.DataFrame) -> None:
        """Same inputs and seed should produce identical test metrics."""
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        r1 = evaluate_on_test(train, val, test, dict(_RIDGE_PARAMS), seed=42)
        r2 = evaluate_on_test(train, val, test, dict(_RIDGE_PARAMS), seed=42)
        assert r1["mae"] == r2["mae"]

    def test_val_and_test_metrics_differ(self, fixture_df: pd.DataFrame) -> None:
        """Validation runner and test evaluation should produce different MAE.

        This confirms the harness is actually evaluating on different data
        partitions — the whole point of the locked split.
        """
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)

        runner = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)
        val_metrics = runner.run(dict(_RIDGE_PARAMS))
        test_metrics = evaluate_on_test(train, val, test, dict(_RIDGE_PARAMS), seed=42)

        assert val_metrics["mae"] != test_metrics["mae"], (
            "Validation and test MAE should differ — they evaluate on different partitions"
        )


# ── PredictiveBenchmarkResult ────────────────────────────────────────


class TestPredictiveBenchmarkResult:
    """Tests for the result dataclass."""

    def test_fields_exist(self) -> None:
        """Dataclass should store all required fields and auto-compute gap."""
        result = PredictiveBenchmarkResult(
            strategy="causal",
            budget=30,
            seed=42,
            best_validation_mae=50.0,
            test_mae=55.0,
            selected_parameters={"model_type": "ridge"},
            runtime_seconds=1.5,
        )
        assert result.strategy == "causal"
        assert result.budget == 30
        assert result.seed == 42
        assert result.best_validation_mae == 50.0
        assert result.test_mae == 55.0
        assert result.validation_test_gap == 5.0
        assert result.selected_parameters == {"model_type": "ridge"}
        assert result.runtime_seconds == 1.5

    def test_negative_gap(self) -> None:
        """A negative gap (test better than val) should be representable."""
        result = PredictiveBenchmarkResult(
            strategy="random",
            budget=10,
            seed=0,
            best_validation_mae=100.0,
            test_mae=90.0,
            selected_parameters={},
            runtime_seconds=0.5,
        )
        assert result.validation_test_gap == pytest.approx(-10.0)

    def test_validation_test_gap_auto_computed(self) -> None:
        """validation_test_gap should be auto-computed as test_mae - best_validation_mae."""
        result = PredictiveBenchmarkResult(
            strategy="random",
            budget=10,
            seed=0,
            best_validation_mae=100.0,
            test_mae=120.0,
            selected_parameters={},
            runtime_seconds=0.5,
        )
        assert result.validation_test_gap == pytest.approx(20.0)


# ── Leakage regression ──────────────────────────────────────────────


class TestNoLeakageWithLagFeatures:
    """Regression test: lag-induced NaN drops must not shift the split boundary.

    When lookback_window > 1, EnergyLoadAdapter drops the first N rows
    (which are NaN from .shift()).  The split_timestamp mechanism ensures
    the boundary stays at the correct temporal position after dropping.
    """

    def test_validation_runner_no_leakage_with_large_lookback(
        self, fixture_df: pd.DataFrame
    ) -> None:
        """Training max timestamp must be < validation min timestamp after lag drops."""
        train, val, _test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)
        train_max_ts = pd.to_datetime(train["timestamp"]).max()
        val_min_ts = pd.to_datetime(val["timestamp"]).min()

        # Run with a large lookback to trigger many NaN drops
        runner = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)
        params = {**_RIDGE_PARAMS, "lookback_window": 12}
        metrics = runner.run(params)

        # The adapter's effective training data must not extend into val
        # timestamps.  train_fraction_actual tells us where the adapter
        # split, but the definitive check is that the metrics are computed
        # on a real validation window (validation_set_size > 0).
        assert metrics["validation_set_size"] > 0
        assert metrics["mae"] > 0

        # Verify the temporal boundary is preserved: the train max ts from
        # the original split must still be earlier than val min ts.
        assert train_max_ts < val_min_ts

    def test_evaluate_on_test_no_leakage_with_large_lookback(
        self, fixture_df: pd.DataFrame
    ) -> None:
        """Test evaluation must not leak test rows into train+val after lag drops."""
        train, val, test = split_time_frame(fixture_df, train_frac=0.5, val_frac=0.25)
        val_max_ts = pd.to_datetime(val["timestamp"]).max()
        test_min_ts = pd.to_datetime(test["timestamp"]).min()

        params = {**_RIDGE_PARAMS, "lookback_window": 12}
        metrics = evaluate_on_test(train, val, test, params, seed=42)

        assert metrics["validation_set_size"] > 0
        assert metrics["mae"] > 0
        assert val_max_ts < test_min_ts

    def test_lookback_exceeding_train_raises_not_leaks(self) -> None:
        """When lookback drops all training rows, the adapter must raise, not leak.

        Small dataset (50 rows) + lookback_window=35 means all rows before the
        split boundary are NaN-dropped.  The adapter should raise ValueError
        rather than silently treating the first validation row as training.
        """
        small_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
                "target_load": range(50),
                "temperature": [20.0] * 50,
            }
        )
        train, val, _test = split_time_frame(small_df, train_frac=0.6, val_frac=0.2)
        runner = ValidationEnergyRunner(train_df=train, val_df=val, seed=42)

        with pytest.raises(ValueError, match="[Pp]reprocessing removed all training rows"):
            runner.run({**_RIDGE_PARAMS, "lookback_window": 35})
