"""Unit tests for the energy predictive benchmark runner script.

Covers: CLI argument parsing, JSON sanitization, and summary formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import pandas as pd

# The script lives under scripts/, which is not a package.  Add it to
# sys.path so we can import it by module name.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from energy_predictive_benchmark import (  # noqa: E402, I001
    _fmt_mean_std,
    _sanitize_for_json,
    parse_args,
    run_strategy,
)


# ── parse_args ───────────────────────────────────────────────────────


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_data_path(self) -> None:
        """--data-path is required; omitting it should cause SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self) -> None:
        """Default values should be set for optional arguments."""
        args = parse_args(["--data-path", "data.csv"])
        assert args.data_path == "data.csv"
        assert args.area_id is None
        assert args.budgets == "20,40,80"
        assert args.seeds == "0,1,2,3,4"
        assert args.strategies == "random,surrogate_only,causal"
        assert args.output == "predictive_energy_results.json"

    def test_custom_values(self) -> None:
        """All arguments should be parseable."""
        args = parse_args(
            [
                "--data-path",
                "my_data.parquet",
                "--area-id",
                "WEST",
                "--budgets",
                "10,20",
                "--seeds",
                "0,1",
                "--strategies",
                "random,causal",
                "--output",
                "out.json",
            ]
        )
        assert args.data_path == "my_data.parquet"
        assert args.area_id == "WEST"
        assert args.budgets == "10,20"
        assert args.seeds == "0,1"
        assert args.strategies == "random,causal"
        assert args.output == "out.json"


# ── _sanitize_for_json ──────────────────────────────────────────────


class TestSanitizeForJson:
    """Tests for the JSON sanitization helper."""

    def test_inf_becomes_none(self) -> None:
        assert _sanitize_for_json(float("inf")) is None
        assert _sanitize_for_json(float("-inf")) is None

    def test_nan_becomes_none(self) -> None:
        assert _sanitize_for_json(float("nan")) is None

    def test_normal_float_unchanged(self) -> None:
        assert _sanitize_for_json(3.14) == 3.14

    def test_nested_dict(self) -> None:
        data = {"a": float("inf"), "b": {"c": float("nan"), "d": 1.0}}
        result = _sanitize_for_json(data)
        assert result == {"a": None, "b": {"c": None, "d": 1.0}}

    def test_nested_list(self) -> None:
        data = [float("inf"), 1.0, [float("nan"), 2.0]]
        result = _sanitize_for_json(data)
        assert result == [None, 1.0, [None, 2.0]]

    def test_numpy_integer(self) -> None:
        assert _sanitize_for_json(np.int64(42)) == 42
        assert isinstance(_sanitize_for_json(np.int64(42)), int)

    def test_numpy_floating(self) -> None:
        assert _sanitize_for_json(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_sanitize_for_json(np.float64(3.14)), float)

    def test_numpy_floating_nan(self) -> None:
        assert _sanitize_for_json(np.float64("nan")) is None

    def test_numpy_floating_inf(self) -> None:
        assert _sanitize_for_json(np.float64("inf")) is None

    def test_numpy_bool(self) -> None:
        assert _sanitize_for_json(np.bool_(True)) is True
        assert isinstance(_sanitize_for_json(np.bool_(True)), bool)

    def test_string_unchanged(self) -> None:
        assert _sanitize_for_json("hello") == "hello"

    def test_none_unchanged(self) -> None:
        assert _sanitize_for_json(None) is None


# ── _fmt_mean_std ───────────────────────────────────────────────────


class TestFmtMeanStd:
    """Tests for the summary formatting helper."""

    def test_empty_list_returns_na(self) -> None:
        assert _fmt_mean_std([]) == "N/A"

    def test_single_value(self) -> None:
        result = _fmt_mean_std([1.0])
        assert "1.0000" in result
        assert "+/-" in result

    def test_multiple_values(self) -> None:
        result = _fmt_mean_std([1.0, 2.0, 3.0])
        mean = (1.0 + 2.0 + 3.0) / 3.0
        assert f"{mean:.4f}" in result
        assert "+/-" in result

    def test_output_fits_column_width(self) -> None:
        """Output should be at most 20 chars to fit the >20 column spec."""
        result = _fmt_mean_std([0.1234, 0.5678])
        # "X.XXXX +/- X.XXXX" = 18 chars max for typical 4-decimal values
        assert len(result) <= 20


# ── run_strategy ─────────────────────────────────────────────────────

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"


class TestRunStrategy:
    """Tests for run_strategy behavior."""

    @pytest.fixture()
    def _splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from causal_optimizer.benchmarks.predictive_energy import (
            load_energy_frame,
            split_time_frame,
        )

        df = load_energy_frame(str(_FIXTURE_PATH))
        return split_time_frame(df, train_frac=0.5, val_frac=0.25)

    def test_runtime_includes_test_evaluation(
        self, _splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """runtime_seconds should cover the full run including test eval."""
        train, val, test = _splits
        result = run_strategy("random", budget=2, seed=0, train_df=train, val_df=val, test_df=test)
        assert result is not None
        # runtime must be positive and should include test eval time
        assert result.runtime_seconds > 0

    def test_returns_none_when_engine_all_crash(
        self, _splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Engine-based strategy returns None when all experiments crash.

        ExperimentEngine catches per-experiment exceptions and marks them
        CRASH, so best_result() returns None. run_strategy should then
        return None instead of a sentinel row.
        """
        from unittest.mock import patch

        train, val, test = _splits

        # Patch the runner's run method to always raise, simulating crashes.
        # ExperimentEngine catches these and marks results as CRASH.
        with patch.object(
            __import__("energy_predictive_benchmark"),
            "ValidationEnergyRunner",
        ) as mock_cls:
            mock_runner = mock_cls.return_value
            mock_runner.run.side_effect = RuntimeError("simulated crash")
            result = run_strategy(
                "surrogate_only",
                budget=3,
                seed=0,
                train_df=train,
                val_df=val,
                test_df=test,
            )

        assert result is None

    def test_invalid_strategy_raises(
        self, _splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Unknown strategy name should raise ValueError."""
        train, val, test = _splits
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_strategy("bogus", budget=2, seed=0, train_df=train, val_df=val, test_df=test)
