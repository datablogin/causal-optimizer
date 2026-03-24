"""Tests for the energy predictive benchmark runner script.

These tests belong to #61 and are written here temporarily for TDD
development of the benchmark runner script (#60).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# The module under test — will be imported as a module
from scripts.energy_predictive_benchmark import (
    _VALID_STRATEGIES,
    main,
    parse_args,
    run_strategy,
)

FIXTURE_PATH = str(Path(__file__).parent.parent / "fixtures" / "energy_load_fixture.csv")


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_data_path_required(self) -> None:
        with pytest.raises(SystemExit):
            parse_args([])

    def test_data_path_only(self) -> None:
        args = parse_args(["--data-path", "data.csv"])
        assert args.data_path == "data.csv"
        assert args.area_id is None
        assert args.budgets == "20,40,80"
        assert args.seeds == "0,1,2,3,4"
        assert args.strategies == "random,surrogate_only,causal"
        assert args.output == "predictive_energy_results.json"

    def test_all_args(self) -> None:
        args = parse_args([
            "--data-path", "data.csv",
            "--area-id", "ZONE_A",
            "--budgets", "10,20",
            "--seeds", "0,1",
            "--strategies", "random,causal",
            "--output", "out.json",
        ])
        assert args.data_path == "data.csv"
        assert args.area_id == "ZONE_A"
        assert args.budgets == "10,20"
        assert args.seeds == "0,1"
        assert args.strategies == "random,causal"
        assert args.output == "out.json"


class TestValidStrategies:
    """Tests for strategy validation."""

    def test_valid_strategies_set(self) -> None:
        assert _VALID_STRATEGIES == {"random", "surrogate_only", "causal"}


class TestRunStrategy:
    """Tests for run_strategy function."""

    @pytest.mark.slow
    def test_run_strategy_random_returns_result(self) -> None:
        from causal_optimizer.benchmarks.predictive_energy import (
            PredictiveBenchmarkResult,
            load_energy_frame,
            split_time_frame,
        )

        df = load_energy_frame(FIXTURE_PATH)
        train_df, val_df, test_df = split_time_frame(df)
        result = run_strategy("random", 3, 0, train_df, val_df, test_df)
        assert isinstance(result, PredictiveBenchmarkResult)
        assert result.strategy == "random"
        assert result.budget == 3
        assert result.seed == 0
        assert result.best_validation_mae > 0
        assert result.test_mae > 0
        assert result.runtime_seconds > 0

    @pytest.mark.slow
    def test_run_strategy_surrogate_only_returns_result(self) -> None:
        from causal_optimizer.benchmarks.predictive_energy import (
            PredictiveBenchmarkResult,
            load_energy_frame,
            split_time_frame,
        )

        df = load_energy_frame(FIXTURE_PATH)
        train_df, val_df, test_df = split_time_frame(df)
        result = run_strategy("surrogate_only", 3, 0, train_df, val_df, test_df)
        assert isinstance(result, PredictiveBenchmarkResult)
        assert result.strategy == "surrogate_only"

    @pytest.mark.slow
    def test_run_strategy_causal_returns_result(self) -> None:
        from causal_optimizer.benchmarks.predictive_energy import (
            PredictiveBenchmarkResult,
            load_energy_frame,
            split_time_frame,
        )

        df = load_energy_frame(FIXTURE_PATH)
        train_df, val_df, test_df = split_time_frame(df)
        result = run_strategy("causal", 3, 0, train_df, val_df, test_df)
        assert isinstance(result, PredictiveBenchmarkResult)
        assert result.strategy == "causal"

    def test_run_strategy_invalid_raises(self) -> None:
        import pandas as pd

        with pytest.raises(ValueError, match="Unknown strategy"):
            run_strategy("invalid", 3, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


class TestMain:
    """Tests for the main function end-to-end."""

    @pytest.mark.slow
    def test_main_writes_json(self, tmp_path: Path) -> None:
        output = tmp_path / "results.json"
        with patch(
            "sys.argv",
            [
                "energy_predictive_benchmark.py",
                "--data-path", FIXTURE_PATH,
                "--budgets", "3",
                "--seeds", "0",
                "--strategies", "random",
                "--output", str(output),
            ],
        ):
            main()
        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy"] == "random"
        assert data[0]["budget"] == 3
