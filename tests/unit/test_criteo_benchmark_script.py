"""Unit tests for ``scripts/criteo_benchmark.py`` CLI helpers.

Covers argument parsing, validation, JSON sanitizer, and end-to-end
smoke run on the CI fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import criteo_benchmark as cb  # noqa: E402


class TestSanitizeForJson:
    def test_numpy_nan_becomes_none(self) -> None:
        assert cb._sanitize_for_json(np.float64("nan")) is None

    def test_numpy_integer(self) -> None:
        out = cb._sanitize_for_json(np.int64(42))
        assert out == 42
        assert isinstance(out, int)

    def test_nested_dict(self) -> None:
        obj = {"a": {"b": np.float64(1.5)}}
        assert cb._sanitize_for_json(obj) == {"a": {"b": 1.5}}

    def test_python_inf_becomes_none(self) -> None:
        assert cb._sanitize_for_json(float("inf")) is None


class TestParseIntList:
    def test_happy_path(self) -> None:
        assert cb._parse_int_list("0,1,2", "seeds") == [0, 1, 2]

    def test_non_integer_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cb._parse_int_list("bad", "seeds")


class TestValidateBudgets:
    def test_accepts_positive(self) -> None:
        cb._validate_budgets([1, 20, 80])

    def test_rejects_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cb._validate_budgets([0])


class TestValidateStrategies:
    def test_accepts_valid(self) -> None:
        cb._validate_strategies(["random", "surrogate_only", "causal"])

    def test_rejects_unknown(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            cb._validate_strategies(["magic"])


class TestParseArgs:
    def test_required_data_path(self) -> None:
        args = cb.parse_args(["--data-path", "x.csv.gz"])
        assert args.data_path == "x.csv.gz"
        assert args.subsample is None
        assert args.null_control is False
        assert args.skip_propensity_gate is False

    def test_subsample_flag(self) -> None:
        args = cb.parse_args(["--data-path", "x.csv", "--subsample", "1000000"])
        assert args.subsample == 1000000

    def test_missing_data_path_errors(self) -> None:
        with pytest.raises(SystemExit):
            cb.parse_args([])


class TestSharedStrategies:
    def test_cli_imports_from_benchmark_module(self) -> None:
        from causal_optimizer.benchmarks.criteo import VALID_STRATEGIES

        assert cb.VALID_STRATEGIES is VALID_STRATEGIES


class TestMainEndToEnd:
    def test_main_writes_json_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        fixture_path = (
            Path(__file__).resolve().parent.parent / "fixtures" / "criteo_uplift_fixture.csv"
        )
        output_path = tmp_path / "smoke.json"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "criteo_benchmark.py",
                "--data-path",
                str(fixture_path),
                "--budgets",
                "3",
                "--seeds",
                "0",
                "--strategies",
                "random",
                "--skip-propensity-gate",
                "--output",
                str(output_path),
            ],
        )

        cb.main()

        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["benchmark"] == "sprint_33_criteo"
        assert len(payload["results"]) == 1
        result = payload["results"][0]
        assert result["strategy"] == "random"
        assert result["is_null_control"] is False
        assert result["budget"] == 3
        assert "criteo" in payload["provenance"]
        criteo_prov = payload["provenance"]["criteo"]
        assert criteo_prov["projected_graph_edge_count"] == 5
        assert criteo_prov["dataset_version"] == "v2.1"


class TestFmtMeanStd:
    def test_empty_list(self) -> None:
        assert cb._fmt_mean_std([]) == "N/A"

    def test_single_value(self) -> None:
        out = cb._fmt_mean_std([1.0])
        assert "1.000000" in out
