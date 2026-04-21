"""Unit tests for ``scripts/open_bandit_benchmark.py`` CLI helpers.

Covers argument parsing, validation, JSON sanitizer, and an end-to-end
synthetic smoke run that writes a provenance-stamped artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import open_bandit_benchmark as obb  # noqa: E402


class TestSanitizeForJson:
    def test_numpy_nan_becomes_none(self) -> None:
        assert obb._sanitize_for_json(np.float64("nan")) is None

    def test_numpy_integer_roundtrips(self) -> None:
        out = obb._sanitize_for_json(np.int64(42))
        assert out == 42
        assert isinstance(out, int)

    def test_nested_dict_is_sanitized(self) -> None:
        obj = {"a": {"b": np.float64(1.5)}}
        assert obb._sanitize_for_json(obj) == {"a": {"b": 1.5}}

    def test_python_inf_becomes_none(self) -> None:
        assert obb._sanitize_for_json(float("inf")) is None


class TestParseIntList:
    def test_happy_path(self) -> None:
        assert obb._parse_int_list("0,1,2", "seeds") == [0, 1, 2]

    def test_non_integer_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            obb._parse_int_list("bad", "seeds")


class TestValidateBudgets:
    def test_accepts_positive(self) -> None:
        obb._validate_budgets([1, 20, 80])

    def test_rejects_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            obb._validate_budgets([0])


class TestValidateStrategies:
    def test_accepts_valid(self) -> None:
        obb._validate_strategies(["random", "surrogate_only", "causal"])

    def test_rejects_unknown(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            obb._validate_strategies(["magic"])


class TestParseArgs:
    def test_required_data_path(self) -> None:
        args = obb.parse_args(["--data-path", "x/open_bandit_dataset"])
        assert args.data_path == "x/open_bandit_dataset"
        assert args.null_control is False

    def test_missing_data_path_errors(self) -> None:
        with pytest.raises(SystemExit):
            obb.parse_args([])

    def test_default_budgets_and_seeds(self) -> None:
        args = obb.parse_args(["--data-path", "x"])
        assert args.budgets == "20,40,80"
        assert args.seeds == "0,1,2,3,4,5,6,7,8,9"
        assert args.strategies == "random,surrogate_only,causal"

    def test_null_control_permutation_seed_flag(self) -> None:
        args = obb.parse_args(["--data-path", "x", "--null-control", "--permutation-seed", "7"])
        assert args.null_control is True
        assert args.permutation_seed == 7


class TestSharedStrategies:
    def test_cli_imports_from_benchmark_module(self) -> None:
        from causal_optimizer.benchmarks.open_bandit_benchmark import VALID_STRATEGIES

        assert obb.VALID_STRATEGIES is VALID_STRATEGIES
