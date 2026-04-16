"""Unit tests for ``scripts/hillstrom_benchmark.py`` CLI helpers.

Covers the pure-logic pieces of the Hillstrom benchmark runner:
argument parsing, validation helpers, and the JSON sanitizer. The
end-to-end CLI run is exercised indirectly by the integration tests
and by a fixture-backed smoke run; these unit tests lock the helpers'
behavior so future edits do not silently drift.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/hillstrom_benchmark.py is not a package module; resolve it
# the same way tests/integration/conftest.py resolves
# scripts/energy_predictive_benchmark.py.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import hillstrom_benchmark as hb  # noqa: E402


class TestSanitizeForJson:
    """``_sanitize_for_json`` must return JSON-safe Python types."""

    def test_passes_through_plain_dict(self) -> None:
        obj = {"a": 1, "b": "two"}
        assert hb._sanitize_for_json(obj) == obj

    def test_recurses_into_nested_dict(self) -> None:
        obj = {"outer": {"inner": 1.5}}
        assert hb._sanitize_for_json(obj) == {"outer": {"inner": 1.5}}

    def test_recurses_into_list(self) -> None:
        assert hb._sanitize_for_json([1, 2, 3]) == [1, 2, 3]

    def test_recurses_into_tuple_returns_list(self) -> None:
        assert hb._sanitize_for_json((1, 2)) == [1, 2]

    def test_numpy_integer_becomes_python_int(self) -> None:
        out = hb._sanitize_for_json(np.int64(42))
        assert out == 42
        assert isinstance(out, int)
        assert not isinstance(out, np.integer)

    def test_numpy_float_becomes_python_float(self) -> None:
        out = hb._sanitize_for_json(np.float64(1.5))
        assert out == 1.5
        assert isinstance(out, float)
        assert not isinstance(out, np.floating)

    def test_numpy_bool_becomes_python_bool(self) -> None:
        out = hb._sanitize_for_json(np.bool_(True))
        assert out is True

    def test_python_nan_becomes_none(self) -> None:
        assert hb._sanitize_for_json(float("nan")) is None

    def test_python_inf_becomes_none(self) -> None:
        assert hb._sanitize_for_json(float("inf")) is None
        assert hb._sanitize_for_json(float("-inf")) is None

    def test_numpy_nan_becomes_none(self) -> None:
        assert hb._sanitize_for_json(np.float64("nan")) is None

    def test_nested_nan_in_dict_is_sanitized(self) -> None:
        out = hb._sanitize_for_json({"loss": float("nan"), "acc": 0.9})
        assert out == {"loss": None, "acc": 0.9}

    def test_hillstrom_provenance_nan_baseline_is_sanitized(self) -> None:
        """Regression: an empty primary slice produces null_baseline=nan.

        If the provenance dict is not passed through _sanitize_for_json
        before json.dump(..., allow_nan=False), the write crashes. This
        test pins the fix by building the exact shape of the provenance
        dict with a nan baseline and verifying the sanitizer converts it
        to None.
        """
        import json

        provenance = hb._sanitize_for_json(
            {
                "slices": ["primary"],
                "null_control_enabled": False,
                "projected_graph_edge_count": 7,
                "projected_graph_edges": [["a", "b"]],
                "null_baselines": {"primary": float("nan")},
            }
        )
        assert provenance["null_baselines"]["primary"] is None
        # Must serialize without error under allow_nan=False
        json.dumps(provenance, allow_nan=False)


class TestParseIntList:
    """``_parse_int_list`` converts comma-separated args into int lists."""

    def test_happy_path(self) -> None:
        assert hb._parse_int_list("0,1,2", "seeds") == [0, 1, 2]

    def test_strips_whitespace(self) -> None:
        assert hb._parse_int_list("0, 1 , 2", "seeds") == [0, 1, 2]

    def test_single_value(self) -> None:
        assert hb._parse_int_list("42", "seeds") == [42]

    def test_non_integer_exits_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            hb._parse_int_list("not-a-number", "seeds")
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "--seeds must be comma-separated integers" in captured.err


class TestValidateBudgets:
    def test_accepts_positive_budgets(self) -> None:
        hb._validate_budgets([1, 20, 80])  # must not raise

    def test_rejects_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            hb._validate_budgets([0])
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "must be positive integers" in captured.err

    def test_rejects_negative(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            hb._validate_budgets([-5])
        captured = capsys.readouterr()
        assert "must be positive integers" in captured.err


class TestValidateStrategies:
    def test_accepts_all_valid(self) -> None:
        hb._validate_strategies(["random", "surrogate_only", "causal"])

    def test_rejects_unknown(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            hb._validate_strategies(["magic"])
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Unknown strategy 'magic'" in captured.err

    def test_error_message_lists_valid_strategies(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            hb._validate_strategies(["bad"])
        captured = capsys.readouterr()
        # Must mention all three valid strategies from the shared
        # VALID_STRATEGIES frozenset, not a local copy.
        assert "random" in captured.err
        assert "surrogate_only" in captured.err
        assert "causal" in captured.err


class TestValidateSlices:
    def test_accepts_primary_and_pooled(self) -> None:
        hb._validate_slices(["primary"])
        hb._validate_slices(["pooled"])
        hb._validate_slices(["primary", "pooled"])

    def test_rejects_unknown_slice(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            hb._validate_slices(["sideways"])
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Unknown slice 'sideways'" in captured.err


class TestParseArgs:
    def test_required_data_path(self) -> None:
        args = hb.parse_args(["--data-path", "x.csv"])
        assert args.data_path == "x.csv"
        assert args.slices == "primary"
        assert args.budgets == "20,40,80"
        assert args.seeds == "0,1,2,3,4"
        assert args.strategies == "random,surrogate_only,causal"
        assert args.null_control is False
        assert args.output == "hillstrom_results.json"

    def test_null_control_flag_enables(self) -> None:
        args = hb.parse_args(["--data-path", "x.csv", "--null-control"])
        assert args.null_control is True

    def test_missing_data_path_errors(self) -> None:
        with pytest.raises(SystemExit):
            hb.parse_args([])


class TestSharedStrategiesSourceOfTruth:
    """Regression guard: the CLI's valid strategies must equal the
    benchmark module's public ``VALID_STRATEGIES`` frozenset. The two
    used to be duplicated; a silent divergence was the motivation for
    consolidation."""

    def test_cli_imports_from_benchmark_module(self) -> None:
        from causal_optimizer.benchmarks.hillstrom import VALID_STRATEGIES

        assert hb.VALID_STRATEGIES is VALID_STRATEGIES


class TestMainEndToEnd:
    """End-to-end CLI smoke test: exercises ``main()`` on the fixture.

    The individual CLI helpers are covered by the tests above; this test
    covers the I/O orchestration path that stitches them together
    (scenario construction, strategy loop, JSON write). A ``random``
    strategy with budget 3 and seeds 0 keeps the runtime well under a
    second so it is safe to include in the fast suite.
    """

    def test_main_writes_json_artifact_with_results_and_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "hillstrom_fixture.csv"
        output_path = tmp_path / "smoke.json"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hillstrom_benchmark.py",
                "--data-path",
                str(fixture_path),
                "--slices",
                "primary",
                "--budgets",
                "3",
                "--seeds",
                "0",
                "--strategies",
                "random",
                "--output",
                str(output_path),
            ],
        )

        hb.main()

        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["benchmark"] == "sprint_31_hillstrom"
        assert len(payload["results"]) == 1
        result = payload["results"][0]
        assert result["strategy"] == "random"
        assert result["slice_type"] == "primary"
        assert result["is_null_control"] is False
        assert result["budget"] == 3
        assert result["seed"] == 0
        # Provenance must include Hillstrom-specific fields
        assert "hillstrom" in payload["provenance"]
        hillstrom_prov = payload["provenance"]["hillstrom"]
        assert hillstrom_prov["projected_graph_edge_count"] == 7
        assert "primary" in hillstrom_prov["null_baselines"]
        assert hillstrom_prov["null_control_enabled"] is False


class TestLoadRaw:
    """``_load_raw`` dispatches on file extension."""

    def test_csv_path(self, tmp_path: Path) -> None:
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)
        df = hb._load_raw(str(csv_path))
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_parquet_path(self, tmp_path: Path) -> None:
        """The ``.parquet`` branch must route to ``pd.read_parquet``.

        We monkey-patch ``pd.read_parquet`` rather than writing a real
        parquet file because parquet support may not be installed in
        every test environment (it requires ``pyarrow`` or ``fastparquet``).
        """
        import pandas as pd

        fake_frame = pd.DataFrame({"segment": ["Womens E-Mail"], "spend": [10.0]})
        original = pd.read_parquet
        calls: list[str] = []

        def fake_read_parquet(path: str) -> pd.DataFrame:
            calls.append(str(path))
            return fake_frame

        pd.read_parquet = fake_read_parquet
        try:
            df = hb._load_raw(str(tmp_path / "data.parquet"))
        finally:
            pd.read_parquet = original

        assert calls == [str(tmp_path / "data.parquet")]
        assert list(df.columns) == ["segment", "spend"]


class TestFmtMeanStd:
    """Summary-table formatting helper."""

    def test_empty_list_returns_na(self) -> None:
        assert hb._fmt_mean_std([]) == "N/A"

    def test_single_value(self) -> None:
        out = hb._fmt_mean_std([1.0])
        assert "1.0000" in out
        assert "+/-" in out

    def test_multiple_values(self) -> None:
        out = hb._fmt_mean_std([1.0, 2.0, 3.0])
        # mean = 2.0, std = ~0.8165
        assert "2.0000" in out

    def test_handles_nonfinite(self) -> None:
        # Infinite values are not filtered by _fmt_mean_std itself (the
        # filtering happens in _print_summary). Sanity check that they
        # do not crash the helper.
        out = hb._fmt_mean_std([1.0, float("inf")])
        assert isinstance(out, str)
        # Output contains either a number or 'inf'; the test only
        # asserts no exception.
        assert out  # not empty
        assert math.isinf(float("inf"))  # sanity anchor
