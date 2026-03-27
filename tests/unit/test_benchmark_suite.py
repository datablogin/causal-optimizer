"""Unit tests for the multi-benchmark suite runner.

Covers: acceptance rules, suite summary schema, and CLI argument parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The script lives under scripts/, which is not a package.  Add it to
# sys.path so we can import it by module name.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from energy_benchmark_suite import (  # noqa: E402, I001
    BenchmarkSummary,
    StrategyStats,
    check_acceptance,
    parse_suite_args,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_stats(mean: float, std: float) -> StrategyStats:
    """Build a StrategyStats with the given test_mae mean and std."""
    return StrategyStats(
        strategy="test",
        budget=80,
        test_mae_mean=mean,
        test_mae_std=std,
        val_mae_mean=mean - 10.0,
        val_mae_std=std,
        n_seeds=5,
    )


def _make_summary(
    dataset_id: str,
    stats: dict[str, StrategyStats],
) -> BenchmarkSummary:
    """Build a BenchmarkSummary from a dict of strategy_name -> StrategyStats."""
    return BenchmarkSummary(dataset_id=dataset_id, strategy_stats=stats)


# ── parse_suite_args ──────────────────────────────────────────────────


class TestParseSuiteArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self) -> None:
        """--datasets, --dataset-ids, and --output-dir are required."""
        with pytest.raises(SystemExit):
            parse_suite_args([])

    def test_defaults(self) -> None:
        args = parse_suite_args(
            [
                "--datasets",
                "a.parquet,b.parquet",
                "--dataset-ids",
                "a,b",
                "--output-dir",
                "/tmp/out",
            ]
        )
        assert args.datasets == "a.parquet,b.parquet"
        assert args.dataset_ids == "a,b"
        assert args.budgets == "20,40,80"
        assert args.seeds == "0,1,2,3,4"
        assert args.strategies == "random,surrogate_only,causal"
        assert args.output_dir == "/tmp/out"


# ── Acceptance rules ──────────────────────────────────────────────────


class TestAcceptanceRulesPass:
    """Causal improves on one benchmark, no regression on other → PASS."""

    def test_all_rules_pass(self) -> None:
        # Benchmark A: causal is best (lower MAE)
        summary_a = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=500.0, std=10.0),
                "surrogate_only": _make_stats(mean=490.0, std=10.0),
                "causal": _make_stats(mean=470.0, std=10.0),
            },
        )
        # Benchmark B: causal is similar (within 2% of best)
        summary_b = _make_summary(
            "bench_b",
            {
                "random": _make_stats(mean=600.0, std=12.0),
                "surrogate_only": _make_stats(mean=595.0, std=12.0),
                "causal": _make_stats(mean=598.0, std=12.0),
            },
        )

        result = check_acceptance([summary_a, summary_b], baseline="random")

        assert result.improved is True
        assert result.no_regression is True
        assert result.stable is True
        assert result.differentiated is True
        assert result.overall == "PASS"


class TestAcceptanceRulesFailRegression:
    """Causal improves one benchmark but regresses other → FAIL."""

    def test_regression_detected(self) -> None:
        # Benchmark A: causal is best
        summary_a = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=500.0, std=10.0),
                "surrogate_only": _make_stats(mean=490.0, std=10.0),
                "causal": _make_stats(mean=470.0, std=10.0),
            },
        )
        # Benchmark B: causal regresses badly (>2% worse than baseline)
        summary_b = _make_summary(
            "bench_b",
            {
                "random": _make_stats(mean=600.0, std=12.0),
                "surrogate_only": _make_stats(mean=595.0, std=12.0),
                "causal": _make_stats(mean=650.0, std=12.0),  # 8.3% worse
            },
        )

        result = check_acceptance([summary_a, summary_b], baseline="random")

        assert result.improved is True
        assert result.no_regression is False
        assert result.overall == "FAIL"


class TestAcceptanceRulesConditional:
    """Strategies produce identical results → CONDITIONAL."""

    def test_undifferentiated(self) -> None:
        # Both benchmarks: all strategies identical
        summary_a = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=500.0, std=10.0),
                "surrogate_only": _make_stats(mean=500.0, std=10.0),
                "causal": _make_stats(mean=500.0, std=10.0),
            },
        )
        summary_b = _make_summary(
            "bench_b",
            {
                "random": _make_stats(mean=600.0, std=12.0),
                "surrogate_only": _make_stats(mean=600.0, std=12.0),
                "causal": _make_stats(mean=600.0, std=12.0),
            },
        )

        result = check_acceptance([summary_a, summary_b], baseline="random")

        assert result.differentiated is False
        assert result.overall == "CONDITIONAL"


class TestAcceptanceRulesConditionalNotImproved:
    """No regression/instability but causal never beats baseline -> CONDITIONAL."""

    def test_not_improved_but_differentiated(self) -> None:
        summary = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=500.0, std=10.0),
                "surrogate_only": _make_stats(mean=505.0, std=10.0),
                "causal": _make_stats(mean=502.0, std=10.0),  # >0.1% diff from surr
            },
        )
        result = check_acceptance([summary], baseline="random")

        assert result.improved is False
        assert result.differentiated is True
        assert result.no_regression is True
        assert result.overall == "CONDITIONAL"


class TestSuiteSummarySchema:
    """Verify the AcceptanceResult contains all required fields."""

    def test_schema_fields(self) -> None:
        summary = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=500.0, std=10.0),
                "causal": _make_stats(mean=490.0, std=9.0),
            },
        )
        result = check_acceptance([summary], baseline="random")

        # Required boolean fields
        assert isinstance(result.improved, bool)
        assert isinstance(result.no_regression, bool)
        assert isinstance(result.stable, bool)
        assert isinstance(result.differentiated, bool)

        # Overall verdict
        assert result.overall in {"PASS", "CONDITIONAL", "FAIL"}

        # Reasons list
        assert isinstance(result.reasons, list)
        assert all(isinstance(r, str) for r in result.reasons)
