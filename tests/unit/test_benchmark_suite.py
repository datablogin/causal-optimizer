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
    build_suite_summary,
    check_acceptance,
    check_coverage,
    parse_suite_args,
)

from causal_optimizer.benchmarks.predictive_energy import (  # noqa: E402
    PredictiveBenchmarkResult,
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


class TestAcceptanceRulesFailInstability:
    """High variance across seeds → FAIL."""

    def test_unstable_detected(self) -> None:
        summary = _make_summary(
            "bench_a",
            {
                "random": _make_stats(mean=100.0, std=2.0),
                "surrogate_only": _make_stats(mean=105.0, std=2.0),
                # CV = 8/99 ≈ 8.1% > 5% threshold
                "causal": _make_stats(mean=99.0, std=8.0),
            },
        )
        result = check_acceptance([summary], baseline="random")

        assert result.stable is False
        assert result.overall == "FAIL"
        assert any("unstable" in r for r in result.reasons)


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


class TestIncompleteCoverage:
    """Incomplete suite coverage must produce FAIL, not CONDITIONAL."""

    def _make_result(self, strategy: str, budget: int, seed: int) -> PredictiveBenchmarkResult:
        """Build a minimal PredictiveBenchmarkResult stub."""
        return PredictiveBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            best_validation_mae=90.0,
            test_mae=105.0,
            selected_parameters={"model_type": "ridge"},
            runtime_seconds=1.0,
        )

    def test_incomplete_coverage_fails(self) -> None:
        """Suite with missing strategies/seeds must be FAIL."""
        # Only one random result, no causal or surrogate_only
        results = [self._make_result("random", 20, 0)]
        all_results = {"bench_a": results}

        coverage = check_coverage(
            all_results,
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert coverage.complete is False
        assert coverage.expected_per_dataset == 6  # 3 strategies * 1 budget * 2 seeds
        assert coverage.actual_per_dataset["bench_a"] == 1
        assert len(coverage.missing) == 5  # 6 expected - 1 actual

    def test_incomplete_coverage_overrides_acceptance(self) -> None:
        """build_suite_summary must override to FAIL on incomplete coverage."""
        results = [self._make_result("random", 20, 0)]
        all_results = {"bench_a": results}

        summary = build_suite_summary(
            all_results,
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert summary["acceptance"]["overall"] == "FAIL"
        assert any("incomplete coverage" in r for r in summary["acceptance"]["reasons"])
        assert summary["coverage"]["complete"] is False

    def test_complete_coverage_does_not_override(self) -> None:
        """Full coverage should not inject a coverage failure."""
        results = [
            self._make_result(s, 20, seed)
            for s in ["random", "surrogate_only", "causal"]
            for seed in [0, 1]
        ]
        all_results = {"bench_a": results}

        summary = build_suite_summary(
            all_results,
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert summary["coverage"]["complete"] is True
        assert summary["acceptance"]["overall"] != "FAIL" or not any(
            "incomplete coverage" in r for r in summary["acceptance"]["reasons"]
        )

    def test_duplicates_do_not_mask_missing_combinations(self) -> None:
        """Row count matching expected but with duplicates must still FAIL."""
        # 6 rows total (matches 3 strategies * 1 budget * 2 seeds = 6)
        # but only 2 unique combos, each repeated 3 times
        results = [
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 1),
            self._make_result("random", 20, 1),
            self._make_result("random", 20, 1),
        ]
        all_results = {"bench_a": results}

        coverage = check_coverage(
            all_results,
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert coverage.complete is False
        assert coverage.actual_per_dataset["bench_a"] == 2  # unique combos
        assert len(coverage.missing) == 4  # surrogate_only*2 + causal*2
        assert len(coverage.duplicates) == 2  # random/20/0 and random/20/1

    def test_duplicates_surfaced_in_suite_summary(self) -> None:
        """build_suite_summary must report FAIL with duplicate details."""
        results = [
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 0),
            self._make_result("random", 20, 1),
            self._make_result("random", 20, 1),
            self._make_result("random", 20, 1),
        ]
        all_results = {"bench_a": results}

        summary = build_suite_summary(
            all_results,
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert summary["acceptance"]["overall"] == "FAIL"
        assert summary["coverage"]["complete"] is False
        assert len(summary["coverage"]["duplicates"]) == 2

    def test_all_combos_present_but_duplicated_still_fails(self) -> None:
        """All expected unique combos present + a duplicate must FAIL."""
        # All 6 expected combos present, plus one extra duplicate
        results = [
            self._make_result(s, 20, seed)
            for s in ["random", "surrogate_only", "causal"]
            for seed in [0, 1]
        ]
        results.append(self._make_result("causal", 20, 0))  # duplicate

        coverage = check_coverage(
            {"bench_a": results},
            strategies=["random", "surrogate_only", "causal"],
            budgets=[20],
            seeds=[0, 1],
        )

        assert coverage.complete is False
        assert len(coverage.missing) == 0
        assert len(coverage.duplicates) == 1
        assert "causal" in coverage.duplicates[0]
