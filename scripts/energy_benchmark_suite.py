"""Multi-benchmark suite runner for energy predictive benchmarks.

Runs the predictive energy benchmark across multiple datasets, aggregates
results, applies acceptance rules, and produces a combined evaluation report.

Usage::

    python scripts/energy_benchmark_suite.py \\
        --datasets data/ercot_north_c.parquet,data/ercot_coast.parquet \\
        --dataset-ids ercot_north_c,ercot_coast \\
        --budgets 20,40,80 \\
        --seeds 0,1,2,3,4 \\
        --strategies random,surrogate_only,causal \\
        --output-dir artifacts/suite_sprint16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from energy_predictive_benchmark import run_strategy

from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    load_energy_frame,
    split_time_frame,
)

logger = logging.getLogger(__name__)


# ── JSON helpers ─────────────────────────────────────────────────────


def _sanitize_for_json(obj: Any) -> Any:
    """Convert nested dicts/lists to JSON-safe Python types.

    Replaces inf/nan with None, converts numpy scalars to Python types.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ── Data models ───────────────────────────────────────────────────────


@dataclass
class StrategyStats:
    """Aggregated statistics for one strategy at one budget level.

    Attributes:
        strategy: Strategy name.
        budget: Experiment budget.
        test_mae_mean: Mean test MAE across seeds.
        test_mae_std: Std of test MAE across seeds.
        val_mae_mean: Mean validation MAE across seeds.
        val_mae_std: Std of validation MAE across seeds.
        n_seeds: Number of seeds that produced results.
    """

    strategy: str
    budget: int
    test_mae_mean: float
    test_mae_std: float
    val_mae_mean: float
    val_mae_std: float
    n_seeds: int


@dataclass
class BenchmarkSummary:
    """Summary of one benchmark (one dataset) across strategies and budgets.

    Attributes:
        dataset_id: Short identifier for the dataset.
        strategy_stats: Mapping from strategy name to StrategyStats.
            When multiple budgets exist, uses the largest budget for
            acceptance rule evaluation.
    """

    dataset_id: str
    strategy_stats: dict[str, StrategyStats]


@dataclass
class CoverageResult:
    """Result of coverage validation for the suite.

    Attributes:
        complete: True if every dataset has all expected unique combinations.
        expected_per_dataset: Expected unique combinations per dataset.
        actual_per_dataset: Actual unique combinations per dataset.
        missing: Human-readable descriptions of missing combinations.
        duplicates: Human-readable descriptions of duplicate combinations.
    """

    complete: bool
    expected_per_dataset: int
    actual_per_dataset: dict[str, int]
    missing: list[str]
    duplicates: list[str] = field(default_factory=list)


@dataclass
class AcceptanceResult:
    """Result of acceptance rule evaluation across benchmarks.

    Attributes:
        improved: At least one benchmark where causal beats baseline.
        no_regression: No material regression (>2% relative) on any benchmark.
        stable: Std across seeds < 5% of mean for all strategies/benchmarks.
        differentiated: Causal and surrogate_only produce different results.
        overall: ``"PASS"`` if all four pass; ``"CONDITIONAL"`` if
            ``no_regression`` and ``stable`` pass but ``improved`` or
            ``differentiated`` fails; ``"FAIL"`` otherwise.
        reasons: Human-readable list of reasons for the verdict.
    """

    improved: bool
    no_regression: bool
    stable: bool
    differentiated: bool
    overall: str
    reasons: list[str] = field(default_factory=list)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_suite_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse suite CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``datasets``, ``dataset_ids``, ``budgets``,
        ``seeds``, ``strategies``, and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Run energy predictive benchmark suite across multiple datasets.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated paths to local Parquet files.",
    )
    parser.add_argument(
        "--dataset-ids",
        required=True,
        help="Comma-separated short IDs matching datasets (e.g., ercot_north_c,ercot_coast).",
    )
    parser.add_argument(
        "--budgets",
        default="20,40,80",
        help="Comma-separated experiment budgets (default: '20,40,80').",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated RNG seeds (default: '0,1,2,3,4').",
    )
    parser.add_argument(
        "--strategies",
        default="random,surrogate_only,causal",
        help="Comma-separated strategies (default: 'random,surrogate_only,causal').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-benchmark and suite artifacts.",
    )
    return parser.parse_args(argv)


# ── Aggregation helpers ───────────────────────────────────────────────


def _compute_strategy_stats(
    results: list[PredictiveBenchmarkResult],
    strategy: str,
    budget: int,
) -> StrategyStats | None:
    """Compute aggregated stats for one strategy/budget from raw results.

    Returns ``None`` if no matching results exist.
    """
    filtered = [r for r in results if r.strategy == strategy and r.budget == budget]
    if not filtered:
        return None
    test_maes = [r.test_mae for r in filtered if math.isfinite(r.test_mae)]
    val_maes = [r.best_validation_mae for r in filtered if math.isfinite(r.best_validation_mae)]
    if not test_maes:
        return None
    arr_test = np.array(test_maes)
    arr_val = np.array(val_maes) if val_maes else np.array([float("nan")])
    return StrategyStats(
        strategy=strategy,
        budget=budget,
        test_mae_mean=float(arr_test.mean()),
        test_mae_std=float(arr_test.std(ddof=1)) if len(arr_test) > 1 else 0.0,
        val_mae_mean=float(arr_val.mean()),
        val_mae_std=float(arr_val.std(ddof=1)) if len(arr_val) > 1 else 0.0,
        n_seeds=len(test_maes),
    )


def build_benchmark_summary(
    dataset_id: str,
    results: list[PredictiveBenchmarkResult],
    strategies: list[str],
    budgets: list[int],
) -> BenchmarkSummary:
    """Build a summary for one dataset using the largest budget for comparison.

    Args:
        dataset_id: Short identifier for the dataset.
        results: All raw results for this dataset.
        strategies: List of strategy names.
        budgets: List of budget levels.

    Returns:
        A :class:`BenchmarkSummary` with stats for each strategy at the
        largest budget.
    """
    max_budget = max(budgets)
    stats: dict[str, StrategyStats] = {}
    for strategy in strategies:
        s = _compute_strategy_stats(results, strategy, max_budget)
        if s is not None:
            stats[strategy] = s
    return BenchmarkSummary(dataset_id=dataset_id, strategy_stats=stats)


# ── Coverage validation ───────────────────────────────────────────────


def check_coverage(
    all_results: dict[str, list[PredictiveBenchmarkResult]],
    strategies: list[str],
    budgets: list[int],
    seeds: list[int],
) -> CoverageResult:
    """Validate that every dataset has the full set of expected unique combinations.

    Compares the expected set of ``(strategy, budget, seed)`` tuples against
    the observed unique set — not raw row count — so duplicate rows cannot
    mask missing combinations.

    Args:
        all_results: Mapping from dataset_id to its raw results.
        strategies: Expected strategy names.
        budgets: Expected budget levels.
        seeds: Expected seed values.

    Returns:
        A :class:`CoverageResult` indicating whether coverage is complete.
    """
    expected_set = {
        (strategy, budget, seed) for strategy in strategies for budget in budgets for seed in seeds
    }
    expected_count = len(expected_set)
    actual: dict[str, int] = {}
    missing: list[str] = []
    duplicates: list[str] = []

    for dataset_id, results in all_results.items():
        present = {(r.strategy, r.budget, r.seed) for r in results}
        actual[dataset_id] = len(present)

        # Detect missing combinations
        for combo in sorted(expected_set - present):
            strategy, budget, seed = combo
            missing.append(f"{dataset_id}: strategy={strategy} budget={budget} seed={seed}")

        # Detect duplicates
        combo_counts = Counter((r.strategy, r.budget, r.seed) for r in results)
        for combo, count in sorted(combo_counts.items()):
            if count > 1:
                strategy, budget, seed = combo
                duplicates.append(
                    f"{dataset_id}: strategy={strategy} budget={budget} seed={seed} ({count} rows)"
                )

    complete = (
        all(count >= expected_count for count in actual.values())
        and len(actual) > 0
        and len(duplicates) == 0
    )
    return CoverageResult(
        complete=complete,
        expected_per_dataset=expected_count,
        actual_per_dataset=actual,
        missing=missing,
        duplicates=duplicates,
    )


# ── Acceptance rules ──────────────────────────────────────────────────

_REGRESSION_THRESHOLD = 0.02  # 2% relative
_STABILITY_THRESHOLD = 0.05  # 5% of mean


def check_acceptance(
    summaries: list[BenchmarkSummary],
    baseline: str = "random",
) -> AcceptanceResult:
    """Evaluate acceptance rules across benchmark summaries.

    Args:
        summaries: List of per-benchmark summaries.
        baseline: Strategy name to use as baseline for comparison.

    Returns:
        An :class:`AcceptanceResult` with per-rule verdicts and overall result.
    """
    reasons: list[str] = []

    # ── improved: causal beats baseline on at least one benchmark ──
    improved = False
    for s in summaries:
        causal = s.strategy_stats.get("causal")
        base = s.strategy_stats.get(baseline)
        if causal is not None and base is not None and causal.test_mae_mean < base.test_mae_mean:
            improved = True
            reasons.append(
                f"improved: causal ({causal.test_mae_mean:.2f}) < "
                f"{baseline} ({base.test_mae_mean:.2f}) on {s.dataset_id}"
            )
    if not improved:
        reasons.append(f"not improved: causal did not beat {baseline} on any benchmark")

    # ── no_regression: no strategy >2% worse than baseline ──
    no_regression = True
    for s in summaries:
        base = s.strategy_stats.get(baseline)
        if base is None:
            continue
        for name, stats in s.strategy_stats.items():
            if name == baseline:
                continue
            if base.test_mae_mean > 0:
                relative_diff = (stats.test_mae_mean - base.test_mae_mean) / base.test_mae_mean
                if relative_diff > _REGRESSION_THRESHOLD:
                    no_regression = False
                    reasons.append(
                        f"regression: {name} is {relative_diff:.1%} worse than "
                        f"{baseline} on {s.dataset_id}"
                    )

    # ── stable: std < 5% of mean for all strategies/benchmarks ──
    stable = True
    for s in summaries:
        for name, stats in s.strategy_stats.items():
            if stats.test_mae_mean > 0:
                cv = stats.test_mae_std / stats.test_mae_mean
                if cv > _STABILITY_THRESHOLD:
                    stable = False
                    reasons.append(
                        f"unstable: {name} on {s.dataset_id} has "
                        f"CV={cv:.1%} (>{_STABILITY_THRESHOLD:.0%})"
                    )

    # ── differentiated: causal and surrogate_only differ ──
    differentiated = False
    for s in summaries:
        causal = s.strategy_stats.get("causal")
        surr = s.strategy_stats.get("surrogate_only")
        # Consider differentiated if means differ by more than 0.1%
        if causal is not None and surr is not None and surr.test_mae_mean > 0:
            diff_pct = abs(causal.test_mae_mean - surr.test_mae_mean) / surr.test_mae_mean
            if diff_pct > 0.001:
                differentiated = True
                reasons.append(
                    f"differentiated: causal vs surrogate_only differ by "
                    f"{diff_pct:.1%} on {s.dataset_id}"
                )
    if not differentiated:
        reasons.append("not differentiated: causal and surrogate_only produce identical results")

    # ── overall verdict ──
    if no_regression and stable and improved and differentiated:
        overall = "PASS"
    elif no_regression and stable and (not differentiated or not improved):
        overall = "CONDITIONAL"
    else:
        overall = "FAIL"
    reasons.append(f"overall: {overall}")

    return AcceptanceResult(
        improved=improved,
        no_regression=no_regression,
        stable=stable,
        differentiated=differentiated,
        overall=overall,
        reasons=reasons,
    )


# ── Per-benchmark execution ──────────────────────────────────────────


def run_single_benchmark(
    data_path: str,
    dataset_id: str,
    budgets: list[int],
    seeds: list[int],
    strategies: list[str],
    output_dir: Path,
) -> list[PredictiveBenchmarkResult]:
    """Run the single-benchmark logic for one dataset.

    Loads data, splits it, runs all strategy/budget/seed combinations,
    and writes a per-benchmark JSON artifact.

    Args:
        data_path: Path to the Parquet or CSV file.
        dataset_id: Short identifier for the dataset.
        budgets: List of experiment budgets.
        seeds: List of RNG seeds.
        strategies: List of strategy names.
        output_dir: Directory to write artifacts.

    Returns:
        List of :class:`PredictiveBenchmarkResult` for this dataset.
    """
    logger.info("Loading dataset %s from %s", dataset_id, data_path)
    df = load_energy_frame(data_path)
    train_df, val_df, test_df = split_time_frame(df)
    logger.info(
        "Split %s: train=%d, val=%d, test=%d rows",
        dataset_id,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    results: list[PredictiveBenchmarkResult] = []
    total = len(budgets) * len(seeds) * len(strategies)
    idx = 0

    for budget in budgets:
        for seed in seeds:
            for strategy in strategies:
                idx += 1
                logger.info(
                    "[%s %d/%d] strategy=%s budget=%d seed=%d",
                    dataset_id,
                    idx,
                    total,
                    strategy,
                    budget,
                    seed,
                )
                try:
                    result = run_strategy(strategy, budget, seed, train_df, val_df, test_df)
                    if result is not None:
                        results.append(result)
                except Exception:
                    logger.warning(
                        "Strategy %s budget=%d seed=%d failed on %s; skipping.",
                        strategy,
                        budget,
                        seed,
                        dataset_id,
                        exc_info=True,
                    )

    # Write per-benchmark JSON
    json_path = output_dir / f"{dataset_id}_results.json"
    output_data = [dataclasses.asdict(r) for r in results]
    output_data = [_sanitize_for_json(d) for d in output_data]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), json_path)

    # Write per-benchmark summary CSV
    _write_summary_csv(results, output_dir / f"{dataset_id}_summary.csv")

    return results


def _write_summary_csv(
    results: list[PredictiveBenchmarkResult],
    path: Path,
) -> None:
    """Write a CSV summary grouped by strategy and budget."""
    groups: dict[tuple[str, int], list[PredictiveBenchmarkResult]] = {}
    for r in results:
        groups.setdefault((r.strategy, r.budget), []).append(r)

    lines = ["strategy,budget,test_mae_mean,test_mae_std,val_mae_mean,val_mae_std,n_seeds"]
    for (strategy, budget), group in sorted(groups.items()):
        test_maes = [r.test_mae for r in group if math.isfinite(r.test_mae)]
        val_maes = [r.best_validation_mae for r in group if math.isfinite(r.best_validation_mae)]
        if test_maes:
            arr_t = np.array(test_maes)
            arr_v = np.array(val_maes) if val_maes else np.array([float("nan")])
            lines.append(
                f"{strategy},{budget},"
                f"{arr_t.mean():.4f},{arr_t.std(ddof=1) if len(arr_t) > 1 else 0.0:.4f},"
                f"{arr_v.mean():.4f},{arr_v.std(ddof=1) if len(arr_v) > 1 else 0.0:.4f},"
                f"{len(test_maes)}"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote summary CSV to %s", path)


# ── Suite summary ─────────────────────────────────────────────────────


def build_suite_summary(
    all_results: dict[str, list[PredictiveBenchmarkResult]],
    strategies: list[str],
    budgets: list[int],
    seeds: list[int] | None = None,
    baseline: str = "random",
) -> dict[str, Any]:
    """Build the full suite summary dict.

    Args:
        all_results: Mapping from dataset_id to its raw results.
        strategies: List of strategy names.
        budgets: List of budget levels.
        seeds: List of seed values. When provided, coverage is validated
            and incomplete suites are marked FAIL.
        baseline: Baseline strategy name.

    Returns:
        Dict with ``per_benchmark``, ``aggregate``, ``acceptance``,
        and ``coverage`` keys.
    """
    summaries: list[BenchmarkSummary] = []
    per_benchmark: list[dict[str, Any]] = []

    for dataset_id, results in all_results.items():
        summary = build_benchmark_summary(dataset_id, results, strategies, budgets)
        summaries.append(summary)
        per_benchmark.append(
            {
                "dataset_id": dataset_id,
                "n_results": len(results),
                "strategy_stats": {
                    name: dataclasses.asdict(s) for name, s in summary.strategy_stats.items()
                },
            }
        )

    # Cross-benchmark rankings: which strategy is best on each benchmark
    rankings: dict[str, list[str]] = {}
    for s in summaries:
        ranked = sorted(
            s.strategy_stats.items(),
            key=lambda x: x[1].test_mae_mean,
        )
        rankings[s.dataset_id] = [name for name, _ in ranked]

    # Coverage validation
    coverage: CoverageResult | None = None
    if seeds is not None:
        coverage = check_coverage(all_results, strategies, budgets, seeds)

    acceptance = check_acceptance(summaries, baseline=baseline)

    # Override acceptance to FAIL if coverage is incomplete
    if coverage is not None and not coverage.complete:
        acceptance = AcceptanceResult(
            improved=acceptance.improved,
            no_regression=acceptance.no_regression,
            stable=acceptance.stable,
            differentiated=acceptance.differentiated,
            overall="FAIL",
            reasons=[
                f"incomplete coverage: expected {coverage.expected_per_dataset} "
                f"results per dataset, got {coverage.actual_per_dataset}",
                *(f"missing: {m}" for m in coverage.missing[:20]),
                *acceptance.reasons,
                "overall: FAIL (incomplete coverage overrides other rules)",
            ],
        )

    result: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "per_benchmark": per_benchmark,
        "aggregate": {
            "rankings": rankings,
            "n_benchmarks": len(summaries),
            "strategies": strategies,
            "budgets": budgets,
            "baseline": baseline,
        },
        "acceptance": dataclasses.asdict(acceptance),
    }
    if coverage is not None:
        result["coverage"] = dataclasses.asdict(coverage)
    return result


# ── Report generation ─────────────────────────────────────────────────


def _fmt(val: float, decimals: int = 2) -> str:
    """Format a float to the given number of decimal places."""
    return f"{val:.{decimals}f}"


def generate_suite_report(
    suite_summary: dict[str, Any],
    all_results: dict[str, list[PredictiveBenchmarkResult]],
    strategies: list[str],
    budgets: list[int],
) -> str:
    """Generate a Markdown suite report from the summary data.

    Args:
        suite_summary: The dict from :func:`build_suite_summary`.
        all_results: Mapping from dataset_id to raw results.
        strategies: List of strategy names.
        budgets: List of budget levels.

    Returns:
        A Markdown string.
    """
    lines: list[str] = []
    acceptance = suite_summary["acceptance"]

    lines.append("# Energy Benchmark Suite Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Date**: {suite_summary['timestamp'][:10]}")
    lines.append(f"- **Benchmarks**: {suite_summary['aggregate']['n_benchmarks']}")
    lines.append(f"- **Strategies**: {', '.join(strategies)}")
    lines.append(f"- **Budgets**: {', '.join(str(b) for b in budgets)}")
    lines.append(f"- **Baseline**: {suite_summary['aggregate']['baseline']}")
    lines.append(f"- **Overall verdict**: **{acceptance['overall']}**")
    lines.append("")

    # Coverage section
    coverage = suite_summary.get("coverage")
    if coverage is not None:
        cov_status = "COMPLETE" if coverage["complete"] else "INCOMPLETE"
        lines.append("## Coverage")
        lines.append("")
        lines.append(f"- **Status**: {cov_status}")
        lines.append(f"- **Expected per dataset**: {coverage['expected_per_dataset']}")
        for ds_id, count in coverage["actual_per_dataset"].items():
            marker = "" if count >= coverage["expected_per_dataset"] else " **MISSING**"
            lines.append(f"- **{ds_id}**: {count}{marker}")
        if coverage["missing"]:
            lines.append("")
            lines.append("Missing combinations:")
            lines.append("")
            for m in coverage["missing"][:20]:
                lines.append(f"- {m}")
            if len(coverage["missing"]) > 20:
                lines.append(f"- ... and {len(coverage['missing']) - 20} more")
        if coverage.get("duplicates"):
            lines.append("")
            lines.append("Duplicate combinations:")
            lines.append("")
            for d in coverage["duplicates"][:20]:
                lines.append(f"- {d}")
            if len(coverage["duplicates"]) > 20:
                lines.append(f"- ... and {len(coverage['duplicates']) - 20} more")
        lines.append("")

    # Per-benchmark comparison tables
    lines.append("## Per-Benchmark Results (Largest Budget)")
    lines.append("")

    for bench in suite_summary["per_benchmark"]:
        dataset_id = bench["dataset_id"]
        lines.append(f"### {dataset_id}")
        lines.append("")
        lines.append("| Strategy | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |")
        lines.append("|----------|------------------------|------------------------|-------|")
        for s_name in strategies:
            if s_name in bench["strategy_stats"]:
                ss = bench["strategy_stats"][s_name]
                lines.append(
                    f"| {s_name} "
                    f"| {_fmt(ss['test_mae_mean'])} +/- {_fmt(ss['test_mae_std'])} "
                    f"| {_fmt(ss['val_mae_mean'])} +/- {_fmt(ss['val_mae_std'])} "
                    f"| {ss['n_seeds']} |"
                )
        lines.append("")

    # Cross-benchmark rankings
    lines.append("## Cross-Benchmark Rankings")
    lines.append("")
    rankings = suite_summary["aggregate"]["rankings"]
    for dataset_id, ranked in rankings.items():
        lines.append(f"- **{dataset_id}**: {' > '.join(ranked)} (best to worst)")
    lines.append("")

    # Full results tables (all budgets)
    lines.append("## Full Results (All Budgets)")
    lines.append("")

    for dataset_id, results in all_results.items():
        lines.append(f"### {dataset_id}")
        lines.append("")
        lines.append(
            "| Strategy | Budget | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |"
        )
        lines.append(
            "|----------|--------|------------------------|------------------------|-------|"
        )
        for budget in budgets:
            for strategy in strategies:
                filtered = [r for r in results if r.strategy == strategy and r.budget == budget]
                if filtered:
                    test_maes = [r.test_mae for r in filtered if math.isfinite(r.test_mae)]
                    val_maes = [
                        r.best_validation_mae
                        for r in filtered
                        if math.isfinite(r.best_validation_mae)
                    ]
                    if test_maes:
                        t_arr = np.array(test_maes)
                        v_arr = np.array(val_maes) if val_maes else np.array([float("nan")])
                        lines.append(
                            f"| {strategy} | {budget} "
                            f"| {_fmt(float(t_arr.mean()))} +/- {_fmt(float(t_arr.std()))} "
                            f"| {_fmt(float(v_arr.mean()))} +/- {_fmt(float(v_arr.std()))} "
                            f"| {len(test_maes)} |"
                        )
        lines.append("")

    # Acceptance rules
    lines.append("## Acceptance Rules")
    lines.append("")
    lines.append("| Rule | Result |")
    lines.append("|------|--------|")
    if coverage is not None:
        lines.append(f"| coverage | {'PASS' if coverage['complete'] else 'FAIL'} |")
    lines.append(f"| improved | {'PASS' if acceptance['improved'] else 'FAIL'} |")
    lines.append(f"| no_regression | {'PASS' if acceptance['no_regression'] else 'FAIL'} |")
    lines.append(f"| stable | {'PASS' if acceptance['stable'] else 'FAIL'} |")
    lines.append(f"| differentiated | {'PASS' if acceptance['differentiated'] else 'FAIL'} |")
    lines.append(f"| **overall** | **{acceptance['overall']}** |")
    lines.append("")

    # Reasons
    lines.append("### Reasons")
    lines.append("")
    for reason in acceptance["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")

    # Key questions
    lines.append("## Key Questions")
    lines.append("")
    lines.append(
        f"1. **Did the causal differentiation change actually change results?** "
        f"{'Yes' if acceptance['differentiated'] else 'No'}"
    )
    lines.append(
        f"2. **Did causal improve on either benchmark?** "
        f"{'Yes' if acceptance['improved'] else 'No'}"
    )
    lines.append(
        f"3. **Did any strategy regress?** {'Yes' if not acceptance['no_regression'] else 'No'}"
    )
    lines.append(
        f"4. **Are results stable across seeds?** {'Yes' if acceptance['stable'] else 'No'}"
    )

    # Overall recommendation
    overall = acceptance["overall"]
    if overall == "PASS":
        recommendation = "PROMOTE"
        rec_detail = (
            "Causal strategy shows improvement on at least one benchmark with no "
            "regressions. Results are stable and differentiated. Recommend promoting "
            "the causal differentiation change."
        )
    elif overall == "CONDITIONAL":
        recommendation = "INVESTIGATE"
        rec_detail = (
            "No regressions detected but either results are not differentiated or "
            "causal did not improve. Further investigation needed to understand "
            "whether the causal graph is providing value."
        )
    else:
        recommendation = "REJECT"
        rec_detail = (
            "Either a regression was detected or results are unstable. "
            "The change should not be promoted without addressing these issues."
        )
    lines.append(f"5. **Overall recommendation**: **{recommendation}**")
    lines.append("")
    lines.append(f"> {rec_detail}")
    lines.append("")

    return "\n".join(lines)


# ── Stdout summary ────────────────────────────────────────────────────


def _print_combined_table(
    all_results: dict[str, list[PredictiveBenchmarkResult]],
    strategies: list[str],
    budgets: list[int],
) -> None:
    """Print a combined comparison table to stdout."""
    print()
    print("=" * 100)
    print("SUITE COMPARISON TABLE")
    print("=" * 100)

    for dataset_id, results in all_results.items():
        print(f"\n--- {dataset_id} ---")
        print(f"{'Strategy':<16} {'Budget':>6}  {'Test MAE':>20}  {'Val MAE':>20}  {'Seeds':>5}")
        print("-" * 75)
        for budget in budgets:
            for strategy in strategies:
                filtered = [r for r in results if r.strategy == strategy and r.budget == budget]
                if filtered:
                    test_maes = [r.test_mae for r in filtered if math.isfinite(r.test_mae)]
                    val_maes = [
                        r.best_validation_mae
                        for r in filtered
                        if math.isfinite(r.best_validation_mae)
                    ]
                    if test_maes:
                        t = np.array(test_maes)
                        v = np.array(val_maes) if val_maes else np.array([float("nan")])
                        print(
                            f"{strategy:<16} {budget:>6}  "
                            f"{t.mean():.4f} +/- {t.std():.4f}  "
                            f"{v.mean():.4f} +/- {v.std():.4f}  "
                            f"{len(test_maes):>5}"
                        )
    print()


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, run benchmarks, produce suite evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_suite_args()

    # Parse comma-separated lists
    datasets = [d.strip() for d in args.datasets.split(",")]
    dataset_ids = [d.strip() for d in args.dataset_ids.split(",")]

    if len(datasets) != len(dataset_ids):
        print(
            f"ERROR: --datasets has {len(datasets)} entries but "
            f"--dataset-ids has {len(dataset_ids)} entries. Must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        budgets = [int(b.strip()) for b in args.budgets.split(",")]
    except ValueError as exc:
        print(f"ERROR: --budgets must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    except ValueError as exc:
        print(f"ERROR: --seeds must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)
    strategies = [s.strip() for s in args.strategies.split(",")]

    # Validate budget values
    for b in budgets:
        if b <= 0:
            print(
                f"ERROR: All budgets must be positive integers, got {b!r}.",
                file=sys.stderr,
            )
            sys.exit(1)

    valid_strategies = frozenset({"random", "surrogate_only", "causal"})
    for s in strategies:
        if s not in valid_strategies:
            print(
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(valid_strategies)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fail-fast: ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each benchmark
    suite_start = time.perf_counter()
    all_results: dict[str, list[PredictiveBenchmarkResult]] = {}

    for data_path, dataset_id in zip(datasets, dataset_ids, strict=True):
        logger.info("Starting benchmark for %s (%s)", dataset_id, data_path)
        results = run_single_benchmark(
            data_path=data_path,
            dataset_id=dataset_id,
            budgets=budgets,
            seeds=seeds,
            strategies=strategies,
            output_dir=output_dir,
        )
        all_results[dataset_id] = results

    suite_runtime = time.perf_counter() - suite_start
    logger.info("Suite completed in %.1f seconds", suite_runtime)

    # Build and write suite summary
    suite_summary = build_suite_summary(all_results, strategies, budgets, seeds=seeds)
    suite_summary["suite_runtime_seconds"] = suite_runtime

    summary_path = output_dir / "suite_summary.json"
    sanitized_summary = _sanitize_for_json(suite_summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_summary, f, indent=2, allow_nan=False)
    logger.info("Wrote suite summary to %s", summary_path)

    # Print combined table
    _print_combined_table(all_results, strategies, budgets)

    # Print acceptance verdict
    acceptance = suite_summary["acceptance"]
    print(f"Acceptance verdict: {acceptance['overall']}")
    for reason in acceptance["reasons"]:
        print(f"  - {reason}")

    # Generate and write report
    report = generate_suite_report(suite_summary, all_results, strategies, budgets)
    report_path = output_dir / "suite_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Wrote suite report to %s", report_path)


if __name__ == "__main__":
    main()
