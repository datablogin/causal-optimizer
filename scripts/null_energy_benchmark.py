"""Null-signal energy benchmark runner -- permuted target negative control.

Loads real energy data, permutes the target column to destroy signal,
then runs the same benchmark harness across strategies, budgets, and seeds.
Used as a negative control to verify no strategy manufactures false wins.

Usage::

    python scripts/null_energy_benchmark.py \\
        --data-path data/energy.parquet \\
        --budgets 20,40,80 \\
        --seeds 0,1,2,3,4 \\
        --strategies random,surrogate_only,causal \\
        --output null_energy_results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from causal_optimizer.benchmarks.null_predictive_energy import (
    _VALID_STRATEGIES as _LIB_VALID_STRATEGIES,
)
from causal_optimizer.benchmarks.null_predictive_energy import (
    check_null_signal,
    permute_target,
    run_null_strategy,
)
from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    load_energy_frame,
    split_time_frame,
)

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = _LIB_VALID_STRATEGIES
_PERMUTATION_SEED_OFFSET = 99999


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert nested dicts/lists to JSON-safe Python types."""
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


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``data_path``, ``area_id``, ``budgets``,
        ``seeds``, ``strategies``, and ``output``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run null-signal energy benchmark (permuted target) "
            "across strategies, budgets, and seeds."
        ),
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to local CSV or Parquet energy data file.",
    )
    parser.add_argument(
        "--area-id",
        default=None,
        help="Filter to one balancing area (optional).",
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
        "--output",
        default="null_energy_results.json",
        help="Output JSON artifact path (default: 'null_energy_results.json').",
    )
    parser.add_argument(
        "--audit-skip-rate",
        type=float,
        default=0.0,
        help="Fraction of skipped candidates to force-evaluate for calibration (default: 0.0).",
    )
    return parser.parse_args(argv)


# ── Summary table ────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    """Format a list of values as ``mean +/- std``."""
    if not values:
        return "N/A"
    arr = np.array(values)
    if len(arr) > 1:
        return f"{arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"
    return f"{arr.mean():.4f} +/- 0.0000"


def _print_summary(results: list[PredictiveBenchmarkResult]) -> None:
    """Print a compact summary table to stdout."""
    groups: dict[tuple[str, int], list[PredictiveBenchmarkResult]] = {}
    for r in results:
        key = (r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    print(f"{'Strategy':<16} {'Budget':>6}  {'Val MAE':>20}  {'Test MAE':>20}  {'Gap':>20}")
    print("-" * 90)

    for (strategy, budget), group in sorted(groups.items()):
        val_maes = [r.best_validation_mae for r in group if math.isfinite(r.best_validation_mae)]
        test_maes = [r.test_mae for r in group if math.isfinite(r.test_mae)]
        gaps = [
            r.validation_test_gap
            for r in group
            if math.isfinite(r.best_validation_mae) and math.isfinite(r.test_mae)
        ]
        print(
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt_mean_std(val_maes):>20}  "
            f"{_fmt_mean_std(test_maes):>20}  "
            f"{_fmt_mean_std(gaps):>20}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, permute target, run strategies, write results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()

    # Parse comma-separated lists
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

    for b in budgets:
        if b <= 0:
            print(
                f"ERROR: All budgets must be positive integers, got {b!r}.",
                file=sys.stderr,
            )
            sys.exit(1)

    for s in strategies:
        if s not in _VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    audit_skip_rate = args.audit_skip_rate
    if not 0.0 <= audit_skip_rate <= 1.0:
        print(
            f"ERROR: --audit-skip-rate must be in [0.0, 1.0], got {audit_skip_rate!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data from %s", args.data_path)
    df = load_energy_frame(args.data_path, area_id=args.area_id)
    logger.info("Loaded %d rows", len(df))

    # Permute target BEFORE splitting — destroys temporal signal completely
    permutation_seed = _PERMUTATION_SEED_OFFSET
    logger.info(
        "Permuting target column 'target_load' with seed=%d (destroying signal)",
        permutation_seed,
    )
    df = permute_target(df, target_col="target_load", seed=permutation_seed)

    # Split after permutation
    train_df, val_df, test_df = split_time_frame(df)
    logger.info(
        "Split: train=%d, val=%d, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Run all combinations
    results: list[PredictiveBenchmarkResult] = []
    total = len(budgets) * len(seeds) * len(strategies)
    idx = 0
    for budget in budgets:
        for seed in seeds:
            for strategy in strategies:
                idx += 1
                logger.info(
                    "[%d/%d] strategy=%s budget=%d seed=%d",
                    idx,
                    total,
                    strategy,
                    budget,
                    seed,
                )
                try:
                    result = run_null_strategy(
                        strategy,
                        budget,
                        seed,
                        train_df,
                        val_df,
                        test_df,
                        audit_skip_rate=audit_skip_rate,
                    )
                    if result is not None:
                        results.append(result)
                except Exception:
                    logger.warning(
                        "Strategy %s budget=%d seed=%d failed; skipping.",
                        strategy,
                        budget,
                        seed,
                        exc_info=True,
                    )

    # Write JSON output
    output_data = [dataclasses.asdict(r) for r in results]
    output_data = [_sanitize_for_json(d) for d in output_data]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), output_path)

    # Print summary table
    if results:
        print()
        print("=" * 90)
        print("NULL-SIGNAL BENCHMARK RESULTS (permuted target)")
        print("=" * 90)
        _print_summary(results)

        # Run null-signal check
        print()
        verdict = check_null_signal(results, strategies)
        print(f"Null-signal verdict: {verdict.verdict}")
        for detail in verdict.details:
            print(f"  - {detail}")

        if verdict.verdict != "PASS":
            sys.exit(1)
    else:
        print("WARNING: No benchmark results collected — all runs failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
