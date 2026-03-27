"""Counterfactual energy benchmark runner -- compare optimization strategies.

Runs random, surrogate-only, and causal strategies on the semi-synthetic
demand-response benchmark using real ERCOT covariates with known treatment
effects.

Usage::

    python scripts/counterfactual_benchmark.py \\
        --data-path data/ercot_north_c_dfw_2022_2024.parquet \\
        --budgets 10,20,40 \\
        --seeds 0,1,2,3,4 \\
        --strategies random,surrogate_only,causal \\
        --output counterfactual_results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    CounterfactualBenchmarkResult,
    DemandResponseScenario,
)
from causal_optimizer.benchmarks.predictive_energy import load_energy_frame

logger = logging.getLogger(__name__)

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert nested dict/list to JSON-safe Python types.

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


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``data_path``, ``area_id``, ``budgets``,
        ``seeds``, ``strategies``, ``treatment_cost``, and ``output``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run counterfactual demand-response benchmark "
            "across strategies, budgets, and seeds."
        ),
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to local CSV or Parquet ERCOT energy data file (covariate source).",
    )
    parser.add_argument(
        "--area-id",
        default=None,
        help="Filter to one balancing area (optional).",
    )
    parser.add_argument(
        "--budgets",
        default="10,20,40",
        help="Comma-separated experiment budgets (default: '10,20,40').",
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
        "--treatment-cost",
        type=float,
        default=50.0,
        help="Fixed cost per demand-response event (default: 50.0).",
    )
    parser.add_argument(
        "--output",
        default="counterfactual_results.json",
        help="Output JSON artifact path (default: 'counterfactual_results.json').",
    )
    return parser.parse_args(argv)


# ── Covariate preparation ────────────────────────────────────────────


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare covariate DataFrame from raw ERCOT data.

    Ensures required columns exist and adds derived features
    (hour_of_day, day_of_week, is_holiday, lag features) if missing.

    Args:
        df: Raw energy data with at least ``timestamp``, ``target_load``,
            and ``temperature`` columns.

    Returns:
        DataFrame with all required covariate columns.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Add calendar features if missing
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    # Add humidity placeholder if missing
    if "humidity" not in df.columns:
        df["humidity"] = 50.0  # neutral default

    # Add lag features if missing
    if "load_lag_1h" not in df.columns:
        df["load_lag_1h"] = df["target_load"].shift(1).bfill()
    if "load_lag_24h" not in df.columns:
        df["load_lag_24h"] = df["target_load"].shift(24).bfill()

    return df


# ── Summary ──────────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    """Format a list of values as ``mean +/- std``."""
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def _print_summary(results: list[CounterfactualBenchmarkResult]) -> None:
    """Print a compact summary table to stdout."""
    groups: dict[tuple[str, int], list[CounterfactualBenchmarkResult]] = {}
    for r in results:
        key = (r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    print(
        f"{'Strategy':<16} {'Budget':>6}  "
        f"{'Policy Value':>20}  {'Regret':>20}  {'Decision Err':>20}"
    )
    print("-" * 90)

    for (strategy, budget), group in sorted(groups.items()):
        pvals = [r.policy_value for r in group if math.isfinite(r.policy_value)]
        regrets = [r.regret for r in group if math.isfinite(r.regret)]
        errs = [r.decision_error_rate for r in group if math.isfinite(r.decision_error_rate)]
        print(
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt_mean_std(pvals):>20}  "
            f"{_fmt_mean_std(regrets):>20}  "
            f"{_fmt_mean_std(errs):>20}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, run strategies, write results."""
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

    # Validate
    for b in budgets:
        if b <= 0:
            print(f"ERROR: All budgets must be positive, got {b!r}.", file=sys.stderr)
            sys.exit(1)
    for s in strategies:
        if s not in _VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. "
                f"Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fail-fast: ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and prepare covariates
    logger.info("Loading covariate data from %s", args.data_path)
    df = load_energy_frame(args.data_path, area_id=args.area_id)
    covariates = prepare_covariates(df)
    logger.info("Prepared %d covariate rows", len(covariates))

    # Create scenario
    scenario = DemandResponseScenario(
        covariates=covariates,
        seed=0,  # data generation seed (fixed for reproducibility)
        treatment_cost=args.treatment_cost,
    )

    # Run all combinations
    results: list[CounterfactualBenchmarkResult] = []
    total = len(budgets) * len(seeds) * len(strategies)
    idx = 0
    t_suite_start = time.perf_counter()

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
                    result = scenario.run_benchmark(
                        budget=budget,
                        seed=seed,
                        strategy=strategy,
                    )
                    results.append(result)
                except Exception:
                    logger.warning(
                        "Strategy %s budget=%d seed=%d failed; skipping.",
                        strategy,
                        budget,
                        seed,
                        exc_info=True,
                    )

    suite_runtime = time.perf_counter() - t_suite_start

    # Write JSON output
    output_data = {
        "benchmark": "counterfactual_demand_response",
        "data_source": str(args.data_path),
        "treatment_cost": args.treatment_cost,
        "n_covariates": len(covariates),
        "suite_runtime_seconds": suite_runtime,
        "results": [_sanitize_for_json(dataclasses.asdict(r)) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), output_path)

    # Print summary table
    if results:
        print()
        _print_summary(results)
        oracle_values = [r.oracle_value for r in results if math.isfinite(r.oracle_value)]
        print(f"\nOracle value (test set): {np.mean(oracle_values):.4f}")
        print(f"Suite runtime: {suite_runtime:.1f}s")


if __name__ == "__main__":
    main()
