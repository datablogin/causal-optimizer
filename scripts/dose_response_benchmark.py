"""Dose-response clinical benchmark runner -- compare optimization strategies.

Runs random, surrogate-only, and causal strategies on the semi-synthetic
dose-response benchmark.  No external data dependency: patient covariates
are generated synthetically.

Usage::

    python scripts/dose_response_benchmark.py \
        --budgets 20,40,80 \
        --seeds 0,1,2,3,4 \
        --strategies random,surrogate_only,causal \
        --output dose_response_results.json
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

from causal_optimizer.benchmarks.dose_response import (
    DoseResponseBenchmarkResult,
    DoseResponseScenario,
)
from causal_optimizer.benchmarks.provenance import collect_provenance

logger = logging.getLogger(__name__)

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert nested dict/list to JSON-safe Python types."""
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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=("Run dose-response clinical benchmark across strategies, budgets, and seeds."),
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=1000,
        help="Number of patients to generate (default: 1000).",
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
        "--treatment-cost",
        type=float,
        default=15.0,
        help="Fixed cost per treatment (default: 15.0).",
    )
    parser.add_argument(
        "--output",
        default="dose_response_results.json",
        help="Output JSON artifact path (default: 'dose_response_results.json').",
    )
    return parser.parse_args(argv)


# ── Summary ──────────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    """Format a list of values as ``mean +/- std``."""
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def _print_summary(results: list[DoseResponseBenchmarkResult]) -> None:
    """Print a compact summary table to stdout."""
    groups: dict[tuple[str, int], list[DoseResponseBenchmarkResult]] = {}
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
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fail-fast: ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create scenario
    scenario = DoseResponseScenario(
        n_patients=args.n_patients,
        seed=0,  # data generation seed (fixed)
        treatment_cost=args.treatment_cost,
    )
    logger.info(
        "Dose-response scenario: %d patients, treatment_cost=%.1f",
        args.n_patients,
        args.treatment_cost,
    )

    # Run all combinations
    results: list[DoseResponseBenchmarkResult] = []
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
        "benchmark": "dose_response_clinical",
        "n_patients": args.n_patients,
        "treatment_cost": args.treatment_cost,
        "suite_runtime_seconds": suite_runtime,
        "results": [_sanitize_for_json(dataclasses.asdict(r)) for r in results],
        "provenance": collect_provenance(
            seeds=seeds,
            budgets=budgets,
            strategies=strategies,
        ),
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
