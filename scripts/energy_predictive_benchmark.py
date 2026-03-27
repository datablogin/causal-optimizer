"""Energy predictive benchmark runner — compare optimization strategies.

Runs random, surrogate-only, and causal strategies across multiple budgets
and seeds against the predictive energy harness, writing results to JSON.

Usage::

    python scripts/energy_predictive_benchmark.py \\
        --data-path data/energy.csv \\
        --budgets 20,40,80 \\
        --seeds 0,1,2,3,4 \\
        --strategies random,surrogate_only,causal \\
        --output predictive_energy_results.json
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

from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    ValidationEnergyRunner,
    evaluate_on_test,
    load_energy_frame,
    split_time_frame,
)
from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter
from causal_optimizer.engine.loop import ExperimentEngine

logger = logging.getLogger(__name__)

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert a nested dict/list to JSON-safe Python types.

    - Replaces ``float('inf')`` and ``float('nan')`` with ``None``
      (RFC 8259 has no representation for Infinity or NaN).
    - Converts numpy scalars (``np.integer``, ``np.floating``, ``np.bool_``)
      to their Python counterparts so ``json.dump`` doesn't choke.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # numpy scalar conversions (must come before float/int checks
    # since np.floating is not always a subclass of float)
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
        description="Run energy predictive benchmark across strategies, budgets, and seeds.",
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
        default="predictive_energy_results.json",
        help="Output JSON artifact path (default: 'predictive_energy_results.json').",
    )
    return parser.parse_args(argv)


# ── Strategy runner ──────────────────────────────────────────────────


def run_strategy(
    strategy: str,
    budget: int,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> PredictiveBenchmarkResult | None:
    """Run one strategy on the predictive energy benchmark.

    Args:
        strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.
        budget: Number of experiments to run.
        seed: Random seed for reproducibility.
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.

    Returns:
        A :class:`PredictiveBenchmarkResult` with validation and test metrics,
        or ``None`` if no valid result was produced (all experiments crashed).

    Raises:
        ValueError: If *strategy* is not in ``_VALID_STRATEGIES``.
    """
    if strategy not in _VALID_STRATEGIES:
        msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(_VALID_STRATEGIES)}."
        raise ValueError(msg)

    t_start = time.perf_counter()

    # Build adapter — needed by all strategies for the search space and runner.
    # Graph and descriptors are only needed for engine-based strategies.
    adapter = EnergyLoadAdapter(data=pd.concat([train_df, val_df], ignore_index=True), seed=seed)
    space = adapter.get_search_space()
    runner = ValidationEnergyRunner(train_df=train_df, val_df=val_df, seed=seed)

    skip_diag = None
    anytime = None

    if strategy == "random":
        best_mae = float("inf")
        best_params: dict[str, Any] | None = None
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            params = sample_random_params(space, rng)
            metrics = runner.run(params)
            mae = metrics.get("mae", float("inf"))
            if mae < best_mae:
                best_mae = mae
                best_params = params
    else:
        # Both "surrogate_only" and "causal" use ExperimentEngine.  The only
        # difference is that "causal" passes the prior graph while
        # "surrogate_only" passes None — the engine then relies on its RF
        # surrogate without causal focus variables.
        graph = adapter.get_prior_graph() if strategy == "causal" else None
        descriptor_names = adapter.get_descriptor_names()
        engine = ExperimentEngine(
            search_space=space,
            runner=runner,
            causal_graph=graph,
            descriptor_names=descriptor_names,
            objective_name="mae",
            minimize=True,
            seed=seed,
        )
        engine.run_loop(budget)
        best_result = engine.log.best_result("mae", minimize=True)
        if best_result is not None:
            best_mae = best_result.metrics.get("mae", float("inf"))
            best_params = best_result.parameters
        else:
            best_mae = float("inf")
            best_params = None

        # Collect skip diagnostics and anytime metrics from the engine
        skip_diag = engine.skip_diagnostics
        _DEFAULT_CHECKPOINTS = [5, 10, 20, 40, 80]
        checkpoints = [c for c in _DEFAULT_CHECKPOINTS if c <= budget] or [budget]
        if budget not in checkpoints:
            checkpoints.append(budget)
        anytime = engine.anytime_metrics(sorted(checkpoints))

    # If no valid result, skip — do not serialize a sentinel row.
    if best_params is None:
        logger.warning(
            "Strategy %r with budget=%d seed=%d produced no valid results; skipping.",
            strategy,
            budget,
            seed,
        )
        return None

    # Evaluate best config on held-out test set
    test_metrics = evaluate_on_test(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        parameters=best_params,
        seed=seed,
    )
    test_mae = test_metrics.get("mae", float("inf"))

    # Timer covers the full strategy run including test evaluation.
    runtime = time.perf_counter() - t_start

    return PredictiveBenchmarkResult(
        strategy=strategy,
        budget=budget,
        seed=seed,
        best_validation_mae=best_mae,
        test_mae=test_mae,
        selected_parameters=best_params,
        runtime_seconds=runtime,
        skip_diagnostics=skip_diag,
        anytime_metrics=anytime,
    )


# ── Summary table ────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    """Format a list of values as ``mean +/- std``."""
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def _print_summary(results: list[PredictiveBenchmarkResult]) -> None:
    """Print a compact summary table to stdout."""
    # Group by (strategy, budget)
    groups: dict[tuple[str, int], list[PredictiveBenchmarkResult]] = {}
    for r in results:
        key = (r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    # Header
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

    # Validate budget values
    for b in budgets:
        if b <= 0:
            print(
                f"ERROR: All budgets must be positive integers, got {b!r}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate strategy names
    for s in strategies:
        if s not in _VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Fail-fast: ensure output directory exists before spending time on computation
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and split data
    logger.info("Loading data from %s", args.data_path)
    df = load_energy_frame(args.data_path, area_id=args.area_id)
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
                    result = run_strategy(strategy, budget, seed, train_df, val_df, test_df)
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

    # Write JSON output — replace inf/nan with None for RFC 8259 compliance
    output_data = [dataclasses.asdict(r) for r in results]
    output_data = [_sanitize_for_json(d) for d in output_data]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), output_path)

    # Print summary table
    if results:
        print()
        _print_summary(results)


if __name__ == "__main__":
    main()
