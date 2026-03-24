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
import sys
import time
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
from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import SearchSpace, VariableType

logger = logging.getLogger(__name__)

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


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


def _sample_random_params(space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
    """Sample uniformly random parameters from a search space."""
    params: dict[str, Any] = {}
    for var in space.variables:
        if var.variable_type == VariableType.CONTINUOUS:
            lower = var.lower if var.lower is not None else -1.0
            upper = var.upper if var.upper is not None else 1.0
            params[var.name] = float(rng.uniform(lower, upper))
        elif var.variable_type == VariableType.INTEGER:
            lower = int(var.lower) if var.lower is not None else 0
            upper = int(var.upper) if var.upper is not None else 10
            params[var.name] = int(rng.integers(lower, upper + 1))
        elif var.variable_type == VariableType.BOOLEAN:
            params[var.name] = bool(rng.choice([True, False]))
        elif var.variable_type == VariableType.CATEGORICAL:
            choices = var.choices or []
            if not choices:
                msg = f"Variable {var.name!r} is CATEGORICAL but has no choices defined."
                raise ValueError(msg)
            params[var.name] = choices[int(rng.integers(0, len(choices)))]
        else:
            msg = f"Unsupported variable type {var.variable_type!r} for variable {var.name!r}."
            raise ValueError(msg)
    return params


def run_strategy(
    strategy: str,
    budget: int,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> PredictiveBenchmarkResult:
    """Run one strategy on the predictive energy benchmark.

    Args:
        strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.
        budget: Number of experiments to run.
        seed: Random seed for reproducibility.
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.

    Returns:
        A :class:`PredictiveBenchmarkResult` with validation and test metrics.

    Raises:
        ValueError: If *strategy* is not in ``_VALID_STRATEGIES``.
    """
    if strategy not in _VALID_STRATEGIES:
        msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(_VALID_STRATEGIES)}."
        raise ValueError(msg)

    t_start = time.perf_counter()

    # Build adapter components
    adapter = EnergyLoadAdapter(data=pd.concat([train_df, val_df], ignore_index=True), seed=seed)
    space = adapter.get_search_space()
    graph = adapter.get_prior_graph() if strategy == "causal" else None
    descriptor_names = adapter.get_descriptor_names()

    runner = ValidationEnergyRunner(train_df=train_df, val_df=val_df, seed=seed)

    if strategy == "random":
        best_mae = float("inf")
        best_params: dict[str, Any] | None = None
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            params = _sample_random_params(space, rng)
            metrics = runner.run(params)
            mae = metrics.get("mae", float("inf"))
            if mae < best_mae:
                best_mae = mae
                best_params = params
    else:
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

    runtime = time.perf_counter() - t_start

    # If no valid result, return sentinel values
    if best_params is None:
        logger.warning(
            "Strategy %r with budget=%d seed=%d produced no valid results; skipping test eval.",
            strategy,
            budget,
            seed,
        )
        return PredictiveBenchmarkResult(
            strategy=strategy,
            budget=budget,
            seed=seed,
            best_validation_mae=float("inf"),
            test_mae=float("inf"),
            selected_parameters={},
            runtime_seconds=runtime,
        )

    # Evaluate best config on held-out test set
    test_metrics = evaluate_on_test(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        parameters=best_params,
        seed=seed,
    )
    test_mae = test_metrics.get("mae", float("inf"))

    return PredictiveBenchmarkResult(
        strategy=strategy,
        budget=budget,
        seed=seed,
        best_validation_mae=best_mae,
        test_mae=test_mae,
        selected_parameters=best_params,
        runtime_seconds=runtime,
    )


# ── Summary table ────────────────────────────────────────────────────


def _print_summary(results: list[PredictiveBenchmarkResult]) -> None:
    """Print a compact summary table to stdout."""
    # Group by (strategy, budget)
    groups: dict[tuple[str, int], list[PredictiveBenchmarkResult]] = {}
    for r in results:
        key = (r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    # Header
    print(
        f"{'Strategy':<16} {'Budget':>6}  "
        f"{'Val MAE':>16}  {'Test MAE':>16}  {'Gap':>16}"
    )
    print("-" * 78)

    for (strategy, budget), group in sorted(groups.items()):
        val_maes = [r.best_validation_mae for r in group if r.best_validation_mae != float("inf")]
        test_maes = [r.test_mae for r in group if r.test_mae != float("inf")]
        gaps = [
            r.validation_test_gap
            for r in group
            if r.best_validation_mae != float("inf") and r.test_mae != float("inf")
        ]

        def _fmt(values: list[float]) -> str:
            if not values:
                return "N/A"
            arr = np.array(values)
            return f"{arr.mean():.4f} +/- {arr.std():.4f}"

        print(
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt(val_maes):>16}  {_fmt(test_maes):>16}  {_fmt(gaps):>16}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, run strategies, write results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()

    # Parse comma-separated lists
    budgets = [int(b.strip()) for b in args.budgets.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]

    # Validate strategy names
    for s in strategies:
        if s not in _VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. "
                f"Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)

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
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info("Wrote %d results to %s", len(results), args.output)

    # Print summary table
    if results:
        print()
        _print_summary(results)


if __name__ == "__main__":
    main()
