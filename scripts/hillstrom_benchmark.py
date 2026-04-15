"""Hillstrom benchmark runner — Sprint 31 first non-energy benchmark.

Runs random, surrogate-only, and causal strategies on a Hillstrom CSV
at one or more budgets and seeds, across the primary and pooled
slices, with an optional permuted-outcome null-control pass. Writes a
provenance-stamped JSON artifact.

The full Hillstrom CSV is local-only. The committed fixture
``tests/fixtures/hillstrom_fixture.csv`` is a small synthetic,
Hillstrom-shaped file intended for CI and smoke runs.

Usage::

    python scripts/hillstrom_benchmark.py \
        --data-path tests/fixtures/hillstrom_fixture.csv \
        --slices primary,pooled \
        --budgets 20,40,80 \
        --seeds 0,1,2,3,4 \
        --strategies random,surrogate_only,causal \
        --null-control \
        --output hillstrom_results.json
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

from causal_optimizer.benchmarks.hillstrom import (
    HillstromBenchmarkResult,
    HillstromScenario,
    HillstromSliceType,
    hillstrom_projected_prior_graph,
)
from causal_optimizer.benchmarks.provenance import collect_provenance

logger = logging.getLogger(__name__)

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})
_VALID_SLICES: frozenset[str] = frozenset({"primary", "pooled"})


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
        description="Run the Sprint 31 Hillstrom benchmark across strategies, budgets, and seeds.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to a Hillstrom CSV file. Must contain segment, spend, visit, "
            "conversion, and history_segment columns with the canonical MineThatData "
            "schema."
        ),
    )
    parser.add_argument(
        "--slices",
        default="primary",
        help=("Comma-separated slices to run (default: 'primary'). Valid values: primary, pooled."),
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
        "--null-control",
        action="store_true",
        help=(
            "When set, also run each strategy on a permuted-outcome copy of the "
            "primary slice at B20 and B40. Null-control seeds are taken from --seeds."
        ),
    )
    parser.add_argument(
        "--output",
        default="hillstrom_results.json",
        help="Output JSON artifact path (default: 'hillstrom_results.json').",
    )
    return parser.parse_args(argv)


def _parse_int_list(raw: str, label: str) -> list[int]:
    try:
        return [int(x.strip()) for x in raw.split(",")]
    except ValueError as exc:
        print(f"ERROR: --{label} must be comma-separated integers: {exc}", file=sys.stderr)
        sys.exit(1)


def _validate_budgets(budgets: list[int]) -> None:
    for b in budgets:
        if b <= 0:
            print(
                f"ERROR: All budgets must be positive integers, got {b!r}.",
                file=sys.stderr,
            )
            sys.exit(1)


def _validate_strategies(strategies: list[str]) -> None:
    for s in strategies:
        if s not in _VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(_VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)


def _validate_slices(slices: list[str]) -> None:
    for s in slices:
        if s not in _VALID_SLICES:
            print(
                f"ERROR: Unknown slice {s!r}. Must be one of {sorted(_VALID_SLICES)}.",
                file=sys.stderr,
            )
            sys.exit(1)


# ── Summary ──────────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def _print_summary(results: list[HillstromBenchmarkResult]) -> None:
    """Print a compact summary grouped by (slice, null_control, strategy, budget)."""
    groups: dict[tuple[str, bool, str, int], list[HillstromBenchmarkResult]] = {}
    for r in results:
        key = (r.slice_type, r.is_null_control, r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    header = (
        f"{'Slice':<10} {'Null':<5} {'Strategy':<16} {'Budget':>6}  "
        f"{'Policy Value':>24}  {'μ (baseline)':>14}"
    )
    print(header)
    print("-" * len(header))
    for (slice_type, is_null, strategy, budget), group in sorted(groups.items()):
        pvals = [r.policy_value for r in group if math.isfinite(r.policy_value)]
        mu = group[0].null_baseline if group[0].null_baseline is not None else float("nan")
        print(
            f"{slice_type:<10} "
            f"{'yes' if is_null else 'no':<5} "
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt_mean_std(pvals):>24}  "
            f"{mu:>14.4f}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def _load_raw(data_path: str) -> pd.DataFrame:
    """Load a Hillstrom CSV or Parquet file."""
    suffix = Path(data_path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def main() -> None:
    """Entry point: parse args, run scenarios, write JSON artifact."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    _validate_slices(slices)

    budgets = _parse_int_list(args.budgets, "budgets")
    _validate_budgets(budgets)

    seeds = _parse_int_list(args.seeds, "seeds")

    strategies = [s.strip() for s in args.strategies.split(",")]
    _validate_strategies(strategies)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_raw(args.data_path)
    logger.info(
        "Loaded Hillstrom frame from %s (%d rows, %d cols)",
        args.data_path,
        len(raw),
        len(raw.columns),
    )

    scenarios: dict[str, HillstromScenario] = {}
    for slice_name in slices:
        slice_type = HillstromSliceType(slice_name)
        scenarios[slice_name] = HillstromScenario(raw, slice_type=slice_type)
        logger.info(
            "Prepared %s slice: %d rows, μ=%.4f",
            slice_name,
            len(scenarios[slice_name].real_slice),
            scenarios[slice_name].null_baseline,
        )

    results: list[HillstromBenchmarkResult] = []
    # Real slice runs
    real_total = len(slices) * len(budgets) * len(seeds) * len(strategies)
    idx = 0
    t_suite_start = time.perf_counter()

    for slice_name in slices:
        scenario = scenarios[slice_name]
        for budget in budgets:
            for seed in seeds:
                for strategy in strategies:
                    idx += 1
                    logger.info(
                        "[%d/%d] slice=%s strategy=%s budget=%d seed=%d",
                        idx,
                        real_total,
                        slice_name,
                        strategy,
                        budget,
                        seed,
                    )
                    try:
                        result = scenario.run_strategy(strategy, budget=budget, seed=seed)
                        results.append(result)
                    except Exception:  # pragma: no cover - defensive
                        logger.warning(
                            "slice=%s strategy=%s budget=%d seed=%d failed; skipping.",
                            slice_name,
                            strategy,
                            budget,
                            seed,
                            exc_info=True,
                        )

    # Null-control runs (primary slice only per Sprint 31 contract)
    if args.null_control and "primary" in scenarios:
        null_budgets = [b for b in budgets if b <= 40] or [min(budgets)]
        primary = scenarios["primary"]
        null_total = len(null_budgets) * len(seeds) * len(strategies)
        null_idx = 0
        for budget in null_budgets:
            for seed in seeds:
                for strategy in strategies:
                    null_idx += 1
                    logger.info(
                        "[null %d/%d] slice=primary strategy=%s budget=%d seed=%d",
                        null_idx,
                        null_total,
                        strategy,
                        budget,
                        seed,
                    )
                    try:
                        result = primary.run_strategy(
                            strategy,
                            budget=budget,
                            seed=seed,
                            null_control=True,
                        )
                        results.append(result)
                    except Exception:  # pragma: no cover - defensive
                        logger.warning(
                            "null slice=primary strategy=%s budget=%d seed=%d failed; skipping.",
                            strategy,
                            budget,
                            seed,
                            exc_info=True,
                        )

    suite_runtime = time.perf_counter() - t_suite_start

    # Assemble JSON artifact with Hillstrom-specific provenance
    projected_graph = hillstrom_projected_prior_graph()
    hillstrom_provenance = {
        "slices": slices,
        "null_control_enabled": bool(args.null_control),
        "projected_graph_edge_count": len(projected_graph.edges),
        "projected_graph_edges": [list(edge) for edge in projected_graph.edges],
        "null_baselines": {name: scenario.null_baseline for name, scenario in scenarios.items()},
    }
    output_data = {
        "benchmark": "sprint_31_hillstrom",
        "suite_runtime_seconds": suite_runtime,
        "results": [_sanitize_for_json(dataclasses.asdict(r)) for r in results],
        "provenance": collect_provenance(
            seeds=seeds,
            budgets=budgets,
            strategies=strategies,
            dataset_path=str(args.data_path),
        )
        | {"hillstrom": hillstrom_provenance},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), output_path)

    if results:
        print()
        _print_summary(results)


if __name__ == "__main__":
    main()
