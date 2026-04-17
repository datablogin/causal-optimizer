"""Criteo uplift benchmark runner — Sprint 33 first large-scale marketing benchmark.

Runs random, surrogate-only, and causal strategies on a Criteo subsample
at one or more budgets and seeds, with an optional permuted-outcome
null-control pass. Writes a provenance-stamped JSON artifact.

The full Criteo dataset is local-only (~297 MB gzip). The committed
fixture ``tests/fixtures/criteo_uplift_fixture.csv`` is a 3,000-row
subsample for CI smoke tests.

Usage::

    python scripts/criteo_benchmark.py \
        --data-path /path/to/criteo-research-uplift-v2.1.csv.gz \
        --subsample 1000000 \
        --budgets 20,40,80 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --strategies random,surrogate_only,causal \
        --null-control \
        --output criteo_results.json
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

from causal_optimizer.benchmarks.criteo import (
    CRITEO_FROZEN_PARAMS,
    CRITEO_PROPENSITY,
    CRITEO_SAMPLE_SEED,
    VALID_STRATEGIES,
    CriteoBenchmarkResult,
    CriteoScenario,
    criteo_projected_prior_graph,
    load_criteo_subsample,
    run_propensity_gate,
)
from causal_optimizer.benchmarks.provenance import collect_provenance

logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(
        description="Run the Sprint 33 Criteo benchmark across strategies, budgets, and seeds.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to a Criteo CSV (.csv, .csv.gz) or Parquet file.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help=(
            "Number of rows to subsample from the full dataset. "
            "If not set, uses the full dataset (or fixture) as-is."
        ),
    )
    parser.add_argument(
        "--budgets",
        default="20,40,80",
        help="Comma-separated experiment budgets (default: '20,40,80').",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated RNG seeds (default: '0,1,2,3,4,5,6,7,8,9').",
    )
    parser.add_argument(
        "--strategies",
        default="random,surrogate_only,causal",
        help="Comma-separated strategies (default: 'random,surrogate_only,causal').",
    )
    parser.add_argument(
        "--null-control",
        action="store_true",
        help="Also run each strategy on permuted-outcome data (B20 and B40 only).",
    )
    parser.add_argument(
        "--synthesize-segment",
        action="store_true",
        help=(
            "Synthesize segment from f0 tertiles (Run 2 heterogeneous surface). "
            "When not set, segment is omitted (Run 1 degenerate surface)."
        ),
    )
    parser.add_argument(
        "--skip-propensity-gate",
        action="store_true",
        help="Skip the propensity heterogeneity gate (for CI fixture runs).",
    )
    parser.add_argument(
        "--output",
        default="criteo_results.json",
        help="Output JSON artifact path (default: 'criteo_results.json').",
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
        if s not in VALID_STRATEGIES:
            print(
                f"ERROR: Unknown strategy {s!r}. Must be one of {sorted(VALID_STRATEGIES)}.",
                file=sys.stderr,
            )
            sys.exit(1)


# ── Summary ──────────────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.6f} +/- {arr.std():.6f}"


def _print_summary(results: list[CriteoBenchmarkResult]) -> None:
    groups: dict[tuple[bool, str, int], list[CriteoBenchmarkResult]] = {}
    for r in results:
        key = (r.is_null_control, r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    header = (
        f"{'Null':<5} {'Strategy':<16} {'Budget':>6}  {'Policy Value':>28}  {'μ (baseline)':>14}"
    )
    print(header)
    print("-" * len(header))
    for (is_null, strategy, budget), group in sorted(groups.items()):
        pvals = [r.policy_value for r in group if math.isfinite(r.policy_value)]
        mu = group[0].null_baseline if group[0].null_baseline is not None else float("nan")
        print(
            f"{'yes' if is_null else 'no':<5} "
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt_mean_std(pvals):>28}  "
            f"{mu:>14.6f}"
        )


# ── Main ─────────────────────────────────────────────────────────────


def _load_raw(data_path: str) -> pd.DataFrame:
    suffix = Path(data_path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(data_path)
    # Handle .csv.gz and .csv
    if data_path.endswith(".csv.gz"):
        return pd.read_csv(data_path, compression="gzip")
    return pd.read_csv(data_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    budgets = _parse_int_list(args.budgets, "budgets")
    _validate_budgets(budgets)

    seeds = _parse_int_list(args.seeds, "seeds")

    strategies = [s.strip() for s in args.strategies.split(",")]
    _validate_strategies(strategies)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_raw(args.data_path)
    logger.info(
        "Loaded Criteo frame from %s (%d rows, %d cols)",
        args.data_path,
        len(raw),
        len(raw.columns),
    )

    if args.subsample is not None and args.subsample < len(raw):
        logger.info(
            "Subsampling to %d rows with seed %d",
            args.subsample,
            CRITEO_SAMPLE_SEED,
        )
        raw = raw.sample(n=args.subsample, random_state=CRITEO_SAMPLE_SEED)
        raw = raw.reset_index(drop=True)

    subsample_stats = {
        "n_rows": len(raw),
        "treatment_ratio": float(raw["treatment"].mean()),
        "visit_rate": float(raw["visit"].mean()),
        "conversion_rate": float(raw["conversion"].mean()),
        "control_count": int((raw["treatment"] == 0).sum()),
        "treated_count": int((raw["treatment"] == 1).sum()),
    }
    logger.info("Subsample stats: %s", subsample_stats)

    # Propensity gate
    propensity_gate_result: dict[str, Any] = {}
    if not args.skip_propensity_gate and "f0" in raw.columns:
        logger.info("Running propensity heterogeneity gate on f0 deciles...")
        gate_passed, gate_details = run_propensity_gate(raw)
        propensity_gate_result = {"passed": gate_passed, **gate_details}
        if gate_passed:
            logger.info(
                "Propensity gate PASSED (max deviation: %.4f)",
                gate_details["max_deviation"],
            )
        else:
            logger.error(
                "Propensity gate FAILED (max deviation: %.4f > 0.02). "
                "Benchmark cannot proceed with constant propensity.",
                gate_details["max_deviation"],
            )
            # Write partial artifact with gate failure
            output_data = {
                "benchmark": "sprint_33_criteo",
                "status": "blocked_propensity_gate",
                "propensity_gate": _sanitize_for_json(propensity_gate_result),
                "subsample_stats": _sanitize_for_json(subsample_stats),
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, allow_nan=False)
            logger.info("Wrote blocked artifact to %s", output_path)
            return
    else:
        propensity_gate_result = {"passed": True, "note": "gate skipped"}

    reshaped = load_criteo_subsample(raw, synthesize_segment=args.synthesize_segment)
    segment_synthesized = "segment" in reshaped.columns
    if args.synthesize_segment:
        if segment_synthesized:
            logger.info("Synthesized segment from f0 tertiles (Run 2 heterogeneous surface)")
        else:
            logger.warning(
                "--synthesize-segment was set but 'f0' is absent from the data; "
                "segment synthesis was skipped. Run will proceed as Run 1 (degenerate surface)."
            )
    scenario = CriteoScenario(reshaped)
    logger.info(
        "Prepared scenario: %d rows, μ=%.6f",
        len(scenario.real_data),
        scenario.null_baseline,
    )

    results: list[CriteoBenchmarkResult] = []
    real_total = len(budgets) * len(seeds) * len(strategies)
    idx = 0
    t_suite_start = time.perf_counter()

    for budget in budgets:
        for seed in seeds:
            for strategy in strategies:
                idx += 1
                logger.info(
                    "[%d/%d] strategy=%s budget=%d seed=%d",
                    idx,
                    real_total,
                    strategy,
                    budget,
                    seed,
                )
                try:
                    result = scenario.run_strategy(strategy, budget=budget, seed=seed)
                    results.append(result)
                except Exception:
                    logger.warning(
                        "strategy=%s budget=%d seed=%d failed; skipping.",
                        strategy,
                        budget,
                        seed,
                        exc_info=True,
                    )

    # Null-control runs
    if args.null_control:
        null_budgets = [b for b in budgets if b <= 40] or [min(budgets)]
        null_total = len(null_budgets) * len(seeds) * len(strategies)
        null_idx = 0
        for budget in null_budgets:
            for seed in seeds:
                for strategy in strategies:
                    null_idx += 1
                    logger.info(
                        "[null %d/%d] strategy=%s budget=%d seed=%d",
                        null_idx,
                        null_total,
                        strategy,
                        budget,
                        seed,
                    )
                    try:
                        result = scenario.run_strategy(
                            strategy,
                            budget=budget,
                            seed=seed,
                            null_control=True,
                        )
                        results.append(result)
                    except Exception:
                        logger.warning(
                            "null strategy=%s budget=%d seed=%d failed; skipping.",
                            strategy,
                            budget,
                            seed,
                            exc_info=True,
                        )

    suite_runtime = time.perf_counter() - t_suite_start

    projected_graph = criteo_projected_prior_graph()
    criteo_provenance = _sanitize_for_json(
        {
            "dataset_version": "v2.1",
            "subsample_seed": CRITEO_SAMPLE_SEED,
            "subsample_stats": subsample_stats,
            "propensity": CRITEO_PROPENSITY,
            "frozen_params": CRITEO_FROZEN_PARAMS,
            "projected_graph_edge_count": len(projected_graph.edges),
            "projected_graph_edges": [list(edge) for edge in projected_graph.edges],
            "null_baseline": scenario.null_baseline,
            "null_control_enabled": bool(args.null_control),
            "synthesize_segment": segment_synthesized,
            "propensity_gate": propensity_gate_result,
        }
    )
    output_data = {
        "benchmark": "sprint_33_criteo",
        "suite_runtime_seconds": suite_runtime,
        "results": [_sanitize_for_json(dataclasses.asdict(r)) for r in results],
        "provenance": collect_provenance(
            seeds=seeds,
            budgets=budgets,
            strategies=strategies,
            dataset_path=str(args.data_path),
        )
        | {"criteo": criteo_provenance},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(results), output_path)

    if results:
        print()
        _print_summary(results)


if __name__ == "__main__":
    main()
