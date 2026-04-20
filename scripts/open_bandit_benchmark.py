"""Sprint 35.C Open Bandit benchmark runner script.

Runs the Men/Random Sprint 34 Open Bandit contract benchmark at
10 seeds x B20/B40/B80 across ``random``, ``surrogate_only``, and
``causal`` strategies, plus an optional Section 7a null control pass.
Writes a provenance-stamped JSON artifact and prints a per-cell summary.

The full Men/Random slice (~452,949 rows per Saito et al. 2021 Table 1)
must be downloaded separately and unzipped locally — see
``thoughts/shared/docs/sprint-31-open-bandit-access-and-gap-audit.md``.
The OBP 0.4.1 wheel bundles a 10,000-row sample that is useful only for
smoke tests; this script reads ``men.csv`` and ``item_context.csv``
directly under ``--data-path`` so the full slice can be evaluated.

Usage
-----

Typical end-to-end invocation::

    python scripts/open_bandit_benchmark.py \
        --data-path /path/to/open_bandit_dataset \
        --budgets 20,40,80 \
        --seeds 0,1,2,3,4,5,6,7,8,9 \
        --strategies random,surrogate_only,causal \
        --null-control \
        --permutation-seed 20260419 \
        --output men_random.json

``--data-path`` is the unzipped root that contains
``random/men/men.csv`` and ``random/men/item_context.csv``.
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

from causal_optimizer.benchmarks.open_bandit import (
    PROPENSITY_SCHEMA_CONDITIONAL,
    ess_gate,
    get_obp_version,
    propensity_sanity_gate,
    snipw_dr_cross_check_gate,
    zero_support_gate,
)
from causal_optimizer.benchmarks.open_bandit_benchmark import (
    OBD_N_POSITIONS,
    VALID_STRATEGIES,
    OpenBanditBenchmarkResult,
    OpenBanditScenario,
    build_policy_action_dist,
    load_men_random_slice,
    summarize_strategy_budget,
)
from causal_optimizer.benchmarks.provenance import collect_provenance, dataset_hash
from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter

logger = logging.getLogger(__name__)

_DEFAULT_PERMUTATION_SEED: int = 20260419
"""Fixed default permutation seed for the Section 7a null control.

One fixed seed per benchmark matches the Criteo and Hillstrom
convention. Multiple permutation seeds are out of scope per Sprint 34
contract Section 7a.
"""


# ── JSON sanitization ───────────────────────────────────────────────


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-safe Python types."""
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
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ── CLI helpers ─────────────────────────────────────────────────────


def _parse_int_list(raw: str, label: str) -> list[int]:
    """Parse a comma-separated int list; exit on malformed input."""
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Sprint 35 Men/Random Open Bandit benchmark.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to the unzipped Open Bandit Dataset root (containing a "
            "``random/men/`` subdirectory with men.csv and item_context.csv)."
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
        help="Also run each strategy on position-stratified permuted rewards.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=_DEFAULT_PERMUTATION_SEED,
        help="Permutation seed for the Section 7a null control.",
    )
    parser.add_argument(
        "--output",
        default="open_bandit_men_random_results.json",
        help="Output JSON artifact path.",
    )
    parser.add_argument(
        "--skip-reward-model",
        action="store_true",
        help="Skip the reward-model fit (disables DM and DR secondary estimates).",
    )
    return parser.parse_args(argv)


# ── Summary printer ─────────────────────────────────────────────────


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.6f} +/- {arr.std(ddof=0):.6f}"


def _print_summary(results: list[OpenBanditBenchmarkResult]) -> None:
    groups: dict[tuple[bool, str, int], list[OpenBanditBenchmarkResult]] = {}
    for r in results:
        key = (r.is_null_control, r.strategy, r.budget)
        groups.setdefault(key, []).append(r)

    header = f"{'Null':<5} {'Strategy':<16} {'Budget':>6}  {'SNIPW':>28}  {'DR (secondary)':>28}"
    print(header)
    print("-" * len(header))
    for (is_null, strategy, budget), group in sorted(groups.items()):
        snipw = [r.policy_value_snipw for r in group if math.isfinite(r.policy_value_snipw)]
        dr = [r.policy_value_dr for r in group if r.policy_value_dr is not None]
        print(
            f"{'yes' if is_null else 'no':<5} "
            f"{strategy:<16} {budget:>6}  "
            f"{_fmt_mean_std(snipw):>28}  "
            f"{_fmt_mean_std(dr):>28}"
        )


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    budgets = _parse_int_list(args.budgets, "budgets")
    _validate_budgets(budgets)
    seeds = _parse_int_list(args.seeds, "seeds")
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    _validate_strategies(strategies)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load and materialize the Men/Random slice ───────────────
    t_load = time.perf_counter()
    bandit_feedback = load_men_random_slice(data_path=Path(args.data_path))
    load_runtime = time.perf_counter() - t_load

    n_rounds = int(bandit_feedback["n_rounds"])
    n_actions = int(bandit_feedback["n_actions"])
    pscore_mean = float(np.asarray(bandit_feedback["pscore"]).mean())
    click_mean = float(np.asarray(bandit_feedback["reward"]).mean())
    logger.info(
        "Loaded Men/Random slice from %s: n_rounds=%d, n_actions=%d, "
        "pscore_mean=%.6f, click_mean=%.6f (%.1fs)",
        args.data_path,
        n_rounds,
        n_actions,
        pscore_mean,
        click_mean,
        load_runtime,
    )

    scenario = OpenBanditScenario(
        bandit_feedback=bandit_feedback,
        use_reward_model=not args.skip_reward_model,
    )

    results: list[OpenBanditBenchmarkResult] = []
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

    # ── Null-control pass ───────────────────────────────────────
    null_results: list[OpenBanditBenchmarkResult] = []
    if args.null_control:
        null_total = len(budgets) * len(seeds) * len(strategies)
        null_idx = 0
        for budget in budgets:
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
                            permutation_seed=int(args.permutation_seed),
                        )
                        null_results.append(result)
                    except Exception:
                        logger.warning(
                            "null strategy=%s budget=%d seed=%d failed; skipping.",
                            strategy,
                            budget,
                            seed,
                            exc_info=True,
                        )

    suite_runtime = time.perf_counter() - t_suite_start
    all_results = results + null_results

    # ── Section 7 gate evaluation ──────────────────────────────
    gate_report = _evaluate_gates(
        scenario=scenario,
        bandit_feedback=bandit_feedback,
        real_results=results,
        null_results=null_results,
        budgets=budgets,
    )

    # ── Serialize ──────────────────────────────────────────────
    summary = summarize_strategy_budget(results)
    summary_serialized = {
        f"{strategy}_b{budget}": cell for (strategy, budget), cell in summary.items()
    }

    output_data = {
        "benchmark": "sprint_35_open_bandit_men_random",
        "suite_runtime_seconds": suite_runtime,
        "load_runtime_seconds": load_runtime,
        "data_provenance": {
            "data_path": str(args.data_path),
            "men_csv_path": str(Path(args.data_path) / "random" / "men" / "men.csv"),
            "men_csv_sha256": dataset_hash(
                str(Path(args.data_path) / "random" / "men" / "men.csv")
            ),
            "item_context_csv_sha256": dataset_hash(
                str(Path(args.data_path) / "random" / "men" / "item_context.csv")
            ),
            "n_rounds": n_rounds,
            "n_actions": n_actions,
            "n_positions": OBD_N_POSITIONS,
            "pscore_mean": pscore_mean,
            "click_mean": click_mean,
            "propensity_schema": PROPENSITY_SCHEMA_CONDITIONAL,
            "position_handling_default": "position_1_only",
            "obp_version": get_obp_version(),
            "min_propensity_clip": scenario.min_propensity_clip,
        },
        "results": [_sanitize_for_json(dataclasses.asdict(r)) for r in all_results],
        "summary": _sanitize_for_json(summary_serialized),
        "gates": _sanitize_for_json(gate_report),
        "provenance": collect_provenance(
            seeds=seeds,
            budgets=budgets,
            strategies=strategies,
            dataset_path=str(Path(args.data_path) / "random" / "men" / "men.csv"),
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote %d results to %s", len(all_results), output_path)

    if all_results:
        print()
        _print_summary(all_results)

    # ── Print gate verdict on stdout ──────────────────────────
    print()
    print(f"Section 7 gates: all_passed={gate_report['all_passed']}")
    for gate_name, gate_data in gate_report["gates"].items():
        print(f"  7{gate_data['gate_letter']} {gate_name}: passed={gate_data['passed']}")


# ── Gates ──────────────────────────────────────────────────────────


def _evaluate_gates(
    *,
    scenario: OpenBanditScenario,
    bandit_feedback: dict[str, Any],
    real_results: list[OpenBanditBenchmarkResult],
    null_results: list[OpenBanditBenchmarkResult],
    budgets: list[int],
) -> dict[str, Any]:
    """Evaluate Section 7 gates against the collected per-seed data.

    The null-control gate (7a) is driven by the ``null_results`` pass;
    ESS (7b), zero-support (7c), and DR/SNIPW cross-check (7e) are
    driven by per-seed diagnostics captured on the **B80** real results
    (the verdict budget per Sprint 34 contract Section 6e). 7d is data-
    level and needs no per-seed input.
    """
    verdict_budget = max(budgets)
    # The Sprint 34 contract Section 6e declares B80 the verdict
    # budget. If the caller asked for a different set of budgets we
    # still gate against `max(budgets)` (so the gate payload is always
    # populated), but log a loud warning so the artifact reader knows
    # the ESS/zero-support/cross-check gates are not at the contract
    # verdict budget.
    if verdict_budget != 80:
        logger.warning(
            "Gate verdict budget is %d, not the Sprint 34 Section 6e default B80. "
            "ESS, zero-support, and DR/SNIPW cross-check gates will run at "
            "max(budgets)=%d; the report's verdict_budget field records this.",
            verdict_budget,
            verdict_budget,
        )

    # ── 7d propensity sanity — operates on pscore directly ──
    pscore = np.asarray(bandit_feedback["pscore"], dtype=float)
    prop_gate = propensity_sanity_gate(
        pscore=pscore,
        schema=PROPENSITY_SCHEMA_CONDITIONAL,
        n_actions=int(bandit_feedback["n_actions"]),
        n_positions=OBD_N_POSITIONS,
    )

    # Helper: select B80 real results for ESS / zero-support / DR.
    #
    # All strategies at the verdict budget are aggregated into one
    # ess/zero-support gate call — the Sprint 34 contract Section 7b
    # defines the ESS floor against the full set of B80 seeds, not
    # per-strategy. The per-strategy median is reported separately in
    # the report's ESS diagnostics table for transparency, but the
    # gate PASS/FAIL boundary is the aggregate median. Keeping the
    # gate aggregate-level also means a high-ESS `random` policy can
    # cover for a lower-ESS optimized policy only if the *aggregate*
    # median still clears the floor, which the Sprint 34 contract
    # explicitly allows.
    b80_results = [r for r in real_results if r.budget == verdict_budget]
    per_seed_ess = [float(r.diagnostics.get("ess", float("nan"))) for r in b80_results]
    per_seed_zero_support = [
        float(r.diagnostics.get("zero_support_fraction", 0.0)) for r in b80_results
    ]

    ess_result = ess_gate(
        per_seed_ess=[e for e in per_seed_ess if math.isfinite(e)],
        n_rows=int(bandit_feedback["n_rounds"]),
    )
    zero_result = zero_support_gate(per_seed_zero_support=per_seed_zero_support)

    # 7e DR/SNIPW — only defined when DR was computed. Build the two
    # lists in lockstep (one SNIPW for each seed that has a DR value)
    # so a single seed with DR=None does not silently collapse the
    # whole gate into "skipped".
    cross_pairs = [
        (r.policy_value_snipw, r.policy_value_dr)
        for r in b80_results
        if r.policy_value_dr is not None
    ]
    if cross_pairs and len(cross_pairs) == len(b80_results):
        snipw_per_seed = [s for s, _ in cross_pairs]
        dr_per_seed = [d for _, d in cross_pairs]
        cross_result = snipw_dr_cross_check_gate(
            snipw_per_seed=snipw_per_seed, dr_per_seed=dr_per_seed
        )
        cross_gate_passed = cross_result.passed
        cross_gate_payload = dataclasses.asdict(cross_result)
    else:
        # DR unavailable on at least one seed — skip cleanly, but record
        # the count of seeds with DR so the report reader can tell the
        # difference between "reward model disabled" and "one seed
        # failed to produce DR".
        cross_gate_passed = False
        cross_gate_payload = {
            "passed": False,
            "skipped": True,
            "n_b80_seeds": len(b80_results),
            "n_dr_available": len(cross_pairs),
            "reason": "reward model disabled or DR unavailable",
        }

    # 7a null control — compare permuted-reward policy values vs the
    # permuted baseline mean.
    if null_results:
        # Null-baseline mu is the mean of the permuted reward column,
        # which equals the raw mean because permutation preserves the
        # marginal. Using the raw reward mean is stable and matches
        # Criteo's convention.
        mu_null = float(np.asarray(bandit_feedback["reward"], dtype=float).mean())
        band = 1.05
        threshold = band * mu_null
        per_cell_values: dict[str, float] = {}
        per_cell_ratios: dict[str, float] = {}
        for r in null_results:
            key = f"{r.strategy}_b{r.budget}_seed{r.seed}"
            per_cell_values[key] = r.policy_value_snipw
            per_cell_ratios[key] = r.policy_value_snipw / mu_null if mu_null > 0 else float("inf")
        # Drop non-finite values (seed produced no best policy, e.g.
        # CRASH during permuted run) before the all() check — NaN <= x
        # is always False in Python and would spuriously fail the gate.
        # The real-results `summarize_strategy_budget` path applies the
        # same math.isfinite guard.
        finite_null_values = [v for v in per_cell_values.values() if math.isfinite(v)]
        n_skipped_null = len(per_cell_values) - len(finite_null_values)
        null_passed = len(finite_null_values) > 0 and all(
            v <= threshold for v in finite_null_values
        )
        null_payload = {
            "passed": null_passed,
            "mu_null": mu_null,
            "band_multiplier": band,
            "threshold": threshold,
            "per_cell_values": per_cell_values,
            "per_cell_ratios": per_cell_ratios,
            "n_cells_non_finite": n_skipped_null,
            "permutation_seed": null_results[0].permutation_seed,
        }
    else:
        null_passed = False
        null_payload = {
            "passed": False,
            "skipped": True,
            "reason": "null control not run (--null-control not set)",
        }

    gates = {
        "null_control": {
            "gate_letter": "a",
            "passed": bool(null_passed),
            **null_payload,
        },
        "ess": {
            "gate_letter": "b",
            "passed": bool(ess_result.passed),
            **dataclasses.asdict(ess_result),
        },
        "zero_support": {
            "gate_letter": "c",
            "passed": bool(zero_result.passed),
            **dataclasses.asdict(zero_result),
        },
        "propensity_sanity": {
            "gate_letter": "d",
            "passed": bool(prop_gate.passed),
            **dataclasses.asdict(prop_gate),
        },
        "dr_cross_check": {
            "gate_letter": "e",
            "passed": bool(cross_gate_passed),
            **cross_gate_payload,
        },
    }
    all_passed = all(g["passed"] for g in gates.values())
    return {
        "all_passed": bool(all_passed),
        "verdict_budget": verdict_budget,
        "gates": gates,
    }


# Keep ``BanditLogAdapter`` and ``build_policy_action_dist`` re-exports
# so the CLI surface references land-of-truth imports rather than
# shadowing them.
__all__ = [
    "BanditLogAdapter",
    "VALID_STRATEGIES",
    "build_policy_action_dist",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
