#!/usr/bin/env python3
"""Interaction policy ablation: early causal pressure components.

Runs the interaction policy benchmark with different combinations of
causal_exploration_weight and causal_softness to isolate which
component of early causal pressure causes the B20 penalty.

Arms:
  1. surrogate_only (no causal graph)
  2. causal_default (weight=0.3, softness=0.5)
  3. no_exploration (weight=0.0, softness=0.5)
  4. no_alignment (weight=0.3, softness=0.0)
  5. graph_only (weight=0.0, softness=0.0)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.interaction_policy import (
    InteractionPolicyRunner,
    InteractionPolicyScenario,
    evaluate_interaction_policy,
)
from causal_optimizer.benchmarks.provenance import collect_provenance
from causal_optimizer.engine import ExperimentEngine

logging.basicConfig(level=logging.WARNING)

ARMS: dict[str, dict] = {
    "surrogate_only": {
        "use_graph": False,
        "causal_exploration_weight": 0.3,  # irrelevant without graph
        "causal_softness": 0.5,  # irrelevant without graph
    },
    "causal_default": {
        "use_graph": True,
        "causal_exploration_weight": 0.3,
        "causal_softness": 0.5,
    },
    "no_exploration": {
        "use_graph": True,
        "causal_exploration_weight": 0.0,
        "causal_softness": 0.5,
    },
    "no_alignment": {
        "use_graph": True,
        "causal_exploration_weight": 0.3,
        "causal_softness": 0.0,
    },
    "graph_only": {
        "use_graph": True,
        "causal_exploration_weight": 0.0,
        "causal_softness": 0.0,
    },
}


def run_one(
    scenario: InteractionPolicyScenario,
    arm_name: str,
    arm_cfg: dict,
    budget: int,
    seed: int,
) -> dict:
    """Run a single arm/budget/seed combination."""
    t_start = time.monotonic()

    data = scenario.generate()
    n = len(data)
    opt_end = int(n * 0.8)
    val_data = data.iloc[:opt_end].reset_index(drop=True)
    test_data = data.iloc[opt_end:].reset_index(drop=True)

    space = scenario.search_space()
    runner = InteractionPolicyRunner(val_data, scenario.treatment_cost)

    graph = scenario.causal_graph() if arm_cfg["use_graph"] else None
    engine = ExperimentEngine(
        search_space=space,
        runner=runner,
        causal_graph=graph,
        objective_name="objective",
        minimize=True,
        seed=seed,
        max_skips=0,
        causal_exploration_weight=arm_cfg["causal_exploration_weight"],
        causal_softness=arm_cfg["causal_softness"],
    )
    engine.run_loop(budget)
    best_result = engine.log.best_result("objective", minimize=True)
    best_params = best_result.parameters if best_result is not None else None

    if best_params is not None:
        policy_value, decision_error = evaluate_interaction_policy(
            test_data, best_params, scenario.treatment_cost
        )
    else:
        policy_value = 0.0
        oracle_treat = test_data["true_treatment_effect"].values > scenario.treatment_cost
        decision_error = float(np.mean(oracle_treat))

    oracle_value = scenario.oracle_policy_value(test_data)
    regret = oracle_value - policy_value
    runtime = time.monotonic() - t_start

    return {
        "arm": arm_name,
        "budget": budget,
        "seed": seed,
        "policy_value": round(policy_value, 4),
        "oracle_value": round(oracle_value, 4),
        "regret": round(regret, 4),
        "decision_error_rate": round(decision_error, 4),
        "runtime_seconds": round(runtime, 1),
        "causal_exploration_weight": arm_cfg["causal_exploration_weight"],
        "causal_softness": arm_cfg["causal_softness"],
        "use_graph": arm_cfg["use_graph"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interaction policy ablation study")
    parser.add_argument("--data-path", type=str, required=True, help="Path to ERCOT parquet")
    parser.add_argument("--budgets", default="20,40,80", help="Comma-separated budgets")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds")
    parser.add_argument(
        "--arms",
        default=",".join(ARMS.keys()),
        help=f"Comma-separated arms (default: all). Options: {','.join(ARMS.keys())}",
    )
    parser.add_argument("--output", default="interaction_ablation_results.json", help="Output path")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    arm_names = [a.strip() for a in args.arms.split(",")]
    for name in arm_names:
        if name not in ARMS:
            parser.error(f"Unknown arm: {name}. Options: {','.join(ARMS.keys())}")

    df = pd.read_parquet(args.data_path)
    scenario = InteractionPolicyScenario(covariates=df, seed=0)

    results: list[dict] = []
    suite_start = time.monotonic()

    total = len(arm_names) * len(budgets) * len(seeds)
    done = 0

    for arm_name in arm_names:
        arm_cfg = ARMS[arm_name]
        for budget in budgets:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] arm={arm_name} budget={budget} seed={seed}")
                result = run_one(scenario, arm_name, arm_cfg, budget, seed)
                results.append(result)

    suite_runtime = time.monotonic() - suite_start

    output = {
        "benchmark": "interaction_policy_ablation",
        "arms": {name: ARMS[name] for name in arm_names},
        "budgets": budgets,
        "seeds": seeds,
        "suite_runtime_seconds": round(suite_runtime, 1),
        "results": results,
        "provenance": collect_provenance(
            seeds=seeds,
            budgets=budgets,
            strategies=arm_names,
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(results)} results to {out_path}")

    # Summary table
    print(f"\n{'Arm':<20} {'Budget':<8} {'Mean Regret':>15} {'Std':>10}")
    print("-" * 60)
    for arm_name in arm_names:
        for budget in budgets:
            rows = [r for r in results if r["arm"] == arm_name and r["budget"] == budget]
            regrets = [r["regret"] for r in rows]
            mean = np.mean(regrets)
            std = np.std(regrets)
            print(f"{arm_name:<20} {budget:<8} {mean:>15.2f} {std:>10.2f}")

    print(f"\nSuite runtime: {suite_runtime:.1f}s")


if __name__ == "__main__":
    main()
