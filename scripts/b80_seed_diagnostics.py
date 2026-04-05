"""B80 seed diagnostics -- trace per-experiment behavior for good vs bad seeds.

Instruments the benchmark to capture per-step parameter trajectories,
objective values, phase transitions, and best-so-far tracking.  Compares
good seeds (regret < 1.0 at B80) vs catastrophic seeds (regret > 10).

Usage::

    uv run python scripts/b80_seed_diagnostics.py \
        --data-path data/ercot_north_c_dfw_2022_2024.parquet \
        --seeds 0,1,3,9 \
        --output artifacts/s23_b80_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.counterfactual_energy import (
    DemandResponseScenario,
    PolicyRunner,
    evaluate_policy,
)
from causal_optimizer.benchmarks.predictive_energy import load_energy_frame
from causal_optimizer.engine.loop import ExperimentEngine

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


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare covariate DataFrame from raw ERCOT data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    if "humidity" not in df.columns:
        df["humidity"] = 50.0
    if "load_lag_1h" not in df.columns:
        df["load_lag_1h"] = df["target_load"].shift(1).bfill()
    if "load_lag_24h" not in df.columns:
        df["load_lag_24h"] = df["target_load"].shift(24).bfill()
    return df


def run_instrumented(
    scenario: DemandResponseScenario,
    budget: int,
    seed: int,
    strategy: str,
) -> dict[str, Any]:
    """Run one benchmark with per-experiment instrumentation.

    Returns a dict with the trajectory of every experiment step including:
    - parameters proposed
    - objective value
    - best-so-far objective and parameters
    - phase
    - whether this step improved best-so-far
    """
    data = scenario.generate()
    n = len(data)
    opt_end = int(n * 0.8)
    val_data = data.iloc[:opt_end].reset_index(drop=True)
    test_data = data.iloc[opt_end:].reset_index(drop=True)

    space = scenario.search_space()
    runner = PolicyRunner(val_data, scenario.treatment_cost)

    graph = scenario.causal_graph() if strategy == "causal" else None
    engine = ExperimentEngine(
        search_space=space,
        runner=runner,
        causal_graph=graph,
        objective_name="objective",
        minimize=True,
        seed=seed,
        max_skips=0,
    )

    # Oracle value is deterministic given the test data -- compute once.
    oracle_value = scenario.oracle_policy_value(test_data)

    trajectory: list[dict[str, Any]] = []
    best_obj = float("inf")
    best_params: dict[str, Any] | None = None
    best_step = -1
    best_policy_value = 0.0
    best_decision_error = 1.0

    for i in range(budget):
        result = engine.step()
        obj = result.metrics.get("objective", float("inf"))
        policy_val = result.metrics.get("policy_value", None)

        improved = False
        if obj < best_obj:
            best_obj = obj
            best_params = dict(result.parameters)
            best_step = i
            improved = True

        # Evaluate the current-step parameters on the test set
        step_policy_value, step_decision_error = evaluate_policy(
            test_data, result.parameters, scenario.treatment_cost
        )

        # Re-evaluate best-so-far on test set only when it changed
        if improved and best_params is not None:
            best_policy_value, best_decision_error = evaluate_policy(
                test_data, best_params, scenario.treatment_cost
            )

        step_record = {
            "step": i,
            "phase": engine.phase,
            "parameters": dict(result.parameters),
            "objective": obj,
            "policy_value_train": policy_val,
            "status": result.status.value,
            "improved_best": improved,
            "best_so_far_objective": best_obj,
            "best_so_far_step": best_step,
            "step_test_policy_value": step_policy_value,
            "step_test_decision_error": step_decision_error,
            "step_test_regret": oracle_value - step_policy_value,
            "best_test_policy_value": best_policy_value,
            "best_test_decision_error": best_decision_error,
            "best_test_regret": oracle_value - best_policy_value,
        }
        trajectory.append(step_record)

    final_regret = oracle_value - best_policy_value

    return {
        "seed": seed,
        "budget": budget,
        "strategy": strategy,
        "final_regret": final_regret,
        "final_policy_value": best_policy_value,
        "oracle_value": oracle_value,
        "final_decision_error": best_decision_error,
        "best_found_at_step": best_step,
        "best_params": best_params,
        "trajectory": trajectory,
    }


def analyze_trajectories(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze trajectories to identify divergence patterns."""
    good_seeds = [r for r in results if r["final_regret"] < 1.0]
    bad_seeds = [r for r in results if r["final_regret"] > 10.0]
    mid_seeds = [r for r in results if 1.0 <= r["final_regret"] <= 10.0]

    analysis: dict[str, Any] = {
        "good_seed_ids": [r["seed"] for r in good_seeds],
        "bad_seed_ids": [r["seed"] for r in bad_seeds],
        "mid_seed_ids": [r["seed"] for r in mid_seeds],
    }

    # Where do bad seeds first diverge from good seeds?
    if good_seeds and bad_seeds:
        max_steps = max(len(r["trajectory"]) for r in results)
        divergence_analysis = []

        for step in range(max_steps):
            good_regrets = []
            bad_regrets = []
            for r in good_seeds:
                if step < len(r["trajectory"]):
                    good_regrets.append(r["trajectory"][step]["best_test_regret"])
            for r in bad_seeds:
                if step < len(r["trajectory"]):
                    bad_regrets.append(r["trajectory"][step]["best_test_regret"])

            if good_regrets and bad_regrets:
                good_mean = float(np.mean(good_regrets))
                bad_mean = float(np.mean(bad_regrets))
                gap = bad_mean - good_mean
                divergence_analysis.append(
                    {
                        "step": step,
                        "good_mean_regret": good_mean,
                        "bad_mean_regret": bad_mean,
                        "gap": gap,
                        "phase": (
                            good_seeds[0]["trajectory"][step]["phase"]
                            if step < len(good_seeds[0]["trajectory"])
                            else "unknown"
                        ),
                    }
                )

        analysis["divergence_by_step"] = divergence_analysis

        # Find first step where gap exceeds 2.0
        first_divergence = None
        for da in divergence_analysis:
            if da["gap"] > 2.0:
                first_divergence = da["step"]
                break
        analysis["first_divergence_step"] = first_divergence

    # Parameter comparison at key steps
    if good_seeds and bad_seeds:
        key_steps = [0, 5, 9, 10, 15, 20, 30, 40, 50, 60, 70, 79]
        param_comparison = []
        for step in key_steps:
            good_params_at_step = []
            bad_params_at_step = []
            for r in good_seeds:
                if step < len(r["trajectory"]):
                    good_params_at_step.append(r["trajectory"][step]["parameters"])
            for r in bad_seeds:
                if step < len(r["trajectory"]):
                    bad_params_at_step.append(r["trajectory"][step]["parameters"])

            if good_params_at_step and bad_params_at_step:
                # Compare parameter distributions
                param_diffs = {}
                for key in good_params_at_step[0]:
                    good_vals = []
                    bad_vals = []
                    for p in good_params_at_step:
                        v = p.get(key)
                        if isinstance(v, (int, float)):
                            good_vals.append(float(v))
                    for p in bad_params_at_step:
                        v = p.get(key)
                        if isinstance(v, (int, float)):
                            bad_vals.append(float(v))
                    if good_vals and bad_vals:
                        param_diffs[key] = {
                            "good_mean": float(np.mean(good_vals)),
                            "bad_mean": float(np.mean(bad_vals)),
                            "good_std": float(np.std(good_vals)),
                            "bad_std": float(np.std(bad_vals)),
                        }
                    else:
                        # Categorical
                        good_cats = [str(p.get(key)) for p in good_params_at_step]
                        bad_cats = [str(p.get(key)) for p in bad_params_at_step]
                        param_diffs[key] = {
                            "good_values": good_cats,
                            "bad_values": bad_cats,
                        }

                param_comparison.append(
                    {
                        "step": step,
                        "param_diffs": param_diffs,
                    }
                )
        analysis["param_comparison"] = param_comparison

    # Best-so-far parameter analysis
    if good_seeds and bad_seeds:
        analysis["good_best_params"] = [
            {"seed": r["seed"], "params": r["best_params"], "best_step": r["best_found_at_step"]}
            for r in good_seeds
        ]
        analysis["bad_best_params"] = [
            {"seed": r["seed"], "params": r["best_params"], "best_step": r["best_found_at_step"]}
            for r in bad_seeds
        ]

    # Phase transition analysis: what happens at step 10 (exploration -> optimization)?
    if results:
        phase_transitions = []
        for r in results:
            is_good = r["final_regret"] < 1.0
            for i, step in enumerate(r["trajectory"]):
                if i > 0 and step["phase"] != r["trajectory"][i - 1]["phase"]:
                    phase_transitions.append(
                        {
                            "seed": r["seed"],
                            "step": step["step"],
                            "from_phase": r["trajectory"][i - 1]["phase"],
                            "to_phase": step["phase"],
                            "best_regret_at_transition": step["best_test_regret"],
                            "is_good_seed": is_good,
                        }
                    )
        analysis["phase_transitions"] = phase_transitions

    # Improvement frequency: how often do good vs bad seeds find improvements?
    if good_seeds and bad_seeds:
        good_improvement_steps = []
        bad_improvement_steps = []
        for r in good_seeds:
            steps = [s["step"] for s in r["trajectory"] if s["improved_best"]]
            good_improvement_steps.append(
                {
                    "seed": r["seed"],
                    "improvement_steps": steps,
                    "n_improvements": len(steps),
                    "last_improvement": steps[-1] if steps else -1,
                }
            )
        for r in bad_seeds:
            steps = [s["step"] for s in r["trajectory"] if s["improved_best"]]
            bad_improvement_steps.append(
                {
                    "seed": r["seed"],
                    "improvement_steps": steps,
                    "n_improvements": len(steps),
                    "last_improvement": steps[-1] if steps else -1,
                }
            )
        analysis["good_improvement_pattern"] = good_improvement_steps
        analysis["bad_improvement_pattern"] = bad_improvement_steps

    return analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B80 seed diagnostics")
    parser.add_argument("--data-path", required=True, help="Path to ERCOT parquet")
    parser.add_argument("--seeds", default="0,1,3,9", help="Seeds to diagnose")
    parser.add_argument("--budget", type=int, default=80, help="Budget (default: 80)")
    parser.add_argument(
        "--output",
        default="b80_diagnostics.json",
        help="Output JSON path",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_energy_frame(args.data_path)
    covariates = prepare_covariates(df)
    logger.info("Prepared %d covariate rows", len(covariates))

    scenario = DemandResponseScenario(
        covariates=covariates,
        seed=0,
        treatment_cost=60.0,
    )

    results = []
    for seed in seeds:
        logger.info("Running instrumented benchmark: seed=%d budget=%d", seed, args.budget)
        t0 = time.perf_counter()
        result = run_instrumented(scenario, args.budget, seed, "causal")
        elapsed = time.perf_counter() - t0
        logger.info(
            "  seed=%d regret=%.4f best_at_step=%d (%.1fs)",
            seed,
            result["final_regret"],
            result["best_found_at_step"],
            elapsed,
        )
        results.append(result)

    # Run analysis
    analysis = analyze_trajectories(results)

    output_data = {
        "diagnostic_type": "b80_seed_trajectory",
        "budget": args.budget,
        "seeds_analyzed": seeds,
        "results": [_sanitize_for_json(r) for r in results],
        "analysis": _sanitize_for_json(analysis),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, allow_nan=False)
    logger.info("Wrote diagnostics to %s", output_path)

    # Print summary
    print("\n=== B80 Seed Diagnostic Summary ===")
    print(f"{'Seed':>6} {'Regret':>10} {'Best@Step':>10} {'Category':>12}")
    print("-" * 42)
    for r in results:
        cat = "GOOD" if r["final_regret"] < 1.0 else ("BAD" if r["final_regret"] > 10.0 else "MID")
        print(f"{r['seed']:>6} {r['final_regret']:>10.4f} {r['best_found_at_step']:>10} {cat:>12}")

    if analysis.get("first_divergence_step") is not None:
        print(f"\nFirst divergence at step: {analysis['first_divergence_step']}")

    # Print best params comparison
    if analysis.get("good_best_params") and analysis.get("bad_best_params"):
        print("\n--- Good Seeds Best Params ---")
        for item in analysis["good_best_params"]:
            print(f"  Seed {item['seed']} (step {item['best_step']}): {item['params']}")
        print("\n--- Bad Seeds Best Params ---")
        for item in analysis["bad_best_params"]:
            print(f"  Seed {item['seed']} (step {item['best_step']}): {item['params']}")


if __name__ == "__main__":
    main()
