"""Null-signal predictive energy benchmark -- permuted target negative control.

Extends the predictive energy benchmark with a target permutation step
that destroys any real signal while preserving marginal distributions.
Used as a negative control to verify the optimizer does not manufacture
false wins from noise.

Public API
----------
- :func:`permute_target` -- permute the target column to destroy signal.
- :func:`run_null_strategy` -- run one strategy on the null benchmark.
- :func:`check_null_signal` -- verify no strategy wins consistently on null data.
- :class:`NullSignalResult` -- result of the null-signal check.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    ValidationEnergyRunner,
    evaluate_on_test,
)
from causal_optimizer.benchmarks.runner import sample_random_params
from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter
from causal_optimizer.engine.loop import ExperimentEngine


@dataclass
class NullSignalResult:
    """Result of checking whether any strategy wins on null data.

    Attributes:
        verdict: ``"PASS"`` if no strategy wins consistently,
            ``"WARN"`` if a strategy appears to win,
            ``"ERROR"`` if no valid results are available.
        has_consistent_winner: True if any strategy beats random by more
            than *threshold* across all seeds.
        best_strategy: Name of the best-performing strategy (may be noise).
        best_improvement: Relative improvement of best strategy over random.
        details: Human-readable details.
    """

    verdict: Literal["PASS", "WARN", "ERROR"]
    has_consistent_winner: bool
    best_strategy: str
    best_improvement: float
    details: list[str] = field(default_factory=list)


# ── Target permutation ────────────────────────────────────────────────


def permute_target(
    df: pd.DataFrame,
    target_col: str = "target_load",
    seed: int = 0,
) -> pd.DataFrame:
    """Permute target column within the dataframe, destroying signal.

    Uses a fixed seed for reproducibility. Permutes only the target
    column, preserving all covariate structure and marginal distribution
    of the target.

    Args:
        df: Input DataFrame containing the target column.
        target_col: Name of the column to permute (default ``"target_load"``).
        seed: Random seed for reproducibility.

    Returns:
        A copy of the DataFrame with the target column permuted.

    Raises:
        ValueError: If *target_col* is not present in the DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in DataFrame")

    result = df.copy()
    rng = np.random.default_rng(seed)
    permuted_values = result[target_col].values.copy()
    rng.shuffle(permuted_values)
    result[target_col] = permuted_values
    return result


# ── Null benchmark strategy runner ────────────────────────────────────

VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})
_DEFAULT_CHECKPOINTS: list[int] = [5, 10, 20, 40, 80]


def run_null_strategy(
    strategy: str,
    budget: int,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audit_skip_rate: float = 0.0,
) -> PredictiveBenchmarkResult | None:
    """Run one strategy on the null (permuted-target) benchmark.

    Same interface as the real benchmark's ``run_strategy``, but operates
    on already-permuted data.  The caller is responsible for permuting
    the target column before splitting.

    Args:
        strategy: One of ``"random"``, ``"surrogate_only"``, ``"causal"``.
        budget: Number of experiments to run.
        seed: Random seed for reproducibility.
        train_df: Training partition (already permuted).
        val_df: Validation partition (already permuted).
        test_df: Test partition (already permuted).
        audit_skip_rate: Fraction of skipped candidates to force-evaluate.

    Returns:
        A :class:`PredictiveBenchmarkResult`, or ``None`` if all experiments crashed.

    Raises:
        ValueError: If *strategy* is not in ``VALID_STRATEGIES``.
    """
    if strategy not in VALID_STRATEGIES:
        msg = f"Unknown strategy {strategy!r}. Must be one of {sorted(VALID_STRATEGIES)}."
        raise ValueError(msg)

    t_start = time.perf_counter()

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
            audit_skip_rate=audit_skip_rate,
        )
        engine.run_loop(budget)
        best_result = engine.log.best_result("mae", minimize=True)
        if best_result is not None:
            best_mae = best_result.metrics.get("mae", float("inf"))
            best_params = best_result.parameters
        else:
            best_mae = float("inf")
            best_params = None

        skip_diag = engine.skip_diagnostics
        checkpoints = [c for c in _DEFAULT_CHECKPOINTS if c <= budget] or [budget]
        if budget not in checkpoints:
            checkpoints.append(budget)
        anytime = engine.anytime_metrics(sorted(checkpoints))

    if best_params is None:
        return None

    test_metrics = evaluate_on_test(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        parameters=best_params,
        seed=seed,
    )
    test_mae = test_metrics.get("mae", float("inf"))
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


# ── Null-signal acceptance check ──────────────────────────────────────


def check_null_signal(
    results: list[PredictiveBenchmarkResult],
    strategies: list[str],
    threshold: float = 0.02,
) -> NullSignalResult:
    """Check that no strategy wins consistently on null data.

    Computes mean test MAE per strategy across all seeds.  Returns PASS
    if no strategy is more than *threshold* (relative) better than the
    ``"random"`` baseline.  Returns WARN if any strategy appears to win.

    Args:
        results: List of benchmark results (across strategies and seeds).
        strategies: List of strategy names to compare.
        threshold: Relative improvement threshold (default 0.02 = 2%).

    Returns:
        A :class:`NullSignalResult` with verdict and details.
    """
    details: list[str] = []

    # Compute mean test MAE per strategy
    strategy_maes: dict[str, list[float]] = {s: [] for s in strategies}
    for r in results:
        if r.strategy in strategy_maes and math.isfinite(r.test_mae):
            strategy_maes[r.strategy].append(r.test_mae)

    strategy_means: dict[str, float] = {}
    strategy_stds: dict[str, float] = {}
    for s, maes in strategy_maes.items():
        if maes:
            arr = np.array(maes)
            strategy_means[s] = float(arr.mean())
            strategy_stds[s] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            details.append(
                f"{s}: mean_test_mae={strategy_means[s]:.4f} "
                f"+/- {strategy_stds[s]:.4f} (n={len(maes)})"
            )

    if not strategy_means:
        return NullSignalResult(
            verdict="ERROR",
            has_consistent_winner=False,
            best_strategy="none",
            best_improvement=0.0,
            details=["No valid results to evaluate"],
        )

    # Use random as baseline; if absent, use the worst (highest MAE) strategy
    baseline_key = (
        "random"
        if "random" in strategy_means
        else max(strategy_means, key=lambda k: strategy_means[k])
    )
    baseline_mae = strategy_means[baseline_key]

    # Find best strategy (lowest mean MAE)
    best_strategy = min(strategy_means, key=lambda k: strategy_means[k])
    best_mae = strategy_means[best_strategy]

    best_improvement = (baseline_mae - best_mae) / baseline_mae if baseline_mae > 0 else 0.0

    # Check if any strategy beats baseline by more than threshold
    has_consistent_winner = False
    for s, mean_mae in strategy_means.items():
        if s == baseline_key:
            continue
        if baseline_mae > 0:
            rel_improvement = (baseline_mae - mean_mae) / baseline_mae
            if rel_improvement > threshold:
                has_consistent_winner = True
                details.append(
                    f"WARNING: {s} beats {baseline_key} by {rel_improvement:.1%} "
                    f"(>{threshold:.0%} threshold) on null data"
                )

    verdict: Literal["PASS", "WARN", "ERROR"]
    if has_consistent_winner:
        verdict = "WARN"
        details.append(
            "A strategy appears to win on null data. This may indicate "
            "overfitting to noise or insufficient seed coverage."
        )
    else:
        verdict = "PASS"
        details.append(
            "No strategy shows consistent improvement beyond noise threshold. Null control passed."
        )

    return NullSignalResult(
        verdict=verdict,
        has_consistent_winner=has_consistent_winner,
        best_strategy=best_strategy,
        best_improvement=best_improvement,
        details=details,
    )
