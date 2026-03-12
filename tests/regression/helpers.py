"""Shared assertion helpers for convergence regression tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_optimizer.benchmarks.runner import BenchmarkResult


def finals_for_strategy(results: list[BenchmarkResult], strategy: str) -> list[float]:
    """Extract final_best values for a given strategy from benchmark results."""
    return [r.final_best for r in results if r.strategy == strategy]


def assert_monotonic_curves(results: list[BenchmarkResult]) -> None:
    """Assert all convergence curves are monotonically non-increasing."""
    for r in results:
        curve = r.convergence_curve
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i - 1] + 1e-12, (
                f"Convergence curve for {r.strategy} seed={r.seed} is not monotonic "
                f"at step {i}: {curve[i - 1]:.6f} -> {curve[i]:.6f}"
            )


def assert_curve_lengths(results: list[BenchmarkResult], expected: int) -> None:
    """Assert all convergence curves have the expected length."""
    for r in results:
        assert len(r.convergence_curve) == expected, (
            f"{r.strategy} seed={r.seed}: expected {expected} steps, got {len(r.convergence_curve)}"
        )


def assert_causal_beats_random(
    results: list[BenchmarkResult],
    n_seeds: int,
    tolerance_fraction: float,
    benchmark_label: str,
) -> None:
    """Assert causal strategy's mean final objective beats random within tolerance."""
    causal_finals = finals_for_strategy(results, "causal")
    random_finals = finals_for_strategy(results, "random")

    strategies_present = sorted({r.strategy for r in results})
    assert len(causal_finals) == n_seeds, (
        f"Expected {n_seeds} causal results, got {len(causal_finals)}. "
        f"Strategies present: {strategies_present}"
    )
    assert len(random_finals) == n_seeds, (
        f"Expected {n_seeds} random results, got {len(random_finals)}. "
        f"Strategies present: {strategies_present}"
    )

    avg_causal = float(np.mean(causal_finals))
    avg_random = float(np.mean(random_finals))

    tolerance = tolerance_fraction * max(abs(avg_random), 1.0)
    assert avg_causal <= avg_random + tolerance, (
        f"Causal ({avg_causal:.4f}) did not beat random ({avg_random:.4f}) "
        f"within {tolerance_fraction:.0%} tolerance on {benchmark_label}. "
        f"Causal finals: {causal_finals}, Random finals: {random_finals}"
    )
