"""Energy predictive benchmark runner — compare optimization strategies.

Runs random, surrogate-only, and causal strategies across multiple budgets
and seeds against the predictive energy harness, writing results to JSON.
"""

from __future__ import annotations

import argparse

import pandas as pd

from causal_optimizer.benchmarks.predictive_energy import PredictiveBenchmarkResult

_VALID_STRATEGIES: frozenset[str] = frozenset({"random", "surrogate_only", "causal"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    raise NotImplementedError


def run_strategy(
    strategy: str,
    budget: int,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> PredictiveBenchmarkResult:
    """Run one strategy and return benchmark result."""
    raise NotImplementedError


def main() -> None:
    """Entry point: parse args, run strategies, write results."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
