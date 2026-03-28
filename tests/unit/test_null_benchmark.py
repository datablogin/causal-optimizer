"""Tests for the null-signal control benchmark.

Verifies that target permutation destroys signal, preserves marginals,
is deterministic, and that the null-signal checker correctly identifies
when no strategy wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.benchmarks.null_predictive_energy import (
    check_null_signal,
    permute_target,
)
from causal_optimizer.benchmarks.predictive_energy import PredictiveBenchmarkResult

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_energy_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic energy DataFrame with real signal in the target."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="h")
    temperature = 60.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 2, n)
    # target_load has real correlation with temperature
    target_load = 1000.0 + 5.0 * temperature + rng.normal(0, 10, n)
    humidity = 50.0 + rng.normal(0, 10, n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "humidity": humidity,
            "target_load": target_load,
        }
    )


def _make_benchmark_result(
    strategy: str,
    budget: int,
    seed: int,
    test_mae: float,
    val_mae: float | None = None,
) -> PredictiveBenchmarkResult:
    """Create a PredictiveBenchmarkResult for testing."""
    return PredictiveBenchmarkResult(
        strategy=strategy,
        budget=budget,
        seed=seed,
        best_validation_mae=val_mae if val_mae is not None else test_mae * 0.9,
        test_mae=test_mae,
        selected_parameters={"model_type": "ridge"},
        runtime_seconds=1.0,
    )


# ── test_permute_target_destroys_correlation ──────────────────────────


def test_permute_target_destroys_correlation() -> None:
    """After permutation, correlation between target and covariates drops to near-zero."""
    df = _make_energy_df(n=500, seed=0)

    # Original correlation should be strong
    original_corr = df["target_load"].corr(df["temperature"])
    assert abs(original_corr) > 0.5, f"Expected strong original correlation, got {original_corr}"

    # After permutation, correlation should be near zero
    permuted = permute_target(df, target_col="target_load", seed=0)
    permuted_corr = permuted["target_load"].corr(permuted["temperature"])
    assert abs(permuted_corr) < 0.15, (
        f"Expected near-zero correlation after permutation, got {permuted_corr}"
    )


# ── test_permute_target_preserves_marginals ───────────────────────────


def test_permute_target_preserves_marginals() -> None:
    """Target's mean, std, min, max are identical before and after permutation."""
    df = _make_energy_df(n=300, seed=1)
    permuted = permute_target(df, target_col="target_load", seed=0)

    original = df["target_load"]
    perm_col = permuted["target_load"]

    assert original.mean() == pytest.approx(perm_col.mean(), abs=1e-10)
    assert original.std() == pytest.approx(perm_col.std(), abs=1e-10)
    assert original.min() == pytest.approx(perm_col.min(), abs=1e-10)
    assert original.max() == pytest.approx(perm_col.max(), abs=1e-10)


# ── test_permute_target_is_deterministic ──────────────────────────────


def test_permute_target_is_deterministic() -> None:
    """Same seed produces same permutation."""
    df = _make_energy_df(n=200, seed=2)
    perm1 = permute_target(df, target_col="target_load", seed=42)
    perm2 = permute_target(df, target_col="target_load", seed=42)
    pd.testing.assert_frame_equal(perm1, perm2)


# ── test_permute_target_different_seeds ───────────────────────────────


def test_permute_target_different_seeds() -> None:
    """Different seeds produce different permutations."""
    df = _make_energy_df(n=200, seed=3)
    perm1 = permute_target(df, target_col="target_load", seed=0)
    perm2 = permute_target(df, target_col="target_load", seed=1)
    # The target columns should differ (extremely unlikely to match by chance)
    assert not perm1["target_load"].equals(perm2["target_load"])


# ── test_permute_target_preserves_covariates ──────────────────────────


def test_permute_target_preserves_covariates() -> None:
    """Covariates are NOT permuted — only the target column changes."""
    df = _make_energy_df(n=200, seed=4)
    permuted = permute_target(df, target_col="target_load", seed=0)

    pd.testing.assert_series_equal(df["temperature"], permuted["temperature"])
    pd.testing.assert_series_equal(df["humidity"], permuted["humidity"])
    pd.testing.assert_series_equal(df["timestamp"], permuted["timestamp"])


# ── test_null_benchmark_smoke ─────────────────────────────────────────


@pytest.mark.slow
def test_null_benchmark_smoke() -> None:
    """Run with budget=3, seed=0 — verify results are produced (no crashes).

    This is a smoke test that exercises the full null benchmark pipeline
    using the synthetic fixture data. It does NOT require real Parquet data.
    """
    from causal_optimizer.benchmarks.null_predictive_energy import run_null_strategy

    df = _make_energy_df(n=200, seed=0)
    permuted = permute_target(df, target_col="target_load", seed=99999)

    from causal_optimizer.benchmarks.predictive_energy import split_time_frame

    train_df, val_df, test_df = split_time_frame(permuted)

    result = run_null_strategy(
        strategy="random",
        budget=3,
        seed=0,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    assert result is not None
    assert result.strategy == "random"
    assert result.budget == 3
    assert result.seed == 0
    assert result.test_mae > 0


# ── test_null_signal_check_no_winner ──────────────────────────────────


def test_null_signal_check_no_winner() -> None:
    """Feed mock results where all strategies perform similarly -> PASS."""
    results = []
    for seed in range(5):
        for strategy in ["random", "surrogate_only", "causal"]:
            results.append(
                _make_benchmark_result(
                    strategy=strategy,
                    budget=40,
                    seed=seed,
                    # All strategies have similar MAE (~100) with small noise
                    test_mae=100.0 + seed * 0.5,
                )
            )

    verdict = check_null_signal(
        results=results,
        strategies=["random", "surrogate_only", "causal"],
        threshold=0.02,
    )
    assert verdict.verdict == "PASS"
    assert not verdict.has_consistent_winner


# ── test_null_signal_check_false_winner ───────────────────────────────


def test_null_signal_check_false_winner() -> None:
    """Feed mock results where one strategy appears better -> WARN."""
    results = []
    for seed in range(5):
        for strategy in ["random", "surrogate_only", "causal"]:
            # Make causal consistently "better" by >5%
            base_mae = 100.0 + seed * 0.5
            mae = base_mae * 0.90 if strategy == "causal" else base_mae
            results.append(
                _make_benchmark_result(
                    strategy=strategy,
                    budget=40,
                    seed=seed,
                    test_mae=mae,
                )
            )

    verdict = check_null_signal(
        results=results,
        strategies=["random", "surrogate_only", "causal"],
        threshold=0.02,
    )
    assert verdict.verdict == "WARN"
    assert verdict.has_consistent_winner


# ── test_null_signal_check_high_variance_winner ───────────────────────


def test_null_signal_check_high_variance_winner() -> None:
    """Strategy appears better on some seeds but not others -> WARN with note."""
    results = []
    for seed in range(5):
        for strategy in ["random", "surrogate_only", "causal"]:
            base_mae = 100.0
            if strategy == "causal":
                # Wins on some seeds, loses on others
                mae = base_mae * (0.85 if seed < 3 else 1.15)
            else:
                mae = base_mae + seed * 0.5
            results.append(
                _make_benchmark_result(
                    strategy=strategy,
                    budget=40,
                    seed=seed,
                    test_mae=mae,
                )
            )

    verdict = check_null_signal(
        results=results,
        strategies=["random", "surrogate_only", "causal"],
        threshold=0.02,
    )
    # causal mean MAE ≈ 97 vs random ≈ 101, ~3.96% improvement exceeds 2% threshold
    assert verdict.verdict == "WARN"
    assert verdict.has_consistent_winner
