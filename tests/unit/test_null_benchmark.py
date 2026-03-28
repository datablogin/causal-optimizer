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
    VALID_STRATEGIES,
    check_null_signal,
    permute_target,
    run_null_strategy,
)
from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    split_time_frame,
)

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

    Only the ``"random"`` strategy is exercised here because ``surrogate_only``
    and ``"causal"`` run ``ExperimentEngine.run_loop()`` which is substantially
    slower and is already covered by separate integration tests. The purpose of
    this test is to verify the null-benchmark plumbing (permutation, splitting,
    runner wiring), not the engine strategies themselves.
    """
    df = _make_energy_df(n=200, seed=0)
    permuted = permute_target(df, target_col="target_load", seed=99999)
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
    # Small per-strategy offsets well within the 2% threshold (max ~0.8% of 100)
    strategy_offsets = {"random": 0.0, "surrogate_only": 0.3, "causal": -0.5}
    results = []
    for seed in range(5):
        for strategy in ["random", "surrogate_only", "causal"]:
            results.append(
                _make_benchmark_result(
                    strategy=strategy,
                    budget=40,
                    seed=seed,
                    # All strategies have similar MAE (~100) with small per-strategy noise
                    test_mae=100.0 + seed * 0.5 + strategy_offsets[strategy],
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


# ── test_permute_target_invalid_column ────────────────────────────────


def test_permute_target_invalid_column() -> None:
    """Permuting a non-existent column raises ValueError."""
    df = _make_energy_df(n=50, seed=0)
    with pytest.raises(ValueError, match="not found"):
        permute_target(df, target_col="nonexistent_column", seed=0)


# ── test_run_null_strategy_invalid_strategy ───────────────────────────


def test_run_null_strategy_invalid_strategy() -> None:
    """Passing an invalid strategy name raises ValueError."""
    df = _make_energy_df(n=50, seed=0)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_null_strategy(
            strategy="invalid_strategy",
            budget=3,
            seed=0,
            train_df=df,
            val_df=df,
            test_df=df,
        )


# ── test_check_null_signal_no_results ─────────────────────────────────


def test_run_null_strategy_returns_result_or_none() -> None:
    """run_null_strategy returns PredictiveBenchmarkResult or None."""
    df = _make_energy_df(n=200, seed=0)
    permuted = permute_target(df, target_col="target_load", seed=99999)
    train_df, val_df, test_df = split_time_frame(permuted)

    # With valid data the random strategy should return a result
    result = run_null_strategy(
        strategy="random",
        budget=2,
        seed=0,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    assert result is None or isinstance(result, PredictiveBenchmarkResult)


def test_check_null_signal_no_results() -> None:
    """check_null_signal with empty results returns ERROR verdict."""
    verdict = check_null_signal(
        results=[],
        strategies=["random", "surrogate_only", "causal"],
    )
    assert verdict.verdict == "ERROR"
    assert not verdict.has_consistent_winner


def test_check_null_signal_flags_zero_result_strategies() -> None:
    """Strategies with no valid results are flagged in details."""
    # Only provide results for 'random'; 'causal' has no results
    results = [
        _make_benchmark_result(strategy="random", budget=40, seed=s, test_mae=100.0)
        for s in range(3)
    ]
    verdict = check_null_signal(
        results=results,
        strategies=["random", "causal"],
    )
    assert verdict.verdict == "PASS"
    assert any("causal: no valid results" in d for d in verdict.details)


# ── test_null_signal_check_budget_masking ─────────────────────────────


def test_null_signal_check_budget_masking() -> None:
    """Per-budget check catches a false win hidden by cross-budget averaging.

    causal is 10% worse than random at budget=20, and 10% better at
    budget=40.  Pooled, these cancel to ~0% difference -> pooled PASS.
    But the per-budget check should flag budget=40 -> overall WARN.
    """
    results = []
    for seed in range(5):
        base_mae = 100.0 + seed * 0.5
        # budget=20: causal 10% WORSE (higher MAE) than random
        results.append(
            _make_benchmark_result(strategy="random", budget=20, seed=seed, test_mae=base_mae)
        )
        results.append(
            _make_benchmark_result(
                strategy="causal", budget=20, seed=seed, test_mae=base_mae * 1.10
            )
        )
        # budget=40: causal 10% BETTER (lower MAE) than random
        results.append(
            _make_benchmark_result(strategy="random", budget=40, seed=seed, test_mae=base_mae)
        )
        results.append(
            _make_benchmark_result(
                strategy="causal", budget=40, seed=seed, test_mae=base_mae * 0.90
            )
        )

    verdict = check_null_signal(
        results=results,
        strategies=["random", "causal"],
        threshold=0.02,
    )
    # The per-budget check should catch the budget=40 false win
    assert verdict.verdict == "WARN", (
        f"Expected WARN due to per-budget false win, got {verdict.verdict}. "
        f"Details: {verdict.details}"
    )
    assert verdict.has_consistent_winner
    # Verify at least one detail line mentions budget=40
    assert any("budget=40" in d and "WARNING" in d for d in verdict.details)


# ── test_run_null_strategy_returns_none_on_crash ──────────────────────


def test_run_null_strategy_returns_none_on_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_null_strategy returns None when all experiments crash (best_params is None)."""
    from unittest.mock import MagicMock

    from causal_optimizer.benchmarks import null_predictive_energy as mod
    from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter

    df = _make_energy_df(n=200, seed=0)
    permuted = permute_target(df, target_col="target_load", seed=99999)
    train_df, val_df, test_df = split_time_frame(permuted)

    # Build a real adapter to get a valid search space
    real_adapter = EnergyLoadAdapter(data=pd.concat([train_df, val_df], ignore_index=True), seed=0)
    real_space = real_adapter.get_search_space()

    # Mock runner whose .run() always returns NaN mae
    mock_runner = MagicMock()
    mock_runner.run.return_value = {"mae": float("nan")}

    # Patch constructors so run_null_strategy uses our mocks
    monkeypatch.setattr(
        mod, "EnergyLoadAdapter", lambda **_kw: MagicMock(get_search_space=lambda: real_space)
    )
    monkeypatch.setattr(mod, "ValidationEnergyRunner", lambda **_kw: mock_runner)

    result = run_null_strategy(
        strategy="random",
        budget=3,
        seed=0,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    # NaN < inf is False, so best_params stays None -> returns None
    assert result is None


# ── test_valid_strategies_is_public ───────────────────────────────────


def test_valid_strategies_is_public() -> None:
    """VALID_STRATEGIES is importable and contains expected members."""
    assert isinstance(VALID_STRATEGIES, frozenset)
    assert {"random", "surrogate_only", "causal"} == VALID_STRATEGIES
