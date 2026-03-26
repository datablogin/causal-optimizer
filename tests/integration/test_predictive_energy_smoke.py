"""Smoke tests for the predictive energy benchmark end-to-end pipeline.

These tests call ``run_strategy`` from the benchmark script on the 200-row
fixture dataset with a tiny budget (3 experiments).  They verify that each
strategy completes without error and produces a fully-populated
:class:`PredictiveBenchmarkResult`.

The fixture dataset (200 rows) uses relaxed split fractions
(0.5/0.25/0.25) to leave each partition with enough rows after
lag-feature creation in ``EnergyLoadAdapter`` (up to 48 rows can be
dropped for ``lookback_window=48``).

All tests are marked ``@pytest.mark.slow`` because engine-based strategies
(surrogate_only, causal) take ~8s each on the fixture data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from energy_predictive_benchmark import run_strategy

from causal_optimizer.benchmarks.predictive_energy import PredictiveBenchmarkResult

if TYPE_CHECKING:
    import pandas as pd

_BUDGET = 3
_SEED = 0


def _assert_valid_result(
    result: PredictiveBenchmarkResult | None,
    strategy: str,
    budget: int,
    seed: int,
) -> None:
    """Shared assertions for any strategy result."""
    assert result is not None, f"run_strategy('{strategy}') returned None — all experiments crashed"
    assert isinstance(result, PredictiveBenchmarkResult)
    assert result.strategy == strategy
    assert result.budget == budget
    assert result.seed == seed
    assert result.test_mae > 0
    assert math.isfinite(result.test_mae), f"test_mae is not finite: {result.test_mae}"
    assert result.best_validation_mae > 0
    assert math.isfinite(result.best_validation_mae), (
        f"best_validation_mae is not finite: {result.best_validation_mae}"
    )
    assert result.runtime_seconds > 0
    assert isinstance(result.selected_parameters, dict)
    assert len(result.selected_parameters) > 0
    # validation_test_gap is auto-computed in __post_init__
    assert result.validation_test_gap == pytest.approx(result.test_mae - result.best_validation_mae)


@pytest.mark.slow
class TestSmokeBenchmarkRandom:
    """Smoke test: random strategy completes and produces valid result."""

    @pytest.fixture(scope="class")
    def random_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> PredictiveBenchmarkResult:
        train, val, test = split_frames
        result: PredictiveBenchmarkResult | None = run_strategy(
            "random", _BUDGET, _SEED, train, val, test
        )
        assert result is not None
        return result

    def test_random_returns_valid_result(self, random_result: PredictiveBenchmarkResult) -> None:
        _assert_valid_result(random_result, "random", _BUDGET, _SEED)

    def test_random_gap_is_computed(self, random_result: PredictiveBenchmarkResult) -> None:
        assert random_result.validation_test_gap == pytest.approx(
            random_result.test_mae - random_result.best_validation_mae
        )


@pytest.mark.slow
class TestSmokeBenchmarkSurrogateOnly:
    """Smoke test: surrogate_only strategy completes and produces valid result."""

    def test_surrogate_only_returns_valid_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("surrogate_only", _BUDGET, _SEED, train, val, test)
        _assert_valid_result(result, "surrogate_only", _BUDGET, _SEED)


@pytest.mark.slow
class TestSmokeBenchmarkCausal:
    """Smoke test: causal strategy completes and produces valid result."""

    def test_causal_returns_valid_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("causal", _BUDGET, _SEED, train, val, test)
        _assert_valid_result(result, "causal", _BUDGET, _SEED)
