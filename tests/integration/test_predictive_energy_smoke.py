"""Smoke tests for the predictive energy benchmark end-to-end pipeline.

These tests call ``run_strategy`` from the benchmark script on the 200-row
fixture dataset with a tiny budget (3 experiments).  They verify that each
strategy completes without error and produces a fully-populated
:class:`PredictiveBenchmarkResult`.

The fixture dataset (200 rows) requires relaxed split fractions
(0.5/0.25/0.25) so that every partition has at least 10 rows after the
``_MIN_PARTITION_ROWS`` check in ``split_time_frame``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from causal_optimizer.benchmarks.predictive_energy import (
    PredictiveBenchmarkResult,
    split_time_frame,
)

# The benchmark script lives in scripts/, not in a package.  Add it to
# sys.path so we can import ``run_strategy`` directly.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from energy_predictive_benchmark import run_strategy  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"

_BUDGET = 3
_SEED = 0
_SPLIT_FRACS = (0.5, 0.25)  # train_frac, val_frac — relaxed for 200-row fixture


@pytest.fixture(scope="module")
def split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load fixture CSV and split into train/val/test with relaxed fractions."""
    df = pd.read_csv(FIXTURE_PATH)
    return split_time_frame(df, train_frac=_SPLIT_FRACS[0], val_frac=_SPLIT_FRACS[1])


class TestSmokeBenchmarkRandom:
    """Smoke test: random strategy completes and produces valid result."""

    def test_random_returns_benchmark_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("random", _BUDGET, _SEED, train, val, test)
        assert isinstance(result, PredictiveBenchmarkResult)

    def test_random_all_fields_populated(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("random", _BUDGET, _SEED, train, val, test)
        assert result is not None
        assert result.strategy == "random"
        assert result.budget == _BUDGET
        assert result.seed == _SEED
        assert result.test_mae > 0
        assert result.best_validation_mae > 0
        assert result.runtime_seconds > 0
        assert isinstance(result.selected_parameters, dict)
        assert len(result.selected_parameters) > 0

    def test_random_gap_is_computed(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("random", _BUDGET, _SEED, train, val, test)
        assert result is not None
        assert result.validation_test_gap == pytest.approx(
            result.test_mae - result.best_validation_mae
        )


class TestSmokeBenchmarkSurrogateOnly:
    """Smoke test: surrogate_only strategy completes and produces valid result."""

    def test_surrogate_only_returns_benchmark_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("surrogate_only", _BUDGET, _SEED, train, val, test)
        assert result is not None
        assert isinstance(result, PredictiveBenchmarkResult)
        assert result.strategy == "surrogate_only"
        assert result.test_mae > 0
        assert result.best_validation_mae > 0
        assert result.runtime_seconds > 0


class TestSmokeBenchmarkCausal:
    """Smoke test: causal strategy completes and produces valid result."""

    def test_causal_returns_benchmark_result(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        train, val, test = split_frames
        result = run_strategy("causal", _BUDGET, _SEED, train, val, test)
        assert result is not None
        assert isinstance(result, PredictiveBenchmarkResult)
        assert result.strategy == "causal"
        assert result.test_mae > 0
        assert result.best_validation_mae > 0
        assert result.runtime_seconds > 0
