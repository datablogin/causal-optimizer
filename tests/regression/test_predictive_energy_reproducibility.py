"""Reproducibility regression test for the predictive energy benchmark.

Runs the benchmark harness twice with identical parameters and asserts that
``best_validation_mae`` and ``test_mae`` are exactly equal across runs.
This catches non-determinism from unseeded RNG, floating-point ordering
changes, or data-dependent race conditions.

Covers ``random`` (direct sampling), ``surrogate_only`` (engine with RF
surrogate), and ``causal`` (engine with prior graph) strategies since they
exercise different code paths.

All tests in this file are marked ``@pytest.mark.slow`` because even a
budget of 3 takes several seconds per strategy on the fixture data.
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

# Import run_strategy from the scripts directory.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from energy_predictive_benchmark import run_strategy  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"

_BUDGET = 3
_SEED = 42
_SPLIT_FRACS = (0.5, 0.25)  # relaxed for 200-row fixture


@pytest.fixture(scope="module")
def split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load fixture CSV and split with relaxed fractions."""
    df = pd.read_csv(FIXTURE_PATH)
    return split_time_frame(df, train_frac=_SPLIT_FRACS[0], val_frac=_SPLIT_FRACS[1])


@pytest.mark.slow
class TestRandomReproducibility:
    """Two identical runs of strategy='random' must produce identical metrics."""

    @pytest.fixture(scope="class")
    def run_pair(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult]:
        """Run strategy='random' twice with the same seed and return both results."""
        train, val, test = split_frames
        r1 = run_strategy("random", _BUDGET, _SEED, train, val, test)
        r2 = run_strategy("random", _BUDGET, _SEED, train, val, test)
        assert r1 is not None
        assert r2 is not None
        return r1, r2

    def test_validation_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.best_validation_mae == r2.best_validation_mae

    def test_test_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.test_mae == r2.test_mae

    def test_selected_parameters_are_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.selected_parameters == r2.selected_parameters


@pytest.mark.slow
class TestSurrogateOnlyReproducibility:
    """Two identical runs of strategy='surrogate_only' must produce identical metrics.

    The surrogate_only strategy exercises ExperimentEngine with an RF surrogate,
    which has more internal RNG state than direct random sampling.
    """

    @pytest.fixture(scope="class")
    def run_pair(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult]:
        """Run strategy='surrogate_only' twice with the same seed."""
        train, val, test = split_frames
        r1 = run_strategy("surrogate_only", _BUDGET, _SEED, train, val, test)
        r2 = run_strategy("surrogate_only", _BUDGET, _SEED, train, val, test)
        assert r1 is not None
        assert r2 is not None
        return r1, r2

    def test_validation_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.best_validation_mae == r2.best_validation_mae

    def test_test_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.test_mae == r2.test_mae

    def test_selected_parameters_are_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.selected_parameters == r2.selected_parameters


@pytest.mark.slow
class TestCausalReproducibility:
    """Two identical runs of strategy='causal' must produce identical metrics.

    The causal strategy exercises ExperimentEngine with the prior graph,
    which activates POMIS-guided focus variables and causal ancestor
    selection — a different code path from surrogate_only.
    """

    @pytest.fixture(scope="class")
    def run_pair(
        self, split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult]:
        """Run strategy='causal' twice with the same seed."""
        train, val, test = split_frames
        r1 = run_strategy("causal", _BUDGET, _SEED, train, val, test)
        r2 = run_strategy("causal", _BUDGET, _SEED, train, val, test)
        assert r1 is not None
        assert r2 is not None
        return r1, r2

    def test_validation_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.best_validation_mae == r2.best_validation_mae

    def test_test_mae_is_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.test_mae == r2.test_mae

    def test_selected_parameters_are_reproducible(
        self,
        run_pair: tuple[PredictiveBenchmarkResult, PredictiveBenchmarkResult],
    ) -> None:
        r1, r2 = run_pair
        assert r1.selected_parameters == r2.selected_parameters
