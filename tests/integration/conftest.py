"""Shared fixtures for integration tests.

Centralizes the ``sys.path`` manipulation needed to import ``run_strategy``
from ``scripts/energy_predictive_benchmark.py`` and the ``split_frames``
fixture used by smoke tests in this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from causal_optimizer.benchmarks.predictive_energy import split_time_frame

# The benchmark script lives in scripts/, not in a package.  Add it to
# sys.path once so all integration tests can import ``run_strategy``.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "energy_load_fixture.csv"
_SPLIT_FRACS = (0.5, 0.25)  # train_frac, val_frac — relaxed for 200-row fixture


@pytest.fixture(scope="module")
def split_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load fixture CSV and split into train/val/test with relaxed fractions.

    The fixture dataset (200 rows) uses relaxed split fractions
    (0.5/0.25/0.25) to leave each partition with enough rows after
    lag-feature creation in ``EnergyLoadAdapter`` (up to 48 rows can be
    dropped for ``lookback_window=48``).
    """
    df = pd.read_csv(FIXTURE_PATH)
    return split_time_frame(df, train_frac=_SPLIT_FRACS[0], val_frac=_SPLIT_FRACS[1])
