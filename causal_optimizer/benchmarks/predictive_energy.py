"""Predictive energy benchmark harness — data loading, splitting, and evaluation.

Stub file: all functions raise NotImplementedError until implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def load_energy_frame(data_path: str, area_id: str | None = None) -> pd.DataFrame:
    """Load CSV or Parquet energy data, optionally filtering by area_id."""
    raise NotImplementedError


def split_time_frame(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train, val, test partitions."""
    raise NotImplementedError


class ValidationEnergyRunner:
    """Validation-phase runner that delegates to EnergyLoadAdapter."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run an experiment using the validation split."""
        raise NotImplementedError


def evaluate_on_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int | None = None,
) -> dict[str, float]:
    """One-shot evaluation on the held-out test set."""
    raise NotImplementedError


@dataclass
class PredictiveBenchmarkResult:
    """Result of running one strategy on the predictive energy benchmark."""

    strategy: str
    budget: int
    seed: int
    best_validation_mae: float
    test_mae: float
    validation_test_gap: float
    selected_parameters: dict[str, Any]
    runtime_seconds: float
