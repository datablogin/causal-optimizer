"""Predictive energy benchmark harness — data loading, splitting, and evaluation.

Provides the data plumbing for benchmarking causal-optimizer strategies on
real energy load forecasting data.  The harness enforces strict temporal
ordering (no leakage) and delegates model training to
:class:`~causal_optimizer.domain_adapters.energy_load.EnergyLoadAdapter`.

Public API
----------
- :func:`load_energy_frame` — load CSV/Parquet, optional area_id filter.
- :func:`split_time_frame` — chronological train/val/test split.
- :class:`ValidationEnergyRunner` — ExperimentRunner for the validation phase.
- :func:`evaluate_on_test` — one-shot held-out test evaluation.
- :class:`PredictiveBenchmarkResult` — result container.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from causal_optimizer.domain_adapters.energy_load import EnergyLoadAdapter

_MIN_PARTITION_ROWS = 10


# ── Data loading ─────────────────────────────────────────────────────


def load_energy_frame(data_path: str, area_id: str | None = None) -> pd.DataFrame:
    """Load CSV or Parquet energy data, optionally filtering by area_id.

    Args:
        data_path: Path to a ``.csv`` or ``.parquet`` file.
        area_id: If provided, filter to rows where ``area_id == area_id``.
            Raises if the column is absent.

    Returns:
        A DataFrame with at least ``timestamp``, ``target_load``, and
        ``temperature`` columns.

    Raises:
        ValueError: On empty result, missing required columns, area_id
            filter requested but column missing, or multi-series data
            without an explicit area_id selection.
    """
    p = Path(data_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Validate required columns
    required = {"timestamp", "target_load", "temperature"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Empty result: file contains no data rows")

    # Handle area_id filtering
    if area_id is not None:
        if "area_id" not in df.columns:
            raise ValueError(
                "area_id filter requested but 'area_id' column is missing from the data"
            )
        df = df[df["area_id"] == area_id].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"Empty result after filtering to area_id={area_id!r}; "
                "no rows match that value"
            )
    elif "area_id" in df.columns and df["area_id"].nunique() > 1:
        raise ValueError(
            f"Data contains {df['area_id'].nunique()} distinct area_id values. "
            "Specify --area-id to select one."
        )

    return df


# ── Time-based splitting ─────────────────────────────────────────────


def split_time_frame(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train, val, test partitions.

    Parses the ``timestamp`` column, sorts chronologically, and splits
    by row position.  No shuffling — this is a blocked time split.

    Args:
        df: Input DataFrame (must contain a ``timestamp`` column).
        train_frac: Fraction of rows for training (default 0.6).
        val_frac: Fraction of rows for validation (default 0.2).
            The test fraction is ``1 - train_frac - val_frac``.

    Returns:
        ``(train_df, val_df, test_df)`` — three non-overlapping DataFrames.

    Raises:
        ValueError: If the DataFrame is empty, contains duplicate timestamps,
            fractions don't leave room for test, or any partition has fewer
            than 10 rows.
    """
    if df.empty:
        raise ValueError("Empty DataFrame: cannot split an empty dataset")

    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"Fractions must leave room for a test set: "
            f"train_frac ({train_frac}) + val_frac ({val_frac}) = "
            f"{train_frac + val_frac:.2f} >= 1.0"
        )

    # Parse and sort by timestamp
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Check for duplicates
    n_dupes = int(df["timestamp"].duplicated().sum())
    if n_dupes > 0:
        raise ValueError(
            f"Found {n_dupes} duplicate timestamp(s). "
            "Data must have unique timestamps for a single-series split."
        )

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # Validate minimum partition sizes
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(part) < _MIN_PARTITION_ROWS:
            raise ValueError(
                f"Partition '{name}' has only {len(part)} rows, "
                f"but the minimum is {_MIN_PARTITION_ROWS}. "
                "Increase the dataset size or adjust fractions."
            )

    return train_df, val_df, test_df


# ── Validation runner ────────────────────────────────────────────────


class ValidationEnergyRunner:
    """Validation-phase runner implementing the ExperimentRunner protocol.

    Concatenates train+val, computes train_ratio from their lengths,
    creates an :class:`EnergyLoadAdapter` with that ratio, and delegates
    to ``adapter.run_experiment()``.

    Args:
        train_df: Training partition (from :func:`split_time_frame`).
        val_df: Validation partition.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        seed: int | None = None,
    ) -> None:
        self._train_df = train_df
        self._val_df = val_df
        self._seed = seed

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Run an experiment using the validation split.

        Internally concatenates train+val, creates an EnergyLoadAdapter
        with the appropriate train_ratio, and runs the experiment.
        """
        combined = pd.concat([self._train_df, self._val_df], ignore_index=True)
        train_ratio = len(self._train_df) / len(combined)
        adapter = EnergyLoadAdapter(data=combined, seed=self._seed, train_ratio=train_ratio)
        return adapter.run_experiment(parameters)


# ── Test evaluation ──────────────────────────────────────────────────


def evaluate_on_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int | None = None,
) -> dict[str, float]:
    """One-shot evaluation on the held-out test set.

    Combines all three partitions, sets ``train_ratio = (train+val) / total``,
    creates an :class:`EnergyLoadAdapter`, and runs the experiment.  The
    adapter's validation window is thus the test set.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        parameters: Model configuration dict.
        seed: Random seed for reproducibility.

    Returns:
        Metrics dict including ``mae``, ``rmse``, ``mape``, etc.
    """
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    train_ratio = (len(train_df) + len(val_df)) / len(combined)
    adapter = EnergyLoadAdapter(data=combined, seed=seed, train_ratio=train_ratio)
    return adapter.run_experiment(parameters)


# ── Result container ─────────────────────────────────────────────────


@dataclass
class PredictiveBenchmarkResult:
    """Result of running one strategy on the predictive energy benchmark.

    Attributes:
        strategy: Optimization strategy name (e.g. ``"causal"``, ``"random"``).
        budget: Number of experiments in the optimization run.
        seed: Random seed used.
        best_validation_mae: Best MAE found during the optimization (on val set).
        test_mae: MAE on the held-out test set using the selected parameters.
        validation_test_gap: ``test_mae - best_validation_mae``.
        selected_parameters: Parameters that achieved the best validation MAE.
        runtime_seconds: Wall-clock time for the full run.
    """

    strategy: str
    budget: int
    seed: int
    best_validation_mae: float
    test_mae: float
    validation_test_gap: float
    selected_parameters: dict[str, Any]
    runtime_seconds: float
