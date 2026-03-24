"""Energy load forecasting optimization adapter.

Optimizes time-series forecasting models for electricity load prediction.
The adapter trains sklearn models (Ridge, RandomForest, GradientBoosting)
on historical load data with configurable features and hyperparameters.

Search space variables control:
  - Model type: ridge, rf, gbm
  - Feature engineering: lookback window, temperature/humidity/calendar toggles
  - Regularization and ensemble size

Prior causal graph encodes domain knowledge:
  lookback_window -> mae (more history = better lag features)
  use_temperature -> mae (temperature drives heating/cooling load)
  use_calendar -> mae (hour-of-day and day-of-week drive demand patterns)
  model_type -> mae, runtime_seconds (model choice affects both)
  n_estimators -> runtime_seconds (more trees = slower)
  regularization -> mae (controls overfitting)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from causal_optimizer.domain_adapters.base import DomainAdapter
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType

logger = logging.getLogger(__name__)


class EnergyLoadAdapter(DomainAdapter):
    """Adapter for energy load forecasting optimization.

    Optimizes time-series model configuration (model type, features,
    hyperparameters) to minimize MAE on a held-out validation window.

    Args:
        data: DataFrame with columns ``timestamp``, ``target_load``, and
            at least one covariate (e.g. ``temperature``).
        data_path: Path to a CSV or Parquet file (alternative to *data*).
        seed: Random seed for reproducibility.
        train_ratio: Fraction of data used for training (default 0.7).
            The remainder is the validation set.  Split is blocked by time
            (no shuffling) to prevent leakage.
        split_timestamp: If provided, the train/validation boundary is
            determined by this timestamp instead of ``train_ratio``.  All
            rows with ``timestamp < split_timestamp`` are training; the
            rest are validation.  This survives lag-induced NaN dropping
            and prevents leakage in benchmark harnesses.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_path: str | None = None,
        seed: int | None = None,
        train_ratio: float = 0.7,
        split_timestamp: pd.Timestamp | None = None,
    ) -> None:
        if data is not None:
            self._data = data.copy()
        elif data_path is not None:
            if Path(data_path).suffix == ".parquet":
                self._data = pd.read_parquet(data_path)
            else:
                self._data = pd.read_csv(data_path)
        else:
            raise ValueError("Either 'data' or 'data_path' must be provided")

        self._seed = seed
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}")
        self._train_ratio = train_ratio

        self._validate_data()

        # Parse, sort, and validate timestamps
        try:
            self._data["timestamp"] = pd.to_datetime(self._data["timestamp"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column 'timestamp' could not be parsed as datetime: {exc}") from exc
        self._data = self._data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        n_dupes = int(self._data["timestamp"].duplicated().sum())
        if n_dupes > 0:
            raise ValueError(
                f"Found {n_dupes} duplicate timestamps. This adapter requires single-series "
                "data with unique timestamps. If you have multi-area data, filter to one area "
                "before passing."
            )

        # Infer dominant cadence and precompute cadence metrics.
        # These reflect the raw input series, not the post-NaN-drop training data.
        diffs = self._data["timestamp"].diff().dropna()
        n_diffs = len(diffs)
        if n_diffs > 0:
            mode_vals = diffs.mode().sort_values().reset_index(drop=True)
            if len(mode_vals) > 1:
                logger.warning(
                    "Ambiguous cadence: %d equally-frequent intervals found; "
                    "using smallest (%s) as dominant cadence.",
                    len(mode_vals),
                    mode_vals.iloc[0],
                )
            self._cadence: pd.Timedelta = mode_vals.iloc[0]
            tolerance = self._cadence * 0.1
            regular_count = int(((diffs - self._cadence).abs() <= tolerance).sum())
            self._cadence_regularity = float(regular_count / n_diffs)
            self._cadence_gaps = float((diffs > self._cadence * 1.5).sum())
        else:
            self._cadence = pd.Timedelta(0)
            self._cadence_regularity = 1.0
            self._cadence_gaps = 0.0

        self._split_timestamp = split_timestamp

        n = len(self._data)
        self._train_end = int(n * self._train_ratio)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_data(self) -> None:
        if self._data.empty:
            raise ValueError("Empty DataFrame: data must contain at least one row")

        required = {"timestamp", "target_load"}
        missing = required - set(self._data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # target_load must be numeric
        if not pd.api.types.is_numeric_dtype(self._data["target_load"]):
            raise ValueError("Column 'target_load' must be numeric")

        # At least one covariate column beyond timestamp and target_load
        covariates = set(self._data.columns) - {"timestamp", "target_load"}
        if not covariates:
            raise ValueError(
                "Data must contain at least one covariate column "
                "besides 'timestamp' and 'target_load'"
            )

    # ------------------------------------------------------------------
    # DomainAdapter interface
    # ------------------------------------------------------------------

    def get_search_space(self) -> SearchSpace:
        return SearchSpace(
            variables=[
                Variable(
                    name="model_type",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["ridge", "rf", "gbm"],
                ),
                Variable(
                    name="lookback_window",
                    variable_type=VariableType.INTEGER,
                    lower=1,
                    upper=48,
                ),
                Variable(
                    name="use_temperature",
                    variable_type=VariableType.BOOLEAN,
                ),
                Variable(
                    name="use_humidity",
                    variable_type=VariableType.BOOLEAN,
                ),
                Variable(
                    name="use_calendar",
                    variable_type=VariableType.BOOLEAN,
                ),
                Variable(
                    name="regularization",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.001,
                    upper=10.0,
                ),
                Variable(
                    name="n_estimators",
                    variable_type=VariableType.INTEGER,
                    lower=10,
                    upper=200,
                ),
            ]
        )

    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Train a model and evaluate on the validation window.

        Steps:
        1. Build a feature matrix from the selected features + lagged load.
        2. Train on the training split, predict on validation.
        3. Return MAE, RMSE, MAPE, runtime_seconds, feature_count,
           validation_set_size, nan_rows_dropped, train_fraction_actual,
           cadence_gaps, cadence_regularity.

        Note: ``cadence_gaps`` and ``cadence_regularity`` are properties of the
        raw input series (computed once at init), not of the post-NaN-drop
        training/validation data.
        """
        t_start = time.monotonic()

        # Extract parameters with defaults
        model_type = parameters.get("model_type", "ridge")
        lookback = int(parameters.get("lookback_window", 3))
        use_temp = parameters.get("use_temperature", True)
        use_humid = parameters.get("use_humidity", False)
        use_cal = parameters.get("use_calendar", True)
        reg = float(parameters.get("regularization", 1.0))
        n_est = int(parameters.get("n_estimators", 50))

        # Build feature matrix
        df = self._data.copy()
        features: list[str] = []

        # Lagged load features (always included — past data only)
        for lag in range(1, lookback + 1):
            col = f"load_lag_{lag}"
            df[col] = df["target_load"].shift(lag)
            features.append(col)

        # Optional features
        if use_temp and "temperature" in df.columns:
            features.append("temperature")
        if use_humid and "humidity" in df.columns:
            features.append("humidity")
        if use_cal:
            for cal_col in ("hour_of_day", "day_of_week", "is_holiday"):
                if cal_col in df.columns:
                    features.append(cal_col)

        if not features:
            raise ValueError("No features selected. Increase lookback_window or enable covariates.")

        # Handle missing values: forward-fill then drop remaining NaN
        df[features] = df[features].ffill()
        mask = df[features + ["target_load"]].notna().all(axis=1)
        # Includes lag-induced NaN rows (first `lookback` rows), so
        # nan_rows_dropped >= lookback even for perfect data.
        n_original = len(df)
        df = df[mask].reset_index(drop=True)
        nan_rows_dropped = float(n_original - len(df))

        # Recompute split after dropping rows.
        # When split_timestamp is set, use it to find the exact boundary
        # in the post-drop frame — this prevents lag-induced NaN drops from
        # shifting the split and leaking validation rows into training.
        if self._split_timestamp is not None and "timestamp" in df.columns:
            train_mask = df["timestamp"] < self._split_timestamp
            train_end = int(train_mask.sum())
            if train_end == 0:
                raise ValueError(
                    "Preprocessing removed all training rows before the split boundary "
                    f"({self._split_timestamp}). Try reducing lookback_window "
                    f"(current: {lookback})."
                )
            if train_end >= len(df):
                raise ValueError(
                    "Preprocessing removed all validation/test rows after the split "
                    f"boundary ({self._split_timestamp}). Try reducing lookback_window "
                    f"(current: {lookback})."
                )
        else:
            train_end = min(self._train_end, len(df) - 1)
            train_end = max(1, train_end)  # ensure at least 1 training row

        x_train = df.loc[: train_end - 1, features].values
        y_train = df.loc[: train_end - 1, "target_load"].values
        x_val = df.loc[train_end:, features].values
        y_val = df.loc[train_end:, "target_load"].values

        if len(x_val) == 0:
            raise ValueError(
                f"Validation set is empty after dropping NaN rows. "
                f"Try reducing lookback_window (current: {lookback}) or train_ratio."
            )

        # Regularization: alpha for Ridge, min_samples_leaf for tree models
        min_leaf = max(1, int(reg))
        if model_type == "ridge":
            model = Ridge(alpha=reg)
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=n_est,
                min_samples_leaf=min_leaf,
                random_state=self._seed,
                n_jobs=1,
            )
        elif model_type == "gbm":
            model = GradientBoostingRegressor(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=4,
                min_samples_leaf=min_leaf,
                random_state=self._seed,
            )
        else:
            logger.warning("Unknown model_type %r, falling back to ridge", model_type)
            model = Ridge(alpha=reg)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)

        # Compute metrics
        errors = y_val - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        # MAPE — guard against zero actuals
        nonzero_mask = np.abs(y_val) > 1e-8
        if nonzero_mask.any():
            mape = float(np.mean(np.abs(errors[nonzero_mask] / y_val[nonzero_mask])) * 100)
        else:
            mape = float("inf")

        runtime = time.monotonic() - t_start
        feature_count = float(len(features))

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "runtime_seconds": float(runtime),
            "feature_count": feature_count,
            "validation_set_size": float(len(x_val)),
            "nan_rows_dropped": nan_rows_dropped,
            "train_fraction_actual": float(train_end / len(df)) if len(df) > 0 else 0.0,
            "cadence_gaps": self._cadence_gaps,
            "cadence_regularity": self._cadence_regularity,
        }

    def get_prior_graph(self) -> CausalGraph:
        """Energy load forecasting causal graph based on domain knowledge.

        Note: ``cadence_gaps`` and ``cadence_regularity`` are diagnostic-only
        metrics not modeled in this graph.  They describe input data quality,
        not outcomes the optimizer should target.
        """
        return CausalGraph(
            edges=[
                ("lookback_window", "mae"),
                ("lookback_window", "runtime_seconds"),
                ("use_temperature", "mae"),
                ("use_humidity", "mae"),
                ("use_calendar", "mae"),
                ("regularization", "mae"),
                ("model_type", "mae"),
                ("model_type", "runtime_seconds"),
                ("n_estimators", "mae"),
                ("n_estimators", "runtime_seconds"),
            ],
        )

    def get_descriptor_names(self) -> list[str]:
        return ["runtime_seconds", "feature_count"]

    def get_objective_name(self) -> str:
        return "mae"

    def get_minimize(self) -> bool:
        return True
