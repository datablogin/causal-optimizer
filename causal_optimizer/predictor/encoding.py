"""Feature encoding for Random Forest surrogate models.

Provides consistent encoding of mixed-type variables (continuous, integer,
categorical, boolean) for use in sklearn RandomForestRegressor. Used by
both the off-policy predictor and the surrogate-guided suggestion strategy.

Note on encoding scheme: categorical variables are label-encoded (mapped to
ordinal integers). Random Forests are robust to this since they split on
thresholds, but it introduces a false ordering. If this module is extended
to other model types, consider one-hot encoding instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from causal_optimizer.types import Variable, VariableType

if TYPE_CHECKING:
    from causal_optimizer.types import SearchSpace

logger = logging.getLogger(__name__)


def encode_dataframe_for_rf(
    df: pd.DataFrame,
    var_names: list[str],
    search_space: SearchSpace,
) -> np.ndarray:
    """Encode a DataFrame of mixed-type variables into a numeric array for RF.

    Categorical variables are label-encoded (mapped to integers based on
    ``var.choices`` ordering). Boolean variables are converted to 0/1.
    Continuous and integer variables are passed through as floats.

    Args:
        df: DataFrame with experiment data.
        var_names: Variable names to include (column subset of df).
        search_space: Search space with variable type metadata.

    Returns:
        A 2D numpy float64 array of shape (n_rows, len(var_names)).
    """
    var_types = {v.name: v for v in search_space.variables}
    encoded_cols: list[np.ndarray] = []

    for name in var_names:
        if name not in df.columns:
            encoded_cols.append(np.zeros(len(df), dtype=np.float64))
            continue

        col = df[name]
        var = var_types.get(name)

        if var is not None and var.variable_type == VariableType.CATEGORICAL:
            choices = _get_categorical_choices(var, name, col)
            mapping = {c: float(i) for i, c in enumerate(choices)}
            encoded_cols.append(col.map(mapping).fillna(0.0).to_numpy(dtype=np.float64))
        elif var is not None and var.variable_type == VariableType.BOOLEAN:
            encoded_cols.append(col.astype(float).fillna(0.0).to_numpy(dtype=np.float64))
        else:
            encoded_cols.append(
                pd.to_numeric(col, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            )

    if not encoded_cols:
        return np.empty((len(df), 0), dtype=np.float64)
    return np.column_stack(encoded_cols)


def encode_params_for_rf(
    parameters: dict[str, Any],
    var_names: list[str],
    search_space: SearchSpace,
) -> np.ndarray:
    """Encode a single parameter dict into a numeric array for RF prediction.

    Uses the same encoding scheme as :func:`encode_dataframe_for_rf` so that
    the feature representation is consistent between fit and predict.

    Returns:
        A 2D numpy float64 array of shape (1, len(var_names)).
    """
    var_types = {v.name: v for v in search_space.variables}
    values: list[float] = []

    for name in var_names:
        raw = parameters.get(name, 0)
        var = var_types.get(name)

        if var is not None and var.variable_type == VariableType.CATEGORICAL:
            choices = var.choices or []
            if not choices:
                logger.warning(
                    "Categorical variable '%s' has no choices defined; encoding as 0.0. "
                    "Set explicit choices on the Variable for consistent encoding.",
                    name,
                )
            mapping = {c: float(i) for i, c in enumerate(choices)}
            values.append(mapping.get(raw, 0.0))
        elif var is not None and var.variable_type == VariableType.BOOLEAN:
            values.append(1.0 if raw else 0.0)
        else:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(0.0)

    return np.array(values, dtype=np.float64).reshape(1, -1)


def _get_categorical_choices(
    var: Variable,
    name: str,
    col: pd.Series,
) -> list[Any]:
    """Get the canonical choice list for a categorical variable.

    Uses ``var.choices`` if defined, otherwise falls back to sorted unique
    values from the observed data. Logs a warning when falling back since
    the data-derived ordering may differ between fit and predict.
    """
    if var.choices:
        return list(var.choices)
    logger.warning(
        "Categorical variable '%s' has no choices defined; deriving order from observed data. "
        "Set explicit choices on the Variable for consistent encoding.",
        name,
    )
    return sorted(col.dropna().unique().tolist())
