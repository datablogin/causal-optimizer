"""Unit tests for causal_optimizer.predictor.encoding.

Tests the feature encoding functions that convert mixed-type variables
(continuous, integer, categorical, boolean) into numeric arrays for
sklearn RandomForestRegressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.predictor.encoding import encode_dataframe_for_rf, encode_params_for_rf
from causal_optimizer.types import SearchSpace, Variable, VariableType


@pytest.fixture()
def mixed_search_space() -> SearchSpace:
    """Search space with all four variable types."""
    return SearchSpace(
        variables=[
            Variable(name="lr", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            Variable(name="epochs", variable_type=VariableType.INTEGER, lower=1, upper=100),
            Variable(
                name="optimizer",
                variable_type=VariableType.CATEGORICAL,
                choices=["adam", "sgd", "rmsprop"],
            ),
            Variable(name="use_bn", variable_type=VariableType.BOOLEAN),
        ]
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame with mixed types matching mixed_search_space."""
    return pd.DataFrame(
        {
            "lr": [0.1, 0.5, 0.9],
            "epochs": [10, 50, 100],
            "optimizer": ["adam", "sgd", "rmsprop"],
            "use_bn": [True, False, True],
        }
    )


class TestEncodeDataframeForRf:
    """Tests for encode_dataframe_for_rf."""

    def test_basic_encoding(self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame) -> None:
        """All four variable types should encode to float64."""
        var_names = ["lr", "epochs", "optimizer", "use_bn"]
        result = encode_dataframe_for_rf(sample_df, var_names, mixed_search_space)

        assert result.dtype == np.float64
        assert result.shape == (3, 4)

    def test_continuous_passthrough(
        self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame
    ) -> None:
        """Continuous variables should pass through as-is."""
        result = encode_dataframe_for_rf(sample_df, ["lr"], mixed_search_space)

        np.testing.assert_array_almost_equal(result[:, 0], [0.1, 0.5, 0.9])

    def test_integer_passthrough(
        self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame
    ) -> None:
        """Integer variables should pass through as floats."""
        result = encode_dataframe_for_rf(sample_df, ["epochs"], mixed_search_space)

        np.testing.assert_array_almost_equal(result[:, 0], [10.0, 50.0, 100.0])

    def test_categorical_label_encoding(
        self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame
    ) -> None:
        """Categorical variables should be label-encoded based on choices order."""
        result = encode_dataframe_for_rf(sample_df, ["optimizer"], mixed_search_space)

        # choices=["adam", "sgd", "rmsprop"] -> adam=0, sgd=1, rmsprop=2
        np.testing.assert_array_almost_equal(result[:, 0], [0.0, 1.0, 2.0])

    def test_boolean_encoding(
        self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame
    ) -> None:
        """Boolean variables should encode as 0.0/1.0."""
        result = encode_dataframe_for_rf(sample_df, ["use_bn"], mixed_search_space)

        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 0.0, 1.0])

    def test_missing_column_fills_zeros(self, mixed_search_space: SearchSpace) -> None:
        """If a var_name is not in df.columns, encode as zeros."""
        df = pd.DataFrame({"lr": [0.1, 0.2]})
        result = encode_dataframe_for_rf(df, ["lr", "missing_col"], mixed_search_space)

        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result[:, 1], [0.0, 0.0])

    def test_empty_dataframe(self, mixed_search_space: SearchSpace) -> None:
        """Empty DataFrame should return empty array with correct shape."""
        df = pd.DataFrame({"lr": pd.Series([], dtype=float)})
        result = encode_dataframe_for_rf(df, ["lr"], mixed_search_space)

        assert result.shape == (0, 1)

    def test_empty_var_names(self, mixed_search_space: SearchSpace) -> None:
        """Empty var_names should return empty array."""
        df = pd.DataFrame({"lr": [0.1]})
        result = encode_dataframe_for_rf(df, [], mixed_search_space)

        assert result.shape == (1, 0)

    def test_unknown_categorical_value(self, mixed_search_space: SearchSpace) -> None:
        """Unknown categorical value (not in choices) should encode as 0.0 (fillna)."""
        df = pd.DataFrame({"optimizer": ["adam", "unknown_opt"]})
        result = encode_dataframe_for_rf(df, ["optimizer"], mixed_search_space)

        # "adam" -> 0.0, "unknown_opt" -> NaN -> fillna(0.0)
        np.testing.assert_array_almost_equal(result[:, 0], [0.0, 0.0])

    def test_single_row(self, mixed_search_space: SearchSpace) -> None:
        """Single-row DataFrame should work (common in predict path)."""
        df = pd.DataFrame({"lr": [0.5], "optimizer": ["sgd"]})
        result = encode_dataframe_for_rf(df, ["lr", "optimizer"], mixed_search_space)

        assert result.shape == (1, 2)
        assert result[0, 0] == pytest.approx(0.5)
        assert result[0, 1] == pytest.approx(1.0)  # sgd is index 1


class TestEncodeParamsForRf:
    """Tests for encode_params_for_rf."""

    def test_basic_encoding(self, mixed_search_space: SearchSpace) -> None:
        """Should encode a parameter dict to a (1, n_vars) array."""
        params = {"lr": 0.5, "epochs": 50, "optimizer": "sgd", "use_bn": True}
        var_names = ["lr", "epochs", "optimizer", "use_bn"]
        result = encode_params_for_rf(params, var_names, mixed_search_space)

        assert result.dtype == np.float64
        assert result.shape == (1, 4)
        assert result[0, 0] == pytest.approx(0.5)
        assert result[0, 1] == pytest.approx(50.0)
        assert result[0, 2] == pytest.approx(1.0)  # sgd is index 1
        assert result[0, 3] == pytest.approx(1.0)  # True -> 1.0

    def test_categorical_encoding_matches_dataframe(
        self, mixed_search_space: SearchSpace, sample_df: pd.DataFrame
    ) -> None:
        """Categorical encoding in params should match DataFrame encoding."""
        df_result = encode_dataframe_for_rf(sample_df, ["optimizer"], mixed_search_space)
        for i, opt in enumerate(["adam", "sgd", "rmsprop"]):
            params_result = encode_params_for_rf(
                {"optimizer": opt}, ["optimizer"], mixed_search_space
            )
            assert params_result[0, 0] == pytest.approx(df_result[i, 0]), (
                f"Mismatch for {opt}: params={params_result[0, 0]}, df={df_result[i, 0]}"
            )

    def test_unknown_categorical_value(self, mixed_search_space: SearchSpace) -> None:
        """Unknown categorical value should encode as 0.0."""
        result = encode_params_for_rf({"optimizer": "unknown"}, ["optimizer"], mixed_search_space)
        assert result[0, 0] == pytest.approx(0.0)

    def test_boolean_false(self, mixed_search_space: SearchSpace) -> None:
        """Boolean False should encode as 0.0."""
        result = encode_params_for_rf({"use_bn": False}, ["use_bn"], mixed_search_space)
        assert result[0, 0] == pytest.approx(0.0)

    def test_missing_parameter_defaults_to_zero(self, mixed_search_space: SearchSpace) -> None:
        """Missing parameter key should default to 0."""
        result = encode_params_for_rf({}, ["lr"], mixed_search_space)
        assert result[0, 0] == pytest.approx(0.0)

    def test_non_numeric_value_defaults_to_zero(self, mixed_search_space: SearchSpace) -> None:
        """Non-numeric value for a continuous var should default to 0.0."""
        result = encode_params_for_rf({"lr": "not_a_number"}, ["lr"], mixed_search_space)
        assert result[0, 0] == pytest.approx(0.0)

    def test_unlisted_choice_encodes_as_zero(self) -> None:
        """Categorical var with a value not in choices should encode as 0.0."""
        space = SearchSpace(
            variables=[
                Variable(
                    name="cat_var",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["a", "b"],
                ),
            ]
        )
        result = encode_params_for_rf({"cat_var": "anything"}, ["cat_var"], space)
        assert result[0, 0] == pytest.approx(0.0)
