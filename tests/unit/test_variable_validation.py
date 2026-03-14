"""Tests for Variable model validation."""

from __future__ import annotations

import pydantic
import pytest

from causal_optimizer.types import Variable, VariableType


class TestVariableValidation:
    """Tests for Variable type-constraint validation."""

    def test_continuous_requires_bounds(self) -> None:
        """CONTINUOUS variable without bounds should raise ValidationError."""
        with pytest.raises(
            pydantic.ValidationError, match="requires both 'lower' and 'upper' bounds"
        ):
            Variable(name="x", variable_type=VariableType.CONTINUOUS)

    def test_continuous_requires_lower_lt_upper(self) -> None:
        """CONTINUOUS variable with lower >= upper should raise ValidationError."""
        with pytest.raises(pydantic.ValidationError, match="lower.*must be < upper"):
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=5.0, upper=5.0)
        with pytest.raises(pydantic.ValidationError, match="lower.*must be < upper"):
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=10.0, upper=5.0)

    def test_integer_requires_bounds(self) -> None:
        """INTEGER variable without bounds should raise ValidationError."""
        with pytest.raises(
            pydantic.ValidationError, match="requires both 'lower' and 'upper' bounds"
        ):
            Variable(name="n", variable_type=VariableType.INTEGER)

    def test_integer_requires_lower_lt_upper(self) -> None:
        """INTEGER variable with lower >= upper should raise ValidationError."""
        with pytest.raises(pydantic.ValidationError, match="lower.*must be < upper"):
            Variable(name="n", variable_type=VariableType.INTEGER, lower=5, upper=5)
        with pytest.raises(pydantic.ValidationError, match="lower.*must be < upper"):
            Variable(name="n", variable_type=VariableType.INTEGER, lower=10, upper=5)

    def test_categorical_requires_choices(self) -> None:
        """CATEGORICAL variable without choices should raise ValidationError."""
        with pytest.raises(pydantic.ValidationError, match="requires non-empty 'choices'"):
            Variable(name="color", variable_type=VariableType.CATEGORICAL)

    def test_categorical_empty_choices_raises(self) -> None:
        """CATEGORICAL variable with empty choices list should raise ValidationError."""
        with pytest.raises(pydantic.ValidationError, match="requires non-empty 'choices'"):
            Variable(name="color", variable_type=VariableType.CATEGORICAL, choices=[])

    def test_boolean_no_bounds_ok(self) -> None:
        """BOOLEAN variable should not require bounds or choices."""
        var = Variable(name="flag", variable_type=VariableType.BOOLEAN)
        assert var.name == "flag"
        assert var.variable_type == VariableType.BOOLEAN

    def test_valid_continuous_passes(self) -> None:
        """CONTINUOUS variable with valid bounds should construct successfully."""
        var = Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)
        assert var.lower == 0.0
        assert var.upper == 10.0

    def test_valid_integer_passes(self) -> None:
        """INTEGER variable with valid bounds should construct successfully."""
        var = Variable(name="n", variable_type=VariableType.INTEGER, lower=1, upper=10)
        assert var.lower == 1
        assert var.upper == 10

    def test_valid_categorical_passes(self) -> None:
        """CATEGORICAL variable with non-empty choices should construct successfully."""
        var = Variable(
            name="color",
            variable_type=VariableType.CATEGORICAL,
            choices=["red", "green", "blue"],
        )
        assert var.choices == ["red", "green", "blue"]
