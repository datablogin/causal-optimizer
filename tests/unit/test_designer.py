"""Tests for experimental design generators."""

from causal_optimizer.designer.factorial import FactorialDesigner
from causal_optimizer.types import SearchSpace, Variable, VariableType


def make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="a", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            Variable(name="b", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            Variable(name="c", variable_type=VariableType.BOOLEAN),
        ]
    )


def test_full_factorial():
    designer = FactorialDesigner(make_search_space())
    designs = designer.full_factorial(levels=2)
    # 3 variables at 2 levels = 8 combinations
    assert len(designs) == 8
    assert all(isinstance(d, dict) for d in designs)
    assert all("a" in d and "b" in d and "c" in d for d in designs)


def test_latin_hypercube():
    designer = FactorialDesigner(make_search_space())
    designs = designer.latin_hypercube(n_samples=20)
    assert len(designs) == 20
    # Values should be within bounds
    for d in designs:
        assert 0.0 <= d["a"] <= 1.0
        assert 0.0 <= d["b"] <= 1.0
        assert isinstance(d["c"], bool)


def test_categorical_variables():
    space = SearchSpace(
        variables=[
            Variable(
                name="method",
                variable_type=VariableType.CATEGORICAL,
                choices=["A", "B", "C"],
            ),
        ]
    )
    designer = FactorialDesigner(space)
    designs = designer.full_factorial(levels=3)
    assert len(designs) == 3
    methods = {d["method"] for d in designs}
    assert methods == {"A", "B", "C"}
