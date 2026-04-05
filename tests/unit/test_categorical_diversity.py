"""Tests for categorical diversity injection in Ax candidate batches (Sprint 24).

The B80 lock-in failure is caused by Ax generating candidates that all share
the same categorical value (e.g., treat_day_filter="weekday"). The
inject_categorical_diversity() function ensures every value of every
categorical variable appears in at least one candidate before alignment-only
re-ranking.

Covers:
1. Categorical diversity guarantee (all values represented after injection)
2. No regression when no categorical variables exist
3. Large categorical domain (more values than batch size)
4. Fallback when Ax returns fewer candidates than expected (single candidate)
5. Multiple categorical variables
6. Integration with _suggest_bayesian (end-to-end Ax path)
"""

from __future__ import annotations

from typing import Any

import pytest

from causal_optimizer.optimizer.suggest import inject_categorical_diversity
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_search_space_with_categorical() -> SearchSpace:
    """Search space with 3 continuous vars + 1 categorical (3 choices)."""
    return SearchSpace(
        variables=[
            Variable(name="temp", variable_type=VariableType.CONTINUOUS, lower=15.0, upper=25.0),
            Variable(name="hour_start", variable_type=VariableType.INTEGER, lower=6, upper=18),
            Variable(name="hour_end", variable_type=VariableType.INTEGER, lower=18, upper=23),
            Variable(
                name="day_filter",
                variable_type=VariableType.CATEGORICAL,
                choices=["all", "weekday", "weekend"],
            ),
        ]
    )


def _make_search_space_no_categorical() -> SearchSpace:
    """Search space with only continuous/integer variables."""
    return SearchSpace(
        variables=[
            Variable(name="X1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X3", variable_type=VariableType.INTEGER, lower=0, upper=20),
        ]
    )


def _make_search_space_large_categorical() -> SearchSpace:
    """Search space with a categorical variable that has 8 choices."""
    return SearchSpace(
        variables=[
            Variable(name="X1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(
                name="color",
                variable_type=VariableType.CATEGORICAL,
                choices=["red", "orange", "yellow", "green", "blue", "indigo", "violet", "black"],
            ),
        ]
    )


def _make_search_space_two_categoricals() -> SearchSpace:
    """Search space with two categorical variables."""
    return SearchSpace(
        variables=[
            Variable(name="X1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(
                name="day_filter",
                variable_type=VariableType.CATEGORICAL,
                choices=["all", "weekday", "weekend"],
            ),
            Variable(
                name="method",
                variable_type=VariableType.CATEGORICAL,
                choices=["A", "B", "C", "D"],
            ),
        ]
    )


def _make_uniform_batch(
    n: int,
    categorical_value: str,
    categorical_name: str = "day_filter",
) -> list[dict[str, Any]]:
    """Create a batch of n candidates all sharing the same categorical value."""
    return [
        {
            "temp": 20.0 + i * 0.1,
            "hour_start": 10,
            "hour_end": 21,
            categorical_name: categorical_value,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: Categorical diversity guarantee
# ---------------------------------------------------------------------------


class TestCategoricalDiversityGuarantee:
    """Given a batch where all candidates share one categorical value,
    the injected batch must contain at least one candidate for every value."""

    def test_all_weekday_gets_all_and_weekend_injected(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = _make_uniform_batch(5, "weekday")

        result = inject_categorical_diversity(batch, ss)

        # All 3 values must appear
        day_values = {c["day_filter"] for c in result}
        assert day_values == {"all", "weekday", "weekend"}

    def test_original_candidates_preserved(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = _make_uniform_batch(5, "weekday")

        result = inject_categorical_diversity(batch, ss)

        # The first 5 candidates should be the originals (unchanged)
        for i in range(5):
            assert result[i] is batch[i]

    def test_diversity_candidates_copy_first_candidate_continuous_values(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = _make_uniform_batch(5, "weekday")

        result = inject_categorical_diversity(batch, ss)

        # Diversity candidates should copy continuous values from the first candidate
        injected = [c for c in result if c["day_filter"] != "weekday"]
        for candidate in injected:
            assert candidate["temp"] == pytest.approx(batch[0]["temp"])
            assert candidate["hour_start"] == batch[0]["hour_start"]
            assert candidate["hour_end"] == batch[0]["hour_end"]

    def test_batch_size_grows_by_missing_count(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = _make_uniform_batch(5, "weekday")

        result = inject_categorical_diversity(batch, ss)

        # 3 choices, 1 already present -> 2 injected -> total 7
        assert len(result) == 7

    def test_no_injection_when_all_values_present(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = [
            {"temp": 20.0, "hour_start": 10, "hour_end": 21, "day_filter": "all"},
            {"temp": 21.0, "hour_start": 11, "hour_end": 21, "day_filter": "weekday"},
            {"temp": 22.0, "hour_start": 12, "hour_end": 21, "day_filter": "weekend"},
            {"temp": 23.0, "hour_start": 10, "hour_end": 20, "day_filter": "all"},
            {"temp": 24.0, "hour_start": 10, "hour_end": 22, "day_filter": "weekday"},
        ]

        result = inject_categorical_diversity(batch, ss)

        # No new candidates needed
        assert len(result) == 5
        assert result is batch  # should return the same list when no changes needed


# ---------------------------------------------------------------------------
# Test 2: No regression when no categorical variables
# ---------------------------------------------------------------------------


class TestNoCategoricalNoOp:
    """When the search space has no categorical variables, the function
    returns the batch unchanged."""

    def test_continuous_only_returns_same_batch(self) -> None:
        ss = _make_search_space_no_categorical()
        batch = [
            {"X1": 3.0, "X2": 7.0, "X3": 10},
            {"X1": 5.0, "X2": 2.0, "X3": 15},
        ]

        result = inject_categorical_diversity(batch, ss)

        assert result is batch
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test 3: Large categorical domain
# ---------------------------------------------------------------------------


class TestLargeCategoricalDomain:
    """When a categorical variable has more values than the original batch
    size, all missing values should still be injected."""

    def test_eight_categories_from_five_candidates(self) -> None:
        ss = _make_search_space_large_categorical()
        batch = [{"X1": float(i), "color": "red"} for i in range(5)]

        result = inject_categorical_diversity(batch, ss)

        # 8 choices, 1 present -> 7 injected -> total 12
        color_values = {c["color"] for c in result}
        expected = {"red", "orange", "yellow", "green", "blue", "indigo", "violet", "black"}
        assert color_values == expected
        assert len(result) == 12

    def test_injected_candidates_from_large_domain_have_correct_continuous(self) -> None:
        ss = _make_search_space_large_categorical()
        batch = [{"X1": 5.5, "color": "red"}]

        result = inject_categorical_diversity(batch, ss)

        for candidate in result:
            assert candidate["X1"] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Test 4: Fallback with single candidate
# ---------------------------------------------------------------------------


class TestSingleCandidateFallback:
    """When Ax returns only 1 candidate, diversity injection should still
    add candidates for all missing categorical values."""

    def test_single_candidate_gets_missing_values_injected(self) -> None:
        ss = _make_search_space_with_categorical()
        batch = [{"temp": 19.5, "hour_start": 10, "hour_end": 21, "day_filter": "weekday"}]

        result = inject_categorical_diversity(batch, ss)

        day_values = {c["day_filter"] for c in result}
        assert day_values == {"all", "weekday", "weekend"}
        assert len(result) == 3

    def test_empty_batch_returns_empty(self) -> None:
        ss = _make_search_space_with_categorical()
        batch: list[dict[str, Any]] = []

        result = inject_categorical_diversity(batch, ss)

        assert result == []


# ---------------------------------------------------------------------------
# Test 5: Multiple categorical variables
# ---------------------------------------------------------------------------


class TestMultipleCategoricals:
    """With two categorical variables, all values of BOTH must be represented."""

    def test_two_categoricals_all_values_covered(self) -> None:
        ss = _make_search_space_two_categoricals()
        batch = [{"X1": float(i), "day_filter": "weekday", "method": "A"} for i in range(5)]

        result = inject_categorical_diversity(batch, ss)

        day_values = {c["day_filter"] for c in result}
        method_values = {c["method"] for c in result}
        assert day_values == {"all", "weekday", "weekend"}
        assert method_values == {"A", "B", "C", "D"}

    def test_two_categoricals_partial_coverage(self) -> None:
        """One categorical fully covered, the other missing some values."""
        ss = _make_search_space_two_categoricals()
        batch = [
            {"X1": 1.0, "day_filter": "all", "method": "A"},
            {"X1": 2.0, "day_filter": "weekday", "method": "A"},
            {"X1": 3.0, "day_filter": "weekend", "method": "B"},
        ]

        result = inject_categorical_diversity(batch, ss)

        day_values = {c["day_filter"] for c in result}
        method_values = {c["method"] for c in result}
        assert day_values == {"all", "weekday", "weekend"}
        assert method_values == {"A", "B", "C", "D"}
        # day_filter already covered, method missing C and D -> 2 injected
        assert len(result) == 5

    def test_injected_for_second_categorical_copies_first_candidate(self) -> None:
        ss = _make_search_space_two_categoricals()
        batch = [
            {"X1": 7.0, "day_filter": "weekday", "method": "A"},
        ]

        result = inject_categorical_diversity(batch, ss)

        # Diversity candidates for day_filter should copy X1=7.0 and method="A" from first
        day_injected = [c for c in result if c["day_filter"] != "weekday"]
        for c in day_injected:
            assert c["X1"] == pytest.approx(7.0)
            # The method value in day_filter-diversity candidates is copied from first
            assert c["method"] == "A"


# ---------------------------------------------------------------------------
# Test 6: Integration with _suggest_bayesian
# ---------------------------------------------------------------------------


class TestBayesianIntegration:
    """End-to-end test: _suggest_bayesian in soft mode with a categorical
    variable should pass through inject_categorical_diversity."""

    def test_bayesian_soft_mode_covers_all_categorical_values(self) -> None:
        """In soft mode with a categorical variable, the Ax path should
        generate candidates covering all categorical values before
        re-ranking selects the best one."""
        pytest.importorskip("ax", reason="ax-platform required for Ax path test")

        from causal_optimizer.optimizer.suggest import _suggest_bayesian

        ss = _make_search_space_with_categorical()
        graph = CausalGraph(
            edges=[("temp", "objective"), ("hour_start", "objective"), ("hour_end", "objective")],
        )

        # Build experiment log with enough data for Ax to fit a GP
        import numpy as np

        rng = np.random.default_rng(42)
        results = []
        for i in range(20):
            params: dict[str, Any] = {
                "temp": float(rng.uniform(15.0, 25.0)),
                "hour_start": int(rng.integers(6, 19)),
                "hour_end": int(rng.integers(18, 24)),
                "day_filter": rng.choice(["all", "weekday", "weekend"]),
            }
            obj = -params["temp"] + params["hour_start"] - params["hour_end"]
            results.append(
                ExperimentResult(
                    experiment_id=str(i),
                    parameters=params,
                    metrics={"objective": obj},
                    status=ExperimentStatus.KEEP,
                )
            )
        log = ExperimentLog(results=results)

        # The result is a single dict (the re-ranked winner), so we cannot
        # directly inspect the candidate pool. But the function should not
        # raise, and the result should be valid.
        result = _suggest_bayesian(
            ss,
            log,
            minimize=True,
            objective_name="objective",
            focus_variables=["temp", "hour_start", "hour_end"],
            causal_softness=0.5,
            causal_graph=graph,
        )

        assert isinstance(result, dict)
        assert "day_filter" in result
        assert result["day_filter"] in ["all", "weekday", "weekend"]
        for v in ss.variables:
            assert v.name in result


class TestInjectCategoricalDiversityNoMutation:
    """inject_categorical_diversity must not mutate the input list."""

    def test_input_list_unchanged_after_injection(self) -> None:
        ss = SearchSpace(
            variables=[
                Variable(
                    name="x",
                    variable_type=VariableType.CONTINUOUS,
                    lower=0.0,
                    upper=1.0,
                ),
                Variable(
                    name="cat",
                    variable_type=VariableType.CATEGORICAL,
                    choices=["a", "b", "c"],
                ),
            ]
        )
        candidates = [{"x": 0.5, "cat": "a"}, {"x": 0.6, "cat": "a"}]
        original_len = len(candidates)

        result = inject_categorical_diversity(candidates, ss)

        # Input list should not be mutated
        assert len(candidates) == original_len, (
            f"Input list was mutated: {len(candidates)} != {original_len}"
        )
        # Result should be a new, longer list
        assert len(result) > original_len
        assert result is not candidates
