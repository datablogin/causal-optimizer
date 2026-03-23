"""Tests for coverage analysis — discarded experiment handling (issue #48)."""

from __future__ import annotations

from causal_optimizer.diagnostics.coverage import analyze_coverage
from causal_optimizer.diagnostics.models import CoverageAnalysis
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


def _make_search_space() -> SearchSpace:
    """Search space with three continuous variables."""
    return SearchSpace(
        variables=[
            Variable(name="X1", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            Variable(name="X3", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
    )


def _make_causal_graph() -> CausalGraph:
    """Graph: X1 -> Y, X2 -> Y, X3 -> Y."""
    return CausalGraph(
        edges=[("X1", "Y"), ("X2", "Y"), ("X3", "Y")],
        nodes=["X1", "X2", "X3", "Y"],
    )


class TestDiscardedExperimentCoverage:
    """Issue #48: discarded experiments should count toward coverage."""

    def test_discard_experiment_not_reported_as_never_intervened(self) -> None:
        """A variable tested in both KEEP and DISCARD should NOT appear in
        ancestors_never_intervened."""
        log = ExperimentLog(
            results=[
                ExperimentResult(
                    experiment_id="1",
                    parameters={"X1": 1.0, "X2": 5.0, "X3": 3.0},
                    metrics={"Y": 10.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="2",
                    parameters={"X1": 2.0, "X2": 5.0, "X3": 7.0},
                    metrics={"Y": 8.0},
                    status=ExperimentStatus.DISCARD,
                ),
            ]
        )
        ss = _make_search_space()
        graph = _make_causal_graph()

        result = analyze_coverage(log, ss, "Y", causal_graph=graph)

        # X1 varied across both experiments (1.0 and 2.0) — should be intervened
        # X3 varied across both experiments (3.0 and 7.0) — should be intervened
        # X2 constant (5.0 in both) — should be never_intervened
        assert "X1" not in (result.ancestors_never_intervened or [])
        assert "X3" not in (result.ancestors_never_intervened or [])
        assert "X1" in (result.ancestors_intervened or [])
        assert "X3" in (result.ancestors_intervened or [])

    def test_only_discard_variable_still_counted(self) -> None:
        """A variable varied ONLY in DISCARD experiments should still count
        as intervened (not crash)."""
        log = ExperimentLog(
            results=[
                ExperimentResult(
                    experiment_id="1",
                    parameters={"X1": 1.0, "X2": 5.0, "X3": 5.0},
                    metrics={"Y": 10.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="2",
                    parameters={"X1": 1.0, "X2": 8.0, "X3": 5.0},
                    metrics={"Y": 8.0},
                    status=ExperimentStatus.DISCARD,
                ),
            ]
        )
        ss = _make_search_space()
        graph = _make_causal_graph()

        result = analyze_coverage(log, ss, "Y", causal_graph=graph)

        # X2 only varied via the DISCARD experiment — should NOT be "never intervened"
        assert "X2" not in (result.ancestors_never_intervened or [])
        assert "X2" in (result.ancestors_intervened or [])


class TestCrashExperimentExclusion:
    """CRASH experiments should still be excluded from coverage."""

    def test_crash_only_variable_reported_as_never_intervened(self) -> None:
        """A variable varied ONLY in CRASH experiments IS reported as never intervened."""
        log = ExperimentLog(
            results=[
                ExperimentResult(
                    experiment_id="1",
                    parameters={"X1": 1.0, "X2": 5.0, "X3": 5.0},
                    metrics={"Y": 10.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="2",
                    parameters={"X1": 1.0, "X2": 9.0, "X3": 5.0},
                    metrics={},
                    status=ExperimentStatus.CRASH,
                ),
            ]
        )
        ss = _make_search_space()
        graph = _make_causal_graph()

        result = analyze_coverage(log, ss, "Y", causal_graph=graph)

        # X2 only varied via the CRASH experiment — should be "never intervened"
        assert "X2" in (result.ancestors_never_intervened or [])


class TestKeptVariedVarsField:
    """The new kept_varied_vars field on CoverageAnalysis."""

    def test_kept_varied_vars_reflects_only_keep_experiments(self) -> None:
        """kept_varied_vars should only contain variables varied in KEEP experiments."""
        log = ExperimentLog(
            results=[
                ExperimentResult(
                    experiment_id="1",
                    parameters={"X1": 1.0, "X2": 5.0, "X3": 5.0},
                    metrics={"Y": 10.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="2",
                    parameters={"X1": 3.0, "X2": 5.0, "X3": 5.0},
                    metrics={"Y": 12.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="3",
                    parameters={"X1": 1.0, "X2": 8.0, "X3": 9.0},
                    metrics={"Y": 6.0},
                    status=ExperimentStatus.DISCARD,
                ),
            ]
        )
        ss = _make_search_space()
        graph = _make_causal_graph()

        result = analyze_coverage(log, ss, "Y", causal_graph=graph)

        # X1 varied in KEEP experiments -> in kept_varied_vars
        # X2, X3 only varied via DISCARD -> NOT in kept_varied_vars
        assert result.kept_varied_vars is not None
        assert "X1" in result.kept_varied_vars
        assert "X2" not in result.kept_varied_vars
        assert "X3" not in result.kept_varied_vars

    def test_kept_varied_vars_is_subset_of_varied(self) -> None:
        """kept_varied_vars should always be a subset of the broader varied set."""
        log = ExperimentLog(
            results=[
                ExperimentResult(
                    experiment_id="1",
                    parameters={"X1": 1.0, "X2": 5.0, "X3": 3.0},
                    metrics={"Y": 10.0},
                    status=ExperimentStatus.KEEP,
                ),
                ExperimentResult(
                    experiment_id="2",
                    parameters={"X1": 2.0, "X2": 7.0, "X3": 3.0},
                    metrics={"Y": 8.0},
                    status=ExperimentStatus.KEEP,
                ),
            ]
        )
        ss = _make_search_space()
        graph = _make_causal_graph()

        result = analyze_coverage(log, ss, "Y", causal_graph=graph)

        # When all experiments are KEEP, kept_varied_vars == varied_vars
        assert result.kept_varied_vars is not None
        assert "X1" in result.kept_varied_vars
        assert "X2" in result.kept_varied_vars

    def test_kept_varied_vars_default_none_on_model(self) -> None:
        """CoverageAnalysis.kept_varied_vars defaults to None."""
        analysis = CoverageAnalysis()
        assert analysis.kept_varied_vars is None
