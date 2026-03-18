"""Tests for the diagnostics module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from causal_optimizer.diagnostics.advisor import ResearchAdvisor, _synthesize_recommendations
from causal_optimizer.diagnostics.convergence import analyze_convergence
from causal_optimizer.diagnostics.coverage import analyze_coverage
from causal_optimizer.diagnostics.models import (
    ConfidenceLevel,
    ConvergenceAnalysis,
    CoverageAnalysis,
    DiagnosticReport,
    Recommendation,
    RecommendationType,
    RobustnessAnalysis,
    VariableSignalAnalysis,
    VariableSignalClass,
    VariableSignalReport,
)
from causal_optimizer.diagnostics.robustness import analyze_robustness
from causal_optimizer.diagnostics.variable_signal import analyze_variable_signal
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


def _search_space(n_vars: int = 3) -> SearchSpace:
    """Create a simple continuous search space with n_vars variables."""
    return SearchSpace(
        variables=[
            Variable(name=f"x{i}", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)
            for i in range(n_vars)
        ]
    )


_result_counter = 0


def _result(
    params: dict,
    objective: float,
    status: ExperimentStatus = ExperimentStatus.KEEP,
    phase: str = "exploration",
) -> ExperimentResult:
    global _result_counter  # noqa: PLW0603
    _result_counter += 1
    return ExperimentResult(
        experiment_id=f"test-exp-{_result_counter}",
        parameters=params,
        metrics={"objective": objective},
        status=status,
        metadata={"phase": phase},
    )


def _log_with_improving_objective(n: int = 20, noise: float = 0.1) -> ExperimentLog:
    """Create a log where objective improves (decreases) over time."""
    rng = np.random.RandomState(42)
    results = []
    for i in range(n):
        x0 = float(rng.uniform(0, 10))
        x1 = float(rng.uniform(0, 10))
        x2 = float(rng.uniform(0, 10))
        # Objective depends on x0 and x1, not x2
        obj = (x0 - 3.0) ** 2 + 0.5 * (x1 - 7.0) ** 2 + rng.normal(0, noise)
        results.append(
            _result(
                {"x0": x0, "x1": x1, "x2": x2},
                obj,
                phase="optimization" if i > 5 else "exploration",
            )
        )
    return ExperimentLog(results=results)


def _log_with_plateau(n: int = 20) -> ExperimentLog:
    """Log where objective improves early then plateaus."""
    results = []
    for i in range(n):
        x0 = float(3.0 + np.random.RandomState(i).normal(0, 0.01))
        x1 = float(7.0 + np.random.RandomState(i + 100).normal(0, 0.01))
        x2 = float(np.random.RandomState(i + 200).uniform(0, 10))
        # First half: improving; second half: flat at minimum
        if i < n // 2:
            obj = 10.0 - i * 0.5
        else:
            obj = 10.0 - (n // 2) * 0.5 + np.random.RandomState(i).normal(0, 0.001)
        results.append(_result({"x0": x0, "x1": x1, "x2": x2}, obj))
    return ExperimentLog(results=results)


def _simple_causal_graph() -> CausalGraph:
    """x0 -> objective, x1 -> objective, x2 is disconnected."""
    return CausalGraph(
        edges=[("x0", "objective"), ("x1", "objective")],
        bidirected_edges=[],
    )


# ---------------------------------------------------------------------------
# Variable Signal Tests
# ---------------------------------------------------------------------------


class TestVariableSignal:
    def test_too_few_experiments_all_untested(self):
        """With < 5 experiments, all variables should be UNTESTED."""
        ss = _search_space(2)
        log = ExperimentLog(
            results=[
                _result({"x0": 1.0, "x1": 2.0}, 5.0),
                _result({"x0": 3.0, "x1": 4.0}, 3.0),
            ]
        )
        analysis = analyze_variable_signal(log, ss, "objective", minimize=True)
        assert analysis.untested_count == 2
        assert analysis.high_signal_count == 0
        assert analysis.low_signal_count == 0

    def test_varied_variable_detected(self):
        """With enough data, a variable with real signal should be HIGH_SIGNAL."""
        ss = _search_space(2)
        log = _log_with_improving_objective(n=30, noise=0.01)
        # Re-create with just x0, x1
        results = []
        for r in log.results:
            results.append(
                _result(
                    {"x0": r.parameters["x0"], "x1": r.parameters["x1"]},
                    r.metrics["objective"],
                )
            )
        log2 = ExperimentLog(results=results)
        analysis = analyze_variable_signal(log2, ss, "objective", minimize=True)
        # At least one variable should be classified
        assert analysis.high_signal_count + analysis.low_signal_count > 0

    def test_constant_variable_is_untested(self):
        """A variable held constant should be UNTESTED."""
        ss = _search_space(2)
        rng = np.random.RandomState(42)
        results = [
            _result({"x0": float(rng.uniform(0, 10)), "x1": 5.0}, float(rng.uniform(0, 10)))
            for _ in range(10)
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_variable_signal(log, ss, "objective", minimize=True)
        x1_report = next(r for r in analysis.variables if r.variable_name == "x1")
        assert x1_report.signal_class == VariableSignalClass.UNTESTED

    def test_missing_variable_column_is_untested(self):
        """If a variable isn't in any result's parameters, it should be UNTESTED."""
        ss = _search_space(3)
        results = [_result({"x0": 1.0, "x1": 2.0}, 5.0) for _ in range(6)]
        log = ExperimentLog(results=results)
        analysis = analyze_variable_signal(log, ss, "objective", minimize=True)
        x2_report = next(r for r in analysis.variables if r.variable_name == "x2")
        assert x2_report.signal_class == VariableSignalClass.UNTESTED

    def test_value_range_reported(self):
        """Numeric variables should have value_range_explored populated."""
        ss = _search_space(1)
        results = [_result({"x0": float(i)}, float(i)) for i in range(10)]
        log = ExperimentLog(results=results)
        analysis = analyze_variable_signal(log, ss, "objective", minimize=True)
        x0_report = analysis.variables[0]
        assert x0_report.value_range_explored is not None
        assert x0_report.value_range_explored[0] == 0.0
        assert x0_report.value_range_explored[1] == 9.0


# ---------------------------------------------------------------------------
# Convergence Tests
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_empty_log(self):
        """Empty log should return safe defaults."""
        log = ExperimentLog(results=[])
        analysis = analyze_convergence(log, "objective", minimize=True)
        assert not analysis.plateaued
        assert not analysis.abandoned_climb
        assert analysis.improvement_rate == 0.0

    def test_few_results_no_plateau(self):
        """With < 4 kept results, no plateau or climb should be detected."""
        log = ExperimentLog(results=[_result({"x0": float(i)}, float(10 - i)) for i in range(3)])
        analysis = analyze_convergence(log, "objective", minimize=True)
        assert not analysis.plateaued
        assert not analysis.abandoned_climb
        assert analysis.best_at_step == 2

    def test_plateau_detected(self):
        """A flat late phase should be detected as a plateau."""
        log = _log_with_plateau(n=20)
        analysis = analyze_convergence(log, "objective", minimize=True)
        assert analysis.plateaued
        assert not analysis.abandoned_climb

    def test_improving_trajectory(self):
        """A steadily improving objective should not be a plateau."""
        results = [_result({"x0": float(i)}, float(100 - i * 5)) for i in range(20)]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=True)
        assert not analysis.plateaued

    def test_abandoned_climb_detection(self):
        """Detecting still-improving objective when run finished at budget."""
        results = [_result({"x0": float(i)}, float(100 - i * 5)) for i in range(20)]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=True, total_budget=20)
        assert analysis.abandoned_climb

    def test_steps_since_improvement(self):
        """steps_since_improvement should reflect the gap from best to end."""
        results = [
            _result({"x0": 1.0}, 5.0),
            _result({"x0": 2.0}, 1.0),  # best at step 1
            _result({"x0": 3.0}, 3.0),
            _result({"x0": 4.0}, 4.0),
            _result({"x0": 5.0}, 6.0),
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=True)
        assert analysis.best_at_step == 1
        assert analysis.steps_since_improvement == 3  # steps 2, 3, 4 after best

    def test_budget_remaining_fraction(self):
        """Budget remaining should be calculated correctly."""
        results = [_result({"x0": float(i)}, float(i)) for i in range(10)]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=True, total_budget=20)
        assert analysis.budget_remaining_fraction == pytest.approx(0.5)

    def test_maximize_direction(self):
        """Maximize mode should track the highest value as best."""
        results = [_result({"x0": float(i)}, float(i * 2)) for i in range(10)]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=False)
        assert analysis.best_at_step == 9
        assert analysis.best_objective == 18.0

    def test_discarded_results_ignored(self):
        """Discarded results should not affect convergence analysis."""
        results = [
            _result({"x0": 1.0}, 10.0),
            _result({"x0": 2.0}, 1.0, status=ExperimentStatus.DISCARD),
            _result({"x0": 3.0}, 5.0),
            _result({"x0": 4.0}, 3.0),
            _result({"x0": 5.0}, 2.0),
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_convergence(log, "objective", minimize=True)
        # Best should be 2.0 (step index 3 among kept: 10, 5, 3, 2)
        assert analysis.best_objective == 2.0


# ---------------------------------------------------------------------------
# Coverage Tests
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_no_optional_inputs(self):
        """With no graph/POMIS/archive, optional fields should be None."""
        ss = _search_space(2)
        log = ExperimentLog(
            results=[_result({"x0": float(i), "x1": float(i)}, float(i)) for i in range(5)]
        )
        analysis = analyze_coverage(log, ss, "objective")
        assert analysis.pomis_sets_total is None
        assert analysis.ancestor_variables is None
        assert analysis.map_elites_coverage is None
        assert analysis.search_space_coverage is not None

    def test_pomis_coverage(self):
        """POMIS sets where all members were varied should be marked explored."""
        ss = _search_space(3)
        rng = np.random.RandomState(42)
        results = [
            _result(
                {"x0": float(rng.uniform(0, 10)), "x1": float(rng.uniform(0, 10)), "x2": 5.0},
                float(rng.uniform(0, 10)),
            )
            for _ in range(5)
        ]
        log = ExperimentLog(results=results)
        pomis = [frozenset({"x0", "x1"}), frozenset({"x0", "x2"})]
        analysis = analyze_coverage(log, ss, "objective", pomis_sets=pomis)
        assert analysis.pomis_sets_total == 2
        # x0 and x1 were varied → first set explored; x2 constant → second not
        assert analysis.pomis_sets_explored == 1
        assert analysis.pomis_sets_unexplored is not None
        assert len(analysis.pomis_sets_unexplored) == 1

    def test_ancestor_coverage(self):
        """Ancestors in the causal graph that were never varied should be flagged."""
        ss = _search_space(3)
        graph = _simple_causal_graph()
        rng = np.random.RandomState(42)
        # Only vary x0, hold x1 constant
        results = [
            _result(
                {"x0": float(rng.uniform(0, 10)), "x1": 5.0, "x2": float(rng.uniform(0, 10))},
                1.0,
            )
            for _ in range(5)
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_coverage(log, ss, "objective", causal_graph=graph)
        assert analysis.ancestor_variables is not None
        assert "x0" in analysis.ancestor_variables
        assert "x1" in analysis.ancestor_variables
        # x0 varied, x1 not
        assert "x0" in (analysis.ancestors_intervened or [])
        assert "x1" in (analysis.ancestors_never_intervened or [])

    def test_search_space_coverage(self):
        """Search space coverage should reflect explored range fraction."""
        ss = SearchSpace(
            variables=[
                Variable(name="x0", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=100.0),
            ]
        )
        # Only explore 0–50 of 0–100
        results = [
            _result({"x0": float(i * 5)}, float(i))
            for i in range(11)  # 0 to 50
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_coverage(log, ss, "objective")
        assert analysis.search_space_coverage is not None
        assert analysis.search_space_coverage == pytest.approx(0.5)

    def test_full_search_space_coverage(self):
        """Full range explored should give coverage of 1.0."""
        ss = SearchSpace(
            variables=[
                Variable(name="x0", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ]
        )
        results = [_result({"x0": 0.0}, 1.0), _result({"x0": 10.0}, 2.0)]
        log = ExperimentLog(results=results)
        analysis = analyze_coverage(log, ss, "objective")
        assert analysis.search_space_coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Robustness Tests
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_insufficient_data(self):
        """With < 4 results, should return not robust with insufficient data message."""
        log = ExperimentLog(results=[_result({"x0": float(i)}, float(i)) for i in range(3)])
        analysis = analyze_robustness(log, "objective", minimize=True)
        assert not analysis.best_result_robust
        assert "Insufficient" in analysis.summary
        assert analysis.top_k_consistency == 0.0

    def test_with_enough_data(self):
        """With enough data, robustness analysis should run without errors."""
        log = _log_with_improving_objective(n=20, noise=0.01)
        ss = _search_space(3)
        analysis = analyze_robustness(log, "objective", minimize=True, search_space=ss)
        # Should produce a valid analysis (specific values depend on data)
        assert isinstance(analysis.signal_to_noise, float)
        assert isinstance(analysis.e_value, float)
        assert isinstance(analysis.top_k_consistency, float)
        assert 0.0 <= analysis.top_k_consistency <= 1.0

    def test_top_k_consistency_identical_params(self):
        """Top-K results with identical parameters should have high consistency."""
        ss = _search_space(2)
        # All results have nearly the same parameters
        results = [_result({"x0": 3.0, "x1": 7.0}, 1.0 + i * 0.001) for i in range(10)]
        log = ExperimentLog(results=results)
        analysis = analyze_robustness(log, "objective", minimize=True, search_space=ss)
        assert analysis.top_k_consistency == pytest.approx(1.0)

    def test_top_k_consistency_diverse_params(self):
        """Top-K results with very different parameters should have low consistency."""
        ss = _search_space(2)
        results = [
            _result({"x0": 0.0, "x1": 0.0}, 1.0),  # best
            _result({"x0": 10.0, "x1": 10.0}, 1.5),  # second
            _result({"x0": 5.0, "x1": 5.0}, 2.0),  # third
            _result({"x0": 3.0, "x1": 7.0}, 10.0),  # worst
        ]
        log = ExperimentLog(results=results)
        analysis = analyze_robustness(log, "objective", minimize=True, search_space=ss)
        assert analysis.top_k_consistency < 0.5


# ---------------------------------------------------------------------------
# Recommendation Synthesis Tests
# ---------------------------------------------------------------------------


class TestRecommendationSynthesis:
    def _base_analyses(self) -> tuple:
        """Create minimal analyses with no issues flagged."""
        signal = VariableSignalAnalysis(
            variables=[
                VariableSignalReport(
                    variable_name="x0",
                    signal_class=VariableSignalClass.HIGH_SIGNAL,
                    importance_score=0.3,
                    n_experiments_varied=20,
                ),
            ],
            high_signal_count=1,
        )
        convergence = ConvergenceAnalysis(
            plateaued=False,
            improvement_rate=0.01,
            improvement_rate_early=0.02,
            improvement_rate_late=0.005,
            best_objective=1.0,
            best_at_step=15,
            steps_since_improvement=3,
            abandoned_climb=False,
        )
        coverage = CoverageAnalysis()
        robustness = RobustnessAnalysis(
            best_result_robust=True,
            signal_to_noise=3.0,
            e_value=5.0,
            effect_size=1.5,
            summary="Strong evidence",
            top_k_consistency=0.8,
        )
        return signal, convergence, coverage, robustness

    def test_no_issues_no_recommendations(self):
        """Clean analyses should produce no recommendations (or only low-priority)."""
        signal, convergence, coverage, robustness = self._base_analyses()
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        # No major issues → no recs
        assert len(recs) == 0

    def test_untested_ancestor_generates_explore(self):
        """Untested causal ancestors should generate EXPLORE recommendations."""
        signal, convergence, coverage, robustness = self._base_analyses()
        coverage.ancestors_never_intervened = ["x2"]
        ss = _search_space(3)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        explore_recs = [r for r in recs if r.rec_type == RecommendationType.EXPLORE]
        assert len(explore_recs) >= 1
        assert "x2" in explore_recs[0].title

    def test_unexplored_pomis_generates_explore(self):
        """Unexplored POMIS sets should generate EXPLORE recommendations."""
        signal, convergence, coverage, robustness = self._base_analyses()
        coverage.pomis_sets_unexplored = [["x1", "x2"]]
        ss = _search_space(3)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        explore_recs = [r for r in recs if r.rec_type == RecommendationType.EXPLORE]
        assert len(explore_recs) >= 1

    def test_abandoned_climb_generates_exploit(self):
        """Abandoned climb should generate EXPLOIT recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        convergence.abandoned_climb = True
        convergence.improvement_rate_late = 0.05
        convergence.improvement_rate_early = 0.1
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        exploit_recs = [r for r in recs if r.rec_type == RecommendationType.EXPLOIT]
        assert len(exploit_recs) == 1
        title = exploit_recs[0].title.lower()
        assert "improvement" in title or "continue" in title

    def test_plateau_generates_pivot(self):
        """Plateau should generate a PIVOT recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        convergence.plateaued = True
        convergence.improvement_rate_late = 0.000001
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        pivot_recs = [r for r in recs if r.rec_type == RecommendationType.PIVOT]
        assert len(pivot_recs) >= 1

    def test_not_robust_generates_pivot(self):
        """Non-robust best result should generate a PIVOT recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        robustness.best_result_robust = False
        robustness.signal_to_noise = 0.5
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        pivot_recs = [r for r in recs if r.rec_type == RecommendationType.PIVOT]
        assert len(pivot_recs) >= 1

    def test_low_signal_variable_generates_drop(self):
        """Low-signal variable should generate a DROP recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        signal.variables.append(
            VariableSignalReport(
                variable_name="x_dead",
                signal_class=VariableSignalClass.LOW_SIGNAL,
                importance_score=0.001,
                effect_significant=False,
                n_experiments_varied=20,
            )
        )
        signal.low_signal_count = 1
        ss = _search_space(2)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        drop_recs = [r for r in recs if r.rec_type == RecommendationType.DROP]
        assert len(drop_recs) == 1
        assert "x_dead" in drop_recs[0].title

    def test_low_map_elites_coverage_generates_explore(self):
        """Low MAP-Elites coverage should generate EXPLORE recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        coverage.map_elites_coverage = 0.1
        coverage.map_elites_filled_cells = 5
        coverage.map_elites_total_cells = 50
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        explore_recs = [r for r in recs if r.rec_type == RecommendationType.EXPLORE]
        assert len(explore_recs) >= 1

    def test_top_k_inconsistency_generates_pivot(self):
        """Low top-K consistency should generate a PIVOT recommendation."""
        signal, convergence, coverage, robustness = self._base_analyses()
        robustness.top_k_consistency = 0.1
        ss = _search_space(1)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        pivot_recs = [r for r in recs if r.rec_type == RecommendationType.PIVOT]
        assert len(pivot_recs) >= 1

    def test_recommendations_ranked_by_info_gain(self):
        """Recommendations should be sorted by expected_info_gain descending."""
        signal, convergence, coverage, robustness = self._base_analyses()
        coverage.ancestors_never_intervened = ["x2"]  # 0.9 gain
        convergence.plateaued = True  # 0.65 gain
        convergence.improvement_rate_late = 0.000001
        ss = _search_space(3)
        recs = _synthesize_recommendations(signal, convergence, coverage, robustness, ss)
        assert len(recs) >= 2
        gains = [r.expected_info_gain for r in recs]
        assert gains == sorted(gains, reverse=True)
        # Ranks should be 1, 2, 3, ...
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))


# ---------------------------------------------------------------------------
# ResearchAdvisor Tests
# ---------------------------------------------------------------------------


class TestResearchAdvisor:
    def test_analyze_from_log(self):
        """ResearchAdvisor.analyze_from_log should produce a valid report."""
        ss = _search_space(3)
        log = _log_with_improving_objective(n=20)
        advisor = ResearchAdvisor(objective_name="objective", minimize=True)
        report = advisor.analyze_from_log(experiment_log=log, search_space=ss)
        assert isinstance(report, DiagnosticReport)
        assert report.n_experiments == 20
        assert isinstance(report.variable_signal, VariableSignalAnalysis)
        assert isinstance(report.convergence, ConvergenceAnalysis)
        assert isinstance(report.coverage, CoverageAnalysis)
        assert isinstance(report.robustness, RobustnessAnalysis)

    def test_analyze_from_log_with_graph(self):
        """Should incorporate causal graph info when provided."""
        ss = _search_space(3)
        graph = _simple_causal_graph()
        log = _log_with_improving_objective(n=20)
        advisor = ResearchAdvisor(objective_name="objective", minimize=True)
        report = advisor.analyze_from_log(experiment_log=log, search_space=ss, causal_graph=graph)
        assert report.coverage.ancestor_variables is not None

    def test_analyze_from_log_empty(self):
        """Should handle an empty log gracefully."""
        ss = _search_space(2)
        log = ExperimentLog(results=[])
        advisor = ResearchAdvisor()
        report = advisor.analyze_from_log(experiment_log=log, search_space=ss)
        assert report.n_experiments == 0
        assert not report.convergence.plateaued


# ---------------------------------------------------------------------------
# DiagnosticReport Tests
# ---------------------------------------------------------------------------


class TestDiagnosticReport:
    def _make_report(self, recs: list[Recommendation] | None = None) -> DiagnosticReport:
        return DiagnosticReport(
            experiment_id="test-123",
            n_experiments=20,
            current_phase="optimization",
            variable_signal=VariableSignalAnalysis(
                variables=[],
                high_signal_count=1,
                low_signal_count=1,
                untested_count=0,
            ),
            convergence=ConvergenceAnalysis(
                plateaued=False,
                improvement_rate=0.01,
                improvement_rate_early=0.02,
                improvement_rate_late=0.005,
                best_objective=1.0,
                best_at_step=15,
                steps_since_improvement=3,
                abandoned_climb=False,
            ),
            coverage=CoverageAnalysis(search_space_coverage=0.75),
            robustness=RobustnessAnalysis(
                best_result_robust=True,
                signal_to_noise=3.0,
                e_value=5.0,
                effect_size=1.5,
                summary="Strong evidence",
                top_k_consistency=0.8,
            ),
            recommendations=recs or [],
        )

    def test_summary_contains_key_info(self):
        report = self._make_report()
        summary = report.summary()
        assert "20 experiments" in summary
        assert "optimization" in summary
        assert "1 high-signal" in summary
        assert "ROBUST" in summary
        assert "75%" in summary

    def test_summary_with_recommendations(self):
        rec = Recommendation(
            rank=1,
            rec_type=RecommendationType.EXPLORE,
            confidence=ConfidenceLevel.HIGH,
            title="Explore x2: causal ancestor never tested",
            description="Variable x2 was never varied.",
            evidence=["x2 is an ancestor"],
            next_step="Run experiments varying x2",
            expected_info_gain=0.9,
            variables_involved=["x2"],
        )
        report = self._make_report(recs=[rec])
        summary = report.summary()
        assert "Research Directions:" in summary
        assert "#1" in summary
        assert "EXPLORE" in summary
        assert "x2" in summary

    def test_summary_no_recommendations(self):
        report = self._make_report()
        summary = report.summary()
        assert "No actionable recommendations" in summary

    def test_summary_plateau_display(self):
        report = self._make_report()
        report.convergence.plateaued = True
        summary = report.summary()
        assert "PLATEAUED" in summary

    def test_summary_abandoned_climb_display(self):
        report = self._make_report()
        report.convergence.abandoned_climb = True
        summary = report.summary()
        assert "ABANDONED CLIMB" in summary

    def test_json_serialization(self):
        """Report should be JSON-serializable via model_dump."""
        report = self._make_report()
        data = report.model_dump()
        json_str = json.dumps(data)
        assert "test-123" in json_str
        assert "optimization" in json_str


# ---------------------------------------------------------------------------
# CLI Integration Test (using _print_diagnostics)
# ---------------------------------------------------------------------------


class TestCLIPrintDiagnostics:
    def test_print_diagnostics(self, capsys):
        """_print_diagnostics should produce output without errors."""
        from causal_optimizer.cli import _print_diagnostics

        ss = _search_space(3)
        log = _log_with_improving_objective(n=10)
        _print_diagnostics(log, ss)
        captured = capsys.readouterr()
        assert "Diagnostic Report" in captured.out

    def test_print_diagnostics_with_objective_flags(self, capsys):
        """_print_diagnostics should pass objective_name and minimize."""
        from causal_optimizer.cli import _print_diagnostics

        ss = _search_space(3)
        log = _log_with_improving_objective(n=10)
        _print_diagnostics(log, ss, objective_name="objective", minimize=False)
        captured = capsys.readouterr()
        assert "Diagnostic Report" in captured.out


# ---------------------------------------------------------------------------
# Engine.diagnose() Integration Test
# ---------------------------------------------------------------------------


class TestEngineDiagnose:
    def test_diagnose_smoke(self):
        """engine.diagnose() should return a valid DiagnosticReport."""
        from unittest.mock import MagicMock

        from causal_optimizer.engine.loop import ExperimentEngine

        ss = _search_space(3)
        runner = MagicMock()
        runner.run.return_value = {"objective": 1.0}
        engine = ExperimentEngine(search_space=ss, runner=runner)
        engine.run_loop(n_experiments=6)
        report = engine.diagnose()
        assert isinstance(report, DiagnosticReport)
        assert report.n_experiments == 6


# ---------------------------------------------------------------------------
# ExperimentStore.load_search_space() Test
# ---------------------------------------------------------------------------


class TestStoreLoadSearchSpace:
    def test_load_search_space(self, tmp_path):
        """load_search_space should round-trip a search space through SQLite."""
        from unittest.mock import MagicMock

        from causal_optimizer.engine.loop import ExperimentEngine
        from causal_optimizer.storage.sqlite import ExperimentStore

        ss = _search_space(2)
        db_path = str(tmp_path / "test.db")
        runner = MagicMock()
        runner.run.return_value = {"objective": 1.0}

        with ExperimentStore(db_path) as store:
            engine = ExperimentEngine(
                search_space=ss,
                runner=runner,
                store=store,
                experiment_id="test-ss",
            )
            engine.run_loop(n_experiments=3)
            loaded_ss = store.load_search_space("test-ss")

        assert len(loaded_ss.variables) == len(ss.variables)
        for orig, loaded in zip(ss.variables, loaded_ss.variables, strict=True):
            assert orig.name == loaded.name
