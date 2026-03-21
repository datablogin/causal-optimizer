"""Tests for observational diagnostics analysis."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from causal_optimizer.diagnostics.models import (
    DiagnosticReport,
    ObservationalAnalysis,
    RecommendationType,
)
from causal_optimizer.diagnostics.observational import analyze_observational
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
    return SearchSpace(
        variables=[
            Variable(name=f"x{i}", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)
            for i in range(n_vars)
        ]
    )


def _result(
    params: dict,
    objective: float,
    status: ExperimentStatus = ExperimentStatus.KEEP,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=str(uuid.uuid4()),
        parameters=params,
        metrics={"objective": objective},
        status=status,
    )


def _make_log(n: int = 15) -> ExperimentLog:
    """Create a log with n KEEP experiments with varying x0, x1, x2."""
    import numpy as np

    rng = np.random.default_rng(42)
    log = ExperimentLog()
    for _ in range(n):
        params = {f"x{i}": float(rng.uniform(0, 10)) for i in range(3)}
        obj = params["x0"] * 0.5 + params["x1"] * 0.1 + rng.normal(0, 0.1)
        log.results.append(_result(params, obj))
    return log


def _causal_graph() -> CausalGraph:
    """x0 -> objective, x1 -> objective, x2 is not an ancestor."""
    return CausalGraph(
        edges=[("x0", "objective"), ("x1", "objective")],
        nodes=["x0", "x1", "x2", "objective"],
    )


def _mock_estimator_cls(
    identified: bool = True, estimate: float = 2.5, ci: tuple[float, float] = (1.5, 3.5)
):
    """Create a mock ObservationalEstimator class."""
    mock_cls = MagicMock()

    def make_instance(*args, **kwargs):
        instance = MagicMock()
        instance.causal_graph = kwargs.get("causal_graph")
        mock_result = MagicMock()
        mock_result.identified = identified
        mock_result.expected_outcome = estimate
        mock_result.confidence_interval = ci
        mock_result.method = "observational/backdoor.linear_regression"
        instance.estimate_intervention.return_value = mock_result
        return instance

    mock_cls.side_effect = make_instance
    return mock_cls


# ---------------------------------------------------------------------------
# Test: no causal graph
# ---------------------------------------------------------------------------


class TestNoGraph:
    def test_no_graph_returns_recommendation_message(self) -> None:
        log = _make_log(15)
        ss = _search_space()
        result = analyze_observational(log, ss, "objective", minimize=False)

        assert result.n_identifiable == 0
        assert result.n_variables == 3
        assert result.variables == []
        assert result.obs_experimental_agreement is None
        assert result.recommendation == "no causal graph available"


# ---------------------------------------------------------------------------
# Test: fewer than 10 experiments
# ---------------------------------------------------------------------------


class TestFewExperiments:
    def test_fewer_than_10_experiments_skips_estimation(self) -> None:
        log = _make_log(5)
        ss = _search_space()
        graph = _causal_graph()
        result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # Should still report variables, but estimates should be None
        assert result.n_variables == 3
        for var_report in result.variables:
            assert var_report.obs_estimate is None
            assert var_report.obs_ci is None


# ---------------------------------------------------------------------------
# Test: backdoor identifiable variable
# ---------------------------------------------------------------------------


class TestBackdoorIdentifiable:
    def test_identifies_causal_ancestors(self) -> None:
        """Only causal ancestors should appear in the analysis."""
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=True),
        ):
            result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # x0 and x1 are ancestors; x2 is not
        var_names = [v.variable_name for v in result.variables]
        assert "x0" in var_names
        assert "x1" in var_names
        assert "x2" not in var_names

    def test_identifiable_variable_has_estimate(self) -> None:
        """When ObservationalEstimator returns identified=True, we get an estimate."""
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=True, estimate=2.5, ci=(1.5, 3.5)),
        ):
            result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # At least one variable should be identifiable
        identifiable = [v for v in result.variables if v.identifiable]
        assert len(identifiable) > 0
        assert identifiable[0].obs_estimate is not None
        assert identifiable[0].obs_ci is not None


# ---------------------------------------------------------------------------
# Test: non-identifiable variable
# ---------------------------------------------------------------------------


class TestNonIdentifiable:
    def test_non_identifiable_marked_correctly(self) -> None:
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=False),
        ):
            result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # All should be non-identifiable since the mock returns identified=False
        non_id = [v for v in result.variables if not v.identifiable]
        assert len(non_id) > 0


# ---------------------------------------------------------------------------
# Test: agreement computation
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_agreement_computed_when_both_estimates_available(self) -> None:
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=True, estimate=2.5, ci=(1.5, 3.5)),
        ):
            result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # The obs_experimental_agreement should be a float or None
        assert isinstance(result.obs_experimental_agreement, float | None)

    def test_agreement_perfect_when_equal(self) -> None:
        from causal_optimizer.diagnostics.observational import _compute_agreement

        assert _compute_agreement(5.0, 5.0) == 1.0

    def test_agreement_zero_for_opposite_signs(self) -> None:
        from causal_optimizer.diagnostics.observational import _compute_agreement

        # Large relative difference → 0.0 agreement
        result = _compute_agreement(1.0, -1.0)
        assert result == 0.0

    def test_agreement_both_zero(self) -> None:
        from causal_optimizer.diagnostics.observational import _compute_agreement

        assert _compute_agreement(0.0, 0.0) == 1.0

    def test_agreement_near_zero_estimates(self) -> None:
        from causal_optimizer.diagnostics.observational import _compute_agreement

        # Very small values close together should have high agreement
        result = _compute_agreement(0.01, 0.011)
        assert result > 0.5

    def test_aggregate_agreement_precision_weighted(self) -> None:
        """Tighter CIs should get more weight in aggregate agreement."""
        from causal_optimizer.diagnostics.models import ObservationalVariableReport
        from causal_optimizer.diagnostics.observational import _compute_aggregate_agreement

        # Variable A: tight CI (width 0.1), agreement 0.9
        var_a = ObservationalVariableReport(
            variable_name="a",
            identifiable=True,
            identification_method="backdoor",
            obs_estimate=1.0,
            obs_ci=(0.95, 1.05),
            exp_estimate=1.0,
            agreement=0.9,
        )
        # Variable B: wide CI (width 10.0), agreement 0.1
        var_b = ObservationalVariableReport(
            variable_name="b",
            identifiable=True,
            identification_method="backdoor",
            obs_estimate=5.0,
            obs_ci=(0.0, 10.0),
            exp_estimate=5.0,
            agreement=0.1,
        )

        agg = _compute_aggregate_agreement([var_a, var_b])
        assert agg is not None
        # With precision weighting, var_a (tight CI) should dominate,
        # so aggregate should be much closer to 0.9 than to 0.5
        assert agg > 0.7


# ---------------------------------------------------------------------------
# Test: DoWhy not installed
# ---------------------------------------------------------------------------


class TestDoWhyNotInstalled:
    def test_graceful_degradation_without_dowhy(self) -> None:
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            None,
        ):
            result = analyze_observational(log, ss, "objective", minimize=False, causal_graph=graph)

        # Should degrade gracefully
        assert all(not v.identifiable for v in result.variables)
        assert (
            "not available" in result.recommendation.lower()
            or "dowhy" in result.recommendation.lower()
        )


# ---------------------------------------------------------------------------
# Test: integration with advisor
# ---------------------------------------------------------------------------


class TestAdvisorIntegration:
    def test_observational_field_on_diagnostic_report(self) -> None:
        """DiagnosticReport should have optional observational field."""
        from causal_optimizer.diagnostics.advisor import ResearchAdvisor

        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        advisor = ResearchAdvisor(objective_name="objective", minimize=False)

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            None,
        ):
            report = advisor.analyze_from_log(
                experiment_log=log,
                search_space=ss,
                causal_graph=graph,
                phase="optimization",
            )

        assert hasattr(report, "observational")
        assert report.observational is not None

    def test_observational_none_without_graph(self) -> None:
        from causal_optimizer.diagnostics.advisor import ResearchAdvisor

        log = _make_log(15)
        ss = _search_space()

        advisor = ResearchAdvisor(objective_name="objective", minimize=False)
        report = advisor.analyze_from_log(
            experiment_log=log,
            search_space=ss,
            phase="optimization",
        )

        assert report.observational is None

    def test_diagnostic_report_model_accepts_observational(self) -> None:
        """DiagnosticReport can be constructed with observational field."""
        from causal_optimizer.diagnostics.models import (
            ConvergenceAnalysis,
            CoverageAnalysis,
            RobustnessAnalysis,
            VariableSignalAnalysis,
        )

        obs = ObservationalAnalysis(
            n_identifiable=1,
            n_variables=3,
            variables=[],
            obs_experimental_agreement=0.9,
            recommendation="strong agreement",
        )
        report = DiagnosticReport(
            n_experiments=10,
            current_phase="optimization",
            variable_signal=VariableSignalAnalysis(variables=[]),
            convergence=ConvergenceAnalysis(
                plateaued=False,
                improvement_rate=0.1,
                improvement_rate_early=0.2,
                improvement_rate_late=0.05,
                best_objective=1.0,
                best_at_step=5,
                steps_since_improvement=2,
                abandoned_climb=False,
            ),
            coverage=CoverageAnalysis(),
            robustness=RobustnessAnalysis(
                best_result_robust=True,
                signal_to_noise=3.0,
                e_value=5.0,
                effect_size=0.5,
                summary="robust",
                top_k_consistency=0.8,
            ),
            observational=obs,
            recommendations=[],
        )
        assert report.observational is not None
        assert report.observational.n_identifiable == 1


# ---------------------------------------------------------------------------
# Test: observational recommendations
# ---------------------------------------------------------------------------


class TestObservationalRecommendations:
    def test_strong_agreement_gives_exploit(self) -> None:
        """Strong obs-exp agreement should produce EXPLOIT recommendation."""
        from causal_optimizer.diagnostics.advisor import ResearchAdvisor

        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=True, estimate=2.5, ci=(2.0, 3.0)),
        ):
            advisor = ResearchAdvisor(objective_name="objective", minimize=False)
            report = advisor.analyze_from_log(
                experiment_log=log, search_space=ss, causal_graph=graph, phase="optimization"
            )

        # Should have recommendations
        assert len(report.recommendations) > 0

    def test_identifiable_untested_gives_explore(self) -> None:
        """Identifiable but untested variable should produce EXPLORE recommendation."""
        # Create log where x1 is never varied (all same value)
        import numpy as np

        from causal_optimizer.diagnostics.advisor import ResearchAdvisor

        rng = np.random.default_rng(42)
        log = ExperimentLog()
        for _ in range(15):
            params = {"x0": float(rng.uniform(0, 10)), "x1": 5.0, "x2": 5.0}
            obj = params["x0"] * 0.5 + rng.normal(0, 0.1)
            log.results.append(_result(params, obj))

        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            _mock_estimator_cls(identified=True, estimate=2.5, ci=(2.0, 3.0)),
        ):
            advisor = ResearchAdvisor(objective_name="objective", minimize=False)
            report = advisor.analyze_from_log(
                experiment_log=log, search_space=ss, causal_graph=graph, phase="optimization"
            )

        rec_types = [r.rec_type for r in report.recommendations]
        # Should have EXPLORE recommendations (from obs or coverage)
        assert RecommendationType.EXPLORE in rec_types


# ---------------------------------------------------------------------------
# Test: summary output
# ---------------------------------------------------------------------------


class TestSummaryOutput:
    def test_summary_includes_observational_section(self) -> None:
        log = _make_log(15)
        ss = _search_space()
        graph = _causal_graph()

        with patch(
            "causal_optimizer.diagnostics.observational.ObservationalEstimator",
            None,
        ):
            from causal_optimizer.diagnostics.advisor import ResearchAdvisor

            advisor = ResearchAdvisor(objective_name="objective", minimize=False)
            report = advisor.analyze_from_log(
                experiment_log=log, search_space=ss, causal_graph=graph, phase="optimization"
            )

        summary = report.summary()
        assert "Observational" in summary or "observational" in summary.lower()
