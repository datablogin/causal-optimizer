"""Tests for observational estimator via DoWhy.

TDD: these tests are written first and must initially fail until the
implementation is created.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_from_scm(
    n: int = 300,
    treatment_effect: float = 3.0,
    confound_strength: float = 0.0,
    seed: int = 42,
) -> ExperimentLog:
    """Generate observational data from X -> Z -> Y SCM.

    Structural equations:
        X ~ N(0, 1)
        Z = 2*X + noise
        Y = treatment_effect * Z + confound_strength * X + noise

    When confound_strength=0, backdoor adjustment is not needed.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = 2.0 * x + rng.normal(0, 0.3, n)
    y = treatment_effect * z + confound_strength * x + rng.normal(0, 0.3, n)

    results = []
    for i in range(n):
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters={"x": float(x[i]), "z": float(z[i])},
                metrics={"objective": float(y[i])},
                status=ExperimentStatus.KEEP,
            )
        )
    return ExperimentLog(results=results)


def _make_confounded_log(
    n: int = 400,
    seed: int = 42,
) -> ExperimentLog:
    """Data with unobserved confounder U -> X, U -> Y; X -> Y.

    True causal effect of X on Y is 1.0 but naive correlation is inflated by U.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n)
    x = u + rng.normal(0, 0.5, n)
    y = 1.0 * x + 3.0 * u + rng.normal(0, 0.3, n)

    results = []
    for i in range(n):
        results.append(
            ExperimentResult(
                experiment_id=str(i),
                parameters={"x": float(x[i])},
                metrics={"objective": float(y[i])},
                status=ExperimentStatus.KEEP,
            )
        )
    return ExperimentLog(results=results)


# ---------------------------------------------------------------------------
# Tests for ObservationalEstimate dataclass
# ---------------------------------------------------------------------------


class TestObservationalEstimate:
    """Tests for the ObservationalEstimate dataclass."""

    def test_dataclass_fields_exist(self) -> None:
        from causal_optimizer.estimator.observational import ObservationalEstimate

        est = ObservationalEstimate(
            expected_outcome=2.5,
            confidence_interval=(1.0, 4.0),
            method="backdoor",
            identified=True,
        )
        assert est.expected_outcome == 2.5
        assert est.confidence_interval == (1.0, 4.0)
        assert est.method == "backdoor"
        assert est.identified is True

    def test_identified_false_case(self) -> None:
        from causal_optimizer.estimator.observational import ObservationalEstimate

        est = ObservationalEstimate(
            expected_outcome=0.0,
            confidence_interval=(float("-inf"), float("inf")),
            method="fallback",
            identified=False,
        )
        assert est.identified is False


# ---------------------------------------------------------------------------
# Tests for CausalGraph -> DoWhy graph conversion
# ---------------------------------------------------------------------------


class TestCausalGraphConversion:
    """Tests for converting CausalGraph to DoWhy-compatible format."""

    def test_directed_edges_converted(self) -> None:
        from causal_optimizer.estimator.observational import causal_graph_to_dowhy_str

        graph = CausalGraph(edges=[("x", "z"), ("z", "y")])
        gml_str = causal_graph_to_dowhy_str(graph)
        # Should produce a digraph string
        assert "x" in gml_str
        assert "z" in gml_str
        assert "y" in gml_str

    def test_bidirected_edges_become_latent_nodes(self) -> None:
        from causal_optimizer.estimator.observational import causal_graph_to_dowhy_str

        graph = CausalGraph(
            edges=[("x", "y")],
            bidirected_edges=[("x", "y")],
        )
        gml_str = causal_graph_to_dowhy_str(graph)
        # Bidirected X <-> Y should become a latent node U -> X, U -> Y
        assert "x" in gml_str
        assert "y" in gml_str
        # The graph string should reference the confounder in some way
        assert gml_str  # non-empty

    def test_graph_roundtrip_identification(self) -> None:
        """DoWhy can identify effects using our converted graph."""
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import (
            causal_graph_to_dowhy_str,
        )

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        gml_str = causal_graph_to_dowhy_str(graph)
        # The string must parse as a valid dowhy graph
        import pandas as pd
        from dowhy import CausalModel

        # Minimal data just to check parsing
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=50),
                "z": rng.normal(size=50),
                "objective": rng.normal(size=50),
            }
        )
        model = CausalModel(data=df, treatment="z", outcome="objective", graph=gml_str)
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        assert identified is not None


# ---------------------------------------------------------------------------
# Tests for ObservationalEstimator.estimate_intervention
# ---------------------------------------------------------------------------


class TestObservationalEstimatorBackdoor:
    """Tests for backdoor adjustment estimation."""

    def test_estimate_returns_observational_estimate(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import (
            ObservationalEstimate,
            ObservationalEstimator,
        )

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm()

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="z",
            treatment_value=2.0,
            objective_name="objective",
        )
        assert isinstance(result, ObservationalEstimate)

    def test_identified_true_for_backdoor_adjustable_graph(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        # X -> Z -> Y: Z effect on Y is identifiable (X is adjustment set)
        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm()

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="z",
            treatment_value=2.0,
            objective_name="objective",
        )
        assert result.identified is True

    @pytest.mark.slow
    def test_backdoor_estimate_close_to_true_effect(self) -> None:
        """Backdoor-adjusted estimate should be close to the analytical solution."""
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        # X -> Z -> Y with Y = 3*Z + noise
        # E[Y | do(Z=2)] should be approx 3*2 = 6
        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm(n=500, treatment_effect=3.0)

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="z",
            treatment_value=2.0,
            objective_name="objective",
        )
        assert result.identified is True
        # True E[Y | do(Z=2)] = 3 * 2 = 6.0 (intercept ≈ 0)
        assert abs(result.expected_outcome - 6.0) < 1.5

    def test_confidence_interval_is_finite(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm(n=300)

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="z",
            treatment_value=1.0,
            objective_name="objective",
        )
        lo, hi = result.confidence_interval
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo < hi

    def test_method_field_reports_backdoor(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm()

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="z",
            treatment_value=1.0,
            objective_name="objective",
        )
        assert "backdoor" in result.method


# ---------------------------------------------------------------------------
# Tests for confounding correction
# ---------------------------------------------------------------------------


class TestConfoundingCorrection:
    """Tests that DoWhy corrects for confounding."""

    @pytest.mark.slow
    def test_naive_correlation_overestimates_in_confounded_graph(self) -> None:
        """Without adjustment, naive regression inflates the estimate."""
        log = _make_confounded_log(n=400)

        df = log.to_dataframe()
        # Naive OLS: regress objective on x
        x_vals = df["x"].values
        y_vals = df["objective"].values
        naive_slope = float(np.cov(x_vals, y_vals)[0, 1] / np.var(x_vals))
        # Naive estimate should be inflated (true = 1.0, naive > 2.0 due to U)
        assert naive_slope > 1.5, f"Expected naive to be inflated, got {naive_slope}"

    @pytest.mark.slow
    def test_identified_false_for_unidentifiable_confounded_graph(self) -> None:
        """When X <-> Y (confounder) and no observed backdoor set, identified=False."""
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        # X <-> Y (bidirected only, no other observed variable to block path)
        graph = CausalGraph(
            edges=[("x", "objective")],
            bidirected_edges=[("x", "objective")],
        )
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_confounded_log(n=200)

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="x",
            treatment_value=1.0,
            objective_name="objective",
        )
        # Effect is not identifiable via backdoor without observed adjustment set
        assert result.identified is False


# ---------------------------------------------------------------------------
# Tests for non-identifiable case
# ---------------------------------------------------------------------------


class TestNonIdentifiable:
    """Tests for graceful handling of non-identifiable effects."""

    def test_identified_false_is_returned(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import (
            ObservationalEstimate,
            ObservationalEstimator,
        )

        # Bidirected edge X <-> Y with no observed variables to adjust
        graph = CausalGraph(
            edges=[("x", "objective")],
            bidirected_edges=[("x", "objective")],
        )
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_confounded_log(n=200)

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="x",
            treatment_value=1.0,
            objective_name="objective",
        )
        assert isinstance(result, ObservationalEstimate)
        assert result.identified is False

    def test_fallback_returns_finite_value(self) -> None:
        """Even when not identified, a finite fallback estimate is returned."""
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.observational import ObservationalEstimator

        graph = CausalGraph(
            edges=[("x", "objective")],
            bidirected_edges=[("x", "objective")],
        )
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_confounded_log(n=200)

        result = estimator.estimate_intervention(
            experiment_log=log,
            treatment_var="x",
            treatment_value=1.0,
            objective_name="objective",
        )
        # Even if not identified, we should return a float (RF fallback)
        assert np.isfinite(result.expected_outcome)


# ---------------------------------------------------------------------------
# Tests for graceful degradation (DoWhy not installed)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests for ImportError when dowhy is not installed."""

    def test_import_error_when_dowhy_missing(self) -> None:
        """ObservationalEstimator raises ImportError with helpful message if dowhy missing."""
        import sys

        obs_mod = "causal_optimizer.estimator.observational"

        # Capture the already-imported observational module (if any) so we can
        # restore it after the test and avoid poisoning the module cache.
        original_obs = sys.modules.get(obs_mod)

        # Patch dowhy as unavailable and force fresh import of observational
        with patch.dict("sys.modules", {"dowhy": None}):
            # Remove stale cached version so we get a fresh import under the patch
            sys.modules.pop(obs_mod, None)

            from causal_optimizer.estimator.observational import ObservationalEstimator

            graph = CausalGraph(edges=[("x", "objective")])
            estimator = ObservationalEstimator(causal_graph=graph)

            log = _make_log_from_scm(n=50)

            with pytest.raises(ImportError, match="dowhy"):
                estimator.estimate_intervention(
                    experiment_log=log,
                    treatment_var="x",
                    treatment_value=1.0,
                    objective_name="objective",
                )

        # Restore original module so dowhy-patched version doesn't leak
        sys.modules.pop(obs_mod, None)
        if original_obs is not None:
            sys.modules[obs_mod] = original_obs


# ---------------------------------------------------------------------------
# Tests for EffectEstimator integration
# ---------------------------------------------------------------------------


class TestEffectEstimatorIntegration:
    """Tests for the 'observational' method in EffectEstimator."""

    def test_observational_method_requires_causal_graph(self) -> None:
        from causal_optimizer.estimator.effects import EffectEstimator

        estimator = EffectEstimator(method="observational")
        log = _make_log_from_scm(n=100)

        with pytest.raises(ValueError, match="causal_graph"):
            estimator.estimate_effect(
                experiment_log=log,
                treatment_param="z",
                treatment_value=1.0,
                control_value=0.0,
                objective_name="objective",
            )

    def test_observational_method_delegates_to_observational_estimator(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.effects import EffectEstimator

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = EffectEstimator(method="observational", causal_graph=graph)

        log = _make_log_from_scm(n=200)

        result = estimator.estimate_effect(
            experiment_log=log,
            treatment_param="z",
            treatment_value=2.0,
            control_value=0.0,
            objective_name="objective",
        )
        # Should return an EffectEstimate with method containing 'observational'
        from causal_optimizer.estimator.effects import EffectEstimate

        assert isinstance(result, EffectEstimate)
        assert "observational" in result.method

    def test_observational_falls_back_to_bootstrap_when_dowhy_unavailable(self) -> None:
        """When dowhy is not available, falls back to bootstrap."""
        import sys

        obs_mod = "causal_optimizer.estimator.observational"
        eff_mod = "causal_optimizer.estimator.effects"

        # Save originals to restore later
        original_obs = sys.modules.get(obs_mod)
        original_eff = sys.modules.get(eff_mod)

        with patch.dict("sys.modules", {"dowhy": None}):
            # Force fresh re-imports under the patch
            sys.modules.pop(obs_mod, None)
            sys.modules.pop(eff_mod, None)

            from causal_optimizer.estimator.effects import EffectEstimator

            graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
            estimator = EffectEstimator(method="observational", causal_graph=graph)

            log = _make_log_from_scm(n=100)

            result = estimator.estimate_effect(
                experiment_log=log,
                treatment_param="z",
                treatment_value=2.0,
                control_value=0.0,
                objective_name="objective",
            )
            # Falls back to bootstrap
            assert result.method == "bootstrap"

        # Restore originals so patched versions don't leak
        sys.modules.pop(obs_mod, None)
        sys.modules.pop(eff_mod, None)
        if original_obs is not None:
            sys.modules[obs_mod] = original_obs
        if original_eff is not None:
            sys.modules[eff_mod] = original_eff

    def test_unknown_method_still_raises(self) -> None:
        """EffectEstimator rejects invalid method at construction time."""
        from causal_optimizer.estimator.effects import EffectEstimator

        with pytest.raises(ValueError, match="method="):
            EffectEstimator(method="totally_unknown")

    def test_observational_point_estimate_is_finite(self) -> None:
        pytest.importorskip("dowhy")
        from causal_optimizer.estimator.effects import EffectEstimator

        graph = CausalGraph(edges=[("x", "z"), ("z", "objective")])
        estimator = EffectEstimator(method="observational", causal_graph=graph)

        log = _make_log_from_scm(n=200)

        result = estimator.estimate_effect(
            experiment_log=log,
            treatment_param="z",
            treatment_value=2.0,
            control_value=0.0,
            objective_name="objective",
        )
        assert np.isfinite(result.point_estimate)
        assert np.isfinite(result.confidence_interval[0])
        assert np.isfinite(result.confidence_interval[1])

    def test_invalid_obs_method_raises_value_error(self) -> None:
        """EffectEstimator rejects invalid obs_method at construction time."""
        from causal_optimizer.estimator.effects import EffectEstimator

        with pytest.raises(ValueError, match="obs_method"):
            EffectEstimator(method="observational", obs_method="invalid_strategy")

    def test_estimate_improvement_with_observational_method_uses_bootstrap(self) -> None:
        """estimate_improvement with method='observational' uses bootstrap path, not DoWhy."""
        from causal_optimizer.estimator.effects import EffectEstimator

        # estimate_improvement doesn't require dowhy — it compares against historical dist
        estimator = EffectEstimator(method="observational")

        log = _make_log_from_scm(n=100)

        # Request with a value clearly better than history (very low for minimize=True)
        result = estimator.estimate_improvement(
            experiment_log=log,
            current_value=-1000.0,
            objective_name="objective",
            minimize=True,
        )

        from causal_optimizer.estimator.effects import EffectEstimate

        assert isinstance(result, EffectEstimate)
        # Method label should be "bootstrap" since no DoWhy ran
        assert result.method == "bootstrap"
        # Very negative value should be clearly significant
        assert result.is_significant is True

    def test_observational_bootstrap_fallback_insufficient_data(self) -> None:
        """_observational_bootstrap_fallback returns insufficient_data when no split works."""
        from causal_optimizer.estimator.effects import EffectEstimator
        from causal_optimizer.types import ExperimentLog, ExperimentResult, ExperimentStatus

        # Build a log with a single row — neither exact-match nor midpoint can produce
        # 2+ samples per group, so the insufficient_data path fires.
        result_single = ExperimentResult(
            experiment_id="r-0",
            parameters={"z": 5.0},
            metrics={"objective": 42.0},
            status=ExperimentStatus.KEEP,
        )
        log = ExperimentLog(results=[result_single])

        graph = CausalGraph(edges=[("z", "objective")])
        estimator = EffectEstimator(method="observational", causal_graph=graph)

        # With a single row, all split strategies fail; result should be insufficient_data
        result = estimator._observational_bootstrap_fallback(  # type: ignore[attr-defined]
            experiment_log=log,
            treatment_param="z",
            treatment_value=10.0,
            control_value=0.0,
            objective_name="objective",
        )

        from causal_optimizer.estimator.effects import EffectEstimate

        assert isinstance(result, EffectEstimate)
        assert result.method == "insufficient_data"
        assert result.is_significant is False

    def test_nan_guard_in_observational_estimator(self) -> None:
        """ObservationalEstimator falls back to RF when DoWhy returns NaN estimate."""
        pytest.importorskip("dowhy")
        from unittest.mock import MagicMock, patch

        from causal_optimizer.estimator.observational import ObservationalEstimator

        graph = CausalGraph(edges=[("x", "objective")])
        estimator = ObservationalEstimator(causal_graph=graph, method="backdoor")

        log = _make_log_from_scm(n=100)

        # Mock DoWhy to return NaN estimate value
        mock_estimate = MagicMock()
        mock_estimate.value = float("nan")

        with patch("dowhy.CausalModel") as mock_model_cls:
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            mock_model.identify_effect.return_value = MagicMock(
                estimands={"backdoor": {"some": "estimand"}}
            )
            mock_model.estimate_effect.return_value = mock_estimate

            result = estimator.estimate_intervention(
                experiment_log=log,
                treatment_var="x",
                treatment_value=1.0,
                objective_name="objective",
            )

        # Should have fallen back to RF surrogate
        assert result.identified is False
        assert np.isfinite(result.expected_outcome)
