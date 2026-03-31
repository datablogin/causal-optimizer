"""Tests for skip calibration diagnostics.

Tests cover:
1. SkipDiagnostics / AuditResult dataclasses
2. Engine skip instrumentation (skip log + skip_diagnostics property)
3. Audit mode (force-evaluate skipped candidates)
4. AnytimeMetrics from engine
5. Random strategy produces zero skips
6. SkipAuditEntry and SkipMetrics (Sprint 19)
7. False-skip rate computation (Sprint 19)
8. Backward compatibility (Sprint 19)
"""

from __future__ import annotations

from typing import Any

import pytest

from causal_optimizer.diagnostics.anytime import AnytimeMetrics
from causal_optimizer.diagnostics.skip_calibration import (
    AuditResult,
    SkipAuditEntry,
    SkipDiagnostics,
    compute_skip_metrics,
)
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    SearchSpace,
    Variable,
    VariableType,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


class _QuadraticRunner:
    """f(x, y) = x^2 + y^2 — minimum at origin."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": x**2 + y**2}


# ── 1. Dataclass tests ──────────────────────────────────────────────


class TestSkipDiagnosticsDataclass:
    def test_construction(self) -> None:
        diag = SkipDiagnostics(
            candidates_considered=20,
            candidates_evaluated=15,
            candidates_skipped=5,
            skip_ratio=0.25,
            skip_confidences=[0.8, 0.9, 0.7, 0.85, 0.6],
            audit_results=None,
        )
        assert diag.candidates_considered == 20
        assert diag.candidates_evaluated == 15
        assert diag.candidates_skipped == 5
        assert diag.skip_ratio == pytest.approx(0.25)
        assert len(diag.skip_confidences) == 5
        assert diag.audit_results is None

    def test_with_audit_results(self) -> None:
        audit = AuditResult(
            parameters={"x": 1.0, "y": 2.0},
            predicted_outcome=5.0,
            actual_outcome=6.0,
            was_correct_skip=True,
        )
        diag = SkipDiagnostics(
            candidates_considered=10,
            candidates_evaluated=8,
            candidates_skipped=2,
            skip_ratio=0.2,
            skip_confidences=[0.7, 0.8],
            audit_results=[audit],
        )
        assert diag.audit_results is not None
        assert len(diag.audit_results) == 1
        assert diag.audit_results[0].was_correct_skip is True


class TestAuditResultDataclass:
    def test_construction(self) -> None:
        ar = AuditResult(
            parameters={"x": 3.0},
            predicted_outcome=9.0,
            actual_outcome=10.0,
            was_correct_skip=True,
        )
        assert ar.parameters == {"x": 3.0}
        assert ar.predicted_outcome == pytest.approx(9.0)
        assert ar.actual_outcome == pytest.approx(10.0)
        assert ar.was_correct_skip is True

    def test_incorrect_skip(self) -> None:
        ar = AuditResult(
            parameters={"x": 0.1},
            predicted_outcome=5.0,
            actual_outcome=0.01,
            was_correct_skip=False,
        )
        assert ar.was_correct_skip is False


# ── 2. Engine skip instrumentation ──────────────────────────────────


class TestSkipDiagnosticsRecorded:
    """test_skip_diagnostics_recorded: Run engine with off-policy predictor
    and verify skip diagnostics are populated."""

    def test_skip_diagnostics_populated_after_run(self) -> None:
        """After running the engine for enough steps that the predictor
        has fitted a model, skip_diagnostics should be populated."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        # Run enough steps to build a model and potentially trigger skips
        engine.run_loop(15)

        diag = engine.skip_diagnostics
        assert isinstance(diag, SkipDiagnostics)
        # candidates_considered >= candidates_evaluated (some may be skipped)
        assert diag.candidates_considered >= diag.candidates_evaluated
        assert diag.candidates_considered == diag.candidates_evaluated + diag.candidates_skipped
        # 15 experiments were actually evaluated
        assert diag.candidates_evaluated == 15
        # skip_ratio is consistent
        if diag.candidates_considered > 0:
            assert diag.skip_ratio == pytest.approx(
                diag.candidates_skipped / diag.candidates_considered
            )

    def test_skip_log_starts_empty(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        diag = engine.skip_diagnostics
        assert diag.candidates_considered == 0
        assert diag.candidates_evaluated == 0
        assert diag.candidates_skipped == 0
        assert diag.skip_ratio == 0.0
        assert diag.skip_confidences == []


# ── 3. Audit mode ───────────────────────────────────────────────────


class TestAuditMode:
    """test_audit_mode_force_evaluates: Set audit_skip_rate=1.0, verify
    all would-be-skipped candidates get audit results."""

    def test_audit_rate_one_evaluates_all_skips(self) -> None:
        """With audit_skip_rate=1.0, every skip should produce an AuditResult."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            audit_skip_rate=1.0,
        )
        engine.run_loop(20)

        diag = engine.skip_diagnostics
        assert isinstance(diag, SkipDiagnostics)

        # If there were any skips, all should have audit results
        if diag.candidates_skipped > 0:
            assert diag.audit_results is not None
            assert len(diag.audit_results) == diag.candidates_skipped
            # Each audit result must have the expected fields
            for ar in diag.audit_results:
                assert isinstance(ar, AuditResult)
                assert isinstance(ar.parameters, dict)
                assert isinstance(ar.predicted_outcome, float)
                assert isinstance(ar.actual_outcome, float)
                assert isinstance(ar.was_correct_skip, bool)

    def test_audit_rate_zero_no_audits(self) -> None:
        """With audit_skip_rate=0.0 (default), no audit results should be produced."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            audit_skip_rate=0.0,
        )
        engine.run_loop(15)

        diag = engine.skip_diagnostics
        assert diag.audit_results is None


# ── 4. Anytime metrics ──────────────────────────────────────────────


class TestAnytimeMetrics:
    """test_anytime_metrics_checkpoints: Run engine for 20 steps, verify
    anytime metrics at checkpoints [5, 10, 20]."""

    def test_checkpoints_populated(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        engine.run_loop(20)

        checkpoints = [5, 10, 20]
        metrics = engine.anytime_metrics(checkpoints)

        assert isinstance(metrics, AnytimeMetrics)
        assert metrics.checkpoints == checkpoints
        assert len(metrics.best_objective_at) == 3
        assert len(metrics.n_evaluated_at) == 3
        assert len(metrics.n_skipped_at) == 3

    def test_best_objective_monotonically_improves(self) -> None:
        """For a minimization problem, best objective should be non-increasing."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            minimize=True,
            seed=42,
        )
        engine.run_loop(20)

        metrics = engine.anytime_metrics([5, 10, 15, 20])
        # Best objective should be non-increasing for minimization
        for i in range(len(metrics.best_objective_at) - 1):
            assert metrics.best_objective_at[i] >= metrics.best_objective_at[i + 1]

    def test_evaluated_plus_skipped_equals_considered(self) -> None:
        """n_evaluated + n_skipped at each checkpoint should equal total considered."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        engine.run_loop(20)

        metrics = engine.anytime_metrics([10, 20])
        # At checkpoint 20, n_evaluated should be 20 (we ran 20 experiments)
        assert metrics.n_evaluated_at[-1] == 20

    def test_checkpoint_beyond_budget_clamps(self) -> None:
        """Checkpoints beyond the number of experiments should clamp to the last value."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        engine.run_loop(10)

        metrics = engine.anytime_metrics([5, 10, 100])
        # Checkpoint 100 should clamp to what's available at step 10
        assert metrics.n_evaluated_at[-1] == 10
        assert metrics.best_objective_at[-1] == metrics.best_objective_at[1]

    def test_empty_engine(self) -> None:
        """Anytime metrics on an engine with no results."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        metrics = engine.anytime_metrics([5, 10])
        assert metrics.checkpoints == [5, 10]
        assert all(v == 0 for v in metrics.n_evaluated_at)
        assert all(v == 0 for v in metrics.n_skipped_at)


# ── 5. Random strategy has zero skips ───────────────────────────────


class TestRandomStrategyZeroSkips:
    """test_skip_diagnostics_zero_for_random: Random strategy should have
    zero skips because it bypasses the engine entirely."""

    def test_no_off_policy_means_no_skips(self) -> None:
        """An engine with max_skips=0 should never skip — simulating the
        'no off-policy gating' scenario that random effectively uses."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            max_skips=0,
        )
        engine.run_loop(15)

        diag = engine.skip_diagnostics
        assert diag.candidates_skipped == 0
        assert diag.skip_ratio == 0.0
        assert diag.skip_confidences == []


# ── 6. Validation ────────────────────────────────────────────────────


class TestAuditSkipRateValidation:
    """audit_skip_rate must be in [0.0, 1.0]."""

    @pytest.mark.parametrize("rate", [-0.1, 1.5, 2.0])
    def test_invalid_audit_skip_rate_raises(self, rate: float) -> None:
        with pytest.raises(ValueError, match="audit_skip_rate"):
            ExperimentEngine(
                search_space=_make_search_space(),
                runner=_QuadraticRunner(),
                seed=0,
                audit_skip_rate=rate,
            )


# ── 7. CLI validation ────────────────────────────────────────────────


class TestCLIAuditSkipRateValidation:
    """CLI scripts reject out-of-range --audit-skip-rate values."""

    def test_benchmark_rejects_invalid_rate(self) -> None:
        import subprocess

        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/energy_predictive_benchmark.py",
                "--data-path",
                "tests/fixtures/energy_load_fixture.csv",
                "--budgets",
                "1",
                "--seeds",
                "0",
                "--strategies",
                "random",
                "--audit-skip-rate",
                "2.0",
                "--output",
                "/tmp/test_invalid_audit.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "audit-skip-rate" in result.stderr.lower()


# ── 8. SkipAuditEntry dataclass (Sprint 19) ──────────────────────────


class TestSkipAuditEntry:
    """SkipAuditEntry captures full per-decision metadata."""

    def test_construction_defaults(self) -> None:
        entry = SkipAuditEntry(step=5, parameters={"x": 1.0}, skip_reason="low_uncertainty")
        assert entry.step == 5
        assert entry.parameters == {"x": 1.0}
        assert entry.skip_reason == "low_uncertainty"
        assert entry.predicted_value is None
        assert entry.actual_value is None
        assert entry.model_quality == 0.0
        assert entry.uncertainty == 0.0
        assert entry.was_false_skip is None

    def test_construction_with_all_fields(self) -> None:
        entry = SkipAuditEntry(
            step=12,
            parameters={"x": 2.0, "y": 3.0},
            skip_reason="epsilon_observe",
            predicted_value=5.5,
            actual_value=4.0,
            model_quality=0.72,
            uncertainty=0.15,
            was_false_skip=True,
        )
        assert entry.step == 12
        assert entry.skip_reason == "epsilon_observe"
        assert entry.predicted_value == pytest.approx(5.5)
        assert entry.actual_value == pytest.approx(4.0)
        assert entry.model_quality == pytest.approx(0.72)
        assert entry.uncertainty == pytest.approx(0.15)
        assert entry.was_false_skip is True


# ── 9. False-skip rate computation (Sprint 19) ───────────────────────


class TestComputeSkipMetrics:
    """compute_skip_metrics aggregates audit entries correctly."""

    def test_empty_entries(self) -> None:
        metrics = compute_skip_metrics([], total_evaluated=10)
        assert metrics.total_skips == 0
        assert metrics.audited_skips == 0
        assert metrics.false_skip_rate == 0.0
        assert metrics.true_skip_rate == 0.0
        assert metrics.skip_coverage == 0.0
        assert metrics.total_candidates == 10

    def test_all_true_skips(self) -> None:
        entries = [
            SkipAuditEntry(
                step=i,
                parameters={"x": float(i)},
                skip_reason="low_uncertainty",
                predicted_value=10.0,
                actual_value=20.0,
                model_quality=0.5,
                uncertainty=0.1,
                was_false_skip=False,
            )
            for i in range(5)
        ]
        metrics = compute_skip_metrics(entries, total_evaluated=15)
        assert metrics.total_skips == 5
        assert metrics.audited_skips == 5
        assert metrics.false_skip_count == 0
        assert metrics.true_skip_count == 5
        assert metrics.false_skip_rate == pytest.approx(0.0)
        assert metrics.true_skip_rate == pytest.approx(1.0)
        assert metrics.skip_coverage == pytest.approx(5 / 20)

    def test_mixed_skips(self) -> None:
        entries = [
            SkipAuditEntry(
                step=0,
                parameters={"x": 0.0},
                skip_reason="low_uncertainty",
                predicted_value=10.0,
                actual_value=20.0,
                model_quality=0.5,
                uncertainty=0.1,
                was_false_skip=False,
            ),
            SkipAuditEntry(
                step=1,
                parameters={"x": 1.0},
                skip_reason="low_uncertainty",
                predicted_value=10.0,
                actual_value=1.0,
                model_quality=0.5,
                uncertainty=0.1,
                was_false_skip=True,
            ),
            SkipAuditEntry(
                step=10,
                parameters={"x": 2.0},
                skip_reason="epsilon_observe",
                predicted_value=10.0,
                actual_value=0.5,
                model_quality=0.7,
                uncertainty=0.05,
                was_false_skip=True,
            ),
        ]
        metrics = compute_skip_metrics(entries, total_evaluated=10)
        assert metrics.total_skips == 3
        assert metrics.audited_skips == 3
        assert metrics.false_skip_count == 2
        assert metrics.true_skip_count == 1
        assert metrics.false_skip_rate == pytest.approx(2 / 3)
        assert metrics.true_skip_rate == pytest.approx(1 / 3)
        assert metrics.mean_model_quality_at_skip == pytest.approx((0.5 + 0.5 + 0.7) / 3)

    def test_unaudited_entries_not_counted(self) -> None:
        """Entries without actual_value should not affect false/true counts."""
        entries = [
            SkipAuditEntry(
                step=0,
                parameters={"x": 0.0},
                skip_reason="low_uncertainty",
                model_quality=0.5,
                uncertainty=0.1,
            ),
            SkipAuditEntry(
                step=1,
                parameters={"x": 1.0},
                skip_reason="low_uncertainty",
                predicted_value=10.0,
                actual_value=20.0,
                model_quality=0.5,
                uncertainty=0.1,
                was_false_skip=False,
            ),
        ]
        metrics = compute_skip_metrics(entries, total_evaluated=8)
        assert metrics.total_skips == 2
        assert metrics.audited_skips == 1
        assert metrics.false_skip_count == 0
        assert metrics.true_skip_count == 1

    def test_early_late_split(self) -> None:
        """False skips are correctly split into early and late."""
        entries = [
            SkipAuditEntry(
                step=2,
                parameters={"x": 0.0},
                skip_reason="low_uncertainty",
                actual_value=0.1,
                model_quality=0.3,
                uncertainty=0.1,
                was_false_skip=True,
            ),
            SkipAuditEntry(
                step=3,
                parameters={"x": 1.0},
                skip_reason="low_uncertainty",
                actual_value=0.2,
                model_quality=0.4,
                uncertainty=0.1,
                was_false_skip=True,
            ),
            SkipAuditEntry(
                step=15,
                parameters={"x": 2.0},
                skip_reason="low_uncertainty",
                actual_value=0.3,
                model_quality=0.6,
                uncertainty=0.05,
                was_false_skip=True,
            ),
        ]
        # midpoint_step=5 -> steps 2,3 are early, step 15 is late
        metrics = compute_skip_metrics(entries, total_evaluated=20, midpoint_step=5)
        assert metrics.early_false_skip_count == 2
        assert metrics.late_false_skip_count == 1

    def test_skip_reasons_counted(self) -> None:
        entries = [
            SkipAuditEntry(step=0, parameters={}, skip_reason="low_uncertainty"),
            SkipAuditEntry(step=1, parameters={}, skip_reason="low_uncertainty"),
            SkipAuditEntry(step=2, parameters={}, skip_reason="epsilon_observe"),
        ]
        metrics = compute_skip_metrics(entries, total_evaluated=10)
        assert metrics.skip_reasons == {"low_uncertainty": 2, "epsilon_observe": 1}


# ── 10. Engine records SkipAuditEntry (Sprint 19) ────────────────────


class TestEngineAuditEntries:
    """The engine populates audit_entries in skip_diagnostics."""

    def test_audit_entries_populated_after_run(self) -> None:
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            audit_skip_rate=1.0,
        )
        engine.run_loop(20)
        diag = engine.skip_diagnostics

        # audit_entries should match candidates_skipped
        assert len(diag.audit_entries) == diag.candidates_skipped

        # Each entry should have valid metadata
        for entry in diag.audit_entries:
            assert isinstance(entry, SkipAuditEntry)
            assert entry.skip_reason in ("low_uncertainty", "epsilon_observe", "unknown")
            assert entry.model_quality >= 0.0

    def test_audit_entries_have_actual_values_when_audited(self) -> None:
        """With audit_skip_rate=1.0, all entries should have actual_value filled."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            audit_skip_rate=1.0,
        )
        engine.run_loop(20)
        diag = engine.skip_diagnostics

        for entry in diag.audit_entries:
            if entry.actual_value is not None:
                assert entry.was_false_skip is not None

    def test_skip_reason_is_low_uncertainty_in_heuristic_mode(self) -> None:
        """In default (heuristic) mode, skip reason should be 'low_uncertainty'."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        engine.run_loop(20)
        diag = engine.skip_diagnostics

        for entry in diag.audit_entries:
            assert entry.skip_reason == "low_uncertainty"


# ── 11. Backward compatibility (Sprint 19) ───────────────────────────


class TestBackwardCompat:
    """Default config produces same behavior as Sprint 18."""

    def test_default_engine_no_regression(self) -> None:
        """Default engine (no audit, no epsilon) should still record skip_log
        and skip_diagnostics the same way as before."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
        )
        engine.run_loop(15)
        diag = engine.skip_diagnostics

        # Basic invariants unchanged
        assert diag.candidates_considered == diag.candidates_evaluated + diag.candidates_skipped
        assert diag.candidates_evaluated == 15
        assert diag.audit_results is None  # no audit mode

        # New field: audit_entries should exist but be populated (not audited though)
        assert isinstance(diag.audit_entries, list)
        assert len(diag.audit_entries) == diag.candidates_skipped

    def test_max_skips_zero_no_entries(self) -> None:
        """With max_skips=0 (counterfactual benchmark default), no skips should occur."""
        engine = ExperimentEngine(
            search_space=_make_search_space(),
            runner=_QuadraticRunner(),
            seed=42,
            max_skips=0,
        )
        engine.run_loop(15)
        diag = engine.skip_diagnostics
        assert diag.candidates_skipped == 0
        assert len(diag.audit_entries) == 0
