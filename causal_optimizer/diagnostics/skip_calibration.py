"""Skip calibration diagnostics -- per-run skip decision tracking.

Provides dataclasses for recording and summarizing off-policy predictor
skip decisions, including optional audit mode for validating skip quality.

The :class:`SkipAuditEntry` captures full per-decision metadata (model
quality, uncertainty, skip reason) for post-hoc analysis.  The
:func:`compute_skip_metrics` helper computes false-skip rate, true-skip
rate, and other calibration statistics from a list of audit results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkipAuditEntry:
    """Full per-decision audit record for a single skip event.

    Captures the predictor state at the moment the skip decision was made,
    enabling post-hoc analysis of skip quality.  Produced by the engine
    when ``audit_skips=True`` or ``audit_skip_rate > 0``.

    Attributes:
        step: The experiment step index at which the skip occurred.
        parameters: The candidate parameters that were skipped.
        skip_reason: Why the experiment was skipped.  One of
            ``"low_uncertainty"`` (heuristic mode, uncertainty below threshold),
            ``"epsilon_observe"`` (epsilon mode chose observe), or
            ``"unknown"`` (fallback).
        predicted_value: The predictor's expected outcome (``None`` if
            prediction was unavailable).
        actual_value: The true outcome from force-evaluation (``None``
            if the candidate was not audited).
        model_quality: Cross-validated R-squared of the surrogate at
            decision time.
        uncertainty: Predicted standard deviation at decision time.
        was_false_skip: Whether the skipped experiment would have improved
            over the current best.  ``None`` if not audited.
    """

    step: int
    parameters: dict[str, Any]
    skip_reason: str
    predicted_value: float | None = None
    actual_value: float | None = None
    model_quality: float = 0.0
    uncertainty: float = 0.0
    was_false_skip: bool | None = None


@dataclass
class AuditResult:
    """Result of force-evaluating a skipped candidate."""

    parameters: dict[str, Any]
    predicted_outcome: float
    actual_outcome: float
    was_correct_skip: bool  # actual was indeed worse than best


@dataclass
class SkipDiagnostics:
    """Per-run skip decision diagnostics."""

    candidates_considered: int  # total suggest() calls
    candidates_evaluated: int  # actually ran
    candidates_skipped: int  # predicted poor -> skipped
    skip_ratio: float  # skipped / considered
    skip_confidences: list[float]  # confidence scores of skip decisions
    audit_results: list[AuditResult] | None  # if audit mode enabled
    audit_entries: list[SkipAuditEntry] = field(default_factory=list)


@dataclass
class SkipMetrics:
    """Aggregated skip calibration metrics computed from audit entries.

    Attributes:
        total_skips: Number of skip decisions recorded.
        audited_skips: Number of skips that were force-evaluated.
        false_skip_count: Skips where running would have improved on best.
        true_skip_count: Skips where running would NOT have improved.
        false_skip_rate: ``false_skip_count / audited_skips`` (0 if none audited).
        true_skip_rate: ``true_skip_count / audited_skips`` (0 if none audited).
        skip_coverage: ``total_skips / total_candidates`` (0 if no candidates).
        total_candidates: Total candidates considered (skipped + evaluated).
        mean_model_quality_at_skip: Mean model R-squared at skip decisions.
        mean_uncertainty_at_skip: Mean uncertainty at skip decisions.
        early_false_skip_count: False skips in the first half of optimization.
        late_false_skip_count: False skips in the second half.
        skip_reasons: Count of each skip reason.
    """

    total_skips: int
    audited_skips: int
    false_skip_count: int
    true_skip_count: int
    false_skip_rate: float
    true_skip_rate: float
    skip_coverage: float
    total_candidates: int
    mean_model_quality_at_skip: float
    mean_uncertainty_at_skip: float
    early_false_skip_count: int
    late_false_skip_count: int
    skip_reasons: dict[str, int] = field(default_factory=dict)


def compute_skip_metrics(
    entries: list[SkipAuditEntry],
    total_evaluated: int,
    midpoint_step: int | None = None,
) -> SkipMetrics:
    """Compute aggregated skip metrics from audit entries.

    Args:
        entries: List of :class:`SkipAuditEntry` from one engine run.
        total_evaluated: Number of experiments that were actually evaluated
            (not skipped).  Used to compute skip coverage.
        midpoint_step: Step index dividing "early" from "late" skips.
            Defaults to half of the maximum step seen.

    Returns:
        A :class:`SkipMetrics` summarizing the skip calibration quality.
    """
    total_skips = len(entries)
    total_candidates = total_skips + total_evaluated

    # Collect audited entries (those with actual_value set)
    audited = [e for e in entries if e.actual_value is not None]
    audited_skips = len(audited)

    false_skip_count = sum(1 for e in audited if e.was_false_skip is True)
    true_skip_count = sum(1 for e in audited if e.was_false_skip is False)

    false_skip_rate = false_skip_count / audited_skips if audited_skips > 0 else 0.0
    true_skip_rate = true_skip_count / audited_skips if audited_skips > 0 else 0.0

    skip_coverage = total_skips / total_candidates if total_candidates > 0 else 0.0

    # Model quality and uncertainty at skip time
    qualities = [e.model_quality for e in entries]
    uncertainties = [e.uncertainty for e in entries]
    mean_quality = sum(qualities) / len(qualities) if qualities else 0.0
    mean_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0

    # Early vs late false skips
    if midpoint_step is None:
        max_step = max((e.step for e in entries), default=0)
        midpoint_step = max_step // 2

    early_false = sum(1 for e in audited if e.was_false_skip is True and e.step <= midpoint_step)
    late_false = sum(1 for e in audited if e.was_false_skip is True and e.step > midpoint_step)

    # Reason counts
    reasons: dict[str, int] = {}
    for e in entries:
        reasons[e.skip_reason] = reasons.get(e.skip_reason, 0) + 1

    return SkipMetrics(
        total_skips=total_skips,
        audited_skips=audited_skips,
        false_skip_count=false_skip_count,
        true_skip_count=true_skip_count,
        false_skip_rate=false_skip_rate,
        true_skip_rate=true_skip_rate,
        skip_coverage=skip_coverage,
        total_candidates=total_candidates,
        mean_model_quality_at_skip=mean_quality,
        mean_uncertainty_at_skip=mean_uncertainty,
        early_false_skip_count=early_false,
        late_false_skip_count=late_false,
        skip_reasons=reasons,
    )
