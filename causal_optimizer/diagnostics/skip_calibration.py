"""Skip calibration diagnostics — per-run skip decision tracking.

Provides dataclasses for recording and summarizing off-policy predictor
skip decisions, including optional audit mode for validating skip quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
