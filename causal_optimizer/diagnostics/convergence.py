"""Convergence analysis — did we plateau or abandon a climb?"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from causal_optimizer.diagnostics.models import ConvergenceAnalysis

if TYPE_CHECKING:
    from causal_optimizer.types import ExperimentLog

logger = logging.getLogger(__name__)

#: Minimum KEEP results needed for meaningful convergence analysis.
_MIN_RESULTS = 4


def analyze_convergence(
    experiment_log: ExperimentLog,
    objective_name: str,
    minimize: bool,
    total_budget: int | None = None,
) -> ConvergenceAnalysis:
    """Analyze optimization convergence: plateau detection and abandoned climbs.

    Computes improvement rates over early and late halves of the run,
    detects plateaus (late improvement near zero), and flags abandoned
    climbs (still improving when budget ran out).
    """
    from causal_optimizer.types import ExperimentStatus

    # Extract kept results' objective values in order
    kept_results = [r for r in experiment_log.results if r.status == ExperimentStatus.KEEP]
    obj_values = [r.metrics.get(objective_name, float("nan")) for r in kept_results]
    obj_values = [v for v in obj_values if not np.isnan(v)]

    n = len(obj_values)
    if n == 0:
        return _empty_analysis(total_budget, len(experiment_log.results))

    # Compute running best
    running_best: list[float] = []
    current_best = obj_values[0]
    best_step = 0
    for i, v in enumerate(obj_values):
        if minimize:
            if v <= current_best:
                current_best = v
                best_step = i
        else:
            if v >= current_best:
                current_best = v
                best_step = i
        running_best.append(current_best)

    # Steps since last improvement
    steps_since = n - 1 - best_step

    if n < _MIN_RESULTS:
        return ConvergenceAnalysis(
            plateaued=False,
            improvement_rate=0.0,
            improvement_rate_early=0.0,
            improvement_rate_late=0.0,
            best_objective=current_best,
            best_at_step=best_step,
            budget_remaining_fraction=_budget_fraction(total_budget, len(experiment_log.results)),
            steps_since_improvement=steps_since,
            abandoned_climb=False,
        )

    # Linear regression of running-best over index
    indices = np.arange(n, dtype=float)
    rb = np.array(running_best)
    # Negate if minimizing so "improvement" is always positive slope
    y = -rb if minimize else rb

    slope = _safe_linregress_slope(indices, y)
    mid = n // 2
    slope_early = _safe_linregress_slope(indices[:mid], y[:mid]) if mid >= 2 else slope
    slope_late = _safe_linregress_slope(indices[mid:], y[mid:]) if (n - mid) >= 2 else slope

    # Plateau detection: late slope within 1% of objective range
    obj_arr = np.array(obj_values)
    obj_range = float(np.max(obj_arr) - np.min(obj_arr)) if len(obj_values) > 1 else 1.0
    threshold = 0.01 * max(obj_range, 1e-10)
    plateaued = abs(slope_late) < threshold

    # Abandoned climb: still improving when budget ran out
    at_budget = total_budget is not None and len(experiment_log.results) >= total_budget
    abandoned_climb = slope_late > threshold and at_budget

    return ConvergenceAnalysis(
        plateaued=plateaued,
        improvement_rate=slope,
        improvement_rate_early=slope_early,
        improvement_rate_late=slope_late,
        best_objective=current_best,
        best_at_step=best_step,
        budget_remaining_fraction=_budget_fraction(total_budget, len(experiment_log.results)),
        steps_since_improvement=steps_since,
        abandoned_climb=abandoned_climb,
    )


def _safe_linregress_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Linear regression slope, returning 0.0 on degenerate input."""
    if len(x) < 2 or (np.max(x) - np.min(x)) == 0:
        return 0.0
    result = stats.linregress(x, y)
    return float(result.slope)


def _budget_fraction(total_budget: int | None, n_results: int) -> float:
    if total_budget is None or total_budget <= 0:
        return 0.0
    return max(0.0, 1.0 - n_results / total_budget)


def _empty_analysis(total_budget: int | None, n_results: int) -> ConvergenceAnalysis:
    return ConvergenceAnalysis(
        plateaued=False,
        improvement_rate=0.0,
        improvement_rate_early=0.0,
        improvement_rate_late=0.0,
        best_objective=float("nan"),
        best_at_step=0,
        budget_remaining_fraction=_budget_fraction(total_budget, n_results),
        steps_since_improvement=0,
        abandoned_climb=False,
    )
