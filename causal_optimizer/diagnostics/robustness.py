"""Robustness analysis — is the best result real or noise?"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from causal_optimizer.diagnostics.models import RobustnessAnalysis

if TYPE_CHECKING:
    from causal_optimizer.types import ExperimentLog, ExperimentResult, SearchSpace

logger = logging.getLogger(__name__)

#: Minimum kept results for robustness analysis.
_MIN_RESULTS = 4


def analyze_robustness(
    experiment_log: ExperimentLog,
    objective_name: str,
    minimize: bool,
    search_space: SearchSpace | None = None,
) -> RobustnessAnalysis:
    """Validate robustness of the best result using sensitivity analysis.

    Splits experiments into early (baseline) and late (improved) halves,
    then runs the sensitivity validator.  Also checks consistency among
    the top-K results to detect noise vs real basins.
    """
    from causal_optimizer.types import ExperimentStatus

    kept = [r for r in experiment_log.results if r.status == ExperimentStatus.KEEP]
    n = len(kept)

    if n < _MIN_RESULTS:
        return RobustnessAnalysis(
            best_result_robust=False,
            signal_to_noise=0.0,
            e_value=1.0,
            effect_size=0.0,
            summary="Insufficient data for robustness analysis",
            top_k_consistency=0.0,
        )

    # Split into early/late halves for sensitivity validation
    mid = n // 2
    early_ids = [r.experiment_id for r in kept[:mid]]
    late_ids = [r.experiment_id for r in kept[mid:]]

    try:
        from causal_optimizer.validator.sensitivity import SensitivityValidator

        validator = SensitivityValidator()
        report = validator.validate_improvement(experiment_log, early_ids, late_ids, objective_name)
        snr = report.signal_to_noise
        e_val = report.e_value
        effect = report.effect_size
        robust = report.is_robust
        summary = report.summary
    except Exception:
        logger.debug("Sensitivity validation failed", exc_info=True)
        snr = 0.0
        e_val = 1.0
        effect = 0.0
        robust = False
        summary = "Sensitivity validation failed"

    # Top-K consistency: do the top 3 results agree on parameter values?
    consistency = _top_k_consistency(kept, objective_name, minimize, search_space)

    return RobustnessAnalysis(
        best_result_robust=robust,
        signal_to_noise=snr,
        e_value=e_val,
        effect_size=effect,
        summary=summary,
        top_k_consistency=consistency,
    )


def _top_k_consistency(
    kept_results: list[ExperimentResult],
    objective_name: str,
    minimize: bool,
    search_space: SearchSpace | None,
    k: int = 3,
) -> float:
    """Compute parameter similarity among the top-K results.

    Returns a value in [0, 1] where 1 means all top-K results have
    identical parameters (a real basin) and 0 means they're completely
    different (noise).
    """
    if len(kept_results) < 2:
        return 0.0

    # Sort by objective
    sorted_results = sorted(
        kept_results,
        key=lambda r: r.metrics.get(objective_name, float("inf") if minimize else float("-inf")),
        reverse=not minimize,
    )
    top_k = sorted_results[: min(k, len(sorted_results))]

    if len(top_k) < 2 or search_space is None:
        return 0.0

    # Compute pairwise similarity
    similarities: list[float] = []
    for i in range(len(top_k)):
        for j in range(i + 1, len(top_k)):
            sim = _param_similarity(top_k[i].parameters, top_k[j].parameters, search_space)
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def _param_similarity(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    search_space: SearchSpace,
) -> float:
    """Fraction of variables within 10% of each other's range."""
    from causal_optimizer.types import VariableType

    matches = 0
    total = 0

    for var in search_space.variables:
        name = var.name
        if name not in params_a or name not in params_b:
            continue
        total += 1

        a, b = params_a[name], params_b[name]
        if var.variable_type in (VariableType.CONTINUOUS, VariableType.INTEGER):
            if var.lower is not None and var.upper is not None:
                var_range = var.upper - var.lower
                if var_range > 0 and abs(a - b) / var_range < 0.1:
                    matches += 1
            elif a == b:
                matches += 1
        elif a == b:
            matches += 1

    return matches / total if total > 0 else 0.0
