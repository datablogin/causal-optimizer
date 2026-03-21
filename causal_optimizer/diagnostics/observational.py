"""Observational signal analysis for experiment diagnostics.

Analyzes identifiability and observational effect estimates for variables
in the search space, comparing them against experimental estimates when
available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from causal_optimizer.diagnostics.models import (
    ObservationalAnalysis,
    ObservationalVariableReport,
)
from causal_optimizer.types import ExperimentStatus

if TYPE_CHECKING:
    import pandas as pd

    from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)

# Minimum number of KEEP experiments required for estimation
_MIN_EXPERIMENTS = 10

# Identification methods tried in order of preference (shared with off_policy.py)
IDENTIFICATION_METHODS: tuple[str, ...] = ("backdoor", "frontdoor", "iv")


def _try_import_estimator() -> type | None:
    """Try to import ObservationalEstimator; return class or None."""
    try:
        from causal_optimizer.estimator.observational import ObservationalEstimator

        return ObservationalEstimator
    except ImportError:
        return None


# Module-level reference that can be patched in tests.
# This avoids repeated try/except and allows clean mocking.
ObservationalEstimator: type | None = _try_import_estimator()


def analyze_observational(
    experiment_log: ExperimentLog,
    search_space: SearchSpace,
    objective_name: str,
    minimize: bool,  # noqa: ARG001 — reserved for future direction-aware logic
    causal_graph: CausalGraph | None = None,
) -> ObservationalAnalysis:
    """Analyze observational signals from experiment data.

    For each search space variable that is a causal ancestor of the objective:
    1. Check identifiability via ObservationalEstimator (backdoor, frontdoor, IV)
    2. If identifiable and >= 10 KEEP experiments: estimate effect, record CI
    3. Compute agreement with experimental estimates if available

    Graceful degradation:
    - No causal graph -> recommendation="no causal graph available"
    - DoWhy not installed -> all identifiable=False
    - < 10 experiments -> skip estimation

    Args:
        experiment_log: Historical experiment data.
        search_space: The search space definition.
        objective_name: Name of the objective metric.
        minimize: Whether the objective is being minimized. Reserved for
            future direction-aware recommendation logic.
        causal_graph: Optional causal graph for identifiability analysis.

    Returns:
        ObservationalAnalysis with per-variable reports and summary.
    """
    n_variables = len(search_space.variables)

    if causal_graph is None:
        return ObservationalAnalysis(
            n_identifiable=0,
            n_variables=n_variables,
            variables=[],
            obs_experimental_agreement=None,
            recommendation="no causal graph available",
        )

    # Get causal ancestors of the objective
    if objective_name in causal_graph.nodes:
        ancestors = causal_graph.ancestors(objective_name)
    else:
        ancestors = set()

    # Filter to search space variables that are ancestors
    ancestor_vars = [v.name for v in search_space.variables if v.name in ancestors]

    if not ancestor_vars:
        return ObservationalAnalysis(
            n_identifiable=0,
            n_variables=n_variables,
            variables=[],
            obs_experimental_agreement=None,
            recommendation="no causal ancestors found in search space",
        )

    # Count KEEP experiments
    keep_results = [r for r in experiment_log.results if r.status == ExperimentStatus.KEEP]
    n_keep = len(keep_results)

    # Check if the estimator class is available
    estimator_cls = ObservationalEstimator
    dowhy_available = estimator_cls is not None

    if not dowhy_available:
        logger.info("DoWhy not available; all variables marked as non-identifiable")

    # Compute dataframe once for reuse across all variable analyses
    df = experiment_log.to_dataframe()

    # Compute experimental effect estimates for comparison
    exp_estimates = _compute_experimental_estimates(df, objective_name, ancestor_vars)

    variables: list[ObservationalVariableReport] = []
    n_identifiable = 0

    for var_name in ancestor_vars:
        report = _analyze_variable(
            var_name=var_name,
            estimator_cls=estimator_cls,
            causal_graph=causal_graph,
            experiment_log=experiment_log,
            df=df,
            objective_name=objective_name,
            n_keep=n_keep,
            exp_estimate=exp_estimates.get(var_name),
        )
        if report.identifiable:
            n_identifiable += 1
        variables.append(report)

    # Compute aggregate agreement
    obs_experimental_agreement = _compute_aggregate_agreement(variables)

    # Generate recommendation
    recommendation = _generate_recommendation(
        n_identifiable=n_identifiable,
        n_ancestor_vars=len(ancestor_vars),
        obs_experimental_agreement=obs_experimental_agreement,
        dowhy_available=dowhy_available,
        n_keep=n_keep,
    )

    return ObservationalAnalysis(
        n_identifiable=n_identifiable,
        n_variables=n_variables,
        variables=variables,
        obs_experimental_agreement=obs_experimental_agreement,
        recommendation=recommendation,
    )


def _analyze_variable(
    var_name: str,
    estimator_cls: type | None,
    causal_graph: CausalGraph,
    experiment_log: ExperimentLog,
    df: pd.DataFrame,
    objective_name: str,
    n_keep: int,
    exp_estimate: float | None,
) -> ObservationalVariableReport:
    """Analyze a single variable's observational identifiability and estimate."""
    if estimator_cls is None or n_keep < _MIN_EXPERIMENTS:
        return ObservationalVariableReport(
            variable_name=var_name,
            identifiable=False,
            identification_method=None,
            obs_estimate=None,
            obs_ci=None,
            exp_estimate=exp_estimate,
            agreement=None,
        )

    # Try each identification method in order of preference
    for method in IDENTIFICATION_METHODS:
        try:
            est = estimator_cls(causal_graph=causal_graph, method=method)
            if var_name not in df.columns:
                continue
            treatment_value = float(df[var_name].median())

            result = est.estimate_intervention(
                experiment_log=experiment_log,
                treatment_var=var_name,
                treatment_value=treatment_value,
                objective_name=objective_name,
            )

            if result.identified:
                agreement = None
                if exp_estimate is not None and result.expected_outcome is not None:
                    agreement = _compute_agreement(result.expected_outcome, exp_estimate)

                return ObservationalVariableReport(
                    variable_name=var_name,
                    identifiable=True,
                    identification_method=result.method,
                    obs_estimate=result.expected_outcome,
                    obs_ci=result.confidence_interval,
                    exp_estimate=exp_estimate,
                    agreement=agreement,
                )
        except (ValueError, RuntimeError, KeyError) as exc:
            logger.debug("Method %s failed for %s: %s", method, var_name, exc)
            continue
        except Exception as exc:
            logger.warning("Unexpected error in method %s for %s: %s", method, var_name, exc)
            continue

    return ObservationalVariableReport(
        variable_name=var_name,
        identifiable=False,
        identification_method=None,
        obs_estimate=None,
        obs_ci=None,
        exp_estimate=exp_estimate,
        agreement=None,
    )


def _compute_experimental_estimates(
    df: pd.DataFrame,
    objective_name: str,
    ancestor_vars: list[str],
) -> dict[str, float]:
    """Compute experimental outcome estimates at each variable's median.

    For each variable, selects experiments where the variable is near its
    median (within +/- 25% of the interquartile range) and returns the mean
    objective value. This produces an estimate comparable to the observational
    ``E[Y | do(T = median)]``.
    """
    estimates: dict[str, float] = {}

    for var_name in ancestor_vars:
        if var_name not in df.columns or objective_name not in df.columns:
            continue
        col = df[var_name]
        if col.nunique() < 2:
            continue
        median = col.median()
        iqr = col.quantile(0.75) - col.quantile(0.25)
        # Use 25% of IQR as the "near median" band; fall back to 10% of
        # median when the IQR is zero (e.g., low-variance variable).
        half_band = max(iqr * 0.25, abs(median) * 0.1, 1e-10)
        near_mask = (col >= median - half_band) & (col <= median + half_band)

        if near_mask.sum() < 2:
            # Not enough data near the median; cannot compute a comparable
            # experimental estimate — skip this variable.
            continue

        mean_obj = df.loc[near_mask, objective_name].mean()

        if np.isfinite(mean_obj):
            estimates[var_name] = float(mean_obj)

    return estimates


def _compute_agreement(obs_est: float, exp_est: float) -> float:
    """Compute agreement between observational and experimental estimates.

    Returns a value in [0, 1] where 1 means perfect agreement.
    Uses a relative difference metric.
    """
    if obs_est == exp_est:
        return 1.0
    denom = max(abs(obs_est), abs(exp_est), 1e-10)
    relative_diff = abs(obs_est - exp_est) / denom
    return float(max(0.0, 1.0 - relative_diff))


def _compute_aggregate_agreement(
    variables: list[ObservationalVariableReport],
) -> float | None:
    """Compute precision-weighted aggregate agreement.

    Variables with tighter observational CIs (more precise estimates) get
    higher weight.  Falls back to unweighted mean when CI data is missing.
    """
    pairs: list[tuple[float, float]] = []  # (agreement, weight)
    for v in variables:
        if v.agreement is None:
            continue
        weight = 1.0
        if v.obs_ci is not None:
            ci_width = v.obs_ci[1] - v.obs_ci[0]
            if np.isfinite(ci_width) and ci_width > 0:
                weight = 1.0 / ci_width
        pairs.append((v.agreement, weight))

    if not pairs:
        return None
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        return None
    return float(sum(a * w for a, w in pairs) / total_weight)


def _generate_recommendation(
    n_identifiable: int,
    n_ancestor_vars: int,
    obs_experimental_agreement: float | None,
    dowhy_available: bool,
    n_keep: int,
) -> str:
    """Generate a summary recommendation string."""
    if not dowhy_available:
        return "DoWhy not available; observational analysis not possible"

    if n_keep < _MIN_EXPERIMENTS:
        return f"insufficient data ({n_keep} experiments; need >= {_MIN_EXPERIMENTS})"

    if n_identifiable == 0:
        return "no identifiable effects found for causal ancestors"

    if obs_experimental_agreement is not None:
        if obs_experimental_agreement >= 0.8:
            return (
                f"strong obs-exp agreement ({obs_experimental_agreement:.0%}); "
                f"observational estimates reliable"
            )
        elif obs_experimental_agreement >= 0.5:
            return (
                f"moderate obs-exp agreement ({obs_experimental_agreement:.0%}); "
                f"use observational estimates cautiously"
            )
        else:
            return (
                f"weak obs-exp agreement ({obs_experimental_agreement:.0%}); "
                f"possible confounding or model misspecification"
            )

    return f"{n_identifiable}/{n_ancestor_vars} ancestors identifiable"
