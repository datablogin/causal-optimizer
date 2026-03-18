"""Variable signal analysis — which variables matter and which are dead weight."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from causal_optimizer.diagnostics.models import (
    VariableSignalAnalysis,
    VariableSignalClass,
    VariableSignalReport,
)

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)

#: Minimum experiments before running screening (RF needs enough data).
_MIN_EXPERIMENTS_FOR_SCREENING = 5

#: Minimum unique values to attempt effect estimation via median split.
_MIN_UNIQUE_FOR_EFFECT = 4

#: Screening importance threshold for HIGH_SIGNAL classification.
_IMPORTANCE_THRESHOLD = 0.05

#: Minimum experiments varying a variable before it counts as "tested".
_MIN_VARIED_FOR_TESTED = 3


def analyze_variable_signal(
    experiment_log: ExperimentLog,
    search_space: SearchSpace,
    objective_name: str,
    minimize: bool,
    causal_graph: CausalGraph | None = None,
) -> VariableSignalAnalysis:
    """Classify each variable as high-signal, low-signal, or untested.

    Uses screening (fANOVA) for importance scores and effect estimation
    for statistical significance.  Degrades gracefully with few experiments.
    """
    from causal_optimizer.types import ExperimentLog as ELog
    from causal_optimizer.types import ExperimentStatus, VariableType

    kept_results = [r for r in experiment_log.results if r.status == ExperimentStatus.KEEP]
    kept_log = ELog(results=kept_results)
    df = kept_log.to_dataframe() if kept_results else experiment_log.to_dataframe()
    n_kept = len(df)

    # Known causal ancestors — avoid classifying these as LOW_SIGNAL
    causal_ancestors: set[str] = set()
    if causal_graph is not None:
        causal_ancestors = causal_graph.ancestors(objective_name) & set(search_space.variable_names)

    # Run screening if enough data
    main_effects: dict[str, float] = {}
    if n_kept >= _MIN_EXPERIMENTS_FOR_SCREENING:
        try:
            from causal_optimizer.designer.screening import ScreeningDesigner

            screener = ScreeningDesigner(search_space)
            screening_result = screener.screen(kept_log, objective_name)
            main_effects = screening_result.main_effects
        except Exception:
            logger.warning("Screening failed, skipping importance scores", exc_info=True)

    reports: list[VariableSignalReport] = []

    for var in search_space.variables:
        name = var.name
        if name not in df.columns:
            reports.append(
                VariableSignalReport(
                    variable_name=name,
                    signal_class=VariableSignalClass.UNTESTED,
                )
            )
            continue

        col = df[name]
        n_unique = col.nunique()
        # Count experiments where this variable was varied (all of them if >1 unique value)
        n_varied = n_kept if n_unique > 1 else 0

        # Compute range explored
        value_range: tuple[float, float] | None = None
        is_numeric = var.variable_type in (VariableType.CONTINUOUS, VariableType.INTEGER)
        if is_numeric and n_unique > 0:
            value_range = (float(col.min()), float(col.max()))

        importance = main_effects.get(name)

        # Effect estimation via median split for numeric variables.
        # NOTE: This is a marginal (unconditional) test — it does not control
        # for confounding from other variables.  In multi-variable optimization
        # the groups may differ on other variables too, so treat p-values as
        # rough screening signals, not causal evidence.
        effect_estimate: float | None = None
        effect_significant: bool | None = None
        enough_unique = n_unique >= _MIN_UNIQUE_FOR_EFFECT
        enough_data = enough_unique and n_kept >= _MIN_EXPERIMENTS_FOR_SCREENING
        if is_numeric and enough_data:
            try:
                median_val = float(col.median())
                above = col[col > median_val]
                below = col[col <= median_val]
                if len(above) >= 2 and len(below) >= 2:
                    from scipy import stats as sp_stats

                    above_obj = df.loc[above.index, objective_name].values
                    below_obj = df.loc[below.index, objective_name].values
                    effect_val = float(np.mean(above_obj) - np.mean(below_obj))
                    _, p_val = sp_stats.ttest_ind(above_obj, below_obj)
                    effect_estimate = effect_val
                    effect_significant = float(p_val) < 0.05
            except Exception:
                logger.warning("Effect estimation failed for %s", name, exc_info=True)

        # Classify — causal ancestors get HIGH_SIGNAL to avoid premature dropping
        is_causal_ancestor = name in causal_ancestors
        if n_unique < 2 or n_kept < _MIN_EXPERIMENTS_FOR_SCREENING:
            signal_class = VariableSignalClass.UNTESTED
        elif (
            (importance is not None and importance > _IMPORTANCE_THRESHOLD)
            or effect_significant is True
            or is_causal_ancestor
        ):
            signal_class = VariableSignalClass.HIGH_SIGNAL
        else:
            signal_class = VariableSignalClass.LOW_SIGNAL

        reports.append(
            VariableSignalReport(
                variable_name=name,
                signal_class=signal_class,
                importance_score=importance,
                effect_estimate=effect_estimate,
                effect_significant=effect_significant,
                n_experiments_varied=n_varied,
                value_range_explored=value_range,
            )
        )

    high = sum(1 for r in reports if r.signal_class == VariableSignalClass.HIGH_SIGNAL)
    low = sum(1 for r in reports if r.signal_class == VariableSignalClass.LOW_SIGNAL)
    untested = sum(1 for r in reports if r.signal_class == VariableSignalClass.UNTESTED)

    return VariableSignalAnalysis(
        variables=reports,
        high_signal_count=high,
        low_signal_count=low,
        untested_count=untested,
    )
