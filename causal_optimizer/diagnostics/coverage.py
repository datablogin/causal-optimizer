"""Coverage analysis — what regions and causal paths were never tested?"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from causal_optimizer.diagnostics.models import CoverageAnalysis

if TYPE_CHECKING:
    from causal_optimizer.evolution.map_elites import MAPElites
    from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace

logger = logging.getLogger(__name__)


def analyze_coverage(
    experiment_log: ExperimentLog,
    search_space: SearchSpace,
    objective_name: str,
    causal_graph: CausalGraph | None = None,
    pomis_sets: list[frozenset[str]] | None = None,
    archive: MAPElites | None = None,
) -> CoverageAnalysis:
    """Analyze coverage of causal structure, POMIS sets, and search space.

    Each sub-analysis is independent and returns ``None`` when its
    required input is unavailable.
    """
    from causal_optimizer.types import ExperimentStatus

    df = experiment_log.to_dataframe()

    # Broad set: all non-crash experiments (KEEP + DISCARD)
    if "status" in df.columns:
        df_non_crash = df[df["status"] != ExperimentStatus.CRASH.value]
        df_keep = df[df["status"] == ExperimentStatus.KEEP.value]
    else:
        # Legacy data without status column — treat all as both non-crash and KEEP
        df_non_crash = df
        df_keep = df

    # Identify which variables were actually varied (have more than one unique value).
    # df_keep is a row-filtered subset of df, so it shares columns with df_non_crash.
    varied_vars: set[str] = set()
    kept_varied_vars: set[str] = set()
    for var in search_space.variables:
        if var.name in df_non_crash.columns:
            if df_non_crash[var.name].nunique() > 1:
                varied_vars.add(var.name)
            if df_keep[var.name].nunique() > 1:
                kept_varied_vars.add(var.name)

    # --- POMIS coverage ---
    pomis_total: int | None = None
    pomis_explored: int | None = None
    pomis_unexplored: list[list[str]] | None = None

    if pomis_sets is not None:
        pomis_total = len(pomis_sets)
        unexplored: list[list[str]] = []
        explored_count = 0
        for pset in pomis_sets:
            # A POMIS set is "explored" if all its members were varied
            if pset <= varied_vars:
                explored_count += 1
            else:
                unexplored.append(sorted(pset))
        pomis_explored = explored_count
        pomis_unexplored = unexplored if unexplored else None

    # --- Ancestor coverage ---
    ancestor_vars: list[str] | None = None
    ancestors_intervened: list[str] | None = None
    ancestors_never: list[str] | None = None

    if causal_graph is not None:
        all_ancestors = causal_graph.ancestors(objective_name)
        # Only consider ancestors that are in the search space
        ancestor_vars = sorted(v for v in all_ancestors if v in search_space.variable_names)
        if ancestor_vars:
            intervened = sorted(v for v in ancestor_vars if v in varied_vars)
            never = sorted(v for v in ancestor_vars if v not in varied_vars)
            ancestors_intervened = intervened
            ancestors_never = never if never else None

    # --- MAP-Elites coverage ---
    me_coverage: float | None = None
    me_filled: int | None = None
    me_total: int | None = None

    if archive is not None and archive.archive is not None:
        me_coverage = archive.coverage
        me_filled = len(archive.archive)
        if archive.descriptor_names:
            me_total = archive.n_bins ** len(archive.descriptor_names)
        else:
            me_total = None

    # --- Search space coverage ---
    from causal_optimizer.types import VariableType

    coverages: list[float] = []
    for var in search_space.variables:
        if var.variable_type not in (VariableType.CONTINUOUS, VariableType.INTEGER):
            continue
        if var.lower is None or var.upper is None:
            continue
        total_range = var.upper - var.lower
        if total_range <= 0 or var.name not in df_non_crash.columns:
            continue
        col = df_non_crash[var.name].dropna()
        if len(col) == 0:
            coverages.append(0.0)
            continue
        explored_range = float(col.max() - col.min())
        coverages.append(min(1.0, explored_range / total_range))

    ss_coverage = float(np.mean(coverages)) if coverages else None

    return CoverageAnalysis(
        pomis_sets_total=pomis_total,
        pomis_sets_explored=pomis_explored,
        pomis_sets_unexplored=pomis_unexplored,
        ancestor_variables=ancestor_vars,
        ancestors_intervened=ancestors_intervened,
        ancestors_never_intervened=ancestors_never,
        map_elites_coverage=me_coverage,
        map_elites_filled_cells=me_filled,
        map_elites_total_cells=me_total,
        search_space_coverage=ss_coverage,
        kept_varied_vars=sorted(kept_varied_vars) if len(df_keep) > 0 else None,
    )
