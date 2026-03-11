"""Learn causal graphs from experimental data.

Wraps causal discovery algorithms (PC, GES, NOTEARS) to infer the causal
structure over optimization variables from experiment logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from causal_optimizer.types import CausalGraph, ExperimentLog

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Default threshold for treating a correlation as a strong association
_DEFAULT_BIDIR_THRESHOLD = 0.7


class GraphLearner:
    """Learns a causal graph over optimization variables from experiment history.

    Supports multiple backends:
    - 'correlation': simple correlation-based graph (always available)
    - 'pc': PC algorithm (requires causal-inference dependency)
    - 'notears': NOTEARS continuous optimization (requires causal-inference dependency)
    """

    def __init__(self, method: str = "correlation", threshold: float = 0.3) -> None:
        self.method = method
        self.threshold = threshold

    def learn(
        self,
        experiment_log: ExperimentLog,
        min_samples: int = 10,
        objective_name: str = "objective",
    ) -> CausalGraph:
        """Learn a causal graph from experiment results.

        Args:
            experiment_log: History of past experiments.
            min_samples: Minimum number of samples required for discovery.
                Returns an empty graph when the log has fewer entries.
            objective_name: Name of the target/outcome variable in the metrics.

        Returns:
            A :class:`CausalGraph` learned from the data.  When there are
            fewer than *min_samples* results the returned graph has no edges.
        """
        if len(experiment_log.results) < min_samples:
            logger.warning(
                "Not enough samples for causal discovery (%d < %d); returning empty graph",
                len(experiment_log.results),
                min_samples,
            )
            return CausalGraph(edges=[], nodes=[])

        df = experiment_log.to_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove metadata columns
        exclude = {"experiment_id"}
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric variables for causal discovery")
            return CausalGraph(edges=[], nodes=numeric_cols)

        if self.method == "correlation":
            return self._learn_correlation(df[numeric_cols], objective_name=objective_name)
        elif self.method == "pc":
            return self._learn_pc(df[numeric_cols], objective_name=objective_name)
        elif self.method == "notears":
            return self._learn_notears(df[numeric_cols], objective_name=objective_name)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

    def _learn_correlation(
        self,
        df: pd.DataFrame,
        objective_name: str = "objective",
    ) -> CausalGraph:
        """Correlation-based graph.

        Directed edges (X → Y) are added for pairs where *one* variable is the
        known outcome (``objective_name``).  All other pairs with absolute
        correlation above ``_DEFAULT_BIDIR_THRESHOLD`` (0.7) receive a
        bidirected edge (X ↔ Y) to indicate a potential confounder.  Pairs
        with moderate correlation (above ``self.threshold`` but at most 0.7)
        are represented as directed edges in index order.
        """
        corr = df.corr().abs()
        cols = df.columns.tolist()

        directed_edges: list[tuple[str, str]] = []
        bidirected_edges: list[tuple[str, str]] = []

        # Identify parameter columns vs. outcome column
        outcome_cols = {objective_name} if objective_name in cols else set()

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                r = corr.loc[c1, c2]
                if r <= self.threshold:
                    continue

                # Decide edge type
                if c2 in outcome_cols:
                    # c1 → objective (standard directed)
                    directed_edges.append((c1, c2))
                elif c1 in outcome_cols:
                    # c2 → objective
                    directed_edges.append((c2, c1))
                elif r > _DEFAULT_BIDIR_THRESHOLD:
                    # Strong correlation between two non-outcome variables and
                    # direction is ambiguous → bidirected (confounder proxy)
                    bidirected_edges.append((c1, c2))
                else:
                    # Moderate correlation with unknown direction → directed by index order
                    directed_edges.append((c1, c2))

        return CausalGraph(edges=directed_edges, bidirected_edges=bidirected_edges, nodes=cols)

    def _learn_pc(
        self,
        df: pd.DataFrame,
        objective_name: str = "objective",
    ) -> CausalGraph:
        """PC algorithm via causal-inference library.

        The PC algorithm outputs a CPDAG (completed partially directed acyclic
        graph) which may contain undirected edges.  Undirected edges are
        converted to bidirected edges (X ↔ Y) because their direction is
        statistically unidentifiable from the data alone.
        """
        try:
            from causal_inference.discovery import PCAlgorithm  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("causal-inference not installed, falling back to correlation")
            return self._learn_correlation(df, objective_name=objective_name)

        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(df)

        directed_edges: list[tuple[str, str]] = []
        bidirected_edges: list[tuple[str, str]] = []

        for edge in result.edges:
            u, v = edge[0], edge[1]
            edge_type = edge[2] if len(edge) > 2 else "directed"
            if edge_type == "undirected":
                bidirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))

        return CausalGraph(
            edges=directed_edges,
            bidirected_edges=bidirected_edges,
            nodes=df.columns.tolist(),
        )

    def _learn_notears(
        self,
        df: pd.DataFrame,
        objective_name: str = "objective",
    ) -> CausalGraph:
        """NOTEARS continuous optimization via causal-inference library."""
        try:
            from causal_inference.discovery import NOTEARS  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("causal-inference not installed, falling back to correlation")
            return self._learn_correlation(df, objective_name=objective_name)

        notears = NOTEARS(threshold=self.threshold)
        result = notears.fit(df)
        return CausalGraph(edges=result.edges, nodes=df.columns.tolist())
