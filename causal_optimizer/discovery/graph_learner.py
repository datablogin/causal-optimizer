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

# Default threshold above which a non-outcome pair gets a bidirected edge
# instead of a directed-by-index-order edge.  Must be ≥ `threshold` to have
# any effect.  Exposed as a default so callers can override per instance.
_DEFAULT_BIDIR_THRESHOLD = 0.7


class GraphLearner:
    """Learns a causal graph over optimization variables from experiment history.

    Supports multiple backends:
    - 'correlation': simple correlation-based graph (always available)
    - 'pc': PC algorithm (requires causal-inference dependency)
    - 'notears': NOTEARS continuous optimization (requires causal-inference dependency)
    """

    def __init__(
        self,
        method: str = "correlation",
        threshold: float = 0.3,
        bidir_threshold: float = _DEFAULT_BIDIR_THRESHOLD,
    ) -> None:
        """Create a GraphLearner.

        Args:
            method: Discovery algorithm (``"correlation"``, ``"pc"``, or ``"notears"``).
            threshold: Minimum absolute correlation to include an edge at all.
            bidir_threshold: For the correlation method, pairs of non-outcome
                variables whose |r| exceeds *this* threshold receive a bidirected
                edge (X ↔ Y) rather than a directed edge.  Must satisfy
                ``bidir_threshold >= threshold`` to have any effect.  Defaults to
                ``0.7`` (``_DEFAULT_BIDIR_THRESHOLD``).
        """
        if bidir_threshold < threshold:
            raise ValueError(
                f"bidir_threshold ({bidir_threshold!r}) must be >= threshold ({threshold!r}); "
                "when bidir_threshold < threshold, no pairs can ever exceed bidir_threshold "
                "(since they must first exceed threshold), making bidirected edges unreachable"
            )
        self.method = method
        self.threshold = threshold
        self.bidir_threshold = bidir_threshold

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
        correlation above ``self.bidir_threshold`` receive a bidirected edge
        (X ↔ Y) as a heuristic for potential confounding — this is *not* a
        rigorous causal identification; it only signals that direction is
        ambiguous and a confounder may exist.  Pairs with moderate correlation
        (above ``self.threshold`` but at most ``self.bidir_threshold``) are
        represented as directed edges in index order.

        .. warning::
            Bidirected edges produced here are heuristic proxies, not
            statistically-identified confounders.  Treat downstream POMIS
            computations that rely on them as approximate guidance only.
        """
        corr = df.corr().abs()
        cols = df.columns.tolist()

        directed_edges: list[tuple[str, str]] = []
        bidirected_edges: list[tuple[str, str]] = []

        # Identify parameter columns vs. outcome column
        if objective_name not in cols:
            logger.warning(
                "objective_name %r not found in data columns %r; "
                "all variable pairs will be treated as non-outcome",
                objective_name,
                cols,
            )
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
                elif r > self.bidir_threshold:
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
        graph) which may contain undirected edges.  Undirected edges indicate
        that the direction is statistically unidentifiable from the data alone;
        they are represented as directed edges in index order (u → v) as a
        conservative default.  They are *not* converted to bidirected edges,
        because an undirected CPDAG edge means ``u → v`` or ``v → u`` — it does
        **not** imply a hidden common cause (which is what a bidirected edge
        X ↔ Y represents in an ADMG).
        """
        try:
            from causal_inference.discovery import PCAlgorithm
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
                # Direction unidentifiable from CPDAG — use index order as a
                # conservative default rather than implying confounding.
                logger.debug(
                    "PC: undirected edge (%r, %r) — orientation ambiguous, using index order",
                    u,
                    v,
                )
                directed_edges.append((u, v))
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
            from causal_inference.discovery import NOTEARS
        except ImportError:
            logger.warning("causal-inference not installed, falling back to correlation")
            return self._learn_correlation(df, objective_name=objective_name)

        notears = NOTEARS(threshold=self.threshold)
        result = notears.fit(df)
        return CausalGraph(edges=result.edges, nodes=df.columns.tolist())
