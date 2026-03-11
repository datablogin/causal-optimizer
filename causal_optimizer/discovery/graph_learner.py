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
            experiment_log: History of experiments to learn from.
            min_samples: Minimum number of results required; returns an empty
                graph if fewer results are available.
            objective_name: Name of the outcome metric column. When present,
                this column is excluded from the set of optimization variables
                so it can serve as the graph's outcome node.
        """
        if len(experiment_log.results) < min_samples:
            logger.warning(
                "Only %d results available (min_samples=%d); returning empty graph.",
                len(experiment_log.results),
                min_samples,
            )
            # Still populate nodes so callers can enumerate variables.
            df_early = experiment_log.to_dataframe()
            early_cols = [
                c
                for c in df_early.select_dtypes(include=[np.number]).columns
                if c != "experiment_id"
            ]
            return CausalGraph(edges=[], nodes=early_cols)

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
            return self._learn_pc(df[numeric_cols])
        elif self.method == "notears":
            return self._learn_notears(df[numeric_cols])
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _learn_correlation(
        self,
        df: pd.DataFrame,
        objective_name: str = "objective",
    ) -> CausalGraph:
        """Simple correlation-based graph (baseline, not truly causal).

        Pairs of non-objective variables correlated above ``self.threshold``
        become directed edges (first alphabetically → second, as a heuristic).
        Variables strongly correlated with the objective are added as direct
        causes of the objective (directed edges ``c → objective_name``).
        """
        corr = df.corr().abs()
        edges: list[tuple[str, str]] = []
        cols = df.columns.tolist()
        # Sort alphabetically so edge orientation matches the documented heuristic:
        # the variable that comes first alphabetically causes the second.
        non_obj = sorted(c for c in cols if c != objective_name)

        for i, c1 in enumerate(non_obj):
            for c2 in non_obj[i + 1 :]:
                if corr.loc[c1, c2] > self.threshold:
                    edges.append((c1, c2))

        # Variables strongly correlated with objective become its direct causes
        if objective_name in cols:
            for c in non_obj:
                if corr.loc[c, objective_name] > self.threshold:
                    edges.append((c, objective_name))

        return CausalGraph(edges=edges, nodes=cols)

    def _learn_pc(self, df: pd.DataFrame) -> CausalGraph:
        """PC algorithm via causal-inference library."""
        try:
            from causal_inference.discovery import PCAlgorithm
        except ImportError:
            logger.warning("causal-inference not installed, falling back to correlation")
            return self._learn_correlation(df)

        pc = PCAlgorithm(alpha=0.05)
        result = pc.fit(df)
        return CausalGraph(edges=result.edges, nodes=df.columns.tolist())

    def _learn_notears(self, df: pd.DataFrame) -> CausalGraph:
        """NOTEARS continuous optimization via causal-inference library."""
        try:
            from causal_inference.discovery import NOTEARS
        except ImportError:
            logger.warning("causal-inference not installed, falling back to correlation")
            return self._learn_correlation(df)

        notears = NOTEARS(threshold=self.threshold)
        result = notears.fit(df)
        return CausalGraph(edges=result.edges, nodes=df.columns.tolist())
