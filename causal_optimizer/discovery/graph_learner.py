"""Learn causal graphs from experimental data.

Wraps causal discovery algorithms (PC, GES, NOTEARS) to infer the causal
structure over optimization variables from experiment logs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from causal_optimizer.types import CausalGraph, ExperimentLog

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

    def learn(self, experiment_log: ExperimentLog) -> CausalGraph:
        """Learn a causal graph from experiment results."""
        df = experiment_log.to_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove metadata columns
        exclude = {"experiment_id"}
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric variables for causal discovery")
            return CausalGraph(edges=[], nodes=numeric_cols)

        if self.method == "correlation":
            return self._learn_correlation(df[numeric_cols])
        elif self.method == "pc":
            return self._learn_pc(df[numeric_cols])
        elif self.method == "notears":
            return self._learn_notears(df[numeric_cols])
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _learn_correlation(self, df: pd.DataFrame) -> CausalGraph:
        """Simple correlation-based graph (baseline, not truly causal)."""
        corr = df.corr().abs()
        edges: list[tuple[str, str]] = []
        cols = df.columns.tolist()

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i < j and corr.loc[c1, c2] > self.threshold:
                    edges.append((c1, c2))

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
