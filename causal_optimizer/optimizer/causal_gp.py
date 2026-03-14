"""Causal GP surrogate — separate GP per causal mechanism, composed for interventional prediction.

This is an EXPERIMENTAL implementation of the CBO (Causal Bayesian Optimization)
architecture. It requires botorch and gpytorch. Falls back gracefully if unavailable.

Simplification: The current implementation uses posterior means for propagation
through non-intervened nodes. The true CBO acquisition marginalizes over
uncertainty in all non-intervened nodes, but using posterior means is a
reasonable first approximation.
"""

from __future__ import annotations

from typing import Any

from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace


class CausalGPSurrogate:
    """Separate GP per causal mechanism, composed for interventional prediction.

    This is an EXPERIMENTAL implementation of the CBO architecture.
    It requires botorch and gpytorch. Falls back gracefully if unavailable.

    Simplification: uses posterior means for propagation through non-intervened
    nodes rather than marginalizing over uncertainty (see module docstring).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        causal_graph: CausalGraph,
        objective_name: str,
        minimize: bool = True,
        seed: int | None = None,
    ) -> None:
        raise NotImplementedError

    def fit(self, experiment_log: ExperimentLog) -> None:
        """Fit one GP per node using observed parent values."""
        raise NotImplementedError

    def predict_interventional(
        self,
        intervention: dict[str, float],
        n_samples: int = 100,
    ) -> tuple[float, float]:
        """Return (mean, std) of E[Y | do(intervention)] via graph composition."""
        raise NotImplementedError

    def suggest(self, n_candidates: int = 100) -> dict[str, Any]:
        """Suggest next intervention using Expected Improvement over do(X)."""
        raise NotImplementedError
