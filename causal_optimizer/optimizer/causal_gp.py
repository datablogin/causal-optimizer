"""Causal GP surrogate — separate GP per causal mechanism, composed for interventional prediction.

This is an EXPERIMENTAL implementation of the CBO (Causal Bayesian Optimization)
architecture.  It requires botorch and gpytorch.  Falls back gracefully if unavailable.

**Simplification**: The current implementation uses posterior means for propagation
through non-intervened nodes.  The true CBO acquisition marginalizes over
uncertainty in all non-intervened nodes, but using posterior means is a
reasonable first approximation.

**CBO Architecture**:
1. Topological order: get nodes in topological sort from ``CausalGraph``
2. Fit phase: for each node X_i, fit ``SingleTaskGP(X=parent_values, y=node_values)``
3. Predict phase (interventional): intervened variables use fixed values;
   non-intervened variables predict from their GP conditioned on parent values
4. Acquisition: Expected Improvement using the objective node's posterior
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
from scipy.stats import norm

from causal_optimizer.types import CausalGraph, ExperimentLog, SearchSpace, VariableType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

try:
    import gpytorch  # noqa: F401
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


class CausalGPSurrogate:
    """Separate GP per causal mechanism, composed for interventional prediction.

    This is an EXPERIMENTAL implementation of the CBO architecture.
    It requires botorch and gpytorch.  Falls back gracefully if unavailable.

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
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "botorch and gpytorch are required for CausalGPSurrogate. "
                "Install them with: uv sync --extra bayesian"
            )

        if objective_name not in causal_graph.nodes:
            raise ValueError(
                f"objective_name={objective_name!r} is not a node in the causal graph. "
                f"Available nodes: {sorted(causal_graph.nodes)}"
            )

        self._search_space = search_space
        self._causal_graph = causal_graph
        self._objective_name = objective_name
        self._minimize = minimize
        self._rng = np.random.default_rng(seed)

        # Compute topological order of all graph nodes
        self._topo_order = self._topological_sort(causal_graph, set(causal_graph.nodes))

        # Per-node GP models, populated by fit()
        self._gp_models: dict[str, Any] = {}
        # Per-node training statistics for normalization
        self._node_stats: dict[str, tuple[float, float]] = {}  # (mean, std)

        # Best observed objective for EI computation
        self._best_objective: float | None = None

    def fit(self, experiment_log: ExperimentLog) -> None:
        """Fit one GP per node using observed parent values."""
        df = experiment_log.to_dataframe()
        if len(df) < 2:
            logger.warning("Not enough data to fit GPs (need >= 2 rows)")
            return

        # Track best observed objective
        if self._objective_name in df.columns:
            obj_vals = df[self._objective_name].values
            if self._minimize:
                self._best_objective = float(np.min(obj_vals))
            else:
                self._best_objective = float(np.max(obj_vals))

        # Fit a GP for each node that has parents
        for node in self._topo_order:
            parents = sorted(self._causal_graph.parents(node))
            if not parents:
                # Root node — no GP needed, but store observed stats for
                # predict_interventional fallback when node is not intervened on.
                if node in df.columns:
                    root_data = df[node].values.astype(np.float64)
                    self._node_stats[node] = (
                        float(np.mean(root_data)),
                        max(float(np.std(root_data)), 1e-10),
                    )
                continue

            # Check all parents and node are in the dataframe
            required_cols = parents + [node]
            if not all(c in df.columns for c in required_cols):
                # Node or parent not in data; skip
                continue

            # Skip non-numeric columns (CausalGP requires continuous data)
            non_numeric = [c for c in required_cols if not np.issubdtype(df[c].dtype, np.number)]
            if non_numeric:
                logger.warning(
                    "Skipping GP for node %s: non-numeric columns %s",
                    node,
                    non_numeric,
                )
                continue

            parent_data = df[parents].values.astype(np.float64)
            node_data = df[node].values.astype(np.float64)

            # Skip GP fitting if constant, but store the constant value
            if np.std(node_data) < 1e-10:
                self._node_stats[node] = (float(np.mean(node_data)), 0.0)
                continue

            # Convert to torch tensors
            train_x = torch.tensor(parent_data, dtype=torch.float64)
            train_y = torch.tensor(node_data, dtype=torch.float64).unsqueeze(-1)

            # Store stats for this node
            self._node_stats[node] = (float(np.mean(node_data)), float(np.std(node_data)))

            # Fit SingleTaskGP
            try:
                model = SingleTaskGP(train_x, train_y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Simple training loop
                model.train()
                mll.train()
                fit_gpytorch_mll(mll)

                model.eval()
                self._gp_models[node] = model
            except Exception as exc:
                logger.warning("Failed to fit GP for node %s: %s", node, exc)

    def predict_interventional(
        self,
        intervention: dict[str, float],
    ) -> tuple[float, float]:
        """Return (mean, std) of E[Y | do(intervention)] via graph composition.

        For intervened variables: use the fixed intervention value (do-operator).
        For non-intervened variables: predict using their GP conditioned on
        already-computed parent values (working in topological order).
        Final prediction: evaluate objective GP at the propagated values.
        """
        # Current values for each node, populated as we traverse topological order
        node_values: dict[str, float] = {}
        node_stds: dict[str, float] = {}

        for node in self._topo_order:
            if node in intervention:
                # Intervened: use fixed value
                node_values[node] = intervention[node]
                node_stds[node] = 0.0
                continue

            parents = sorted(self._causal_graph.parents(node))

            if not parents or node not in self._gp_models:
                # Root node or no GP fitted — fall back to observed statistics
                if node not in self._node_stats:
                    logger.warning(
                        "No observed statistics for node %s; using default (0.0, 1.0)",
                        node,
                    )
                mean, std = self._node_stats.get(node, (0.0, 1.0))
                node_values[node] = mean
                node_stds[node] = std
                continue

            # Predict using GP: parent values → node value
            parent_vals = [node_values.get(p, 0.0) for p in parents]
            x_test = torch.tensor([parent_vals], dtype=torch.float64)

            model = self._gp_models[node]
            with torch.no_grad():
                posterior = model.posterior(x_test)
                pred_mean = float(posterior.mean.squeeze().item())
                pred_std = float(posterior.variance.squeeze().sqrt().item())

            node_values[node] = pred_mean
            node_stds[node] = pred_std

        # Final prediction is the objective node's value
        obj_mean = node_values.get(self._objective_name, 0.0)
        obj_std = node_stds.get(self._objective_name, 1.0)

        return (obj_mean, obj_std)

    def suggest(self, n_candidates: int = 100) -> dict[str, Any]:
        """Suggest next intervention using Expected Improvement over do(X).

        Generates random candidates in the search space, evaluates the
        interventional prediction for each, and selects the one with
        the highest Expected Improvement.
        """
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")

        if not self._gp_models:
            logger.warning(
                "No GPs were fitted — suggest() will return a random candidate "
                "(call fit() with sufficient data first)"
            )

        # Generate random candidates
        candidates: list[dict[str, float]] = []
        for _ in range(n_candidates):
            params: dict[str, float] = {}
            for var in self._search_space.variables:
                if var.variable_type == VariableType.CONTINUOUS:
                    lo = var.lower if var.lower is not None else 0.0
                    hi = var.upper if var.upper is not None else 1.0
                    params[var.name] = float(self._rng.uniform(lo, hi))
                elif var.variable_type == VariableType.INTEGER:
                    lo = int(var.lower) if var.lower is not None else 0
                    hi = int(var.upper) if var.upper is not None else 10
                    params[var.name] = float(self._rng.integers(lo, hi + 1))
                elif var.variable_type == VariableType.BOOLEAN:
                    params[var.name] = float(self._rng.choice([0.0, 1.0]))
                elif var.variable_type == VariableType.CATEGORICAL and var.choices:
                    # Use index for categorical
                    params[var.name] = float(self._rng.integers(0, len(var.choices)))
            candidates.append(params)

        # Evaluate EI for each candidate
        best_ei = float("-inf")
        best_candidate = candidates[0]

        for candidate in candidates:
            mean, std = self.predict_interventional(candidate)
            ei = self._expected_improvement(mean, std)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        # Convert back to proper types
        result: dict[str, Any] = {}
        for var in self._search_space.variables:
            if var.name in best_candidate:
                val = best_candidate[var.name]
                if var.variable_type == VariableType.INTEGER:
                    result[var.name] = int(val)
                elif var.variable_type == VariableType.BOOLEAN:
                    result[var.name] = bool(val > 0.5)
                elif var.variable_type == VariableType.CATEGORICAL and var.choices:
                    idx = int(val) % len(var.choices)
                    result[var.name] = var.choices[idx]
                else:
                    result[var.name] = float(val)

        return result

    def _expected_improvement(self, mean: float, std: float) -> float:
        """Compute Expected Improvement."""
        if self._best_objective is None:
            # No best observed yet — just use predicted value
            return -mean if self._minimize else mean

        if std < 1e-10:
            # No uncertainty — EI is 0 unless mean is better
            if self._minimize:
                return max(0.0, self._best_objective - mean)
            else:
                return max(0.0, mean - self._best_objective)

        if self._minimize:
            z = (self._best_objective - mean) / std
        else:
            z = (mean - self._best_objective) / std

        ei: float = std * (z * norm.cdf(z) + norm.pdf(z))
        return ei

    @staticmethod
    def _topological_sort(graph: CausalGraph, node_set: set[str]) -> list[str]:
        """Topological sort of nodes using Kahn's algorithm.

        Returns nodes in topological order (roots first).
        """
        if not node_set:
            return []

        nodes = sorted(node_set)
        in_degree: dict[str, int] = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = {n: [] for n in nodes}

        for u, v in graph.edges:
            if u in node_set and v in node_set:
                in_degree[v] += 1
                adj[u].append(v)

        queue = deque(sorted(n for n in nodes if in_degree[n] == 0))
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(adj[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            sorted_queue = sorted(queue)
            queue.clear()
            queue.extend(sorted_queue)

        if len(result) != len(node_set):
            raise ValueError("Graph contains a cycle; topological sort is not possible.")

        return result
