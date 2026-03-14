"""Core types and data models for the optimization engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    """Status of a completed experiment."""

    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"
    PENDING = "pending"


class VariableType(str, Enum):
    """Type of an optimization variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Variable(BaseModel):
    """A single variable in the optimization space."""

    name: str
    variable_type: VariableType
    lower: float | None = None
    upper: float | None = None
    choices: list[Any] | None = None

    def validate_value(self, value: Any) -> bool:
        if self.variable_type == VariableType.CATEGORICAL:
            return value in (self.choices or [])
        if self.variable_type == VariableType.BOOLEAN:
            return isinstance(value, bool)
        if self.lower is not None and value < self.lower:
            return False
        return not (self.upper is not None and value > self.upper)


class SearchSpace(BaseModel):
    """Defines the space of possible interventions."""

    variables: list[Variable]

    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def dimensionality(self) -> int:
        return len(self.variables)


class ObjectiveSpec(BaseModel):
    """Specification for a single optimization objective."""

    name: str
    minimize: bool = True
    weight: float = 1.0  # for scalarization fallback


class Constraint(BaseModel):
    """Constraint on a metric value.

    A result violates this constraint if:
    - ``upper_bound`` is set and the metric exceeds it, or
    - ``lower_bound`` is set and the metric is below it.
    """

    metric_name: str
    upper_bound: float | None = None
    lower_bound: float | None = None

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        """Return True if the constraint is satisfied by the given metrics."""
        value = metrics.get(self.metric_name)
        if value is None:
            return False
        if self.upper_bound is not None and value > self.upper_bound:
            return False
        return not (self.lower_bound is not None and value < self.lower_bound)


class ExperimentResult(BaseModel):
    """Result of a single experiment."""

    experiment_id: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    status: ExperimentStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParetoResult(BaseModel):
    """Non-dominated Pareto front from multi-objective optimization."""

    front: list[ExperimentResult]

    @staticmethod
    def dominated_by(
        candidate: ExperimentResult,
        other: ExperimentResult,
        objectives: list[ObjectiveSpec],
    ) -> bool:
        """Return True if *other* dominates *candidate* on all objectives.

        *other* dominates *candidate* iff *other* is at least as good on every
        objective and strictly better on at least one.
        """
        all_at_least_as_good = True
        strictly_better = False
        for obj in objectives:
            # Missing metrics default to the *worst* value for the direction
            missing_default = float("inf") if obj.minimize else float("-inf")
            c_val = candidate.metrics.get(obj.name, missing_default)
            o_val = other.metrics.get(obj.name, missing_default)
            if obj.minimize:
                if o_val > c_val:
                    all_at_least_as_good = False
                if o_val < c_val:
                    strictly_better = True
            else:
                if o_val < c_val:
                    all_at_least_as_good = False
                if o_val > c_val:
                    strictly_better = True
        return all_at_least_as_good and strictly_better


@dataclass
class CausalGraph:
    """Causal graph with directed and bidirected edges.

    Directed edges (X → Y) represent direct causal effects.
    Bidirected edges (X ↔ Y) represent unobserved confounders —
    a hidden variable U that causes both X and Y. Bidirected edges
    are required for POMIS computation, which identifies the minimal
    intervention sets worth experimenting with.
    """

    edges: list[tuple[str, str]]
    bidirected_edges: list[tuple[str, str]] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            node_set: set[str] = set()
            for u, v in self.edges:
                node_set.add(u)
                node_set.add(v)
            for u, v in self.bidirected_edges:
                node_set.add(u)
                node_set.add(v)
            self.nodes = sorted(node_set)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Adjacency matrix for directed edges only."""
        n = len(self.nodes)
        idx = {name: i for i, name in enumerate(self.nodes)}
        mat = np.zeros((n, n), dtype=int)
        for u, v in self.edges:
            mat[idx[u], idx[v]] = 1
        return mat

    @property
    def has_confounders(self) -> bool:
        return len(self.bidirected_edges) > 0

    def c_components(self) -> list[set[str]]:
        """Connected components via bidirected edges (confounder clusters).

        Two nodes are in the same c-component if they are connected by a path
        of bidirected edges. Each c-component represents a set of variables
        that share unobserved common causes.
        """
        if not self.bidirected_edges:
            return [{n} for n in self.nodes]

        # Union-Find
        parent: dict[str, str] = {n: n for n in self.nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for u, v in self.bidirected_edges:
            union(u, v)

        components: dict[str, set[str]] = {}
        for n in self.nodes:
            root = find(n)
            if root not in components:
                components[root] = set()
            components[root].add(n)

        return list(components.values())

    def ancestors(self, target: str) -> set[str]:
        """All ancestors of target (nodes with a directed path to target)."""
        result: set[str] = set()
        frontier = {target}
        while frontier:
            node = frontier.pop()
            for u, v in self.edges:
                if v == node and u not in result:
                    result.add(u)
                    frontier.add(u)
        return result

    def descendants(self, target: str) -> set[str]:
        """All descendants of target (nodes reachable via directed path from target)."""
        result: set[str] = set()
        frontier = {target}
        while frontier:
            node = frontier.pop()
            for u, v in self.edges:
                if u == node and v not in result:
                    result.add(v)
                    frontier.add(v)
        return result

    def parents(self, target: str) -> set[str]:
        """Direct parents of target."""
        return {u for u, v in self.edges if v == target}

    def children(self, target: str) -> set[str]:
        """Direct children of target."""
        return {v for u, v in self.edges if u == target}

    def do(self, interventions: set[str]) -> CausalGraph:
        """Graph surgery: remove all incoming directed edges to intervened variables."""
        new_edges = [(u, v) for u, v in self.edges if v not in interventions]
        new_bidirected = [
            (u, v)
            for u, v in self.bidirected_edges
            if u not in interventions and v not in interventions
        ]
        return CausalGraph(
            edges=new_edges,
            bidirected_edges=new_bidirected,
            nodes=list(self.nodes),
        )

    def subgraph(self, node_set: set[str]) -> CausalGraph:
        """Induced subgraph on the given node set."""
        new_edges = [(u, v) for u, v in self.edges if u in node_set and v in node_set]
        new_bidirected = [
            (u, v) for u, v in self.bidirected_edges if u in node_set and v in node_set
        ]
        return CausalGraph(
            edges=new_edges,
            bidirected_edges=new_bidirected,
            nodes=sorted(node_set & set(self.nodes)),
        )


class ExperimentLog(BaseModel):
    """Full log of all experiments run by the optimizer."""

    results: list[ExperimentResult] = Field(default_factory=list)

    def best_result(
        self,
        objective_name: str = "objective",
        minimize: bool = True,
    ) -> ExperimentResult | None:
        """Return the best kept result by the given objective.

        Args:
            objective_name: Metric key to compare results on.
            minimize: If ``True`` return the result with the smallest value,
                otherwise return the result with the largest value.
        """
        kept = [r for r in self.results if r.status == ExperimentStatus.KEEP]
        if not kept:
            return None
        if minimize:
            return min(kept, key=lambda r: r.metrics.get(objective_name, float("inf")))
        return max(kept, key=lambda r: r.metrics.get(objective_name, float("-inf")))

    def pareto_front(self, objectives: list[ObjectiveSpec]) -> list[ExperimentResult]:
        """Return the non-dominated subset of KEEP results.

        A result A dominates B if A is at least as good on all objectives
        and strictly better on at least one.
        """
        kept = [r for r in self.results if r.status == ExperimentStatus.KEEP]
        if not kept:
            return []

        front: list[ExperimentResult] = []
        for candidate in kept:
            dominated = False
            for other in kept:
                if other is candidate:
                    continue
                if ParetoResult.dominated_by(candidate, other, objectives):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        return front

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row: dict[str, object] = {
                "experiment_id": r.experiment_id,
                "status": r.status.value,
            }
            row.update(r.parameters)
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
