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
        if self.upper is not None and value > self.upper:
            return False
        return True


class SearchSpace(BaseModel):
    """Defines the space of possible interventions."""

    variables: list[Variable]

    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def dimensionality(self) -> int:
        return len(self.variables)


class ExperimentResult(BaseModel):
    """Result of a single experiment."""

    experiment_id: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    status: ExperimentStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class CausalGraph:
    """Lightweight causal graph representation wrapping networkx."""

    edges: list[tuple[str, str]]
    nodes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            node_set: set[str] = set()
            for u, v in self.edges:
                node_set.add(u)
                node_set.add(v)
            self.nodes = sorted(node_set)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        idx = {name: i for i, name in enumerate(self.nodes)}
        mat = np.zeros((n, n), dtype=int)
        for u, v in self.edges:
            mat[idx[u], idx[v]] = 1
        return mat


class ExperimentLog(BaseModel):
    """Full log of all experiments run by the optimizer."""

    results: list[ExperimentResult] = Field(default_factory=list)

    @property
    def best_result(self) -> ExperimentResult | None:
        kept = [r for r in self.results if r.status == ExperimentStatus.KEEP]
        if not kept:
            return None
        return min(kept, key=lambda r: r.metrics.get("objective", float("inf")))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"experiment_id": r.experiment_id, "status": r.status.value}
            row.update(r.parameters)
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
