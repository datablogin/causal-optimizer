"""Causal Optimizer — causally-informed experiment optimization engine."""

from __future__ import annotations

from causal_optimizer.engine import ExperimentEngine, ValidationRecord
from causal_optimizer.types import (
    CausalGraph,
    Constraint,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    ObjectiveSpec,
    SearchSpace,
    Variable,
    VariableType,
)

__all__ = [
    "CausalGraph",
    "Constraint",
    "ExperimentEngine",
    "ExperimentLog",
    "ExperimentResult",
    "ExperimentStatus",
    "ObjectiveSpec",
    "SearchSpace",
    "ValidationRecord",
    "Variable",
    "VariableType",
]

__version__ = "0.1.0"
