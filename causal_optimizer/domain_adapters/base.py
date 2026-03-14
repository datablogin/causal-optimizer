"""Base domain adapter and example implementations.

Domain adapters translate between the optimizer's abstract interface
(SearchSpace, ExperimentRunner) and concrete domains (marketing campaigns,
ML training, manufacturing processes, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, Constraint, ObjectiveSpec, SearchSpace


class DomainAdapter(ABC):
    """Base class for domain-specific experiment adapters."""

    @abstractmethod
    def get_search_space(self) -> SearchSpace:
        """Define the optimization variables for this domain."""
        ...

    @abstractmethod
    def run_experiment(self, parameters: dict[str, Any]) -> dict[str, float]:
        """Execute an experiment and return metrics."""
        ...

    def get_prior_graph(self) -> CausalGraph | None:
        """Return a prior causal graph if domain knowledge is available."""
        return None

    def get_descriptor_names(self) -> list[str]:
        """Return behavioral descriptor names for MAP-Elites diversity."""
        return []

    def get_objective_name(self) -> str:
        """Return the primary objective metric name.

        Default is ``"objective"``.  Override to match domain-specific metric names.
        """
        return "objective"

    def get_minimize(self) -> bool:
        """Return whether the objective should be minimized.

        Default is ``True``.  Override to ``False`` for maximization problems.
        """
        return True

    def get_strategy(self) -> str:
        """Return the optimization strategy to use.

        Default is ``"bayesian"``.  Valid values: ``"bayesian"``, ``"causal_gp"``.
        """
        return "bayesian"

    def get_objectives(self) -> list[ObjectiveSpec] | None:
        """Return multi-objective specifications, or None for single-objective.

        Default is ``None`` (single-objective mode).
        """
        return None

    def get_constraints(self) -> list[Constraint] | None:
        """Return optimization constraints, or None for unconstrained.

        Default is ``None`` (unconstrained).
        """
        return None

    def get_discovery_method(self) -> str | None:
        """Return the causal discovery method to use, or None to disable.

        Default is ``None`` (disabled).  Valid values: ``"correlation"``,
        ``"pc"``, ``"notears"``.
        """
        return None
