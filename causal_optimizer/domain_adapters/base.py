"""Base domain adapter and example implementations.

Domain adapters translate between the optimizer's abstract interface
(SearchSpace, ExperimentRunner) and concrete domains (marketing campaigns,
ML training, manufacturing processes, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph, SearchSpace


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
