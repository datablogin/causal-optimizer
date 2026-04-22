"""Tests for the Sprint 37 preregistered Open Bandit prior graph.

The graph is preregistered in
``thoughts/shared/plans/26-sprint-36-recommendation.md`` (Minimal
Preregistered Graph section):

- 7 nodes: ``tau``, ``eps``, ``w_item_feature_0``, ``w_user_item_affinity``,
  ``w_item_popularity``, ``position_handling_flag``, ``policy_value``.
- 6 directed edges: every search variable points to ``policy_value``.
- 0 bidirected edges.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from causal_optimizer.domain_adapters.bandit_log import BanditLogAdapter
from causal_optimizer.types import CausalGraph


def _tiny_bandit_feedback() -> dict[str, Any]:
    """Minimal bandit_feedback for adapter construction."""
    n_rounds = 50
    n_actions = 4
    rng = np.random.default_rng(0)
    return {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "action": rng.integers(0, n_actions, size=n_rounds).astype(np.int64),
        "position": np.zeros(n_rounds, dtype=np.int64),
        "reward": rng.integers(0, 2, size=n_rounds).astype(float),
        "pscore": np.full(n_rounds, 1.0 / n_actions, dtype=float),
        "context": rng.normal(size=(n_rounds, n_actions)).astype(float),
        "action_context": rng.normal(size=(n_actions, 3)).astype(float),
    }


@pytest.fixture
def adapter() -> BanditLogAdapter:
    return BanditLogAdapter(bandit_feedback=_tiny_bandit_feedback(), seed=0)


_SEARCH_VARIABLES = (
    "tau",
    "eps",
    "w_item_feature_0",
    "w_user_item_affinity",
    "w_item_popularity",
    "position_handling_flag",
)


class TestPreregisteredPriorGraph:
    def test_get_prior_graph_returns_causal_graph(self, adapter: BanditLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert isinstance(graph, CausalGraph)

    def test_graph_has_seven_nodes(self, adapter: BanditLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        expected_nodes = set(_SEARCH_VARIABLES) | {"policy_value"}
        assert set(graph.nodes) == expected_nodes
        assert len(graph.nodes) == 7

    def test_graph_has_six_directed_edges(self, adapter: BanditLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert len(graph.edges) == 6

    def test_graph_has_zero_bidirected_edges(self, adapter: BanditLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert graph.bidirected_edges == []

    def test_every_search_variable_directly_parents_policy_value(
        self, adapter: BanditLogAdapter
    ) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        parents = graph.parents("policy_value")
        assert parents == set(_SEARCH_VARIABLES)

    def test_every_search_variable_is_an_ancestor_of_policy_value(
        self, adapter: BanditLogAdapter
    ) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        ancestors = graph.ancestors("policy_value")
        assert ancestors == set(_SEARCH_VARIABLES)

    def test_no_chain_edges_between_search_variables(self, adapter: BanditLogAdapter) -> None:
        """Search variables are independently sampled by the optimizer (knobs)."""
        graph = adapter.get_prior_graph()
        assert graph is not None
        for u, v in graph.edges:
            # Every directed edge must terminate at policy_value.
            assert v == "policy_value", f"unexpected non-terminal edge {u}->{v}"

    def test_graph_matches_search_space_variable_names(self, adapter: BanditLogAdapter) -> None:
        """The graph nodes must match the actual adapter search space exactly."""
        graph = adapter.get_prior_graph()
        space = adapter.get_search_space()
        assert graph is not None
        graph_search_nodes = set(graph.nodes) - {"policy_value"}
        assert graph_search_nodes == set(space.variable_names)

    def test_graph_objective_name_matches_adapter(self, adapter: BanditLogAdapter) -> None:
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert adapter.get_objective_name() in graph.nodes
