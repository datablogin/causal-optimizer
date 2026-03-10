"""POMIS: Possibly Optimal Minimal Intervention Sets.

Implements the algorithm from Lee & Bareinboim (NeurIPS 2018) for identifying
the minimal sets of variables worth intervening on in a causal graph to
optimize an outcome variable.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal_optimizer.types import CausalGraph


def compute_pomis(graph: CausalGraph, outcome: str) -> list[frozenset[str]]:
    """Compute all Possibly Optimal Minimal Intervention Sets for the given outcome.

    Args:
        graph: A CausalGraph with directed and optionally bidirected edges.
        outcome: The name of the outcome variable to optimize.

    Returns:
        A list of frozensets, each representing a minimal intervention set.

    Raises:
        ValueError: If outcome is not a node in the graph.
    """
    if outcome not in graph.nodes:
        raise ValueError(f"Outcome '{outcome}' is not a node in the graph.")

    # _muct handles ancestor restriction internally, which is needed for the
    # recursive _sub_pomis case where the graph isn't pre-restricted.
    muct = _muct(graph, outcome)
    ib = _interventional_border(graph, muct)

    # H = G.do(IB) restricted to MUCT ∪ IB
    g_do_ib = graph.do(ib)
    h = g_do_ib.subgraph(muct | ib)

    # If Y has no ancestors, the only POMIS is the empty set
    if len(muct) == 1 and not ib:
        return [frozenset()]

    # Reverse topological order of MUCT \ {Y} (leaves first)
    order = _topological_sort(h, muct - {outcome})
    order.reverse()

    # Filter order to only include nodes present in the subgraph
    order = [w for w in order if w in set(h.nodes)]

    result = _sub_pomis(h, outcome, order)
    # The interventional border is always a valid POMIS member — it represents
    # intervening on all parents of the confounded territory, which by definition
    # controls all confounding.
    result.add(frozenset(ib))

    return sorted(result, key=lambda s: (len(s), sorted(s)))


def _bidirected_reachable(graph: CausalGraph, node: str) -> set[str]:
    """Find all nodes reachable from node via bidirected edges."""
    # Build adjacency list for bidirected edges for O(V+E) traversal
    bi_adj: dict[str, set[str]] = {}
    for u, v in graph.bidirected_edges:
        bi_adj.setdefault(u, set()).add(v)
        bi_adj.setdefault(v, set()).add(u)

    visited = {node}
    frontier = {node}
    while frontier:
        current = frontier.pop()
        for neighbor in bi_adj.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.add(neighbor)
    return visited


def _muct(graph: CausalGraph, outcome: str) -> set[str]:
    """Compute the Minimal Unobserved Confounder's Territory.

    First restricts the graph to ancestors of the outcome (plus outcome itself).
    Then uses a frontier-based approach: start from {outcome}, expand each
    node's c-component (nodes reachable via bidirected edges), then add their
    descendants to the frontier. Continue until no new nodes are found.

    Within the ancestral subgraph G[An(Y)], we expand MUCT by adding descendants
    of c-component members. This is correct per Lee & Bareinboim (2018) — the
    ancestral restriction ensures descendants are still relevant to the outcome.
    """
    # Restrict to ancestors of Y first
    ancestor_nodes = graph.ancestors(outcome) | {outcome}
    h = graph.subgraph(ancestor_nodes)
    h_nodes = set(h.nodes)

    ts: set[str] = {outcome}
    frontier: set[str] = {outcome}

    while frontier:
        node = frontier.pop()
        # Find c-component of node in the ancestral subgraph
        cc = _bidirected_reachable(h, node) & h_nodes
        ts |= cc
        # Add descendants of the c-component within the ancestral subgraph
        desc: set[str] = set()
        for v in cc:
            desc |= h.descendants(v)
        new_nodes = (desc & h_nodes) - ts
        frontier |= new_nodes
        ts |= new_nodes

    return ts


def _interventional_border(graph: CausalGraph, muct: set[str]) -> set[str]:
    """Compute the Interventional Border: parents of MUCT that are not in MUCT."""
    parents: set[str] = set()
    for node in muct:
        parents |= graph.parents(node)
    return parents - muct


def _topological_sort(graph: CausalGraph, node_set: set[str]) -> list[str]:
    """Topological sort of a subset of nodes using Kahn's algorithm.

    Only considers directed edges between nodes in node_set.
    Returns nodes in topological order (roots first).

    Raises:
        ValueError: If the subgraph induced by node_set contains a cycle.
    """
    if not node_set:
        return []

    nodes = sorted(node_set)  # deterministic ordering
    # Build in-degree map restricted to node_set
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    adj: dict[str, list[str]] = {n: [] for n in nodes}

    for u, v in graph.edges:
        if u in node_set and v in node_set:
            in_degree[v] += 1
            adj[u].append(v)

    # Start with zero in-degree nodes
    queue = deque(sorted(n for n in nodes if in_degree[n] == 0))
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in sorted(adj[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        # Re-sort deque for deterministic ordering
        sorted_queue = sorted(queue)
        queue.clear()
        queue.extend(sorted_queue)

    if len(result) != len(node_set):
        raise ValueError("Graph contains a cycle; topological sort is not possible.")

    return result


def _sub_pomis(
    graph: CausalGraph,
    outcome: str,
    order: list[str],
    obs: set[str] | None = None,
) -> set[frozenset[str]]:
    """Recursively enumerate sub-POMIS sets.

    For each variable W_i in the (reverse topological) order, intervene on it,
    recompute MUCT and IB. If the IB doesn't overlap with already-processed
    variables, add it as a POMIS candidate and recurse.

    Args:
        graph: The current causal graph.
        outcome: The outcome variable.
        order: Variables in reverse topological order to iterate over.
        obs: Set of variables already processed (to avoid redundant sets).
    """
    if obs is None:
        obs = set()

    result: set[frozenset[str]] = set()

    for i, w_i in enumerate(order):
        g_do_wi = graph.do({w_i})
        muct_i = _muct(g_do_wi, outcome)
        ib_i = _interventional_border(g_do_wi, muct_i)

        new_obs = obs | set(order[:i])

        # Only add if IB doesn't overlap with already-observed variables
        if not (ib_i & new_obs):
            result.add(frozenset(ib_i))

            # Filter remaining order to nodes still in MUCT
            new_order = [w for w in order[i + 1 :] if w in muct_i]

            if new_order:
                # Recurse on restricted graph: do(IB) on original G, then subgraph
                g_do_ib = graph.do(ib_i)
                h_i = g_do_ib.subgraph(muct_i | ib_i)
                sub_results = _sub_pomis(h_i, outcome, new_order, new_obs)
                result |= sub_results

    return result
