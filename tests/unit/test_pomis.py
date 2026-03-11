"""Tests for the POMIS (Possibly Optimal Minimal Intervention Sets) algorithm."""

from __future__ import annotations

import pytest

from causal_optimizer.optimizer.pomis import (
    _interventional_border,
    _muct,
    _topological_sort,
    compute_pomis,
)
from causal_optimizer.types import CausalGraph


class TestToyGraph:
    """X -> Z -> Y, no confounders. POMIS should be [{z}]."""

    def test_pomis(self) -> None:
        graph = CausalGraph(edges=[("x", "z"), ("z", "y")])
        result = compute_pomis(graph, "y")
        assert result == [frozenset({"z"})]


class TestSimpleConfounder:
    """X -> Y with X <-> Y confounder. POMIS should include {x} and empty set."""

    def test_pomis_includes_x(self) -> None:
        graph = CausalGraph(
            edges=[("x", "y")],
            bidirected_edges=[("x", "y")],
        )
        result = compute_pomis(graph, "y")
        assert frozenset({"x"}) in result

    def test_pomis_includes_empty(self) -> None:
        """With a confounder, the empty set (observe only) is a valid POMIS."""
        graph = CausalGraph(
            edges=[("x", "y")],
            bidirected_edges=[("x", "y")],
        )
        result = compute_pomis(graph, "y")
        assert frozenset() in result


class TestCompleteGraph:
    """7-variable graph with 2 confounders (CBO/Aglietti graph).

    edges: f->a, a->e, b->c, c->d, c->e, d->y, e->y
    bidirected: a<->y, b<->y

    Expected POMIS (from reference implementation by Lee & Bareinboim):
    [{a}, {e}, {f}, {a,d}, {d,e}, {d,f}, {c,d,f}] — 7 sets
    """

    @pytest.fixture()
    def graph(self) -> CausalGraph:
        return CausalGraph(
            edges=[
                ("f", "a"),
                ("a", "e"),
                ("b", "c"),
                ("c", "d"),
                ("c", "e"),
                ("d", "y"),
                ("e", "y"),
            ],
            bidirected_edges=[("a", "y"), ("b", "y")],
        )

    def test_pomis_count(self, graph: CausalGraph) -> None:
        result = compute_pomis(graph, "y")
        assert len(result) == 7

    def test_pomis_sets(self, graph: CausalGraph) -> None:
        result = compute_pomis(graph, "y")
        expected = {
            frozenset({"a"}),
            frozenset({"e"}),
            frozenset({"f"}),
            frozenset({"a", "d"}),
            frozenset({"d", "e"}),
            frozenset({"d", "f"}),
            frozenset({"c", "d", "f"}),
        }
        assert set(result) == expected


class TestNoConfoundersChain:
    """A -> B -> C -> Y, no confounders. POMIS = [{c}]."""

    def test_pomis(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c"), ("c", "y")])
        result = compute_pomis(graph, "y")
        assert result == [frozenset({"c"})]


class TestEmptyGraph:
    """Just Y, no edges. POMIS = [frozenset()]."""

    def test_pomis(self) -> None:
        graph = CausalGraph(edges=[], nodes=["y"])
        result = compute_pomis(graph, "y")
        assert result == [frozenset()]


class TestDiamondWithConfounder:
    """Diamond: X->M1, X->M2, M1->Y, M2->Y with X<->Y confounder.

    An(Y) = {x, m1, m2, y}. MUCT({y}): C-comp(y) via bidirected = {x, y}.
    De({x,y}) in An(Y) = {m1, m2, y}. So MUCT = {x, y, m1, m2} = all nodes.
    IB = Pa(MUCT) \\ MUCT = {} (x has no external parents).
    SubPOMIS then produces {m1}, {m2}, {m1,m2} via intervening on mediators.
    Together with the top-level empty IB: {}, {m1}, {m2}, {m1,m2}.
    """

    def test_pomis_exact(self) -> None:
        graph = CausalGraph(
            edges=[("x", "m1"), ("x", "m2"), ("m1", "y"), ("m2", "y")],
            bidirected_edges=[("x", "y")],
        )
        result = compute_pomis(graph, "y")
        expected = {
            frozenset(),
            frozenset({"m1"}),
            frozenset({"m2"}),
            frozenset({"m1", "m2"}),
        }
        assert set(result) == expected


class TestMUCTFixedPoint:
    """Test that MUCT correctly reaches fixed point with cascading expansions."""

    def test_muct_convergence_with_confounder(self) -> None:
        # A -> B -> Y, A <-> Y
        # After restricting to An(Y), MUCT should expand:
        # Y -> CC(Y)={A,Y} -> De({A,Y})={B} -> T={A,B,Y}
        graph = CausalGraph(
            edges=[("a", "b"), ("b", "y")],
            bidirected_edges=[("a", "y")],
        )
        muct, _g_an = _muct(graph, "y")
        assert "a" in muct
        assert "b" in muct
        assert "y" in muct

    def test_muct_no_confounders(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "y")])
        muct, _g_an = _muct(graph, "y")
        # Without confounders, MUCT is just {Y}
        assert muct == {"y"}

    def test_muct_restricts_to_ancestors(self) -> None:
        """MUCT should only include ancestors of Y (and Y itself)."""
        # A -> Y, A <-> Y, B -> A, B <-> C, C (not ancestor of Y via directed)
        # B is an ancestor of Y (B->A->Y), C is not.
        # In ancestral subgraph {A, B, Y}: bidirected A<->Y pulls A into MUCT,
        # then De(A) = {Y} (already in). B<->C is dropped since C not ancestor.
        graph = CausalGraph(
            edges=[("a", "y"), ("b", "a")],
            bidirected_edges=[("a", "y"), ("b", "c")],
            nodes=["a", "b", "c", "y"],
        )
        muct, _g_an = _muct(graph, "y")
        assert "c" not in muct
        assert "a" in muct
        assert "y" in muct


class TestValueErrorOnMissingOutcome:
    """Should raise ValueError if outcome is not in the graph."""

    def test_raises(self) -> None:
        graph = CausalGraph(edges=[("a", "b")])
        with pytest.raises(ValueError, match="not a node"):
            compute_pomis(graph, "y")


class TestSingleNodeGraph:
    """Graph with only the outcome node."""

    def test_pomis(self) -> None:
        graph = CausalGraph(edges=[], nodes=["y"])
        result = compute_pomis(graph, "y")
        assert result == [frozenset()]


class TestTopologicalSort:
    """Test topological sort correctness."""

    def test_simple_chain(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c"), ("c", "d")])
        order = _topological_sort(graph, {"a", "b", "c", "d"})
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_diamond(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        order = _topological_sort(graph, {"a", "b", "c", "d"})
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_empty_set(self) -> None:
        graph = CausalGraph(edges=[("a", "b")])
        order = _topological_sort(graph, set())
        assert order == []

    def test_subset(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c"), ("c", "d")])
        order = _topological_sort(graph, {"b", "c"})
        assert order == ["b", "c"]

    def test_cycle_raises(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c"), ("c", "a")])
        with pytest.raises(ValueError, match="cycle"):
            _topological_sort(graph, {"a", "b", "c"})


class TestInterventionalBorder:
    """Test interventional border computation."""

    def test_simple(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c")])
        muct = {"c"}
        ib = _interventional_border(graph, muct)
        assert ib == {"b"}

    def test_muct_includes_parents(self) -> None:
        graph = CausalGraph(edges=[("a", "b"), ("b", "c")])
        muct = {"b", "c"}
        ib = _interventional_border(graph, muct)
        assert ib == {"a"}


class TestIBConsistencyFullVsAncestral:
    """Verify that IB computed on the full graph equals IB on the ancestor subgraph.

    In a DAG, Pa(MUCT) is always a subset of An(Y) because MUCT is a subset of An(Y)
    and parents of ancestors are themselves ancestors. So _interventional_border yields
    the same result whether computed on the full graph or the ancestral subgraph.
    """

    def test_ib_matches_on_graph_with_non_ancestor_nodes(self) -> None:
        """Graph with nodes that are NOT ancestors of Y.

        w -> x -> y, z (isolated non-ancestor).
        IB on full graph and ancestral subgraph should both be {x}.
        """
        graph = CausalGraph(
            edges=[("w", "x"), ("x", "y")],
            nodes=["w", "x", "y", "z"],
        )
        muct_full, g_an = _muct(graph, "y")
        ib_full = _interventional_border(graph, muct_full)
        ib_ancestral = _interventional_border(g_an, muct_full)
        assert ib_full == ib_ancestral

    def test_ib_matches_with_confounders_and_extra_nodes(self) -> None:
        """Graph with confounders and extra non-ancestor nodes.

        a -> b -> y, a <-> y, c -> d (c, d not ancestors of y).
        Non-ancestors should not appear in IB regardless of which graph is used.
        """
        graph = CausalGraph(
            edges=[("a", "b"), ("b", "y"), ("c", "d")],
            bidirected_edges=[("a", "y")],
            nodes=["a", "b", "c", "d", "y"],
        )
        muct, g_an = _muct(graph, "y")
        ib_full = _interventional_border(graph, muct)
        ib_ancestral = _interventional_border(g_an, muct)
        assert ib_full == ib_ancestral
        assert "c" not in ib_full
        assert "d" not in ib_full

    def test_ancestor_subgraph_excludes_non_ancestors(self) -> None:
        """Confirm _muct returns an ancestor subgraph that excludes non-ancestors."""
        graph = CausalGraph(
            edges=[("a", "b"), ("b", "y"), ("c", "d")],
            nodes=["a", "b", "c", "d", "y"],
        )
        _muct_set, g_an = _muct(graph, "y")
        an_nodes = set(g_an.nodes)
        assert "c" not in an_nodes
        assert "d" not in an_nodes
        assert "a" in an_nodes
        assert "b" in an_nodes
        assert "y" in an_nodes
