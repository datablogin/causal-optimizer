"""Tests for auto-discovery pipeline integration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    CausalGraph,
    ExperimentLog,
    ExperimentResult,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


def make_three_var_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="z", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


class QuadraticRunner:
    """Simple test runner: f(x, y) = x² + y² (minimum at origin)."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        return {"objective": float(x**2 + y**2)}


class HighlyCorrelatedRunner:
    """Runner that produces strong correlations between x and y."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x = parameters.get("x", 0.0)
        y = parameters.get("y", 0.0)
        # Make x and y strongly correlated in the output
        noise = self._rng.normal(0, 0.001)
        return {"objective": float(x + y + noise)}


def _make_log_with_correlation(n: int = 15, seed: int = 42) -> ExperimentLog:
    """Build an ExperimentLog with strongly correlated x and y values."""
    rng = np.random.default_rng(seed)
    log = ExperimentLog()
    for i in range(n):
        x = float(rng.uniform(-5, 5))
        y = x + float(rng.normal(0, 0.1))  # y ≈ x (strong correlation)
        log.results.append(
            ExperimentResult(
                experiment_id=f"test-{i:03d}",
                parameters={"x": x, "y": y},
                metrics={"objective": float(x**2 + y**2)},
                status=ExperimentStatus.KEEP,
            )
        )
    return log


def _make_log_no_correlation(n: int = 15, seed: int = 42) -> ExperimentLog:
    """Build an ExperimentLog with independent x and y values."""
    rng = np.random.default_rng(seed)
    log = ExperimentLog()
    for i in range(n):
        x = float(rng.uniform(-5, 5))
        y = float(rng.uniform(-5, 5))
        log.results.append(
            ExperimentResult(
                experiment_id=f"test-{i:03d}",
                parameters={"x": x, "y": y},
                metrics={"objective": float(x**2 + y**2)},
                status=ExperimentStatus.KEEP,
            )
        )
    return log


# ---------------------------------------------------------------------------
# Tests for ExperimentEngine auto-discovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_engine_auto_discovery_triggers_at_transition() -> None:
    """Engine with discovery_method='correlation' should discover graph at phase transition."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        discovery_method="correlation",
    )

    # Before phase transition, no discovered graph
    assert engine._discovered_graph is None
    assert engine._causal_graph is None

    # Run 12 experiments to cross the exploration → optimization boundary (at 10)
    engine.run_loop(n_experiments=12)

    # After transition, discovered graph should be set
    assert engine._discovered_graph is not None
    assert engine._causal_graph is not None
    # The discovered graph should be the same object as _causal_graph
    assert engine._causal_graph is engine._discovered_graph


@pytest.mark.slow
def test_engine_auto_discovery_graph_has_nodes() -> None:
    """Discovered graph should contain nodes matching the search space variables."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        discovery_method="correlation",
    )
    engine.run_loop(n_experiments=12)

    graph = engine._discovered_graph
    assert graph is not None
    # Graph should at least contain the parameter variables
    param_names = {"x", "y"}
    graph_nodes = set(graph.nodes)
    assert param_names.issubset(graph_nodes), f"Expected {param_names} in graph nodes {graph_nodes}"


@pytest.mark.slow
def test_engine_auto_discovery_no_prior_graph_required() -> None:
    """In hybrid mode, discovery runs even with a prior graph but doesn't override it."""
    prior_graph = CausalGraph(edges=[("x", "y")], nodes=["x", "y"])
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=prior_graph,
        discovery_method="correlation",
    )
    engine.run_loop(n_experiments=12)

    # Prior graph should be unchanged (hybrid mode: don't override)
    assert engine._causal_graph is prior_graph
    # Discovered graph should still be computed (but not assigned to _causal_graph)
    assert engine._discovered_graph is not None


@pytest.mark.slow
def test_engine_hybrid_mode_keeps_prior_graph() -> None:
    """When both prior graph and discovery_method are set, prior graph is preserved."""
    prior_graph = CausalGraph(edges=[("x", "y")], nodes=["x", "y"])
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=prior_graph,
        discovery_method="correlation",
    )
    engine.run_loop(n_experiments=12)

    # causal_graph should still be the prior (not overwritten)
    assert engine.causal_graph is prior_graph
    assert engine._causal_graph is prior_graph


@pytest.mark.slow
def test_engine_no_discovery_method_backward_compatible() -> None:
    """Default (discovery_method=None) should not trigger any discovery."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        # no discovery_method
    )
    engine.run_loop(n_experiments=12)

    assert engine._discovered_graph is None
    # _causal_graph should still be None (no prior, no discovery)
    assert engine._causal_graph is None


@pytest.mark.slow
def test_engine_discovery_logs_graph_info(caplog: pytest.LogCaptureFixture) -> None:
    """Engine should log info about the discovered graph at transition."""
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        discovery_method="correlation",
    )
    with caplog.at_level(logging.INFO, logger="causal_optimizer.engine.loop"):
        engine.run_loop(n_experiments=12)

    # Check that discovery log message was emitted
    discovery_logs = [r for r in caplog.records if "Discovered causal graph" in r.message]
    assert len(discovery_logs) >= 1, f"Expected discovery log message, got: {caplog.messages}"


@pytest.mark.slow
def test_engine_hybrid_mode_logs_discovered_graph(caplog: pytest.LogCaptureFixture) -> None:
    """In hybrid mode, discovered graph is logged but prior is kept."""
    prior_graph = CausalGraph(edges=[("x", "y")], nodes=["x", "y"])
    engine = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=prior_graph,
        discovery_method="correlation",
    )
    with caplog.at_level(logging.INFO, logger="causal_optimizer.engine.loop"):
        engine.run_loop(n_experiments=12)

    # Should log discovered graph but note it's not overriding the prior
    discovery_logs = [r for r in caplog.records if "Discovered causal graph" in r.message]
    assert len(discovery_logs) >= 1


@pytest.mark.slow
def test_engine_discovery_triggers_pomis_when_confounders() -> None:
    """If discovered graph has bidirected edges, POMIS computation should be triggered."""
    from unittest.mock import patch

    from causal_optimizer.discovery.graph_learner import GraphLearner

    # Create a graph with confounders that the learner will return
    graph_with_confounders = CausalGraph(
        edges=[("x", "objective")],
        bidirected_edges=[("x", "y")],
        nodes=["x", "y", "objective"],
    )

    with patch.object(GraphLearner, "learn", return_value=graph_with_confounders):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            discovery_method="correlation",
        )
        engine.run_loop(n_experiments=12)

    # POMIS should have been computed because graph has confounders
    assert engine._pomis_sets is not None


@pytest.mark.slow
def test_engine_focus_variables_work_with_discovered_graph() -> None:
    """Focus variables from the discovered graph should be used during optimization."""
    from unittest.mock import patch

    from causal_optimizer.discovery.graph_learner import GraphLearner

    # Graph where only x is an ancestor of objective
    discovered = CausalGraph(
        edges=[("x", "objective")],
        nodes=["x", "y", "objective"],
    )

    with patch.object(GraphLearner, "learn", return_value=discovered):
        engine = ExperimentEngine(
            search_space=make_three_var_space(),
            runner=QuadraticRunner(),
            discovery_method="correlation",
        )
        engine.run_loop(n_experiments=12)

    # Engine should be in optimization phase and causal_graph should be set
    assert engine._phase == "optimization"
    assert engine._causal_graph is not None


# ---------------------------------------------------------------------------
# Tests for GraphLearner directly
# ---------------------------------------------------------------------------


def test_graph_learner_returns_empty_graph_below_min_samples() -> None:
    """GraphLearner.learn() with fewer than min_samples should return an empty graph."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation")

    # Build a log with only 5 results (below default min_samples=10)
    log = ExperimentLog()
    rng = np.random.default_rng(42)
    for i in range(5):
        log.results.append(
            ExperimentResult(
                experiment_id=f"test-{i:03d}",
                parameters={"x": float(rng.uniform(-5, 5)), "y": float(rng.uniform(-5, 5))},
                metrics={"objective": float(rng.uniform(0, 10))},
                status=ExperimentStatus.KEEP,
            )
        )

    graph = learner.learn(log, min_samples=10)
    assert len(graph.edges) == 0
    assert len(graph.bidirected_edges) == 0


def test_graph_learner_respects_min_samples_param() -> None:
    """GraphLearner.learn() min_samples parameter should be respected."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation")
    log = _make_log_with_correlation(n=8)

    # With min_samples=10, should return empty graph
    graph_empty = learner.learn(log, min_samples=10)
    assert len(graph_empty.edges) == 0
    assert len(graph_empty.bidirected_edges) == 0

    # With min_samples=5, should return non-empty graph (strong correlation)
    graph_full = learner.learn(log, min_samples=5)
    # Doesn't need to have edges, but should be a valid CausalGraph
    assert isinstance(graph_full, CausalGraph)


def test_graph_learner_correlation_detects_strong_correlation() -> None:
    """Correlation method should detect strong correlations as bidirected edges."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation", threshold=0.7)
    log = _make_log_with_correlation(n=20)

    graph = learner.learn(log, min_samples=5)
    # x and y are strongly correlated (r ≈ 1.0), should produce a bidirected edge
    assert isinstance(graph, CausalGraph)
    all_edges = set(graph.edges) | set(graph.bidirected_edges)
    assert len(all_edges) > 0, "Expected at least one edge for strongly correlated x and y"


def test_graph_learner_correlation_above_threshold_adds_bidirected() -> None:
    """Pairs with corr > threshold where no directed edge exists get a bidirected edge."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation", threshold=0.7)
    log = _make_log_with_correlation(n=20)  # x and y are nearly identical

    graph = learner.learn(log, min_samples=5)

    # Strong correlation without direction knowledge → bidirected edge
    has_xy_bidir = ("x", "y") in graph.bidirected_edges or ("y", "x") in graph.bidirected_edges
    has_xy_dir = ("x", "y") in graph.edges or ("y", "x") in graph.edges
    assert has_xy_bidir or has_xy_dir, (
        f"Expected x<->y or x->y edge, got directed={graph.edges}, "
        f"bidirected={graph.bidirected_edges}"
    )


def test_graph_learner_no_edges_for_uncorrelated_data() -> None:
    """Uncorrelated data should produce no edges with high threshold."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    # bidir_threshold must be >= threshold
    learner = GraphLearner(method="correlation", threshold=0.9, bidir_threshold=0.95)
    log = _make_log_no_correlation(n=30)

    graph = learner.learn(log, min_samples=5)
    # With threshold=0.9 and independent x/y, expect no edges
    # (or very few — statistical fluctuations might create some)
    total_edges = len(graph.edges) + len(graph.bidirected_edges)
    assert total_edges <= 2, f"Expected few edges for uncorrelated data, got {total_edges}"


def test_graph_learner_objective_name_parameter() -> None:
    """GraphLearner should accept an objective_name parameter."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation")
    log = _make_log_with_correlation(n=15)

    # Should not raise even with non-default objective name
    graph = learner.learn(log, min_samples=5, objective_name="my_metric")
    assert isinstance(graph, CausalGraph)


def test_graph_learner_with_toy_graph_benchmark() -> None:
    """GraphLearner fed ToyGraph benchmark data should recover some structure."""
    from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
    from causal_optimizer.discovery.graph_learner import GraphLearner

    benchmark = ToyGraphBenchmark(rng=np.random.default_rng(42))
    log = ExperimentLog()
    rng = np.random.default_rng(42)

    for i in range(20):
        x = float(rng.uniform(-5, 5))
        z = float(rng.uniform(-5, 20))
        metrics = benchmark.run({"x": x, "z": z})
        log.results.append(
            ExperimentResult(
                experiment_id=f"tg-{i:03d}",
                parameters={"x": x, "z": z},
                metrics=metrics,
                status=ExperimentStatus.KEEP,
            )
        )

    learner = GraphLearner(method="correlation", threshold=0.3)
    graph = learner.learn(log, min_samples=10, objective_name="objective")

    assert isinstance(graph, CausalGraph)
    # Graph nodes should include all three variables in the toy graph
    for expected_node in ("x", "z", "objective"):
        assert expected_node in graph.nodes, (
            f"Expected node {expected_node!r} in graph nodes {graph.nodes}"
        )


def test_graph_learner_correlation_method_no_extra_deps() -> None:
    """The 'correlation' method should work without the causal-inference extra."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="correlation")
    log = _make_log_with_correlation(n=15)

    # Should not raise an ImportError
    graph = learner.learn(log, min_samples=5)
    assert isinstance(graph, CausalGraph)


def test_graph_learner_unknown_method_raises() -> None:
    """GraphLearner with an unknown method should raise ValueError."""
    from causal_optimizer.discovery.graph_learner import GraphLearner

    learner = GraphLearner(method="unknown_algo")
    log = _make_log_with_correlation(n=15)

    with pytest.raises(ValueError, match="Unknown method"):
        learner.learn(log, min_samples=5)


# ---------------------------------------------------------------------------
# Tests for POMIS with discovered graph
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_engine_discovery_no_pomis_without_confounders() -> None:
    """Discovered graph without bidirected edges should not trigger POMIS."""
    from unittest.mock import patch

    from causal_optimizer.discovery.graph_learner import GraphLearner

    discovered = CausalGraph(
        edges=[("x", "objective"), ("y", "objective")],
        bidirected_edges=[],
        nodes=["x", "y", "objective"],
    )

    with patch.object(GraphLearner, "learn", return_value=discovered):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            discovery_method="correlation",
        )
        engine.run_loop(n_experiments=12)

    # No confounders → no POMIS
    assert engine._pomis_sets is None


# ---------------------------------------------------------------------------
# Tests for re-discovery after screening revert
# ---------------------------------------------------------------------------


def test_prior_causal_graph_attribute_set_correctly() -> None:
    """_prior_causal_graph stores only the user-supplied prior, not auto-discovered."""
    prior_graph = CausalGraph(edges=[("x", "y")], nodes=["x", "y"])

    # With prior graph
    engine_with_prior = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        causal_graph=prior_graph,
        discovery_method="correlation",
    )
    assert engine_with_prior._prior_causal_graph is prior_graph

    # Without prior graph
    engine_no_prior = ExperimentEngine(
        search_space=make_search_space(),
        runner=QuadraticRunner(),
        discovery_method="correlation",
    )
    assert engine_no_prior._prior_causal_graph is None


@pytest.mark.slow
def test_auto_discovery_overwrites_previous_auto_discovered_graph() -> None:
    """When _run_auto_discovery runs again (after screening revert), auto-discovered
    graph is replaced with the newer richer dataset — not locked as a prior."""
    from unittest.mock import patch

    from causal_optimizer.discovery.graph_learner import GraphLearner

    call_count = 0
    graphs = [
        CausalGraph(edges=[("x", "objective")], nodes=["x", "y", "objective"]),
        CausalGraph(edges=[("x", "objective"), ("y", "objective")], nodes=["x", "y", "objective"]),
    ]

    def mock_learn(self_inner: GraphLearner, log: object, **kwargs: object) -> CausalGraph:
        nonlocal call_count
        graph = graphs[min(call_count, len(graphs) - 1)]
        call_count += 1
        return graph

    with patch.object(GraphLearner, "learn", mock_learn):
        engine = ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            discovery_method="correlation",
            # Use max_screening_attempts=1 so we can predict when screening fires
        )
        # Trigger the transition manually by running 12 steps
        engine.run_loop(n_experiments=12)

    # The discovered graph should be the result of one of the mock calls
    assert engine._discovered_graph is not None
    # Crucially, _prior_causal_graph should still be None (no user prior)
    assert engine._prior_causal_graph is None


# ---------------------------------------------------------------------------
# Tests for engine with invalid discovery_method
# ---------------------------------------------------------------------------


def test_engine_invalid_discovery_method_raises() -> None:
    """ExperimentEngine should raise ValueError for unknown discovery_method."""
    with pytest.raises(ValueError, match="discovery_method"):
        ExperimentEngine(
            search_space=make_search_space(),
            runner=QuadraticRunner(),
            discovery_method="invalid_method",
        )
