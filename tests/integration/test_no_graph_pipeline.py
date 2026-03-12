"""Integration test: pipeline without prior causal graph.

Runs ExperimentEngine.run_loop(n=60) on ToyGraphBenchmark WITHOUT a prior
causal graph, but WITH graph discovery enabled. Verifies that:
- Graph discovery kicks in at the exploration -> optimization transition
- The discovered graph is non-empty (has edges)
- The engine still completes the full loop successfully
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.validator.sensitivity import RobustnessReport


@pytest.fixture()
def engine_no_graph() -> ExperimentEngine:
    """Create an ExperimentEngine with ToyGraph but no prior causal graph.

    Uses correlation-based discovery to learn the graph from data.
    """
    bench = ToyGraphBenchmark(noise_scale=0.1, rng=np.random.default_rng(42))
    engine = ExperimentEngine(
        search_space=ToyGraphBenchmark.search_space(),
        runner=bench,
        causal_graph=None,  # No prior graph
        objective_name="objective",
        minimize=True,
        seed=42,
        max_skips=3,
        discovery_method="correlation",
        discovery_threshold=0.1,  # Low threshold to ensure edges are discovered from 10 samples
    )
    return engine


def test_no_graph_discovery_produces_graph(engine_no_graph: ExperimentEngine) -> None:
    """Without a prior graph, auto-discovery should produce a non-empty graph."""
    engine_no_graph.run_loop(n_experiments=60)

    # The discovered graph should exist and have edges
    assert engine_no_graph._discovered_graph is not None, (
        "Expected auto-discovery to produce a graph"
    )
    assert len(engine_no_graph._discovered_graph.edges) > 0, (
        "Discovered graph should have at least one edge"
    )
    assert len(engine_no_graph._discovered_graph.nodes) > 0, "Discovered graph should have nodes"


def test_no_graph_active_graph_is_discovered(engine_no_graph: ExperimentEngine) -> None:
    """Without a prior, the active causal graph should be the discovered one."""
    engine_no_graph.run_loop(n_experiments=60)

    # Active graph should be set from auto-discovery
    assert engine_no_graph.causal_graph is not None, (
        "Active causal graph should be set after auto-discovery"
    )


def test_no_graph_completes_loop(engine_no_graph: ExperimentEngine) -> None:
    """The engine should complete all 60 experiments without errors."""
    log = engine_no_graph.run_loop(n_experiments=60)
    assert len(log.results) == 60


def test_no_graph_phase_transitions(engine_no_graph: ExperimentEngine) -> None:
    """Phase transitions should still happen without a prior graph."""
    log = engine_no_graph.run_loop(n_experiments=60)

    phases = [r.metadata.get("phase") for r in log.results]
    unique_phases = set(phases)
    assert "exploration" in unique_phases


def test_no_graph_robustness_report(engine_no_graph: ExperimentEngine) -> None:
    """At least one RobustnessReport should be generated even without a prior graph."""
    engine_no_graph.run_loop(n_experiments=60)

    assert hasattr(engine_no_graph, "validation_results")
    assert len(engine_no_graph.validation_results) >= 1, (
        "Expected at least one RobustnessReport from phase transitions"
    )
    for report in engine_no_graph.validation_results:
        assert isinstance(report, RobustnessReport)


def test_no_graph_reasonable_result(engine_no_graph: ExperimentEngine) -> None:
    """Even without a prior graph, the optimizer should find reasonable results."""
    log = engine_no_graph.run_loop(n_experiments=60)

    best = log.best_result("objective", minimize=True)
    assert best is not None
    # Should still find a decent result (maybe not as good as with a prior)
    assert best.metrics["objective"] < 1.0, (
        f"Expected reasonable objective, got {best.metrics['objective']}"
    )
