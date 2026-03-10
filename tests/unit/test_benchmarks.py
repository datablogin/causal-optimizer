"""Tests for benchmark structural causal models."""

from __future__ import annotations

import numpy as np
import pytest

from causal_optimizer.benchmarks.complete_graph import CompleteGraphBenchmark
from causal_optimizer.benchmarks.interaction import InteractionBenchmark
from causal_optimizer.benchmarks.toy_graph import ToyGraphBenchmark
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import CausalGraph, SearchSpace, VariableType

ALL_BENCHMARKS = [ToyGraphBenchmark, CompleteGraphBenchmark, InteractionBenchmark]


class TestToyGraph:
    """Tests for the ToyGraphBenchmark."""

    def test_run_returns_valid_metrics(self) -> None:
        bench = ToyGraphBenchmark(rng=np.random.default_rng(42))
        result = bench.run({"x": 1.0, "z": 2.0})
        assert "objective" in result
        assert isinstance(result["objective"], float)

    def test_search_space_valid(self) -> None:
        space = ToyGraphBenchmark.search_space()
        assert isinstance(space, SearchSpace)
        names = space.variable_names
        assert "x" in names
        assert "z" in names
        assert len(names) == 2

    def test_causal_graph_valid(self) -> None:
        graph = ToyGraphBenchmark.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert ("x", "z") in graph.edges
        assert ("z", "objective") in graph.edges
        assert len(graph.edges) == 2

    def test_known_pomis(self) -> None:
        pomis = ToyGraphBenchmark.known_pomis()
        assert pomis == [frozenset({"z"})]

    def test_partial_intervention_only_z(self) -> None:
        """When only z is provided, x is sampled from its prior."""
        bench = ToyGraphBenchmark(noise_scale=0.0, rng=np.random.default_rng(42))
        result = bench.run({"z": 0.0})
        assert "objective" in result

    def test_partial_intervention_only_x(self) -> None:
        """When only x is provided, z follows structural equation."""
        bench = ToyGraphBenchmark(noise_scale=0.0, rng=np.random.default_rng(42))
        result = bench.run({"x": 0.0})
        # z = exp(-0) = 1.0, y = cos(1) - exp(-1/20) ~ 0.5403 - 0.9512
        assert "objective" in result

    def test_determinism_with_rng(self) -> None:
        bench1 = ToyGraphBenchmark(rng=np.random.default_rng(123))
        bench2 = ToyGraphBenchmark(rng=np.random.default_rng(123))
        r1 = bench1.run({"x": 1.0, "z": 2.0})
        r2 = bench2.run({"x": 1.0, "z": 2.0})
        assert r1["objective"] == r2["objective"]


class TestCompleteGraph:
    """Tests for the CompleteGraphBenchmark."""

    def test_run_returns_valid_metrics(self) -> None:
        bench = CompleteGraphBenchmark(rng=np.random.default_rng(42))
        params = {"f": 0.0, "a": 1.0, "b": 0.5, "c": 1.0, "d": 0.1, "e": 0.5}
        result = bench.run(params)
        assert "objective" in result
        assert isinstance(result["objective"], float)

    def test_search_space_valid(self) -> None:
        space = CompleteGraphBenchmark.search_space()
        assert isinstance(space, SearchSpace)
        names = space.variable_names
        expected = {"f", "a", "b", "c", "d", "e"}
        assert set(names) == expected
        assert len(names) == 6

    def test_causal_graph_valid(self) -> None:
        graph = CompleteGraphBenchmark.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert ("f", "a") in graph.edges
        assert ("d", "objective") in graph.edges
        assert ("e", "objective") in graph.edges
        assert graph.has_confounders
        assert ("a", "objective") in graph.bidirected_edges
        assert ("b", "objective") in graph.bidirected_edges

    def test_known_pomis(self) -> None:
        pomis = CompleteGraphBenchmark.known_pomis()
        expected = [
            frozenset({"b"}),
            frozenset({"d"}),
            frozenset({"e"}),
            frozenset({"b", "d"}),
            frozenset({"d", "e"}),
        ]
        assert len(pomis) == len(expected)
        for p in expected:
            assert p in pomis

    def test_partial_intervention_subset(self) -> None:
        """Providing only some variables works; others follow structural eqs."""
        bench = CompleteGraphBenchmark(noise_scale=0.0, rng=np.random.default_rng(42))
        result = bench.run({"b": 1.0, "d": 0.5})
        assert "objective" in result

    def test_determinism_with_rng(self) -> None:
        params = {"f": 1.0, "a": 2.0, "b": 0.0, "c": 1.0, "d": 0.1, "e": 0.5}
        bench1 = CompleteGraphBenchmark(rng=np.random.default_rng(456))
        bench2 = CompleteGraphBenchmark(rng=np.random.default_rng(456))
        r1 = bench1.run(params)
        r2 = bench2.run(params)
        assert r1["objective"] == r2["objective"]


class TestInteraction:
    """Tests for the InteractionBenchmark."""

    def test_run_returns_valid_metrics(self) -> None:
        bench = InteractionBenchmark(rng=np.random.default_rng(42))
        result = bench.run({"use_a": True, "use_b": True, "c_value": 0.5})
        assert "objective" in result
        assert isinstance(result["objective"], float)

    def test_search_space_valid(self) -> None:
        space = InteractionBenchmark.search_space()
        assert isinstance(space, SearchSpace)
        names = space.variable_names
        assert "use_a" in names
        assert "use_b" in names
        assert "c_value" in names
        assert len(names) == 3

    def test_causal_graph_valid(self) -> None:
        graph = InteractionBenchmark.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert ("use_a", "objective") in graph.edges
        assert ("use_b", "objective") in graph.edges
        assert ("c_value", "objective") in graph.edges
        assert not graph.has_confounders

    def test_known_pomis(self) -> None:
        pomis = InteractionBenchmark.known_pomis()
        assert pomis == [frozenset({"use_a", "use_b", "c_value"})]

    def test_interaction_effect(self) -> None:
        """Both together should produce a lower objective than either alone."""
        bench = InteractionBenchmark(rng=np.random.default_rng(42))
        n_samples = 200
        both_results = []
        a_only_results = []
        b_only_results = []

        for _ in range(n_samples):
            both_results.append(
                bench.run({"use_a": True, "use_b": True, "c_value": 0.5})["objective"]
            )
            a_only_results.append(
                bench.run({"use_a": True, "use_b": False, "c_value": 0.5})["objective"]
            )
            b_only_results.append(
                bench.run({"use_a": False, "use_b": True, "c_value": 0.5})["objective"]
            )

        # Both together should be lower (better for minimization) than either alone
        assert np.mean(both_results) < np.mean(a_only_results)
        assert np.mean(both_results) < np.mean(b_only_results)

    def test_partial_intervention_defaults(self) -> None:
        """Running with no parameters should use defaults."""
        bench = InteractionBenchmark(rng=np.random.default_rng(42))
        result = bench.run({})
        assert "objective" in result

    def test_determinism_with_rng(self) -> None:
        params = {"use_a": True, "use_b": False, "c_value": 0.3}
        bench1 = InteractionBenchmark(rng=np.random.default_rng(789))
        bench2 = InteractionBenchmark(rng=np.random.default_rng(789))
        r1 = bench1.run(params)
        r2 = bench2.run(params)
        assert r1["objective"] == r2["objective"]


@pytest.mark.parametrize(
    "benchmark_cls",
    ALL_BENCHMARKS,
    ids=["toy_graph", "complete_graph", "interaction"],
)
class TestBenchmarkProtocol:
    """Tests that all benchmarks satisfy the ExperimentRunner protocol."""

    def test_run_produces_objective(self, benchmark_cls: type) -> None:
        bench = benchmark_cls(rng=np.random.default_rng(42))
        space = benchmark_cls.search_space()
        # Build default parameters from search space
        params = _default_params(space)
        result = bench.run(params)
        assert "objective" in result
        assert isinstance(result["objective"], float)

    def test_search_space_returns_search_space(self, benchmark_cls: type) -> None:
        space = benchmark_cls.search_space()
        assert isinstance(space, SearchSpace)
        assert len(space.variables) > 0

    def test_causal_graph_returns_causal_graph(self, benchmark_cls: type) -> None:
        graph = benchmark_cls.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert len(graph.edges) > 0

    def test_known_pomis_returns_frozensets(self, benchmark_cls: type) -> None:
        pomis = benchmark_cls.known_pomis()
        assert isinstance(pomis, list)
        assert len(pomis) > 0
        for p in pomis:
            assert isinstance(p, frozenset)

    def test_engine_smoke_test(self, benchmark_cls: type) -> None:
        """Run 5 steps with ExperimentEngine to verify integration."""
        bench = benchmark_cls(rng=np.random.default_rng(42))
        space = benchmark_cls.search_space()
        graph = benchmark_cls.causal_graph()

        engine = ExperimentEngine(
            search_space=space,
            runner=bench,
            causal_graph=graph,
        )

        for _ in range(5):
            result = engine.step()
            assert "objective" in result.metrics
            assert isinstance(result.metrics["objective"], float)

        assert len(engine.log.results) == 5


def _default_params(space: SearchSpace) -> dict[str, object]:
    """Build default parameters from a search space for testing."""
    params: dict[str, object] = {}
    for var in space.variables:
        if var.variable_type == VariableType.BOOLEAN:
            params[var.name] = False
        elif var.variable_type == VariableType.CONTINUOUS:
            lower = var.lower if var.lower is not None else -1.0
            upper = var.upper if var.upper is not None else 1.0
            params[var.name] = (lower + upper) / 2.0
        elif var.variable_type == VariableType.INTEGER:
            lower = var.lower if var.lower is not None else 0
            upper = var.upper if var.upper is not None else 10
            params[var.name] = int((lower + upper) / 2)
        elif var.variable_type == VariableType.CATEGORICAL:
            params[var.name] = var.choices[0] if var.choices else "default"
    return params
