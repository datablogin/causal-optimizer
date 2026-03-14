"""Tests for example scripts — verify each example runs without error."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from causal_optimizer.benchmarks.toy_graph import ToyGraphBiObjective
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    Constraint,
    ObjectiveSpec,
    SearchSpace,
    Variable,
    VariableType,
)

# Add examples directory to sys.path so imports work
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


def _import_example(name: str) -> Any:
    """Import an example module by name."""
    qualified_name = f"_examples_test.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None, f"Could not find {name}.py in {_EXAMPLES_DIR}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestQuickstartRuns:
    """Test that quickstart.py runs to completion."""

    def test_quickstart_runs(self) -> None:
        mod = _import_example("quickstart")
        # The main() function should run without raising
        mod.main()


class TestMultiObjectiveRuns:
    """Test that multi_objective.py runs and produces a non-empty Pareto front."""

    def test_multi_objective_runs(self) -> None:
        mod = _import_example("multi_objective")
        mod.main()

    def test_multi_objective_pareto_front_nonempty(self) -> None:
        bench = ToyGraphBiObjective(noise_scale=0.05)
        objectives = [
            ObjectiveSpec(name="objective", minimize=True),
            ObjectiveSpec(name="cost", minimize=True),
        ]

        class Runner:
            def run(self, parameters: dict[str, Any]) -> dict[str, float]:
                return bench.run(parameters)

        engine = ExperimentEngine(
            search_space=bench.search_space(),
            runner=Runner(),
            objective_name="objective",
            minimize=True,
            causal_graph=bench.causal_graph(),
            objectives=objectives,
            seed=42,
        )
        engine.run_loop(n_experiments=15)
        assert len(engine.pareto_front) > 0


class TestConstrainedRuns:
    """Test that constrained.py runs and exercises constraint logic."""

    def test_constrained_runs(self) -> None:
        mod = _import_example("constrained")
        mod.main()

    def test_constrained_has_violations(self) -> None:
        """Verify that at least one experiment is marked constraint_violated."""

        class Runner:
            def run(self, parameters: dict[str, Any]) -> dict[str, float]:
                x1, x2 = parameters["x1"], parameters["x2"]
                return {
                    "objective": float((1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2),
                    "cost": float((x1**2 + x2**2) ** 0.5),
                }

        engine = ExperimentEngine(
            search_space=SearchSpace(
                variables=[
                    Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-2, upper=2),
                    Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=-2, upper=2),
                ]
            ),
            runner=Runner(),
            objective_name="objective",
            minimize=True,
            constraints=[Constraint(metric_name="cost", upper_bound=2.0)],
            seed=42,
        )
        engine.run_loop(n_experiments=15)
        violated = [r for r in engine.log.results if r.metadata.get("constraint_violated")]
        # With [-2,2]^2 and cost=sqrt(x1^2+x2^2), corners have cost ~2.83 > 2.0.
        # LHS exploration over 15 experiments reliably hits high-cost regions.
        assert len(violated) >= 1, "Expected at least one constraint violation"
        assert len(engine.log.results) == 15


class TestAutoDiscoveryRuns:
    """Test that auto_discovery.py runs and discovers a graph."""

    def test_auto_discovery_runs(self) -> None:
        mod = _import_example("auto_discovery")
        mod.main()

    def test_auto_discovery_produces_graph(self) -> None:
        """Verify that discovery produces a non-None causal graph."""
        import numpy as np

        class Runner:
            def __init__(self) -> None:
                self._rng = np.random.default_rng(42)

            def run(self, parameters: dict[str, Any]) -> dict[str, float]:
                x1, x2 = parameters["x1"], parameters["x2"]
                _ = parameters["x3"]
                x2_effect = x2 + 0.5 * x1
                return {"objective": float((x2_effect - 2.0) ** 2 + self._rng.normal(0, 0.1))}

        engine = ExperimentEngine(
            search_space=SearchSpace(
                variables=[
                    Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-5, upper=5),
                    Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=-5, upper=5),
                    Variable(name="x3", variable_type=VariableType.CONTINUOUS, lower=-5, upper=5),
                ]
            ),
            runner=Runner(),
            objective_name="objective",
            minimize=True,
            discovery_method="correlation",
            discovery_threshold=0.3,
            seed=42,
        )
        engine.run_loop(n_experiments=20)
        assert engine.causal_graph is not None


class TestDemoAdapter:
    """Test that DemoAdapter implements the DomainAdapter protocol."""

    def test_demo_adapter_works(self) -> None:
        mod = _import_example("demo_adapter")
        adapter = mod.DemoAdapter()

        # Verify it has the required methods
        space = adapter.get_search_space()
        assert space is not None
        assert len(space.variables) > 0

        # Verify run_experiment works
        params: dict[str, Any] = {}
        for var in space.variables:
            if var.lower is not None and var.upper is not None:
                params[var.name] = (var.lower + var.upper) / 2.0
        metrics = adapter.run_experiment(params)
        assert "objective" in metrics
        assert isinstance(metrics["objective"], float)

        # DemoAdapter always provides a prior graph; verify it has edges
        graph = adapter.get_prior_graph()
        assert graph is not None
        assert hasattr(graph, "edges")
