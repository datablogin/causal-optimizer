"""Tests for example scripts — verify each example runs without error."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

# Add examples directory to sys.path so imports work
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


def _import_example(name: str) -> Any:
    """Import an example module by name."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES_DIR / f"{name}.py")
    assert spec is not None, f"Could not find {name}.py in {_EXAMPLES_DIR}"
    assert spec.loader is not None
    qualified_name = f"_examples_test.{name}"
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
    """Test that multi_objective.py runs to completion."""

    def test_multi_objective_runs(self) -> None:
        mod = _import_example("multi_objective")
        mod.main()


class TestConstrainedRuns:
    """Test that constrained.py runs to completion."""

    def test_constrained_runs(self) -> None:
        mod = _import_example("constrained")
        mod.main()


class TestAutoDiscoveryRuns:
    """Test that auto_discovery.py runs to completion."""

    def test_auto_discovery_runs(self) -> None:
        mod = _import_example("auto_discovery")
        mod.main()


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

        # Verify optional methods
        graph = adapter.get_prior_graph()
        # DemoAdapter always provides a prior graph; verify it has edges
        assert graph is not None
        assert hasattr(graph, "edges")
