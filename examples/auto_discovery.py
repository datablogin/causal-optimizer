"""Auto-discovery: learn a causal graph from experiment data.

Runs the engine WITHOUT a prior causal graph, using correlation-based
auto-discovery. After exploration the engine infers which variables
matter and uses the discovered graph for optimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import SearchSpace, Variable, VariableType


class _DiscoveryRunner:
    """3-variable SCM: X1 -> X2 -> objective. X3 is noise (irrelevant).

    The engine should discover that X1 and X2 matter but X3 does not.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        _ = parameters["x3"]  # irrelevant noise — should not appear in discovered graph
        x2_effect = x2 + 0.5 * x1
        objective = float((x2_effect - 2.0) ** 2 + self._rng.normal(0, 0.1))
        return {"objective": objective}


def main() -> None:
    search_space = SearchSpace(
        variables=[
            Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="x3", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )

    # Run WITHOUT a prior graph — discovery will learn one
    engine = ExperimentEngine(
        search_space=search_space,
        runner=_DiscoveryRunner(seed=42),
        objective_name="objective",
        minimize=True,
        discovery_method="correlation",
        discovery_threshold=0.3,
        seed=42,
    )

    engine.run_loop(n_experiments=20)

    # Show the discovered graph (if any)
    graph = engine.causal_graph
    if graph is not None:
        print("\n=== Discovered causal graph ===")
        print(f"  Nodes: {graph.nodes}")
        print(f"  Directed edges: {graph.edges}")
        if graph.bidirected_edges:
            print(f"  Bidirected edges: {graph.bidirected_edges}")
        # Show which variables are ancestors of the objective
        if "objective" in graph.nodes:
            ancestors = graph.ancestors("objective")
            print(f"  Ancestors of 'objective': {ancestors}")
    else:
        print("\nNo graph was discovered (not enough data or discovery disabled).")

    # Show best result
    best = engine.log.best_result("objective", minimize=True)
    if best:
        print("\n=== Best result ===")
        print(f"  objective={best.metrics['objective']:.4f}")
        print(f"  params={best.parameters}")

    print(f"\nTotal experiments: {len(engine.log.results)}")


if __name__ == "__main__":
    main()
