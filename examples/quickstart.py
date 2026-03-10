"""Quickstart: optimize a simple function using the causal optimizer.

Demonstrates the full loop on a synthetic Branin function —
a standard benchmark with multiple local minima.
"""

from typing import Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import CausalGraph, SearchSpace, Variable, VariableType


class BraninRunner:
    """Branin function — standard optimization benchmark with 3 global minima."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x1 = parameters["x1"]
        x2 = parameters["x2"]

        # Branin function
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        result = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

        return {
            "objective": float(result),
            "x1_magnitude": abs(x1),
        }


def main() -> None:
    # Define search space
    search_space = SearchSpace(
        variables=[
            Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=10.0),
            Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=0.0, upper=15.0),
        ]
    )

    # Optional: provide prior causal knowledge
    prior_graph = CausalGraph(
        edges=[
            ("x1", "objective"),
            ("x2", "objective"),
        ]
    )

    # Create and run the optimizer
    engine = ExperimentEngine(
        search_space=search_space,
        runner=BraninRunner(),
        objective_name="objective",
        minimize=True,
        causal_graph=prior_graph,
    )

    # Run 30 experiments
    import logging

    logging.basicConfig(level=logging.INFO)

    log = engine.run_loop(n_experiments=30)

    # Report results
    best = log.best_result
    if best:
        print(f"\nBest result: objective={best.metrics['objective']:.6f}")
        print(f"Parameters: {best.parameters}")
        print("(Branin global minimum ≈ 0.397887)")

    print(f"\nTotal experiments: {len(log.results)}")
    kept = sum(1 for r in log.results if r.status.value == "keep")
    print(f"Kept: {kept}, Discarded: {len(log.results) - kept}")


if __name__ == "__main__":
    main()
