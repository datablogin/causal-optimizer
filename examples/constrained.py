"""Constrained optimization: keep only experiments that satisfy bounds.

Demonstrates how constraints mark experiments as DISCARD with
``constraint_violated`` metadata when metric bounds are exceeded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import (
    Constraint,
    ExperimentStatus,
    SearchSpace,
    Variable,
    VariableType,
)


class _ConstrainedRunner:
    """Simple function with a secondary 'cost' metric to constrain."""

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        # Objective: Rosenbrock-like (minimize)
        objective = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
        # Cost: increases with distance from origin
        cost = float(np.sqrt(x1**2 + x2**2))
        return {"objective": float(objective), "cost": cost}


def main() -> None:
    search_space = SearchSpace(
        variables=[
            Variable(name="x1", variable_type=VariableType.CONTINUOUS, lower=-2.0, upper=2.0),
            Variable(name="x2", variable_type=VariableType.CONTINUOUS, lower=-2.0, upper=2.0),
        ]
    )

    # Constraint: cost must stay below 2.0
    constraints = [Constraint(metric_name="cost", upper_bound=2.0)]

    engine = ExperimentEngine(
        search_space=search_space,
        runner=_ConstrainedRunner(),
        objective_name="objective",
        minimize=True,
        constraints=constraints,
        seed=42,
    )

    engine.run_loop(n_experiments=20)

    # Summarize kept vs discarded
    kept = [r for r in engine.log.results if r.status == ExperimentStatus.KEEP]
    discarded = [r for r in engine.log.results if r.status == ExperimentStatus.DISCARD]
    violated = [r for r in engine.log.results if r.metadata.get("constraint_violated")]

    print("\n=== Constraint summary ===")
    print(f"  Total experiments: {len(engine.log.results)}")
    print(f"  Kept:              {len(kept)}")
    print(f"  Discarded:         {len(discarded)}")
    print(f"  Constraint violated: {len(violated)}")

    # Show a few violated experiments
    if violated:
        print("\n=== Violated experiments (first 3) ===")
        for r in violated[:3]:
            cost = r.metrics.get("cost", float("nan"))
            print(f"  {r.experiment_id}: cost={cost:.4f} > 2.0  params={r.parameters}")

    # Show best kept result
    best = engine.log.best_result("objective", minimize=True)
    if best:
        print("\n=== Best kept result ===")
        print(f"  objective={best.metrics['objective']:.4f}  cost={best.metrics['cost']:.4f}")
        print(f"  params={best.parameters}")


if __name__ == "__main__":
    main()
