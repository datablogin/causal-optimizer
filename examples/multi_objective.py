"""Multi-objective optimization: minimize cost and maximize performance.

Demonstrates how to define two objectives and inspect the Pareto front
using the ToyGraphBiObjective benchmark (X -> Z -> Y with cost).
"""

from __future__ import annotations

from typing import Any

from causal_optimizer.benchmarks.toy_graph import ToyGraphBiObjective
from causal_optimizer.engine.loop import ExperimentEngine
from causal_optimizer.types import ObjectiveSpec


class _BiObjectiveRunner:
    """Wraps ToyGraphBiObjective into the ExperimentRunner protocol."""

    def __init__(self, bench: ToyGraphBiObjective) -> None:
        self._bench = bench

    def run(self, parameters: dict[str, Any]) -> dict[str, float]:
        return self._bench.run(parameters)


def main() -> None:
    bench = ToyGraphBiObjective(noise_scale=0.05)
    search_space = bench.search_space()
    graph = bench.causal_graph()

    # Define two objectives — both minimized
    objectives = [
        ObjectiveSpec(name="objective", minimize=True),
        ObjectiveSpec(name="cost", minimize=True),
    ]

    engine = ExperimentEngine(
        search_space=search_space,
        runner=_BiObjectiveRunner(bench),
        objective_name="objective",
        minimize=True,
        causal_graph=graph,
        objectives=objectives,
        seed=42,
    )

    # Run 20 experiments
    engine.run_loop(n_experiments=20)

    # Retrieve the Pareto front
    front = engine.pareto_front
    print(f"\n=== Pareto front ({len(front)} non-dominated solutions) ===")
    for i, result in enumerate(front):
        obj = result.metrics.get("objective", float("nan"))
        cost = result.metrics.get("cost", float("nan"))
        print(f"  [{i + 1}] objective={obj:.4f}  cost={cost:.4f}  params={result.parameters}")

    # Show trade-off: sort by objective to illustrate cost increasing
    sorted_front = sorted(front, key=lambda r: r.metrics.get("objective", float("inf")))
    print("\n=== Trade-off (sorted by objective) ===")
    for r in sorted_front:
        obj = r.metrics.get("objective", float("nan"))
        cost = r.metrics.get("cost", float("nan"))
        print(f"  objective={obj:.4f}  cost={cost:.4f}")

    total = len(engine.log.results)
    print(f"\nTotal experiments: {total}")


if __name__ == "__main__":
    main()
