"""ML hyperparameter tuning with the causal optimizer.

Demonstrates the full pipeline using MLTrainingAdapter:
- 9-variable search space (continuous, integer, categorical)
- Prior causal graph modeling training dynamics
- 50 experiments through exploration -> optimization -> exploitation phases
- Diagnostic report with research recommendations

Usage:
    python examples/ml_hyperparameter_tuning.py
"""

from __future__ import annotations

import logging

from causal_optimizer.domain_adapters.ml_training import MLTrainingAdapter
from causal_optimizer.engine.loop import ExperimentEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Create the adapter — it provides search space, causal graph, and simulator
    adapter = MLTrainingAdapter(seed=42)

    # Build the engine
    engine = ExperimentEngine(
        search_space=adapter.get_search_space(),
        runner=adapter,
        objective_name=adapter.get_objective_name(),
        minimize=adapter.get_minimize(),
        causal_graph=adapter.get_prior_graph(),
        descriptor_names=adapter.get_descriptor_names(),
        epsilon_mode=False,
        seed=42,
    )

    # Run 50 experiments
    print("Running 50 ML hyperparameter tuning experiments...")
    print(f"Objective: minimize {adapter.get_objective_name()}")
    print(f"Variables: {adapter.get_search_space().variable_names}")
    print()

    log = engine.run_loop(n_experiments=50)

    # Report results
    best = log.best_result(adapter.get_objective_name(), minimize=adapter.get_minimize())
    if best:
        print(f"\nBest result: val_loss = {best.metrics['val_loss']:.6f}")
        print("  Parameters:")
        for k, v in sorted(best.parameters.items()):
            print(f"    {k}: {v}")
        print(f"  Memory usage: {best.metrics.get('memory_usage', 0):.2f} MB")
        print(f"  Model capacity: {best.metrics.get('model_capacity', 0):.2f}")

    print(f"\nTotal experiments: {len(log.results)}")
    kept = sum(1 for r in log.results if r.status.value == "keep")
    discarded = sum(1 for r in log.results if r.status.value == "discard")
    print(f"  Kept: {kept}, Discarded: {discarded}")

    # Phase information
    phases = {}
    for r in log.results:
        phase = r.metadata.get("phase", "unknown")
        phases[phase] = phases.get(phase, 0) + 1
    print(f"  Phases: {phases}")

    # POMIS sets
    if engine.pomis_sets:
        print(f"\nPOMIS intervention sets ({len(engine.pomis_sets)}):")
        for pset in engine.pomis_sets:
            search_vars = pset & set(adapter.get_search_space().variable_names)
            if search_vars:
                print(f"  {sorted(search_vars)}")

    # Diagnostic report
    print("\n--- Research Advisor Report ---")
    report = engine.diagnose(total_budget=50)
    for rec in report.recommendations:
        print(f"  [{rec.recommendation_type.value}] {rec.summary}")
        print(f"    Rationale: {rec.rationale}")


if __name__ == "__main__":
    main()
