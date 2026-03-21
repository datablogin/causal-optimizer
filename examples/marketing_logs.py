"""Offline marketing policy optimization from logged data.

Demonstrates the full pipeline using MarketingLogAdapter:
- 6-variable continuous search space (policy parameters)
- Prior causal graph with 13 directed edges
- IPS/IPW-weighted counterfactual policy evaluation
- 20 experiments through exploration -> optimization phases
- Diagnostic report with research recommendations

Usage:
    python examples/marketing_logs.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from causal_optimizer.domain_adapters.marketing_logs import MarketingLogAdapter
from causal_optimizer.engine.loop import ExperimentEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fixture_path = (
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "marketing_log_fixture.csv"
    )
    adapter = MarketingLogAdapter(data_path=str(fixture_path), seed=42)

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

    # Run 20 experiments
    print("Running 20 offline marketing policy optimization experiments...")
    print(f"Objective: maximize {adapter.get_objective_name()}")
    print(f"Variables: {adapter.get_search_space().variable_names}")
    print()

    log = engine.run_loop(n_experiments=20)

    # Report results
    best = log.best_result(adapter.get_objective_name(), minimize=adapter.get_minimize())
    if best:
        print(f"\nBest result: policy_value = {best.metrics['policy_value']:.4f}")
        print(f"  Parameters: {best.parameters}")
        print(f"  Total cost: {best.metrics.get('total_cost', 0):.2f}")
        print(f"  Treated fraction: {best.metrics.get('treated_fraction', 0):.4f}")
        print(f"  Effective sample size: {best.metrics.get('effective_sample_size', 0):.1f}")

    print(f"\nTotal experiments: {len(log.results)}")
    kept = sum(1 for r in log.results if r.status.value == "keep")
    discarded = sum(1 for r in log.results if r.status.value == "discard")
    print(f"  Kept: {kept}, Discarded: {discarded}")

    # Phase information
    phases: dict[str, int] = {}
    for r in log.results:
        phase = r.metadata.get("phase", "unknown")
        phases[phase] = phases.get(phase, 0) + 1
    print(f"  Phases: {phases}")

    # POMIS sets
    if engine.pomis_sets:
        print(f"\nPOMIS intervention sets ({len(engine.pomis_sets)}):")
        for pset in engine.pomis_sets:
            search_vars = pset & set(adapter.get_search_space().variable_names)
            print(f"  {sorted(search_vars)}")

    # Diagnostic report
    print("\n--- Research Advisor Report ---")
    report = engine.diagnose(total_budget=30)
    for rec in report.recommendations:
        print(f"  [{rec.recommendation_type.value}] {rec.summary}")
        print(f"    Rationale: {rec.rationale}")


if __name__ == "__main__":
    main()
