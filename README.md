# causal-optimizer

Causally informed experiment optimization for teams trying to learn from experiments without fooling themselves.

## Project Status

This project has **characterized the boundary of causal advantage** across a 7-benchmark suite spanning 4 domain families.

What we can say confidently today:

1. **Causal guidance wins on noisy, categorical-barrier landscapes.** On the demand-response family, medium-noise (9D, p=0.002) and high-noise (15D, p=0.001) show statistically significant causal wins. Base (5D) has an improved mean (1.01) but lost significance after the Sprint 29 default change (p=0.112, was 0.045).
2. **Causal guidance wins on smooth dose-response landscapes under Ax/BoTorch.** On the dose-response benchmark (Emax curve, 6D), causal achieves near-zero regret (0.22, p=0.003, 9/10 wins). Under RF fallback, surrogate-only still wins — this row is Ax-primary.
3. **Interaction is now near-parity.** Sprint 29 identified and removed harmful causal-weighted exploration, flipping the interaction row from surrogate-only advantage (p=0.014) to near-parity (causal 1.90 vs s.o. 2.18, p=0.225). No benchmark row now has a statistically significant surrogate-only advantage under Ax.
4. **The crossover is structural, not dimensional.** The boundary depends on landscape family (categorical barriers, noise-to-signal ratio), not a single noise-dimension threshold.
5. **Null control has been clean for 11 runs across Sprints 18--29** (S26 did not re-run), confirming the optimizer does not manufacture false signal.
6. Sprint 30 produced the **first real-world causal vs surrogate-only differentiation** on ERCOT: COAST certified (p=0.008, two-sided MWU, 5 seeds) and NORTH_C trending (p=0.059). Causal still does not beat random. Results are at 5 seeds and confined to energy forecasting.
7. The benchmark stack is disciplined enough to catch false stories -- Sprint 21 rejected an attractive reranking idea that did not survive locked A/B attribution.

## What This Repo Is

`causal-optimizer` is an experiment loop that combines:

1. search-space exploration
2. surrogate or Bayesian optimization
3. causal structure and screening signals
4. off-policy skip logic
5. benchmark, provenance, and audit tooling

The goal is not just to find a good next experiment. The goal is to make claims that survive controlled re-evaluation.

For system architecture details (7-stage pipeline, graceful degradation, MAP-Elites, POMIS, observation-intervention tradeoff), see [Architecture](docs/architecture.md).

## What We Have Learned So Far

### 1. Benchmark discipline matters as much as optimizer cleverness

The project now has:

1. locked chronological train/validation/test splits for real forecasting benchmarks
2. positive controls, negative/null controls, and provenance capture
3. controlled A/B comparison infrastructure
4. a 7-benchmark suite across 4 domain families

That has been a major success. It means failed ideas are now informative rather than ambiguous.

### 2. Causal advantage has a characterized boundary

Sprint 27 completed a noise-dimension gradient study across the demand-response family:

| Variant | Dimensions | Noise Dims | B80 Causal Wins | Two-Sided p |
|---------|-----------|-----------|-----------------|-------------|
| Base | 5 | 2 | 7/10 | 0.112 |
| Medium-noise | 9 | 6 | 10/10 | 0.002 |
| High-noise | 15 | 12 | 10/10 | 0.001 |

Causal pruning provides stable performance across noise levels (B80 mean regret: 1.01, 1.19, 1.08). Surrogate-only degrades sharply (4.98, 9.61, 15.23). On the smooth dose-response landscape (6D), causal also wins under Ax/BoTorch (0.22, p=0.003) but surrogate-only wins under RF fallback.

Sprint 29 removed harmful causal-weighted exploration, improving causal on every row. The interaction benchmark (7D, 3-way interaction surface) flipped from surrogate-only advantage to near-parity. No benchmark row now has a statistically significant surrogate-only advantage under Ax.

### 3. First real-world signal on ERCOT (Sprint 30)

Sprint 30 retested the two real ERCOT forecasting benchmarks under the Sprint 29 default (`causal_exploration_weight=0.0`):

1. **COAST B80**: causal certified better than surrogate-only (MAE 104.88 vs 105.72, p=0.008, 5/5 wins)
2. **NORTH_C B80**: causal trending better than surrogate-only (MAE 132.48 vs 132.98, p=0.059, 4/5 wins)
3. `causal` still does not statistically beat `random` on either dataset
4. all strategies still converge to `ridge`; improvement is in hyperparameters, not model class
5. results at 5 seeds only; 10-seed rerun recommended for Sprint 31

This breaks the long-standing "causal identical to surrogate-only on real data" result from Sprint 16. It is the first real-world evidence of causal vs surrogate-only differentiation, but not a full causal advantage claim.

### 4. Attribution discipline rejected a false story

Sprint 20's post-merge rerun looked better. Sprint 21's locked A/B rerun then showed that the improvement was **not attributable** to balanced Ax reranking -- alignment-only was equal or better. That rejection strengthened the project's evidence standards and led to reverting balanced reranking in Sprint 22.

### 5. Stability took four targeted fixes to achieve

The base-B80 catastrophic-seed problem (seeds locking into a bad categorical value during exploitation) resisted four fixes across Sprints 22--24 before Sprint 25's exploitation-phase categorical sweep resolved it (0/10 catastrophic, mean 1.13, std 1.40). The key insight was targeting the correct optimizer phase (exploitation, not optimization).

## Current Read

If you want the shortest honest summary:

1. we have **statistically significant causal wins on 3 of 7 benchmarks** under Ax/BoTorch (medium p=0.002, high p=0.001, dose-response p=0.003), with base trending (p=0.112, mean improved from 1.13 to 1.01 but no longer significant after the Sprint 29 default change)
2. **no benchmark row has a statistically significant surrogate-only advantage** under Ax — Sprint 29 flipped the interaction row from s.o. winning to near-parity
3. backend matters: dose-response and base energy are Ax-primary (surrogate-only wins under RF fallback)
4. Sprint 30 produced the **first real-world causal vs surrogate-only differentiation** on ERCOT (COAST p=0.008, NORTH_C p=0.059) but causal does not yet beat random
5. we have enough rigor to reject optimizer stories that do not survive attribution (Sprint 21) and stability gates that took 4 targeted fixes to pass (Sprint 25)

## Key Documents

1. [Sprint 30 Reality-and-Generalization Scorecard](thoughts/shared/docs/sprint-30-reality-and-generalization-scorecard.md) -- REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC
2. [Sprint 30 ERCOT Reality Report](thoughts/shared/docs/sprint-30-ercot-reality-report.md) -- first real-world causal vs s.o. differentiation
3. [Sprint 30 Portability Brief](thoughts/shared/docs/sprint-30-general-causal-portability-brief.md) -- domain-agnostic re-anchoring
4. [Sprint 29 Optimizer-Core Scorecard](thoughts/shared/docs/sprint-29-optimizer-core-scorecard.md) -- GENERALITY IMPROVED after removing causal-weighted exploration
5. [Sprint 28 Backend Baseline Scorecard](thoughts/shared/docs/sprint-28-backend-baseline-scorecard.md) -- Ax-primary vs RF-secondary classification
6. [Sprint 27 Crossover Scorecard](thoughts/shared/docs/sprint-27-crossover-scorecard.md) -- where causal wins, ties, and loses
7. [Sprint 25 Stability Scorecard](thoughts/shared/docs/sprint-25-stability-scorecard.md) -- exploitation-phase fix that resolved B80 catastrophic seeds
8. [Sprint 26 Expansion Scorecard](thoughts/shared/docs/sprint-26-expansion-scorecard.md) -- benchmark expansion to interaction policy and dose-response
9. [Sprint 21 Attribution Scorecard](thoughts/shared/docs/sprint-21-attribution-scorecard.md) -- locked A/B reranking attribution
10. [Benchmark State](thoughts/shared/plans/07-benchmark-state.md)

## Install

### From PyPI

```bash
pip install causal-optimizer
pip install causal-optimizer[all]
```

### From source

```bash
git clone https://github.com/datablogin/causal-optimizer.git
cd causal-optimizer
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra bayesian
uv sync --extra doe
uv sync --extra causal
uv sync --extra all
```

## Quick Start

```python
from causal_optimizer.engine import ExperimentEngine
from causal_optimizer.types import SearchSpace, Variable, VariableType

# Define what you're optimizing
search_space = SearchSpace(variables=[
    Variable(name="learning_rate", variable_type=VariableType.CONTINUOUS, lower=1e-5, upper=1e-1),
    Variable(name="batch_size", variable_type=VariableType.INTEGER, lower=8, upper=512),
])

# Define how to run an experiment
class MyRunner:
    def run(self, parameters):
        return {"objective": some_metric}

# Run the optimization loop
engine = ExperimentEngine(search_space=search_space, runner=MyRunner())
log = engine.run_loop(n_experiments=50)
print(f"Best: {log.best_result().metrics}")
```

See [examples/quickstart.py](examples/quickstart.py) for a full runnable example.

## Repo Layout

```text
causal_optimizer/
  types.py           Core data models (SearchSpace, CausalGraph, ExperimentLog)
  engine/            Experiment loop orchestrator
  discovery/         Causal graph learning (correlation, PC, NOTEARS)
  designer/          DoE: full/fractional factorial, LHS, screening
  estimator/         Treatment effect estimation (difference, bootstrap, AIPW)
  optimizer/         Candidate suggestion, surrogate/Bayesian, reranking
  evolution/         MAP-Elites population diversity
  predictor/         Off-policy evaluation and skip logic
  validator/         Sensitivity analysis (E-values, SNR, robustness)
  benchmarks/        Predictive, counterfactual, null-control, provenance
  diagnostics/       Profiler, skip calibration, anytime metrics
  domain_adapters/   Marketing, ML training, energy load (extensible)
  storage/           SQLite persistence
thoughts/            Plans, prompts, reports, and research notes
scripts/             Benchmark runners, A/B harness, CLI tools
examples/            Small runnable demos
tests/               Unit and integration coverage
docs/                Architecture and design documentation
```

## License

MIT
