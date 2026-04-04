# causal-optimizer

Causally informed experiment optimization for teams trying to learn from experiments without fooling themselves.

## Project Status

This project is now in a stronger place as a **research system** than as a proven **real-world optimizer win**.

What we can say confidently today:

1. The benchmark stack is real and trustworthy enough to catch false stories.
2. The project can exploit signal on controlled positive benchmarks.
3. The project can avoid claiming wins on null-signal controls.
4. The project has **not yet** shown a reliable causal advantage on the real ERCOT forecasting tasks.
5. Sprint 21's locked A/B rerun found that Sprint 20's apparent improvement was **not attributable** to balanced Ax reranking.

Current recommendation:

1. revert or disable balanced reranking back to alignment-only
2. confirm that alignment-only still delivers the stronger positive-control behavior
3. only then decide whether to expand to new benchmark families or resume optimizer-core tuning

## What This Repo Is

`causal-optimizer` is an experiment loop that combines:

1. search-space exploration
2. surrogate or Bayesian optimization
3. causal structure and screening signals
4. off-policy skip logic
5. benchmark, provenance, and audit tooling

The goal is not just to find a good next experiment. The goal is to make claims that survive controlled re-evaluation.

## What We Have Learned So Far

### 1. Benchmark discipline matters as much as optimizer cleverness

The project now has:

1. locked chronological train/validation/test splits for real forecasting benchmarks
2. positive controls
3. negative/null controls
4. provenance capture
5. controlled A/B comparison infrastructure

That has been a major success. It means failed ideas are now informative rather than ambiguous.

### 2. Real predictive wins are still unproven

On the two real ERCOT forecasting benchmarks:

1. `random` was marginally better than the engine-based strategies
2. `causal` and `surrogate_only` were effectively identical
3. all strategies converged to `ridge`
4. the results were very stable across seeds

Those results matter. They tell us the original causal differentiation was too weak to matter on the real task.

### 3. Positive and negative controls changed the project

Sprint 18 established a much better evidence standard:

1. the repaired counterfactual benchmark became a valid positive control
2. the null-signal benchmark passed
3. the time-series profiler correctly surfaced calendar / timezone / DST issues

That is the point where this stopped being just “interesting optimizer ideas” and became a real research program.

### 4. Sprint 19 produced the first meaningful causal progress

The first convincing gains came from the soft-causal optimizer changes:

1. causal improved on the base counterfactual
2. causal improved on the high-noise counterfactual
3. the null control stayed clean

This is the strongest evidence so far that the optimizer can use causal structure in benchmark settings where it should matter.

### 5. Sprint 20 looked better, but Sprint 21 forced attribution

Sprint 20's post-merge rerun looked materially better. Sprint 21 then asked the harder question: **what actually caused the improvement?**

The locked A/B rerun found:

1. alignment-only reranking matched or beat balanced reranking everywhere that mattered
2. on base B80, alignment-only was much better
3. on high-noise, the two were mostly indistinguishable
4. the null control was identical on both sides

That is why Sprint 21's final verdict is: **NOT ATTRIBUTED**.

This was a good outcome for the project even though it was a negative outcome for the balanced reranking idea. The system is getting better at rejecting attractive but unsupported explanations.

## Current Read

If you want the shortest honest summary:

1. we have a much better **automated research harness** than we did at the start
2. we have some evidence of causal advantage on controlled benchmarks
3. we do **not** yet have strong evidence of causal advantage on the real forecasting tasks
4. we now have enough rigor to reject optimizer stories that do not survive attribution

## Key Documents

1. [Sprint 18 Discovery Trust Scorecard](thoughts/shared/docs/sprint-18-discovery-trust-scorecard.md)
2. [Sprint 19 Differentiation Scorecard](thoughts/shared/docs/sprint-19-differentiation-scorecard.md)
3. [Sprint 20 Post-Ax Controlled Rerun Report](thoughts/shared/docs/sprint-20-post-ax-rerun-report.md)
4. [Sprint 21 Controlled A/B Rerun Report](thoughts/shared/docs/sprint-21-controlled-ab-rerun-report.md)
5. [Sprint 21 Attribution Scorecard](thoughts/shared/docs/sprint-21-attribution-scorecard.md)
6. [Benchmark State](thoughts/shared/plans/07-benchmark-state.md)

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

search_space = SearchSpace(variables=[
    Variable(name="learning_rate", variable_type=VariableType.CONTINUOUS, lower=1e-5, upper=1e-1),
    Variable(name="batch_size", variable_type=VariableType.INTEGER, lower=8, upper=512),
])

class MyRunner:
    def run(self, parameters):
        return {"objective": some_metric}

engine = ExperimentEngine(search_space=search_space, runner=MyRunner())
log = engine.run_loop(n_experiments=50)
print(log.best_result().metrics)
```

See [examples/quickstart.py](examples/quickstart.py) for a full runnable example.

## Repo Layout

```text
causal_optimizer/
  engine/          Experiment loop orchestrator
  optimizer/       Candidate suggestion and reranking
  predictor/       Off-policy estimation and skip logic
  validator/       Robustness and sensitivity checks
  benchmarks/      Predictive, counterfactual, and null-control benchmark support
  diagnostics/     Profiler and calibration diagnostics
thoughts/          Plans, prompts, reports, and research notes
examples/          Small runnable demos
tests/             Unit and integration coverage
```

## License

MIT
