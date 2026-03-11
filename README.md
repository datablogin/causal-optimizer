# causal-optimizer

**Causally-informed experiment optimization engine.** Decides *what to try next* using causal inference, Bayesian optimization, and evolutionary strategies.

## The Premise

Most optimization systems treat experiments as black boxes: try something, measure the result, try something else. This is how Bayesian optimization, random search, and greedy hill-climbing all work. They ask *"what happened?"* but never *"why did it happen?"*

Causal inference asks a fundamentally different question: *"what would happen if I intervened?"* This distinction matters because:

1. **Correlation misleads.** A parameter may correlate with good performance because both are caused by a third factor. Optimizing that parameter directly wastes experiments.
2. **Interactions are invisible to one-at-a-time testing.** Changes A and B may each hurt individually but help together. Greedy hill-climbing will never discover this.
3. **Noise masquerades as signal.** Without statistical rigor, you keep changes that "improved" the metric by chance and discard changes that would have helped.
4. **Not all experiments are equally informative.** Some candidate experiments can be evaluated cheaply from existing data (observational estimation); others require actual execution (intervention). Knowing when to observe vs. intervene saves enormous cost.

This project bridges two fields that have developed largely in isolation:
- **Causal inference** — a mature statistical framework for reasoning about cause and effect from data (propensity scores, doubly-robust estimation, structural causal models, do-calculus)
- **Automated optimization** — systems that search for the best configuration of a function, model, or process (Bayesian optimization, evolutionary strategies, AutoML)

The result is an optimization engine that doesn't just search — it *reasons* about why experiments succeed or fail, and uses that reasoning to design better experiments.

## Architectural Principles

### 1. Separate "what to try" from "how to try it"

The engine orchestrates the optimization loop. Domain adapters handle the specifics of running experiments. This means the same causal optimization logic works whether you're tuning marketing campaigns, ML hyperparameters, manufacturing processes, or drug candidates.

```
Engine (domain-agnostic)          Domain Adapter (domain-specific)
┌─────────────────────┐           ┌─────────────────────┐
│ Discover → Screen → │           │ Marketing           │
│ Estimate → Prioritize → ───────▶│ ML Training         │
│ Evolve → Predict →  │           │ Manufacturing       │
│ Validate             │           │ Drug Discovery      │
└─────────────────────┘           └─────────────────────┘
```

### 2. Seven-stage optimization loop

Each iteration progresses through stages, any of which can be skipped if unnecessary:

| Stage | Module | Question Answered |
|-------|--------|-------------------|
| **Discover** | `discovery/` | What is the causal structure among variables? |
| **Screen** | `designer/` | Which variables and interactions matter most? |
| **Estimate** | `estimator/` | Did past changes truly help, or was it noise? |
| **Prioritize** | `optimizer/` | What should the next experiment be? |
| **Evolve** | `evolution/` | Are we maintaining diverse solutions, or stuck in a local optimum? |
| **Predict** | `predictor/` | Can we estimate this experiment's outcome without running it? |
| **Validate** | `validator/` | Is this finding robust to confounding and noise? |

### 3. Progressive sophistication

The system starts simple and adds complexity only when warranted:

- **Phase 1 — Exploration** (experiments 1–10): Space-filling designs (Latin Hypercube). No model needed. Goal: cover the search space.
- **Phase 2 — Optimization** (experiments 11–50): Surrogate models guide search. Causal graph (if available) focuses attention on ancestors of the objective. Factorial screening identifies interactions.
- **Phase 3 — Exploitation** (experiments 50+): Local perturbation around the best known configuration. Robust estimation confirms findings. Sensitivity analysis validates that improvements are real.

### 4. Graceful degradation

Every module has a built-in fallback:

| Full capability | Fallback (no optional deps) |
|---|---|
| AIPW/TMLE estimation | Bootstrap confidence intervals |
| PC/NOTEARS causal discovery | Correlation-based graph |
| Ax/BoTorch Bayesian optimization | Random forest surrogate |
| pyDOE3 fractional factorial | Latin Hypercube sampling |
| Causal forests (HTE) | Random forest feature importance |

Core functionality works with only numpy, pandas, scipy, and scikit-learn. Optional extras unlock the full causal inference stack.

### 5. The observation-intervention tradeoff

Inspired by Causal Bayesian Optimization (Aglietti et al., AISTATS 2020), the system balances two ways of learning:

- **Observation**: estimate the effect of a candidate experiment from existing data using do-calculus or surrogate models. Free, but uncertain.
- **Intervention**: actually run the experiment. Expensive, but definitive.

The `predictor/` module fits a surrogate model to experiment history and estimates uncertainty. When uncertainty is low and the model is reliable, the experiment is skipped. When uncertainty is high, the experiment runs. This can dramatically reduce the number of expensive experiments needed.

### 6. Diversity preservation via MAP-Elites

Greedy optimization converges to a single solution — often a local optimum. MAP-Elites (Mouret & Clune, 2015) maintains an archive of diverse high-quality solutions indexed by behavioral descriptors. For example, in ML training optimization, solutions might be indexed by (model_size, memory_usage), ensuring the archive contains good solutions across different size/memory tradeoffs. This prevents premature convergence and enables discovering solutions that combine strategies from different regions of the search space.

### 7. Causal graphs as first-class citizens

The system can accept, learn, or operate without causal graphs:

- **Prior knowledge**: domain adapters can supply a causal DAG based on expert knowledge
- **Data-driven**: the discovery module learns graphs from experiment logs (PC, GES, NOTEARS)
- **No graph**: the system works without one, falling back to correlation-based importance

When a causal graph is available, it enables:
- **POMIS** (Possibly-Optimal Minimum Intervention Sets): identifies which variable subsets are worth experimenting with, pruning the search space
- **Ancestor identification**: focuses optimization on variables that causally affect the objective, ignoring downstream effects
- **Counterfactual reasoning**: estimates "what would have happened if we changed X instead of Y?" without running the experiment

## Install

```bash
uv sync
```

With optional dependencies:

```bash
uv sync --extra bayesian   # Ax/BoTorch for Bayesian optimization
uv sync --extra doe         # pyDOE3 for factorial designs
uv sync --extra causal      # causal-inference-marketing library
uv sync --extra all         # everything
```

## Quick start

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
        # Your experiment logic here
        return {"objective": some_metric}

# Run the optimization loop
engine = ExperimentEngine(search_space=search_space, runner=MyRunner())
log = engine.run_loop(n_experiments=50)
print(f"Best: {log.best_result().metrics}")
```

See [examples/quickstart.py](examples/quickstart.py) for a complete working example using the Branin benchmark function.

## Project structure

```
causal_optimizer/
    types.py             # Core data models (SearchSpace, CausalGraph, ExperimentLog)
    engine/              # Experiment loop orchestrator
    discovery/           # Causal graph learning (correlation, PC, NOTEARS)
    designer/            # DoE: full/fractional factorial, LHS, screening
    estimator/           # Treatment effect estimation (difference, bootstrap, AIPW)
    optimizer/           # Parameter suggestion (exploration → surrogate/Bayesian → local)
    evolution/           # MAP-Elites population diversity
    predictor/           # Off-policy evaluation, observation-intervention tradeoff
    validator/           # Sensitivity analysis (E-values, SNR, robustness)
    domain_adapters/     # Marketing, ML training (extensible)
thoughts/                # Research notes, literature review, design rationale
examples/                # Working examples and benchmarks
tests/                   # Unit and integration tests
```

## Domain adapters

The optimizer is domain-agnostic. Plug in different domains via adapters:

- **Marketing** — campaign optimization, media mix, audience targeting
- **ML Training** — hyperparameter and architecture optimization
- **Manufacturing** — process parameter optimization (planned)
- **Drug Discovery** — compound/dosing optimization (planned)

Each adapter defines a search space, a runner, and optionally a prior causal graph based on domain knowledge.

## Relationship to causal-inference-marketing

This project optionally depends on [causal-inference-marketing](https://github.com/datablogin/causal-inference-marketing) for advanced causal estimators (AIPW, TMLE, causal forests) and discovery algorithms (PC, NOTEARS). The CI library provides the *analysis* tools; this project provides the *optimization loop* that uses those tools to design and sequence experiments.

Core functionality works without it — built-in bootstrap estimation and correlation-based graphs cover the basics.

## Research

Design rationale, literature review, and exploratory notes are in the [thoughts/](thoughts/) directory.

## License

MIT
