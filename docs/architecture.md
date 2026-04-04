# Architecture

## The Premise

Most optimization systems treat experiments as black boxes: try something, measure the result, try something else. Causal inference asks a different question: *what would happen if I intervened?*

This project bridges causal inference (propensity scores, doubly-robust estimation, structural causal models, do-calculus) with automated optimization (Bayesian optimization, evolutionary strategies, surrogate models).

## Engine and Domain Adapters

The engine orchestrates the optimization loop. Domain adapters handle the specifics of running experiments.

```
Engine (domain-agnostic)          Domain Adapter (domain-specific)
+-----------------------+         +-----------------------+
| Discover -> Screen -> |         | Marketing             |
| Estimate -> Prioritize -> ---->-| ML Training           |
| Evolve -> Predict ->  |         | Energy Load           |
| Validate              |         | (extensible)          |
+-----------------------+         +-----------------------+
```

## Seven-Stage Optimization Loop

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

## Progressive Sophistication

The system starts simple and adds complexity only when warranted:

- **Phase 1 -- Exploration** (experiments 1-10): Space-filling designs (Latin Hypercube). No model needed. Goal: cover the search space.
- **Phase 2 -- Optimization** (experiments 11-50): Surrogate models guide search. Causal graph (if available) focuses attention on ancestors of the objective. Factorial screening identifies interactions.
- **Phase 3 -- Exploitation** (experiments 50+): Local perturbation around the best known configuration. Robust estimation confirms findings. Sensitivity analysis validates that improvements are real.

## Graceful Degradation

Every module has a built-in fallback:

| Full capability | Fallback (no optional deps) |
|---|---|
| AIPW/TMLE estimation | Bootstrap confidence intervals |
| PC/NOTEARS causal discovery | Correlation-based graph |
| Ax/BoTorch Bayesian optimization | Random forest surrogate |
| pyDOE3 fractional factorial | Latin Hypercube sampling |
| Causal forests (HTE) | Random forest feature importance |

Core functionality works with only numpy, pandas, scipy, and scikit-learn. Optional extras unlock the full causal inference stack.

## The Observation-Intervention Tradeoff

Inspired by Causal Bayesian Optimization (Aglietti et al., AISTATS 2020), the system balances two ways of learning:

- **Observation**: estimate the effect of a candidate experiment from existing data using do-calculus or surrogate models. Free, but uncertain.
- **Intervention**: actually run the experiment. Expensive, but definitive.

The `predictor/` module fits a surrogate model to experiment history and estimates uncertainty. When uncertainty is low and the model is reliable, the experiment is skipped. When uncertainty is high, the experiment runs.

## Diversity Preservation via MAP-Elites

Greedy optimization converges to a single solution -- often a local optimum. MAP-Elites (Mouret & Clune, 2015) maintains an archive of diverse high-quality solutions indexed by behavioral descriptors. This prevents premature convergence and enables discovering solutions that combine strategies from different regions of the search space.

## Causal Graphs as First-Class Citizens

The system can accept, learn, or operate without causal graphs:

- **Prior knowledge**: domain adapters can supply a causal DAG based on expert knowledge
- **Data-driven**: the discovery module learns graphs from experiment logs (PC, GES, NOTEARS)
- **No graph**: the system works without one, falling back to correlation-based importance

When a causal graph is available, it enables:

- **POMIS** (Possibly-Optimal Minimum Intervention Sets): identifies which variable subsets are worth experimenting with, pruning the search space
- **Ancestor identification**: focuses optimization on variables that causally affect the objective, ignoring downstream effects
- **Counterfactual reasoning**: estimates "what would have happened if we changed X instead of Y?" without running the experiment
