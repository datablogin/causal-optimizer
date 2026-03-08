# Causal Bayesian Optimization: Deep Dive

## Core Paper: Aglietti et al. (AISTATS 2020)

"Causal Bayesian Optimization" — the foundational paper for this project's optimization strategy.

### Problem Setup

Standard Bayesian optimization models y = f(x) as a black box and uses a Gaussian Process
surrogate with acquisition functions (Expected Improvement, etc.) to select the next x to evaluate.

CBO adds a **causal graph** over the variables. Instead of searching the full space of possible
interventions, it exploits the graph structure to:

1. **Prune the search space** via POMIS
2. **Balance observation vs. intervention** via do-calculus
3. **Maintain separate surrogate models** per causal mechanism

### POMIS: Possibly-Optimal Minimum Intervention Sets

The key theoretical contribution. Given a causal DAG:

- An **intervention set** is a subset of variables you can set (do-operator)
- A **Minimum Intervention Set (MIS)** is the smallest subset that achieves a distinct
  interventional distribution on the target
- **POMIS** = the MIS sets that *could* be optimal under some parameterization of the SCM

POMIS drastically prunes the search space. Instead of searching over all 2^|X| possible
intervention subsets, you only need to consider the POMIS members.

**Example**: with 10 variables, naive search considers 1024 subsets. If POMIS identifies 5
relevant subsets, you've reduced the search by 200x.

### Observation-Intervention Tradeoff

CBO introduces a *new* tradeoff beyond exploration-exploitation:

- **Observe**: use do-calculus + existing observational data to *estimate* the interventional
  effect. Free, but subject to model uncertainty and data coverage.
- **Intervene**: actually run the experiment. Expensive, but gives ground truth.

An epsilon parameter controls this based on observational data coverage. When you have enough
data to be confident in the observational estimate, skip the experiment. When the region is
poorly covered, intervene.

This is directly relevant to our `predictor/off_policy.py` module.

### Separate GP Surrogates

Instead of one GP for the full space, CBO maintains separate GPs for each element of the
exploration set. Each models a distinct causal mechanism. This is more statistically efficient
because each GP only needs to model a lower-dimensional relationship.

### Concrete Application: Apache Spark Tuning

The paper applied CBO to optimizing Apache Spark configurations:
- Causal graph generated via static code analysis (cDEP tool)
- Variables: memory allocation, parallelism, serialization settings, etc.
- CBO converged faster than vanilla BO by exploiting the graph structure

This is directly analogous to our ML training and code optimization use cases.

## Extensions

### Dynamic CBO (Aglietti et al., NeurIPS 2021)
- Handles **time-varying** causal relationships
- Integrates CBO with dynamic Bayesian networks
- Relevant for: marketing campaigns where channel effects change seasonally,
  ML training where optimal hyperparameters shift as the model trains

### Constrained CBO (cCBO, ICML 2023)
- Adds **constraints** on intervention outcomes
- e.g., "maximize conversion rate but keep cost below $X"
- Relevant for: budget-constrained marketing, VRAM-constrained ML training

### Adversarial CBO (Sussex et al., ICLR 2024)
- Models **uncontrollable external factors** that affect the outcome
- Robust optimization under environmental uncertainty
- Relevant for: manufacturing with varying raw material quality,
  marketing with competitor actions

### CBO with Unknown Graphs (2025)
- Removes the requirement for a **pre-specified causal graph**
- Learns the graph jointly with optimization
- Directly relevant to our approach: start with no graph, learn it from experiments

### Contextual CBO (2023)
- Adds **context variables** that affect optimal intervention
- e.g., "the best learning rate depends on the dataset size"
- Enables conditional optimization strategies

## Implementation Strategy for causal-optimizer

### Phase 1 (Current)
- Use surrogate models (random forest) as a fallback for Ax/BoTorch
- Causal graph guides which variables to focus on (ancestor identification)
- Simple observation-intervention tradeoff via uncertainty thresholding

### Phase 2 (Planned)
- Implement POMIS computation from the causal graph
- Add GP-based surrogates per causal mechanism (separate from the full-space surrogate)
- Implement proper do-calculus for observational estimation
- Add the epsilon-based observation-intervention controller

### Phase 3 (Future)
- Dynamic CBO for time-varying settings
- Constrained CBO for budget/resource constraints
- Joint graph learning + optimization (CBO with unknown graphs)

## Key References

- Aglietti et al. (2020). "Causal Bayesian Optimization." AISTATS.
  https://proceedings.mlr.press/v108/aglietti20a.html
- Code: https://github.com/VirgiAgl/CausalBayesianOptimization
- Aglietti et al. (2021). "Dynamic Causal Bayesian Optimization." NeurIPS.
- Sussex et al. (2024). "Adversarial Causal Bayesian Optimization." ICLR.
- CBO Unknown Graphs (2025). https://arxiv.org/html/2503.19554v1
