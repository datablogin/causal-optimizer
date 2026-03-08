# Roadmap

## Phase 1: Foundation (Current)

What's built:
- [x] Core types (SearchSpace, Variable, CausalGraph, ExperimentLog)
- [x] Engine loop with phase transitions (exploration → optimization → exploitation)
- [x] Factorial designer (full, fractional, LHS)
- [x] Screening designer (fANOVA-style variable importance + interaction detection)
- [x] Effect estimator (difference, bootstrap, AIPW adapter)
- [x] Parameter suggestion (LHS → surrogate RF → local perturbation)
- [x] MAP-Elites archive
- [x] Off-policy predictor with observation-intervention tradeoff
- [x] Sensitivity validator (E-values, SNR)
- [x] Domain adapters (marketing, ML training)
- [x] Working quickstart example (Branin function)
- [x] 12 passing unit tests

## Phase 2: Causal Integration

Priority: connect causal reasoning to the optimization loop.

- [ ] POMIS computation from CausalGraph (graphical criterion)
- [ ] Causal graph integration into suggestion strategy (focus on ancestors)
- [ ] Joint graph learning + optimization (learn graph from early experiments,
      use it for later experiments)
- [ ] Proper do-calculus for observational estimation (beyond simple surrogate)
- [ ] Integration tests with causal-inference-marketing (PC, NOTEARS, AIPW)
- [ ] Benchmark: causal-aware vs. causal-agnostic on synthetic problems with
      known causal structure

## Phase 3: Bayesian Optimization

- [ ] Ax/BoTorch integration (currently stubbed, falls back to RF surrogate)
- [ ] Separate GP surrogates per causal mechanism (CBO paper approach)
- [ ] Acquisition function aware of causal structure
- [ ] Multi-objective optimization support (Pareto front)
- [ ] Constrained optimization (budget limits, memory limits)

## Phase 4: Advanced Evolution

- [ ] Island model (multiple MAP-Elites archives with migration)
- [ ] Descriptor auto-discovery from experiment data
- [ ] Population-based training (PBT) integration
- [ ] LLM-guided mutations for code optimization adapter

## Phase 5: Production Hardening

- [ ] Persistent experiment storage (SQLite/Postgres)
- [ ] Experiment resumption (save/restore optimizer state)
- [ ] Async experiment execution (run multiple experiments in parallel)
- [ ] Visualization dashboard (experiment history, causal graph, MAP-Elites archive)
- [ ] CLI interface for running optimization from the command line
- [ ] API service (FastAPI, like causal-inference-marketing)

## Phase 6: Domain Adapters

- [ ] Manufacturing adapter (real process variables, sensor integration)
- [ ] Drug discovery adapter (molecular descriptors, QSAR integration)
- [ ] Supply chain adapter (inventory policies, routing)
- [ ] Education adapter (adaptive learning interventions)
- [ ] Autoresearch adapter (code modification via LLM, like Karpathy's project but
      with causal reasoning)

## Open Questions

1. **How to construct causal graphs for code?** Static analysis (like cDEP for Spark)?
   LLM-assisted graph construction? Learned from ablation experiments?

2. **When does causal reasoning help most?** Need benchmarks comparing causal-aware vs.
   causal-agnostic optimization across different problem structures (linear, nonlinear,
   high-dimensional, sparse effects, dense interactions).

3. **How to handle the DAG being wrong?** Sensitivity to graph misspecification. The CBO
   paper assumes the graph is correct — what happens when it's not?

4. **Scaling POMIS computation.** For large graphs (50+ variables), POMIS enumeration may
   become expensive. Need efficient algorithms or approximations.

5. **LLM integration.** Should the optimizer use LLMs to propose experiments (like
   AlphaEvolve), or stick to mathematical optimization? Probably both — LLMs for code
   domains, mathematical optimization for numerical domains.
