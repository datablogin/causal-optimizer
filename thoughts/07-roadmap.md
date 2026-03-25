# Roadmap

Updated: 2026-03-25 (after Sprint 14)

## Phase 1: Foundation — COMPLETE

- [x] Core types (SearchSpace, Variable, CausalGraph, ExperimentLog)
- [x] Engine loop with phase transitions (exploration → optimization → exploitation)
- [x] Factorial designer (full, fractional, LHS)
- [x] Screening designer (fANOVA-style variable importance + interaction detection)
- [x] Effect estimator (difference, bootstrap, AIPW adapter)
- [x] Parameter suggestion (LHS → surrogate RF → local perturbation)
- [x] MAP-Elites archive
- [x] Off-policy predictor with observation-intervention tradeoff
- [x] Sensitivity validator (E-values, SNR)
- [x] Domain adapters (marketing simulator, ML training simulator)
- [x] Working quickstart example (Branin function)

Sprints 1–2.

## Phase 2: Causal Integration — COMPLETE

- [x] POMIS computation from CausalGraph (graphical criterion)
- [x] Causal graph integration into suggestion strategy (focus on ancestors)
- [x] Joint graph learning + optimization (GraphLearner: correlation, PC, NOTEARS)
- [x] Proper do-calculus for observational estimation (DoWhy integration)
- [x] Integration tests with causal-inference methods (PC, NOTEARS, AIPW)
- [x] Benchmark: causal-aware vs. causal-agnostic on synthetic problems (ToyGraph, CompleteGraph, HighDimensionalSparse, Interaction benchmarks)
- [x] Sensitivity validation wiring (E-values, SNR)
- [x] Bidirected edge semantics for confounders, c-components

Sprints 2–5. Key results: causal beats random on ToyGraph; POMIS prunes to 5 members vs 64 naive on CompleteGraph; screening detects interactions that greedy misses.

## Phase 3: Bayesian Optimization — COMPLETE

- [x] Ax/BoTorch integration with graceful fallback to RF surrogate
- [x] Causal GP surrogates per causal mechanism (CBO paper approach)
- [x] POMIS-aware acquisition function
- [x] Multi-objective optimization support (Pareto front)
- [x] Constrained optimization (budget limits, memory limits)

Sprints 6–7.

## Phase 4: Advanced Evolution — PARTIAL

- [x] MAP-Elites archive with diversity sampling in exploitation phase
- [ ] Island model (multiple MAP-Elites archives with migration)
- [ ] Descriptor auto-discovery from experiment data
- [ ] Population-based training (PBT) integration
- [ ] LLM-guided mutations for code optimization adapter

Sprint 1 (MAP-Elites). Remaining items are future work.

## Phase 5: Production Hardening — MOSTLY COMPLETE

- [x] SQLite persistent experiment storage
- [x] Experiment resumption (save/restore optimizer state)
- [x] CLI interface (`uv run causal-optimizer`)
- [x] Research advisor diagnostics module (`diagnose()`)
- [x] PyPI publishing pipeline (LICENSE, CHANGELOG, publish workflow)
- [ ] Async experiment execution (run multiple experiments in parallel)
- [ ] Visualization dashboard (experiment history, causal graph, MAP-Elites archive)
- [ ] API service (FastAPI)

Sprints 7–8 (persistence, CLI), Sprint 9 (diagnostics), Sprint 11 (PyPI).

## Phase 6: Domain Adapters — IN PROGRESS

### Simulators (complete)

- [x] Marketing simulator — saturating effects, confounders, interior optimum
- [x] ML training simulator — divergence/overfit/underfit failure modes

### Real-data adapters (complete)

- [x] EnergyLoadAdapter — day-ahead electricity load forecasting (Ridge/RF/GBM, blocked time split, 7 search vars)
- [x] MarketingLogAdapter — offline marketing policy evaluation with IPS/IPW weighting (simplex channel parameterization, 6 search vars)
- [x] Adapter hardening — Parquet support, timestamp validation, warning metrics (Sprint 12b)
- [x] Marketing adapter hardening — input validation, zero-support guard, perf optimization (Sprint 13)

### Predictive-model benchmarks (complete)

- [x] Energy predictive benchmark harness — locked 3-way chronological split, `ValidationEnergyRunner`, held-out test evaluation
- [x] Benchmark runner CLI — strategy dispatch (random/surrogate_only/causal), JSON artifact output, summary tables
- [x] Benchmark tests — smoke tests, reproducibility regression tests, benchmark documentation

Sprints 10 (simulators), 12–12b (real-data adapters), 13 (marketing hardening), 14 (predictive benchmark).

### Not started

- [ ] Manufacturing adapter (real process variables, sensor integration)
- [ ] Drug discovery adapter (molecular descriptors, QSAR integration)
- [ ] Supply chain adapter (inventory policies, routing)
- [ ] Education adapter (adaptive learning interventions)
- [ ] Autoresearch adapter (code modification via LLM)

## Phase 7: Benchmark Validation — NEXT

Priority: run the energy benchmark on a real dataset and evaluate results.

- [ ] Run energy predictive benchmark on a real hourly load dataset (1+ year)
- [ ] Produce benchmark iteration report using the standardized template
- [ ] Evaluate success criteria: does causal/surrogate beat random on held-out test MAE?
- [ ] Investigate failure signals: val-test gap, seed variance, strategy indistinguishability
- [ ] Second real benchmark: tabular classification/regression on a public dataset

## Open Questions

1. **When does causal reasoning help most?** Synthetic benchmarks show wins. The energy predictive benchmark will test this on real data. Need results across problem structures (linear, nonlinear, high-dimensional, sparse, dense interactions).

2. **How to handle the DAG being wrong?** Sensitivity to graph misspecification. The CBO paper assumes the graph is correct — what happens when it's not? SensitivityValidator provides partial answers.

3. **Scaling POMIS computation.** For large graphs (50+ variables), POMIS enumeration may become expensive. Need efficient algorithms or approximations.

4. **LLM integration.** Should the optimizer use LLMs to propose experiments (like AlphaEvolve), or stick to mathematical optimization? Probably both — LLMs for code domains, mathematical optimization for numerical domains.

5. **How to construct causal graphs for code?** Static analysis (like cDEP for Spark)? LLM-assisted graph construction? Learned from ablation experiments?

## Test Coverage

- ~739 fast tests passing on main (`uv run pytest -m "not slow"`)
- Slow regression tests for convergence and benchmark reproducibility (`uv run pytest -m slow`)
- CI runs fast tests only (~2 min per matrix job)
