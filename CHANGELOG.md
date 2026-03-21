# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-21

### Added

- Three-phase optimization loop: exploration (LHS), optimization (Bayesian/RF),
  exploitation (MAP-Elites diversity sampling)
- Causal graph integration: focus variables from DAG ancestors, POMIS pruning
  of intervention sets
- Bayesian optimization via Ax/BoTorch with POMIS-aware acquisition functions
- Causal GP surrogates for causally-informed surrogate modeling
- Multi-objective optimization with Pareto dominance and scalarization
- Constrained optimization with constraint satisfaction checking
- Effect estimation with bootstrap confidence intervals for keep/discard decisions
- Off-policy prediction with RF surrogate to skip predicted-poor experiments
- Observational estimation via DoWhy (backdoor/frontdoor/IV adjustment)
- Observational-enhanced off-policy predictions (tighter/wider CI based on agreement)
- fANOVA-based screening at phase transitions to identify important variables
- MAP-Elites archive for diversity tracking with behavioral descriptors
- Epsilon controller for observation-intervention tradeoff
- Causal discovery from experiment data (correlation, PC, NOTEARS methods)
- Sensitivity validation for causal effect robustness checking
- Research advisor diagnostics with four analyses + recommendation synthesis
  (EXPLOIT/EXPLORE/DROP/PIVOT) and observational signal analysis
- Domain adapters: marketing campaign optimization, ML hyperparameter tuning
  (both with realistic simulators, confounders, and failure modes)
- SQLite persistence with resume support
- CLI: run, resume, report, list commands
- 530+ tests across unit, integration, and regression suites
