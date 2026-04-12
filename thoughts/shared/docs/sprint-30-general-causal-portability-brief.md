# Sprint 30 General-Causal Portability Brief

**Date**: 2026-04-12
**Sprint**: 30 (General Causal Autoresearch: Reality Check And Portability)
**Issue**: #163
**Branch**: `sprint-30/general-causal-portability-brief`
**Base commit**: `4294a9e` (Sprint 29 optimizer-core regression gate merged)

## 1. Project Thesis

The causal-optimizer is not an energy forecasting tool.  It is a
**domain-agnostic automated research organization** that combines causal
reasoning, experiment design, benchmark discipline, and institutional
memory to decide what experiment to run next and whether the result is
real.

The origin document (00-origin.md) states the gap clearly: no existing
system combines causal discovery, experimental design, doubly-robust
estimation, causal Bayesian optimization, evolutionary diversity,
off-policy evaluation, and sensitivity analysis into one production
system.  The adapters listed — marketing, ML training, manufacturing,
drug discovery — are validation surfaces, not the product identity.

ERCOT energy is the first and most exercised validation surface.  It is
not the project.

## 2. Energy-Specific vs Domain-Portable Components

### 2a. Domain-Portable (Core Engine)

These components work on any domain that provides a search space, a
runner, and optionally a causal graph:

| Component | Path | Portability |
|-----------|------|-------------|
| ExperimentEngine | `engine/loop.py` | Fully generic |
| suggest_parameters | `optimizer/suggest.py` | Fully generic |
| CausalGraph / SearchSpace / types | `types.py` | Fully generic |
| EffectEstimator | `estimator/effects.py` | Fully generic |
| OffPolicyPredictor | `predictor/off_policy.py` | Fully generic |
| ScreeningDesigner | `designer/screening.py` | Fully generic |
| MAPElites | `evolution/map_elites.py` | Fully generic |
| GraphLearner | `discovery/graph_learner.py` | Fully generic |
| DomainAdapter base | `domain_adapters/base.py` | Fully generic |
| BenchmarkRunner | `benchmarks/runner.py` | Fully generic |
| Provenance | `benchmarks/provenance.py` | Fully generic |
| Bayesian optimizer | `optimizer/bayesian.py` | Fully generic |

### 2b. Domain Adapters (Reusable Contracts, Domain-Specific Data)

| Adapter | Path | Status | Domain |
|---------|------|--------|--------|
| EnergyLoadAdapter | `domain_adapters/energy_load.py` | Shipped, tested | Energy forecasting |
| MarketingLogAdapter | `domain_adapters/marketing_logs.py` | Shipped, tested | Marketing policy |
| MLTrainingAdapter | `domain_adapters/ml_training.py` | Shipped | ML hyperparameter |
| MarketingAdapter | `domain_adapters/marketing.py` | Shipped | Marketing (older, simulated) |

Note: `MarketingAdapter` is an older simulated marketing adapter.
`MarketingLogAdapter` is the newer logged-data / IPS-weighted adapter
recommended for the next benchmark.  Both implement the `DomainAdapter`
contract but serve different use cases (simulated vs logged data).

Each adapter implements the same `DomainAdapter` contract:
`get_search_space()`, `run_experiment()`, `get_prior_graph()`,
`get_objective_name()`, `get_minimize()`.

### 2c. Energy-Specific Benchmarks

These are the components tied to energy data or ERCOT-specific logic:

| Component | Path | Tied To |
|-----------|------|---------|
| DemandResponseScenario | `benchmarks/counterfactual_energy.py` | ERCOT energy covariates |
| CounterfactualVariants | `benchmarks/counterfactual_variants.py` | ERCOT energy covariates |
| InteractionPolicyScenario | `benchmarks/interaction_policy.py` | ERCOT energy covariates |
| NullSignalResult (module) | `benchmarks/null_predictive_energy.py` | ERCOT energy data |
| ValidationEnergyRunner / PredictiveBenchmarkResult | `benchmarks/predictive_energy.py` | ERCOT energy data |
| counterfactual_benchmark.py | `scripts/counterfactual_benchmark.py` | Energy CLI |
| null_energy_benchmark.py | `scripts/null_energy_benchmark.py` | Energy CLI |

### 2d. Domain-Portable Benchmarks

| Component | Path | Domain-Free? |
|-----------|------|-------------|
| DoseResponseScenario | `benchmarks/dose_response.py` | Yes — synthetic clinical |
| CompleteGraphBenchmark | `benchmarks/complete_graph.py` | Yes — synthetic |
| ToyGraphBenchmark | `benchmarks/toy_graph.py` | Yes — synthetic |
| HighDimensionalSparseBenchmark | `benchmarks/high_dimensional.py` | Yes — synthetic |
| InteractionSCM | `benchmarks/interaction_scm.py` | Yes — synthetic |
| InteractionBenchmark | `benchmarks/interaction.py` | Yes — synthetic |

### 2e. Summary

The core engine is **fully domain-portable**.  The benchmark portfolio
is **heavily energy-weighted**: of the 7 active rows in the Ax-primary
regression gate, **6 use ERCOT energy data** (base, medium-noise,
high-noise, confounded, null control, interaction — all built on ERCOT
covariates).  Dose-response is the only non-energy active row.

Several additional domain-portable benchmarks exist in the codebase
(complete_graph, toy_graph, high_dimensional, interaction_scm,
interaction) but are not part of the active regression gate.

The MarketingLogAdapter exists and is extensively tested but has no benchmark
contract or Sprint evidence attached to it.

## 3. Recommended Next Non-Energy Benchmark Contract

### 3a. Candidate: Marketing Offline Policy Benchmark

**Why marketing:**

1. The `MarketingLogAdapter` already exists, is extensively tested
   (66 unit + 5 integration tests), and satisfies the full adapter
   contract
2. Marketing policy evaluation is intervention-oriented: the engine
   searches over treatment policies, not passive predictions
3. The logged-action / IPS evaluation paradigm is a natural fit for the
   engine's causal framework — propensity scores, treatment effects, and
   policy value are first-class concepts
4. The adapter already has a causal graph (14 edges), search space
   (6 continuous variables), and a fixture dataset (300 rows)
5. This is the next non-energy domain recommended in the real-data
   adapter requirements doc (04-real-data-adapter-requirements.md)

**What the benchmark would test:**

1. Can the engine find better marketing policies than random search
   under a fixed experiment budget?
2. Does causal guidance (graph-based focus on policy levers) improve
   sample efficiency compared to surrogate-only search?
3. Does the null-control discipline transfer — i.e., does the optimizer
   avoid manufacturing false signal on a permuted-outcome marketing
   dataset?

### 3b. Benchmark Contract Shape

| Element | Specification |
|---------|--------------|
| Data | `tests/fixtures/marketing_log_fixture.csv` (300 rows) for CI; real marketing log for extended evaluation. **Known limitation:** 300 rows may be marginal for stable IPS-weighted policy evaluation across 10 seeds. The benchmark should report ESS (effective sample size) per seed and note variance if ESS is consistently low. |
| Search space | 6 continuous variables: `eligibility_threshold`, `email_share`, `social_share_of_remainder`, `min_propensity_clip`, `regularization`, `treatment_budget_pct` |
| Objective | `policy_value`, maximize |
| Strategies | random, surrogate_only, causal |
| Seeds | 10 |
| Budgets | 20, 40, 80 |
| Causal graph | 14-edge prior from adapter |
| Null control | Permuted outcome column, same search space |
| Success criterion | Causal >= surrogate_only on policy_value at B80 |
| Failure criterion | Causal <= random at B80, or null control fails |

### 3c. Acceptance Rules

1. The benchmark must run on the committed fixture data without network
   access
2. Results must be reproducible within tolerance under fixed seed (Ax/BoTorch
   has known cross-platform non-determinism; strict determinism applies
   only to the RF surrogate path)
3. The null control must use outcome permutation, not label shuffling
4. The report must separate observed policy improvement from causal
   attribution
5. Provenance must record optimizer path and adapter version

### 3d. What This Would Prove

A passing marketing benchmark would show that the engine's causal
advantage is not specific to energy demand-response surfaces.  It would
demonstrate that:

1. the intervention-oriented framing works on logged-action data
2. the causal graph provides useful variable pruning on a non-energy
   search space
3. the Sprint 29 default (causal_exploration_weight=0.0) is not
   harmful on non-energy surfaces and transfers without
   domain-specific tuning

A failing benchmark would be equally informative: it would identify
whether the engine needs adaptation for IPS-weighted objectives,
non-Gaussian metrics, or policy-evaluation-specific challenges.

## 4. Recommended Benchmark Portfolio Shape

For a general causal research assistant, the benchmark portfolio should
contain three research modes:

### Mode 1: Predictive Real-Data Benchmarks

**Purpose:** ground truth on real forecasting tasks.

**Current coverage:** ERCOT NORTH_C and COAST (energy).

**Gap:** no non-energy real-data predictive benchmark.

**Future candidates:** retail demand forecasting, web traffic
prediction, or any domain with a clear time-series prediction task and
available covariates.

### Mode 2: Intervention / Offline-Policy Benchmarks

**Purpose:** test whether the engine can find better intervention
policies from logged or experimental data.

**Current coverage:** none active (MarketingLogAdapter exists but has
no benchmark evidence).

**Next step:** the marketing offline policy benchmark described above.

**Future candidates:** clinical trial dose optimization (real logged
data), A/B test configuration optimization, ad bidding policy
evaluation.

### Mode 3: Controlled Positive / Negative Controls

**Purpose:** mechanism testing with known ground truth.

**Current coverage:**
- Demand-response family (3 noise levels + confounded): energy-specific
  covariates
- Dose-response (synthetic clinical): domain-free
- Interaction policy: energy-specific covariates
- Null control: energy-specific data

**Gap:** no non-energy controlled positive control.

**Future candidates:** synthetic marketing uplift with known CATE,
synthetic manufacturing yield optimization.

### Portfolio Balance Target

| Mode | Current | Target |
|------|---------|--------|
| Predictive real-data | 1 domain (energy) | 2+ domains |
| Intervention / offline-policy | 0 active | 1+ active |
| Controlled positive/negative | 7 rows (6 energy-tied) | 7+ rows (3+ non-energy) |

## 5. Sprint 31 Recommendation

**Start the marketing offline policy benchmark as a concrete non-energy
validation surface.**

Sprint 31 should:

1. Write a marketing offline policy benchmark scenario class modeled on
   `DoseResponseScenario` (which is already non-energy and works well)
2. Run it on the fixture data with 10 seeds, B20/B40/B80
3. Add a marketing null control (permuted outcomes)
4. Compare causal vs surrogate_only vs random
5. Publish a benchmark report with the same evidence standards used for
   energy (provenance, MWU tests, per-seed detail)

This should happen alongside or after the ERCOT rerun (Issue #162),
not instead of it.  The ERCOT rerun tests whether the Sprint 29
optimizer change matters on real data.  The marketing benchmark tests
whether the engine's causal advantage is domain-portable.

If the marketing benchmark passes, the project will have demonstrated
causal advantage on three domain families (energy demand-response,
clinical dose-response, marketing policy) instead of the current two.

If it fails, the failure will be specific and diagnosable — and the
project will have honestly tested its generality claim instead of
assuming it from adapter existence alone.
