# Sprint 26: Interaction Policy Benchmark Report

**Date:** 2026-04-05
**Branch:** `sprint-26/positive-control-expansion`
**Benchmark variant:** `interaction` (multi-threshold interaction policy)

## Summary

Added a new positive-control benchmark family (`InteractionPolicyScenario`)
with a fundamentally different causal structure from the existing
DemandResponseScenario family.  The key challenge is a **nonlinear
interaction surface** rather than a categorical trap.

## Benchmark Design

### Causal structure

The treatment effect function depends on **three-way interactions** between
temperature, humidity, and hour-of-day.  Each marginal effect is small
(~20-32 MW), but the combined effect when all three conditions align
is large (~400 MW peak).  This super-additive structure means an optimizer
must get multiple thresholds right simultaneously.

### Search space (7 dimensions)

- 4 real policy variables: `policy_temp_threshold`, `policy_humidity_threshold`,
  `policy_hour_start`, `policy_hour_end` -- all continuous/integer, no categoricals.
- 3 noise dimensions: `noise_wind_speed`, `noise_pressure`, `noise_cloud_cover`
  -- continuous variables with zero effect on outcome.

### Intended causal advantage

The causal graph connects the 4 real variables to `objective` and routes
the 3 noise variables to an isolated `weather_noise` node.  A causal-aware
optimizer focuses on the 4-D real subspace; surrogate_only must search 7-D
with a complex interaction landscape.

### Oracle

- **Oracle treat rate:** 21.3% (within target 20-40%)
- **Oracle value:** 19.88 (test set, average net benefit)
- **Treatment cost:** 120.0 MW

## Results

Evaluated with: budgets 20, 40, 80; seeds 0-9; strategies random, surrogate_only, causal.
Data source: ERCOT North C / DFW 2022-2024 (26,291 hourly rows).
Suite runtime: 1,085 seconds (~18 minutes).

### Regret table (mean +/- std, 10 seeds)

| Strategy       | B20            | B40            | B80            |
|----------------|----------------|----------------|----------------|
| random         | 10.13 +/- 4.73 | 8.37 +/- 3.81  | 5.85 +/- 1.85  |
| surrogate_only | 5.75 +/- 1.56  | 4.54 +/- 1.88  | 2.19 +/- 0.75  |
| causal         | 12.26 +/- 5.46 | 7.04 +/- 4.79  | 2.85 +/- 1.78  |

### Decision error rate (mean +/- std, 10 seeds)

| Strategy       | B20            | B40            | B80            |
|----------------|----------------|----------------|----------------|
| random         | 0.191 +/- 0.053| 0.190 +/- 0.059| 0.154 +/- 0.037|
| surrogate_only | 0.153 +/- 0.037| 0.128 +/- 0.040| 0.092 +/- 0.020|
| causal         | 0.227 +/- 0.055| 0.158 +/- 0.059| 0.107 +/- 0.037|

### Statistical significance (B80, Mann-Whitney U, two-sided)

| Comparison                | U-statistic | p-value |
|---------------------------|-------------|---------|
| causal vs random          | 12.0        | 0.0046  |
| surrogate_only vs random  | 2.0         | 0.0003  |
| causal vs surrogate_only  | 56.0        | 0.6767  |

### B80 win rate

- causal beats surrogate_only: 6/10 seeds
- surrogate_only beats random: 10/10 seeds

## Interpretation

1. **Both guided strategies significantly beat random** at B80 (p < 0.01),
   confirming this is a genuine positive-control benchmark.

2. **Surrogate_only and causal converge at B80** (regret 2.19 vs 2.85,
   p=0.68 -- not significantly different).  This is expected: with no
   categorical trap, the main challenge is modeling the interaction surface,
   which the RF surrogate can also learn.

3. **Early-budget (B20) shows causal weakness**: causal has higher regret
   than surrogate_only (12.26 vs 5.75) and even random (10.13) at B20.
   This is because the engine's exploration phase (first 10 experiments)
   fills a large 7-D LHS which is sparse, and the causal focus on 4
   variables does not help enough when the interaction surface is poorly
   sampled.  Surrogate_only benefits from Ax/BoTorch's Sobol initialization
   which adapts better in early budget.

4. **Convergence trend**: causal shows the steepest improvement trajectory
   (B20 regret 12.26 -> B40 7.04 -> B80 2.85), suggesting the causal
   focus variable pruning kicks in strongly after the exploration phase.

5. **Structural difference from DemandResponse family**: This benchmark
   has zero categorical variables.  The challenge is the multi-threshold
   interaction surface, not a categorical lock-in trap.  This provides
   complementary coverage to the existing benchmark family.

## Verdict

**PASS as positive control** -- the benchmark discriminates meaningfully
between strategies (guided >> random), has a non-degenerate oracle (21.3%
treat rate), and runs in under 20 minutes for a full 10-seed grid.

**Causal advantage**: marginal.  The interaction structure benefits any
guided optimizer equally.  The causal graph helps prune noise dimensions
but does not provide information about the interaction structure itself.
At high budget, causal and surrogate_only converge.

## Files

- Benchmark: `causal_optimizer/benchmarks/interaction_policy.py`
- Tests: `tests/unit/test_interaction_policy_benchmark.py`
- CLI: `scripts/counterfactual_benchmark.py` (extended with `--variant interaction`)
- Results: `interaction_benchmark_results.json`
