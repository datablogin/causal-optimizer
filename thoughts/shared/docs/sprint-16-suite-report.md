# Energy Benchmark Suite Report

## Metadata

- **Date**: 2026-03-27
- **Benchmarks**: 2
- **Strategies**: random, surrogate_only, causal
- **Budgets**: 20, 40, 80
- **Baseline**: random
- **Overall verdict**: **CONDITIONAL**

## Per-Benchmark Results (Largest Budget)

### ercot_north_c

| Strategy | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|------------------------|------------------------|-------|
| random | 132.35 +/- 0.10 | 124.86 +/- 0.06 | 5 |
| surrogate_only | 132.72 +/- 0.37 | 124.94 +/- 0.17 | 5 |
| causal | 132.55 +/- 0.34 | 124.87 +/- 0.15 | 5 |

### ercot_coast

| Strategy | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|------------------------|------------------------|-------|
| random | 105.21 +/- 0.21 | 90.20 +/- 0.04 | 5 |
| surrogate_only | 105.27 +/- 0.49 | 90.26 +/- 0.14 | 5 |
| causal | 105.68 +/- 0.21 | 90.29 +/- 0.14 | 5 |

## Cross-Benchmark Rankings

- **ercot_north_c**: random > causal > surrogate_only (best to worst)
- **ercot_coast**: random > surrogate_only > causal (best to worst)

## Full Results (All Budgets)

### ercot_north_c

| Strategy | Budget | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|--------|------------------------|------------------------|-------|
| random | 20 | 132.50 +/- 0.22 | 124.92 +/- 0.05 | 5 |
| surrogate_only | 20 | 133.14 +/- 0.01 | 125.15 +/- 0.03 | 5 |
| causal | 20 | 132.89 +/- 0.43 | 125.07 +/- 0.18 | 5 |
| random | 40 | 132.50 +/- 0.22 | 124.92 +/- 0.05 | 5 |
| surrogate_only | 40 | 133.15 +/- 0.01 | 125.14 +/- 0.02 | 5 |
| causal | 40 | 132.90 +/- 0.41 | 125.05 +/- 0.19 | 5 |
| random | 80 | 132.35 +/- 0.10 | 124.86 +/- 0.06 | 5 |
| surrogate_only | 80 | 132.72 +/- 0.37 | 124.94 +/- 0.17 | 5 |
| causal | 80 | 132.55 +/- 0.34 | 124.87 +/- 0.15 | 5 |

### ercot_coast

| Strategy | Budget | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|--------|------------------------|------------------------|-------|
| random | 20 | 105.14 +/- 0.32 | 90.27 +/- 0.05 | 5 |
| surrogate_only | 20 | 105.70 +/- 0.21 | 90.47 +/- 0.04 | 5 |
| causal | 20 | 105.90 +/- 0.05 | 90.50 +/- 0.06 | 5 |
| random | 40 | 105.20 +/- 0.27 | 90.25 +/- 0.04 | 5 |
| surrogate_only | 40 | 105.61 +/- 0.23 | 90.45 +/- 0.04 | 5 |
| causal | 40 | 105.91 +/- 0.04 | 90.50 +/- 0.06 | 5 |
| random | 80 | 105.21 +/- 0.21 | 90.20 +/- 0.04 | 5 |
| surrogate_only | 80 | 105.27 +/- 0.49 | 90.26 +/- 0.14 | 5 |
| causal | 80 | 105.68 +/- 0.21 | 90.29 +/- 0.14 | 5 |

## Acceptance Rules

| Rule | Result |
|------|--------|
| improved | FAIL |
| no_regression | PASS |
| stable | PASS |
| differentiated | PASS |
| **overall** | **CONDITIONAL** |

### Reasons

- not improved: causal did not beat random on any benchmark
- differentiated: causal vs surrogate_only differ by 0.1% on ercot_north_c
- differentiated: causal vs surrogate_only differ by 0.4% on ercot_coast
- overall: CONDITIONAL

## Key Questions

1. **Did the causal differentiation change actually change results?** Yes
2. **Did causal improve on either benchmark?** No
3. **Did any strategy regress?** No
4. **Are results stable across seeds?** Yes
5. **Overall recommendation**: **INVESTIGATE**

> No regressions detected but either results are not differentiated or causal did not improve. Further investigation needed to understand whether the causal graph is providing value.

## Sprint 16 Analysis

### What changed vs Sprint 15

Sprint 15 showed `causal == surrogate_only` on every seed, budget, and metric. Sprint 16's
causal fallback differentiation (PR #70) successfully broke that identity:

- **NORTH_C**: causal now beats surrogate_only at every budget (132.55 vs 132.72 at B80)
- **COAST**: causal is slightly worse than surrogate_only (105.68 vs 105.27 at B80)
- **The strategies are now genuinely different** — the `differentiated` rule passes

### Why causal still loses to random

The causal graph for the energy adapter is a star (all 7 search variables are direct parents
of `mae`). This means the targeted intervention candidates perturb 1-2 of the same variables
that random search explores uniformly. The graph provides structural information (which
variables are direct causes) but the star topology provides no *selective* information (which
variables to prioritize over others).

For causal guidance to outperform random, we likely need either:

1. A graph with non-trivial structure (some variables are not direct parents)
2. Edge weights or effect sizes that let the optimizer prioritize stronger causes
3. A benchmark where the causal structure matters (e.g., confounders, mediators)

### Implications for next sprint

The suite infrastructure works and the acceptance rules correctly classify this as CONDITIONAL
(differentiated but not improved). The next step should be either:

- A semi-synthetic benchmark with known causal structure where causal guidance has a
  plausible advantage (Plan 08, Agent 5)
- Edge-weighted causal guidance that uses effect estimates, not just graph topology

## Artifact Paths

- Suite summary: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint16/suite_summary.json`
- Per-benchmark results: `suite_sprint16/ercot_north_c_results.json`, `suite_sprint16/ercot_coast_results.json`
- Per-benchmark CSVs: `suite_sprint16/ercot_north_c_summary.csv`, `suite_sprint16/ercot_coast_summary.csv`
- Suite report: `suite_sprint16/suite_report.md`
