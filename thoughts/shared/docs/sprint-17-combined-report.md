# Sprint 17 Combined Benchmark Report

## Metadata

- **Date**: 2026-04-08
- **Sprint**: 17 (suite re-run with skip calibration + counterfactual benchmark)
- **Datasets**: ercot_north_c, ercot_coast
- **Strategies**: random, surrogate_only, causal
- **Budgets**: 20, 40, 80
- **Seeds**: 0, 1, 2, 3, 4
- **Audit skip rate**: 0.1
- **PRs included**: #76 (counterfactual benchmark), #77 (skip calibration)

## 1. Real Benchmark Results

### Per-Benchmark Results (Budget = 80)

#### ercot_north_c

| Strategy | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|------------------------|------------------------|-------|
| random | 132.35 +/- 0.11 | 124.86 +/- 0.07 | 5 |
| surrogate_only | 132.95 +/- 0.32 | 125.06 +/- 0.17 | 5 |
| causal | 132.83 +/- 0.26 | 125.00 +/- 0.14 | 5 |

#### ercot_coast

| Strategy | Test MAE (mean +/- std) | Val MAE (mean +/- std) | Seeds |
|----------|------------------------|------------------------|-------|
| random | 105.21 +/- 0.24 | 90.20 +/- 0.04 | 5 |
| surrogate_only | 105.58 +/- 0.15 | 90.25 +/- 0.13 | 5 |
| causal | 105.65 +/- 0.19 | 90.31 +/- 0.16 | 5 |

### Cross-Benchmark Rankings

- **ercot_north_c**: random > causal > surrogate_only (best to worst)
- **ercot_coast**: random > surrogate_only > causal (best to worst)

### Acceptance Rules

| Rule | Result |
|------|--------|
| coverage | PASS |
| improved | FAIL |
| no_regression | PASS |
| stable | PASS |
| differentiated | FAIL |
| **overall** | **CONDITIONAL** |

### Key Observations vs Sprint 16

Sprint 16 showed `differentiated: PASS` because causal beat surrogate_only on ercot_north_c.
Sprint 17 shows `differentiated: FAIL` -- causal and surrogate_only are identical at budget=20
and budget=40 on both datasets. At budget=80, small differences emerge but the suite
considers them identical because at smaller budgets the strategies produce byte-identical
results (both converge to the same initial exploration then follow the same fallback path).

The causal fallback differentiation from PR #70 (Sprint 16) is only active once the engine
reaches the optimization phase (budget > 10 experiments), and the effect is small enough
that at budget=20 and budget=40 the engine does not yet diverge. Only at budget=80 does
the longer optimization run produce measurably different results.

### Full Results (All Budgets)

#### ercot_north_c

| Strategy | Budget | Test MAE (mean +/- std) | Val MAE (mean +/- std) |
|----------|--------|------------------------|------------------------|
| random | 20 | 132.50 +/- 0.25 | 124.92 +/- 0.06 |
| surrogate_only | 20 | 133.32 +/- 0.04 | 125.26 +/- 0.04 |
| causal | 20 | 133.32 +/- 0.04 | 125.26 +/- 0.04 |
| random | 40 | 132.50 +/- 0.25 | 124.92 +/- 0.06 |
| surrogate_only | 40 | 133.32 +/- 0.04 | 125.26 +/- 0.04 |
| causal | 40 | 133.32 +/- 0.04 | 125.26 +/- 0.04 |
| random | 80 | 132.35 +/- 0.11 | 124.86 +/- 0.07 |
| surrogate_only | 80 | 132.95 +/- 0.32 | 125.06 +/- 0.17 |
| causal | 80 | 132.83 +/- 0.26 | 125.00 +/- 0.14 |

#### ercot_coast

| Strategy | Budget | Test MAE (mean +/- std) | Val MAE (mean +/- std) |
|----------|--------|------------------------|------------------------|
| random | 20 | 105.14 +/- 0.36 | 90.27 +/- 0.06 |
| surrogate_only | 20 | 105.84 +/- 0.00 | 90.52 +/- 0.07 |
| causal | 20 | 105.84 +/- 0.00 | 90.52 +/- 0.07 |
| random | 40 | 105.20 +/- 0.30 | 90.25 +/- 0.05 |
| surrogate_only | 40 | 105.84 +/- 0.00 | 90.52 +/- 0.07 |
| causal | 40 | 105.84 +/- 0.00 | 90.52 +/- 0.07 |
| random | 80 | 105.21 +/- 0.24 | 90.20 +/- 0.04 |
| surrogate_only | 80 | 105.58 +/- 0.15 | 90.25 +/- 0.13 |
| causal | 80 | 105.65 +/- 0.19 | 90.31 +/- 0.16 |

## 2. Skip Calibration Analysis

### Skip Ratio by Strategy (Budget = 80)

#### ercot_north_c

| Strategy | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 | Mean |
|----------|--------|--------|--------|--------|--------|------|
| random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| surrogate_only | 0.0% | 20.8% | 0.0% | 36.0% | 0.0% | 11.4% |
| causal | 0.0% | 0.0% | 0.0% | 27.3% | 0.0% | 5.5% |

#### ercot_coast

| Strategy | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 | Mean |
|----------|--------|--------|--------|--------|--------|------|
| random | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| surrogate_only | 29.2% | 3.6% | 0.0% | 0.0% | 0.0% | 6.6% |
| causal | 0.0% | 0.0% | 31.0% | 37.5% | 0.0% | 13.7% |

### Skip Distribution Pattern

Random never skips (as expected -- no predictor model).
Engine-based strategies (surrogate_only, causal) skip 0-37% of candidates depending on seed.
Skip behavior is highly variable across seeds: most seeds skip nothing, while a few seeds
trigger aggressive skipping (20-37%).

### Audit False-Skip Rate

Audit results are sparse (only available when skipping occurs and a random audit fires):

| Dataset | Strategy | Audits | Correct | False-Skip Rate |
|---------|----------|--------|---------|-----------------|
| ercot_north_c | surrogate_only | 6 | 4 | 33% |
| ercot_north_c | causal | 3 | 3 | 0% |
| ercot_coast | surrogate_only | 5 | 2 | 60% |
| ercot_coast | causal | 10 | 7 | 30% |

Combined false-skip rate across all audits: **33% (8/24 audits were false skips)**.

### Skip Calibration Assessment

1. **Skip ratio is low overall**: Mean skip rate is under 14% for any strategy, meaning
   the off-policy predictor is conservative.
2. **False-skip rate is concerning**: 33% of audited skips were incorrect (the skipped
   candidate would have been better than expected). This is too high for production use.
3. **Skipping does not improve outcomes**: The strategies that skip more do not produce
   better results -- random (which never skips) wins on both datasets.
4. **Skip behavior is seed-dependent**: Skipping is triggered by early exploration results
   that happen to create overconfident surrogate predictions.

## 3. Counterfactual Benchmark Results

### Overview

The counterfactual demand-response benchmark ran with treatment_cost=50.0 using real
ERCOT NORTH_C covariates with synthetic treatment effects.

### Results

| Strategy | Budget | Policy Value | Regret | Decision Error Rate |
|----------|--------|-------------|--------|---------------------|
| random | 20 | 0.0000 | 0.0000 | 0.0% |
| surrogate_only | 20 | 0.0000 | 0.0000 | 0.0% |
| causal | 20 | 0.0000 | 0.0000 | 0.0% |
| random | 40 | 0.0000 | 0.0000 | 0.0% |
| surrogate_only | 40 | 0.0000 | 0.0000 | 0.0% |
| causal | 40 | 0.0000 | 0.0000 | 0.0% |
| random | 80 | 0.0000 | 0.0000 | 0.0% |
| surrogate_only | 80 | 0.0000 | 0.0000 | 0.0% |
| causal | 80 | 0.0000 | 0.0000 | 0.0% |

Oracle value: 0.0000 (across all seeds and budgets).

### Counterfactual Benchmark Assessment

**The benchmark is degenerate.** The oracle policy is "never treat" because the
treatment cost (50.0) exceeds the treatment benefit for every covariate pattern in the
dataset. All strategies correctly find this trivial optimum (regret=0 everywhere).

This makes the counterfactual benchmark uninformative for comparing strategies. The
benchmark needs to be re-parameterized with either:

1. A lower treatment cost so that treatment is sometimes beneficial
2. Stronger treatment effects that sometimes exceed the cost
3. A different cost-benefit structure that creates a non-trivial optimal policy

### Runtime Cost

The causal strategy on the counterfactual benchmark was extraordinarily slow:
- random budget=80: ~0.04s per run
- surrogate_only budget=80: ~32s per run
- causal budget=80: ~5000s (83 min) per run

The 100x slowdown for causal vs surrogate_only is caused by the screening designer
generating enormous numbers of formula permutations during the optimization phase.

## 4. Combined Promotion Assessment

### Does causal guidance help when the graph has non-trivial structure?

**Not yet testable.** The real benchmark uses a star graph (all variables are direct parents
of MAE), which provides no selective information. The counterfactual benchmark was degenerate
(oracle=never treat). We do not yet have a benchmark where causal structure matters.

### Are skip decisions trustworthy?

**No.** The 33% false-skip rate means roughly one in three skipped candidates would have
been worth evaluating. The off-policy predictor needs calibration improvements before
skip decisions can be trusted for cost savings.

### Updated Recommendation: **INVESTIGATE**

The promotion criteria from Plan 08 are:

| Criterion | Status |
|-----------|--------|
| causal materially different from surrogate_only | FAIL (identical at B20/B40) |
| change improves held-out performance on at least one benchmark | FAIL |
| no material regression on other benchmarks | PASS |
| results stable across seeds | PASS |
| speedup explainable by trustworthy skip diagnostics | FAIL (33% false-skip rate) |

**Verdict: INVESTIGATE.** No regressions, but causal guidance does not improve outcomes on
any available benchmark. The skip calibration infrastructure is working but the false-skip
rate is too high. The counterfactual benchmark needs re-parameterization.

## 5. What This Means for the Next Sprint

### Priority 1: Fix the Counterfactual Benchmark

The counterfactual benchmark is the most promising path to testing causal advantage, but
it needs non-trivial ground truth. The treatment cost should be lowered or the treatment
effect strengthened so that the oracle policy selectively treats some conditions.

### Priority 2: Improve Skip Calibration

The 33% false-skip rate indicates the off-policy predictor's confidence estimates are
poorly calibrated. Options:

1. Require higher confidence thresholds before skipping
2. Use calibrated probabilities (Platt scaling or isotonic regression)
3. Add a burn-in period before enabling skipping

### Priority 3: Address Causal Differentiation at Low Budgets

At budget=20 and budget=40, causal and surrogate_only are byte-identical. The causal
fallback path only activates in the optimization phase, which does not start until
experiment 11. This means at budget=20, only 10 experiments benefit from causal guidance.
The engine may need earlier causal influence (e.g., during exploration-phase candidate
selection).

### Priority 4: Performance

The causal strategy is 100x slower than surrogate_only on the counterfactual benchmark
due to screening formula enumeration. This needs profiling and optimization before the
counterfactual benchmark can be run at scale.

## Artifact Paths

- Suite summary: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17/suite_summary.json`
- Suite report: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17/suite_report.md`
- ercot_north_c results: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17/ercot_north_c_results.json`
- ercot_coast results: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint17/ercot_coast_results.json`
- Counterfactual results: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/counterfactual_sprint17_results.json`
