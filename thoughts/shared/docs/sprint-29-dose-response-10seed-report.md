# Sprint 29 Dose-Response 10-Seed Certification Report

**Date**: 2026-04-10
**Sprint**: 29 (Adaptive Causal Guidance)
**Issue**: #152
**Branch**: `sprint-29/dose-response-10seed`
**Base commit**: `ff4af00` (Sprint 29 trajectory diagnosis merged to main)
**Optimizer path**: ax_botorch (ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0)

## Verdict

**CAUSAL WIN CERTIFIED** -- the Sprint 28 dose-response causal trend is
confirmed as a statistically significant win at 10 seeds.  Causal beats
surrogate-only at B80 with two-sided MWU p=0.002, 9/10 seed wins,
sample-pooled Cohen's d=1.46.

## 1. Executive Summary

The Sprint 29 trajectory diagnosis identified dose-response as a real
but underpowered causal trend at n=5 (p=0.142, d~1.5).  This report
reruns the dose-response benchmark with 10 seeds under the Ax/BoTorch
primary path to certify or refute that trend.

**Result:** the trend is strongly confirmed.  At B80, causal converges
to near-zero regret (0.19, std 0.03) across all 10 seeds, while
surrogate-only is higher and more variable (0.92, std 0.66).  The
two-sided MWU p-value drops from 0.142 (5 seeds) to 0.002 (10 seeds),
crossing the significance threshold by a wide margin.  Causal also wins
at B40 (p=0.004).

This converts dose-response from an "Ax-primary mean-regret direction"
row to a **certified Ax-primary causal win**, narrowing the
backend-sensitive gap identified in the Sprint 28 scorecard.

## 2. B80 Results (10 Seeds)

| Strategy | Mean Regret | Std | Per-Seed |
|----------|------------|-----|----------|
| random | 9.08 | 0.71 | 8.79, 9.10, 8.58, 9.40, 10.20, 7.90, 8.38, 10.12, 8.73, 9.58 |
| surrogate_only | 0.92 | 0.66 | 1.49, 1.89, 0.30, 0.15, 2.11, 0.71, 0.45, 0.45, 0.45, 1.15 |
| causal | 0.19 | 0.03 | 0.22, 0.17, 0.23, 0.20, 0.20, 0.20, 0.21, 0.17, 0.15, 0.15 |

MWU causal vs s.o.: U=9.0, one-sided p=0.001, two-sided p=0.002
Causal wins: 9/10 seeds
Cohen's d (sample-pooled): 1.46

## 3. Full Budget Trajectory

| Budget | Causal Mean (Std) | S.O. Mean (Std) | MWU Two-Sided p | Causal Wins |
|--------|-------------------|------------------|-----------------|-------------|
| B20 | 6.20 (3.89) | 7.59 (2.80) | 0.909 | 6/10 |
| B40 | 0.73 (1.50) | 6.37 (3.22) | **0.004** | 8/10 |
| B80 | 0.19 (0.03) | 0.92 (0.66) | **0.002** | 9/10 |

At B20, both strategies are noisy and not significantly different.
At B40, causal already wins decisively (p=0.004, 8/10 seeds) -- only
one laggard seed (seed 2, regret 5.23) has not yet converged.
At B80, all 10 causal seeds converge to the 0.15-0.23 range with
near-zero variance.

## 4. Comparison to Sprint 28 (5-Seed Baseline)

| Metric | Sprint 28 (5 seeds) | Sprint 29 (10 seeds) | Change |
|--------|--------------------|--------------------|--------|
| Causal B80 mean | 0.20 | 0.19 | -0.01 (stable) |
| Causal B80 std | 0.02 | 0.03 | +0.01 (stable) |
| S.O. B80 mean | 1.19 | 0.92 | -0.27 (new seeds slightly better) |
| S.O. B80 std | 0.81 | 0.66 | -0.15 |
| Two-sided MWU p | 0.142 | **0.002** | Significant |
| Causal wins | 3/5 pairwise | 9/10 | Decisive |
| Cohen's d | ~1.5 | 1.46 | Stable |

**Seeds 0-4 reproduced exactly** for causal (0.22, 0.17, 0.23, 0.20,
0.20 -- identical to Sprint 28).  Seeds 5-9 are new and all converge to
the same narrow band (0.15-0.21).  The causal result is deterministic
and reproducible.

Surrogate-only seeds 0-4 also reproduced exactly (1.49, 1.89, 0.30,
0.15, 2.11).  The new seeds 5-9 are generally better (0.71, 0.45, 0.45,
0.45, 1.15), which slightly reduces the s.o. mean from 1.19 to 0.92.
The causal advantage persists regardless.

## 5. Per-Seed B80 Comparison

| Seed | Causal | S.O. | Causal Wins? |
|------|--------|------|-------------|
| 0 | 0.22 | 1.49 | Yes |
| 1 | 0.17 | 1.89 | Yes |
| 2 | 0.23 | 0.30 | Yes |
| 3 | 0.20 | 0.15 | No |
| 4 | 0.20 | 2.11 | Yes |
| 5 | 0.20 | 0.71 | Yes |
| 6 | 0.21 | 0.45 | Yes |
| 7 | 0.17 | 0.45 | Yes |
| 8 | 0.15 | 0.45 | Yes |
| 9 | 0.15 | 1.15 | Yes |

Seed 3 is the only seed where surrogate-only beats causal (0.15 vs 0.20),
and by a margin of only 0.05.  In all other 9 seeds, causal wins by
0.07-1.91 regret points.

## 6. Mechanism Confirmation

The trajectory diagnosis (PR #155) attributed the dose-response causal
advantage to dimensionality reduction: the causal graph correctly prunes
3 noise dimensions (bmi_threshold, age_threshold, comorbidity_threshold),
letting the GP model the smooth 3D Emax surface with higher sample
efficiency.

The 10-seed data confirms this mechanism:
1. Causal variance is near-zero at B80 (std 0.03) -- the 3D GP reliably
   finds the optimum
2. S.O. variance remains 20x higher (std 0.66) -- the 6D GP sometimes
   wastes budget on noise dimensions
3. The advantage is visible at B40 (p=0.004), confirming faster
   convergence, not just a final-value difference
4. All 10 causal seeds land in a 0.08-point band (0.15-0.23), while
   s.o. spans a 1.96-point band (0.15-2.11)

## 7. Updated Backend Classification

The Sprint 28 scorecard classified dose-response as "Ax-primary,
mean-regret direction reverses."  With 10-seed certification, the
updated classification is:

| Benchmark | Old Classification | New Classification |
|-----------|-------------------|-------------------|
| Dose-response | Ax-primary (mean-regret direction) | **Ax-primary (certified causal win, p=0.002)** |

The project now has **4 certified causal wins** under Ax/BoTorch (base
energy p=0.045, medium-noise p=0.007, high-noise p=0.014, dose-response
p=0.002) and 4 backend-invariant rows.  Only the interaction row remains
as a surrogate-only advantage.

## 8. Provenance

### Environment

- Python 3.13.12
- numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0
- **ax-platform: 1.2.4**, **botorch: 0.17.2**, **torch: 2.10.0**
- git SHA: ff4af00

### Optimizer Path

All runs confirmed `optimizer_path: "ax_botorch"` in provenance metadata.

### Artifacts

Local (not committed):
```
artifacts/sprint-29-dose-10seed/dose_response_10seed_results.json
```

Regeneration command:
```bash
uv sync --extra bayesian
uv run python3 scripts/dose_response_benchmark.py \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --budgets 20,40,80 \
  --output dose_response_10seed_results.json
```

Runtime: 823 seconds (~14 minutes).

## 9. Sprint 29 Recommendation

With dose-response now certified, the remaining optimizer-frontier row is
interaction only.  Issue #153 (adaptive causal guidance) should focus
exclusively on:

1. **Interaction ablation**: test whether reducing
   `causal_exploration_weight` and/or delaying alignment bonus improves
   the interaction B20/B80 gap
2. **Regression gate**: confirm demand-response and dose-response wins
   are preserved after any interaction-targeted change
3. **Do not pursue dose-response improvements** -- the row is solved
   (regret 0.19, p=0.002)
