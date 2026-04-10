# Sprint 28 Ax/BoTorch Regression Gate Report

## Metadata

- **Date**: 2026-04-09
- **Sprint**: 28 (Ax/BoTorch Regression Gate)
- **Issue**: #147
- **Branch**: `sprint-28/ax-regression-gate`
- **Base commit**: `f715173` (Sprint 28 optimizer-path provenance merged to main)
- **Benchmarks run**: 7 of 7
- **Optimizer path**: ax_botorch (ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0)

## Verdict

**PASS -- Ax/BoTorch regression gate intact.** The trusted Ax priors
are reproduced exactly: base and high-noise match Sprint 25, medium-noise
matches Sprint 27 (when it was introduced).  The base-B80 stability gate holds
(0/10 catastrophic, mean 1.13, std 1.40).  Null control is clean.  All
7 benchmarks completed successfully with provenance confirming
`optimizer_path: "ax_botorch"` in every artifact.

## 1. Executive Summary

This is the direct Ax/BoTorch comparison that Sprint 27's combined
regression gate could not provide (it ran on RF fallback).  With the
same package versions as Sprint 25 (ax-platform 1.2.4, botorch 0.17.2,
torch 2.10.0) and the same code (plus Sprint 28 provenance tracking),
the core energy benchmarks reproduce their trusted Ax references to the
second decimal place (base and high-noise vs Sprint 25; medium-noise vs
Sprint 27, when it was introduced).

The key finding: **the optimizer backend is the primary driver of
absolute regret differences between Sprint 25 and Sprint 27.**
The code is stable; the numbers change only when the backend changes.

## 2. Benchmark Coverage

| # | Benchmark | Budgets | Seeds | Strategies | Runtime |
|---|-----------|---------|-------|------------|---------|
| 1 | Base energy | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 2147s |
| 2 | Medium-noise energy | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 2181s |
| 3 | High-noise energy | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 2171s |
| 4 | Confounded energy | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 2090s |
| 5 | Null control | 20, 40 | 0-2 | random, surrogate_only, causal | ~7200s |
| 6 | Interaction policy | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 2047s |
| 7 | Dose-response | 20, 40, 80 | 0-4 | random, surrogate_only, causal | 817s |

Total wall-clock time: approximately 2 hours running 7 benchmarks in parallel.

## 3. Core Questions

### 3a. Did the trusted base-B80 stability gate remain intact under the Ax path?

**Yes, exactly.**

| Metric | S25 (Ax/BoTorch) | S28 (Ax/BoTorch) | S27 (RF fallback) |
|--------|------------------|------------------|-------------------|
| B80 causal mean regret | 1.13 | 1.13 | 7.32 |
| B80 causal std | 1.40 | 1.40 | 10.14 |
| Catastrophic (>10) | 0/10 | 0/10 | 3/10 |
| Seeds < 1.0 | 8/10 | 8/10 | 3/10 |
| Causal wins vs s.o. | 9/10 | 9/10 | 5/10 |
| MWU two-sided p | 0.045 | 0.045 | 0.970 |

The base-B80 numbers are identical to Sprint 25 to the second decimal
place.  The stability gate criteria (0/10 catastrophic, mean < 2.0,
std < 3.0) are fully met.  This confirms that the Sprint 27 RF fallback
deviation (mean 7.32, 3/10 catastrophic) was entirely attributable to
the optimizer backend, not code drift.

Per-seed B80 regrets: 0.35, 0.36, 0.35, 3.52, 0.60, 0.61, 0.36, 4.28, 0.36, 0.51

### 3b. Did medium-noise and high-noise preserve the demand-response story?

**Yes, exactly.**

| Metric | S27 Medium | S28 Medium | S25 High | S28 High |
|--------|-----------|-----------|---------|---------|
| B80 causal mean | 1.87 | 1.87 | 2.57 | 2.57 |
| B80 causal std | 1.74 | 1.74 | 2.28 | 2.28 |
| Catastrophic | 0/10 | 0/10 | 0/10 | 0/10 |
| Causal wins vs s.o. | 10/10 | 10/10 | 8/10 | 8/10 |
| MWU two-sided p | 0.007 | 0.007 | 0.014 | 0.014 |

Medium-noise reproduces the Sprint 27 Ax-primary reference exactly
(medium-noise was introduced in Sprint 27, not Sprint 25).  High-noise
reproduces the Sprint 25 trusted prior exactly.  The
noise-dimension gradient remains smooth and monotonic under Ax/BoTorch:

| Variant | Dims | Noise dims | B80 causal mean | B80 s.o. mean | Causal wins | Two-sided p |
|---------|------|-----------|-----------------|---------------|-------------|-------------|
| Base | 5 | 2 | 1.13 | 4.98 | 9/10 | 0.045 |
| Medium | 9 | 6 | 1.87 | 9.61 | 10/10 | 0.007 |
| High | 15 | 12 | 2.57 | 15.23 | 8/10 | 0.014 |

The causal advantage strengthens as dimensionality increases: causal
remains low (1.13 to 2.57 mean regret), while surrogate_only degrades
sharply (4.98 to 15.23).  The surrogate_only B80 regret is actually
worse than random at 15D (15.23 vs 10.71), confirming the curse of
dimensionality.

### 3c. Did null control remain clean?

**Yes. PASS.**

| Budget | S.O. vs Random Delta | Causal vs Random Delta |
|--------|---------------------|----------------------|
| B20 | 0.2% | 0.1% |
| B40 | 0.2% | 0.1% |

Maximum strategy difference: **0.2%** (within the 2% threshold).
This is the **10th clean null run** (S18-S25, S27, S28).

### 3d. Which benchmark rows match the Sprint 27 RF-fallback gate, and which differ?

**Qualitative ordering matches; absolute numbers differ significantly
for base, medium-noise, and dose-response.**

| Benchmark | S28 B80 Winner | S27 B80 Winner | Match? | Key difference |
|-----------|---------------|---------------|--------|----------------|
| Base | Causal (p=0.045) | Tie (p=0.97) | **Different** | Causal wins only under Ax |
| Medium-noise | Causal (p=0.007) | Causal (p=0.026) | Match | Stronger under Ax |
| High-noise | Causal (p=0.014) | Causal (p=0.017) | Match | Similar p-values |
| Confounded | None (all misled) | None (all misled) | Match | Same behavior |
| Null control | PASS (0.2%) | PASS (0.2%) | Match | Identical |
| Interaction | S.O. (p=0.014) | S.O. (p=0.0006) | Match | S.O. wins in both |
| Dose-response | Causal (p=0.142) | S.O. (p=0.008) | **Different** | Causal wins under Ax |

Two benchmarks flip their winner when switching backends:

1. **Base energy**: Causal wins decisively under Ax (p=0.045) but is
   at parity under RF (p=0.97).  The GP-based BO is critical for the
   exploitation-phase sweep to converge.

2. **Dose-response**: Causal achieves near-zero regret under Ax
   (mean 0.20) vs surrogate_only (1.19), reversing the RF finding where
   surrogate_only won (2.80 vs 7.99).  The GP surrogate is much more
   effective at modeling the smooth Emax dose-response surface than the
   RF surrogate.

### 3e. Are any family-level conclusions changed by restoring the Ax path?

**Yes, one significant change:**

- **Dose-response**: Under RF fallback, surrogate_only won decisively
  (Sprint 26/27).  Under Ax/BoTorch, causal achieves near-zero B80
  regret (0.20) and beats surrogate_only (1.19), though not statistically
  significant at this sample size (p=0.142 two-sided).  Both guided
  strategies converge much faster under Ax.  This reversal suggests the
  Sprint 26 conclusion ("smooth Emax favors direct surrogate modeling
  over causal pruning") was RF-specific rather than a fundamental
  property.

Other family-level conclusions:
- Core energy: causal advantage confirmed (3/3 variants under Ax, 2/3 under RF)
- Confounded: all strategies misled (unchanged)
- Null control: no false positives (unchanged)
- Interaction: surrogate_only wins under both backends, but the gap
  narrows under Ax (S26 was a tie at p=0.68; S28 Ax shows s.o. wins at
  p=0.014).  This benchmark is backend-sensitive in magnitude, with
  surrogate_only consistently favored

## 4. Per-Benchmark Detailed Results

### 4a. Base Energy (B80, 10 seeds)

| Strategy | Mean Regret | Std | Catastrophic | Per-Seed |
|----------|------------|-----|-------------|----------|
| random | 7.77 | 2.83 | 3/10 | 10.21, 11.01, 8.57, 11.35, 4.64, 7.50, 4.63, 9.44, 7.82, 2.52 |
| surrogate_only | 4.98 | 5.32 | 3/10 | 0.78, 1.59, 0.36, 0.35, 0.93, 11.71, 8.29, 11.30, 13.90, 0.63 |
| causal | 1.13 | 1.40 | 0/10 | 0.35, 0.36, 0.35, 3.52, 0.60, 0.61, 0.36, 4.28, 0.36, 0.51 |

MWU causal vs s.o.: U=23.0, one-sided p=0.022, two-sided p=0.045

### 4b. Medium-Noise Energy (B80, 10 seeds)

| Strategy | Mean Regret | Std | Catastrophic | Per-Seed |
|----------|------------|-----|-------------|----------|
| random | 9.36 | 4.30 | 3/10 | 17.49, 9.60, 14.74, 9.69, 9.60, 2.25, 11.01, 4.72, 8.92, 5.56 |
| surrogate_only | 9.61 | 5.38 | 7/10 | 2.93, 0.58, 1.05, 13.90, 12.23, 12.23, 14.64, 12.23, 13.90, 12.39 |
| causal | 1.87 | 1.74 | 0/10 | 1.44, 0.35, 0.35, 3.20, 3.52, 0.51, 0.36, 3.21, 0.35, 5.43 |

MWU causal vs s.o.: U=14.0, one-sided p=0.004, two-sided p=0.007

### 4c. High-Noise Energy (B80, 10 seeds)

| Strategy | Mean Regret | Std | Catastrophic | Per-Seed |
|----------|------------|-----|-------------|----------|
| random | 10.71 | 3.53 | 5/10 | 9.70, 4.25, 11.93, 9.83, 7.99, 15.86, 7.92, 16.29, 13.25, 10.03 |
| surrogate_only | 15.23 | 11.36 | 7/10 | 1.51, 0.65, 1.05, 28.78, 12.23, 26.58, 28.15, 12.23, 28.78, 12.39 |
| causal | 2.57 | 2.28 | 0/10 | 3.33, 0.36, 4.28, 0.41, 0.96, 7.52, 0.73, 3.34, 4.31, 0.41 |

MWU causal vs s.o.: U=17.0, one-sided p=0.007, two-sided p=0.014

### 4d. Confounded Energy (B80, 10 seeds)

| Strategy | Mean Regret | Std | Catastrophic | Per-Seed |
|----------|------------|-----|-------------|----------|
| random | 20.66 | 9.40 | 10/10 | 10.21, 20.27, 21.10, 11.35, 24.11, 21.44, 15.50, 21.68, 15.39, 45.54 |
| surrogate_only | 20.65 | 0.00 | 10/10 | 20.65, 20.65, 20.65, 20.65, 20.65, 20.65, 20.65, 20.65, 20.65, 20.65 |
| causal | 20.83 | 0.39 | 10/10 | 20.65, 20.65, 20.85, 20.82, 20.65, 21.96, 20.65, 20.71, 20.65, 20.65 |

All strategies converge to the same wrong optimum due to confounding.
Surrogate_only deterministically converges to the confounded surface
optimum (20.65) with zero variance, matching Sprint 27.

### 4e. Null Control (B20/B40, 3 seeds)

| Budget | S.O. vs Random | Causal vs Random |
|--------|---------------|-----------------|
| B20 | 0.2% | 0.1% |
| B40 | 0.2% | 0.1% |

**Verdict: PASS.** Maximum delta 0.2%, well within the 2% threshold.
10th clean null run (S18-S25, S27, S28).

### 4f. Interaction Policy (B80, 10 seeds)

| Strategy | Mean Regret | Std | Catastrophic | Per-Seed |
|----------|------------|-----|-------------|----------|
| random | 5.85 | 1.85 | 0/10 | 5.76, 5.35, 4.02, 9.86, 6.70, 4.76, 6.94, 2.81, 7.21, 5.14 |
| surrogate_only | 2.18 | 0.75 | 0/10 | 1.79, 1.79, 1.79, 2.31, 1.78, 2.31, 4.34, 1.80, 1.93, 1.93 |
| causal | 3.17 | 1.61 | 0/10 | 1.96, 3.41, 1.79, 2.57, 2.36, 6.30, 2.25, 2.41, 2.38, 6.28 |

MWU causal vs s.o.: U=83.0, one-sided p=0.994, two-sided p=0.014

Note: one-sided p=0.994 reflects H_a: causal < s.o. (wrong direction).
The two-sided p=0.014 confirms surrogate_only wins.

Under Ax/BoTorch the causal-vs-s.o. gap narrows compared to RF
fallback (3.17 vs 2.18 under Ax; 4.31 vs 1.76 under RF).  The
Sprint 26 tie (p=0.68) is not replicated; surrogate_only still wins
at B80, but the gap is modest.

### 4g. Dose-Response (B80, 5 seeds)

| Strategy | Mean Regret | Std | Per-Seed |
|----------|------------|-----|----------|
| random | 9.22 | 0.57 | 8.79, 9.10, 8.58, 9.40, 10.20 |
| surrogate_only | 1.19 | 0.81 | 1.49, 1.89, 0.30, 0.15, 2.11 |
| causal | 0.20 | 0.02 | 0.22, 0.17, 0.23, 0.20, 0.20 |

Oracle value: 10.51 (matching Sprint 27).

MWU causal vs s.o.: U=5.0, one-sided p=0.071, two-sided p=0.142

Under Ax/BoTorch, causal converges to near-zero regret (0.20) with
almost zero variance (std 0.02).  This is a dramatic improvement over
both RF fallback (7.99) and surrogate_only (1.19).  The GP surrogate
models the smooth Emax surface much more effectively than the RF,
and causal pruning to the dose variable further accelerates convergence.

## 5. Strategy Ordering Summary

| Benchmark | B80 Winner | Causal Advantage? | Consistent with Ax prior? | Changed from S27 RF? |
|-----------|-----------|-------------------|--------------------------|---------------------|
| Base energy | Causal (p=0.045) | Yes | Yes, matches S25 | Yes (was tie) |
| Medium-noise | Causal (p=0.007) | Yes, strong | Yes, matches S27 | No (was also causal) |
| High-noise | Causal (p=0.014) | Yes, strong | Yes, matches S25 | No (was also causal) |
| Confounded | None (all misled) | No | Yes | No |
| Null control | None (PASS) | N/A | Yes | No |
| Interaction | S.O. (p=0.014) | No | Closer to S26 tie | Yes (was stronger s.o.) |
| Dose-response | Causal (p=0.142) | Trending yes | **New finding** | Yes (was s.o. win) |

## 6. Backend-Sensitive Differences (S28 Ax vs S27 RF)

| Metric | S28 Ax/BoTorch | S27 RF Fallback | Delta |
|--------|---------------|-----------------|-------|
| Base B80 causal mean | 1.13 | 7.32 | -84.6% |
| Base B80 catastrophic | 0/10 | 3/10 | -3 |
| Medium B80 causal mean | 1.87 | 4.63 | -59.6% |
| High B80 causal mean | 2.57 | 3.62 | -29.0% |
| Dose B80 causal mean | 0.20 | 7.99 | -97.5% |
| Dose B80 s.o. mean | 1.19 | 2.80 | -57.5% |
| Interaction B80 causal mean | 3.17 | 4.31 | -26.5% |

The Ax/BoTorch path produces substantially lower regret across all
benchmarks.  The improvement is largest where the GP surrogate has the
most to contribute: base energy (-84.6%), dose-response (-97.5%), and
medium-noise (-59.6%).  The improvement is smallest where
dimensionality makes the GP less effective and causal pruning does the
heavy lifting: high-noise (-29.0%).

## 7. Provenance

### 7a. Environment

- Python 3.13.12
- numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0
- **ax-platform: 1.2.4** (installed and confirmed)
- **botorch: 0.17.2** (installed and confirmed)
- **torch: 2.10.0** (installed and confirmed)
- **gpytorch: 1.15.2**
- git SHA: f715173

### 7b. Optimizer Path

All runs confirmed `optimizer_path: "ax_botorch"` in provenance
metadata.  This is verified by the `detect_optimizer_path()` function
added in Sprint 28 PR #149.

### 7c. Data

- Energy data: `ercot_north_c_dfw_2022_2024.parquet`
- Dose-response: synthetic (1000 patients, generated per-seed)
- Oracle values: Base/Medium/High/Confounded = 48.41, Interaction = 19.88, Dose = 10.51

### 7d. Artifacts

All JSON result files written to (local path, not checked into repo):
`/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-28-ax-gate/`

- `base_results.json` (90 results)
- `medium_noise_results.json` (90 results)
- `high_noise_results.json` (90 results)
- `confounded_results.json` (90 results)
- `null_control_results.json` (18 results)
- `interaction_results.json` (90 results)
- `dose_response_results.json` (45 results)

## 8. Answers to Required Questions

### 8a. Did the trusted base-B80 stability gate remain intact under the Ax path?

**Yes.** The Sprint 25 numbers are reproduced exactly: mean 1.13,
std 1.40, catastrophic 0/10, causal wins 9/10 (two-sided p=0.045).
The stability gate criteria (0/10 catastrophic, mean < 2.0, std < 3.0)
are fully met.

### 8b. Did medium-noise and high-noise preserve the demand-response story?

**Yes, exactly.** Medium-noise: mean 1.87, std 1.74, 10/10 causal wins
(p=0.007) — matches the Sprint 27 Ax-primary reference.  High-noise:
mean 2.57, std 2.28, 8/10 causal wins (p=0.014) — matches the Sprint 25
trusted prior.  Both match their respective Ax references to the second
decimal place.

### 8c. Did null control remain clean?

**Yes.** PASS with 0.2% maximum delta.  10th clean null run.

### 8d. Which benchmark rows match the S27 RF-fallback gate, and which differ?

Five of seven match qualitatively (medium-noise, high-noise, confounded,
null control, interaction).  Two change winner: base energy flips from
tie to causal-wins, and dose-response flips from s.o.-wins to
causal-wins.  Both flips are attributable to the GP surrogate being
more effective than the RF surrogate at low-to-moderate dimensionality.

### 8e. Are any family-level conclusions changed by restoring the Ax path?

One change: dose-response now favors causal under Ax (mean 0.20 vs 1.19),
reversing the Sprint 26/27 conclusion.  All other family-level conclusions
are unchanged.  The noise-dimension gradient story is strengthened:
causal advantage is statistically significant at all three noise levels
under Ax/BoTorch.

## 9. Recommendation

The Ax/BoTorch regression gate **passes**.  Sprint 25 trusted priors
are reproduced exactly.  The code is stable across Sprint 25 through
Sprint 28; all observed variation between sprints was attributable to
the optimizer backend (Ax/BoTorch vs RF fallback), not to code changes.
No further action required for this gate.
