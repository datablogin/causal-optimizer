# Sprint 24 Post-Fix Stability Report

## Metadata

- **Date**: 2026-04-06
- **Sprint**: 24 (Post-Fix Stability Sweep)
- **Issue**: #126
- **Branch**: `sprint-24/post-fix-stability-sweep`
- **Base commit**: `dacd8fe` (PR #128 categorical diversity fix merged to main)
- **Predecessor**: Sprint 23 stability scorecard (#124)

## Provenance

- **Git SHA**: `dacd8fe7c4b3a01593bed3ca6402551537b2788e`
- **Python**: 3.13.12
- **Key packages**: ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0, numpy 2.4.2
- **Dataset**: ERCOT NORTH_C + DFW 2022-2024 (hash `be4af8b3...`)
- **Artifacts**: local `/artifacts/sprint-24/` (not committed)

## 1. Executive Summary

**The categorical diversity fix (PR #128) did not meet the stability gate.**

Base B80 results are unchanged from Sprint 23 hardened: 3/10 catastrophic
seeds, mean regret 5.30, std 6.82.  The fix had no measurable effect on the
bimodal failure mode.

High-noise remains directionally strong (causal wins 7/10 at B80,
two-sided p=0.037).  Null control passes cleanly (max delta 0.2%).

The fix was correctly implemented and tested but is insufficient: the
categorical lock-in that traps bad seeds persists despite diversity
injection into the Ax candidate batch.

## 2. Benchmark Configuration

### 2a. Base Counterfactual

- Variant: `base` (3 causal parents + 2 noise dimensions)
- Budgets: 20, 40, 80
- Seeds: 0-9
- Strategies: random, surrogate_only, causal
- Total runs: 90
- Runtime: 1167s

### 2b. High-Noise Counterfactual

- Variant: `high_noise` (3 causal parents + 2 noise + 10 nuisance dimensions)
- Budgets: 20, 40, 80
- Seeds: 0-9
- Strategies: random, surrogate_only, causal
- Total runs: 90
- Runtime: 1263s

### 2c. Null Control

- Variant: null-signal (permuted target, seed 99999)
- Budgets: 20, 40
- Seeds: 0, 1, 2
- Strategies: random, surrogate_only, causal
- Total runs: 18

## 3. Base Counterfactual Results

### 3a. Summary Table (mean regret +/- std over 10 seeds)

| Strategy | B20 | B40 | B80 |
|----------|-----|-----|-----|
| random | 20.58 +/- 11.34 | 12.75 +/- 9.27 | 7.77 +/- 2.83 |
| surrogate_only | 23.00 +/- 5.33 | 20.40 +/- 4.82 | 6.13 +/- 5.09 |
| causal | 16.31 +/- 10.72 | 9.80 +/- 6.78 | **5.30 +/- 6.82** |

### 3b. Base B80 Causal Per-Seed Breakdown

| Seed | Regret | Status |
|------|--------|--------|
| 0 | 0.35 | good |
| 1 | 0.36 | good |
| 2 | 0.35 | good |
| 3 | **16.96** | catastrophic |
| 4 | 0.41 | good |
| 5 | 0.58 | good |
| 6 | 0.36 | good |
| 7 | 3.93 | mediocre |
| 8 | **14.80** | catastrophic |
| 9 | **14.94** | catastrophic |

- Catastrophic (regret > 10): **3/10**
- Good (regret < 1.0): **6/10**
- Oracle value: 48.41

### 3c. Causal vs Surrogate-Only at B80

| Metric | Causal | Surrogate-Only |
|--------|--------|----------------|
| Mean regret | 5.30 | 6.13 |
| Std | 6.82 | 5.09 |
| Catastrophic | 3/10 | 3/10 |
| Seeds < 1.0 | 6/10 | 3/10 |
| Wins | 7/10 | 3/10 |
| MWU p (one-sided) | 0.154 | — |
| MWU p (two-sided) | 0.307 | — |

Causal wins 7/10 seeds but the advantage is not statistically significant
at B80 (one-sided p=0.154, two-sided p=0.307) due to the high variance
from catastrophic seeds.  All Mann-Whitney U tests in this report use the
directional hypothesis H_a: causal regret < surrogate_only regret.

### 3d. Comparison to Sprint 22 and Sprint 23 Baselines

| Session | Mean | Std | Catastrophic | Seeds < 1.0 |
|---------|------|-----|-------------|-------------|
| S22 | 3.26 | 5.78 | 2/10 | 8/10 |
| S23 benchmark | 1.81 | 4.35 | 1/10 | 9/10 |
| S23 hardened | 5.30 | 6.82 | 3/10 | 6/10 |
| **S24 (this run)** | **5.30** | **6.82** | **3/10** | **6/10** |

S24 results are numerically identical to S23 hardened.  This is not
coincidence: the seed-forwarding fix retained from Sprint 23 makes Ax
candidate generation deterministic per seed, so the same seeds (3, 8, 9)
fail identically.  The categorical diversity injection did not change which
candidates the alignment-only reranker selects.

## 4. High-Noise Counterfactual Results

### 4a. Summary Table (mean regret +/- std over 10 seeds)

| Strategy | B20 | B40 | B80 |
|----------|-----|-----|-----|
| random | 23.99 +/- 8.95 | 18.14 +/- 7.90 | 10.71 +/- 3.53 |
| surrogate_only | 26.94 +/- 4.97 | 25.07 +/- 5.17 | 15.30 +/- 11.47 |
| causal | 22.46 +/- 3.97 | 16.78 +/- 6.95 | **4.56 +/- 5.92** |

### 4b. High-Noise B80 Per-Seed

| Seed | Causal Regret | Surrogate-Only Regret | Winner |
|------|--------------|----------------------|--------|
| 0 | 3.33 | 1.51 | surrogate_only |
| 1 | 0.44 | 0.65 | causal |
| 2 | 3.31 | 0.78 | surrogate_only |
| 3 | 0.36 | 28.66 | causal |
| 4 | 0.41 | 12.23 | causal |
| 5 | 4.72 | 28.07 | causal |
| 6 | 0.65 | 28.16 | causal |
| 7 | **16.96** | 12.39 | surrogate_only |
| 8 | **15.01** | 28.16 | causal |
| 9 | 0.41 | 12.39 | causal |

- Causal catastrophic: 2/10
- Surrogate-only catastrophic: 7/10
- Causal wins: 7/10
- Mann-Whitney U: one-sided p=0.019, two-sided p=0.037

Note: Seed 8 is catastrophic for causal on both base and high-noise.  Seed 7
is catastrophic on high-noise but only mediocre (3.93) on base.  The partial
overlap suggests the failure is seed-specific (bad initial LHS draw or GP
trajectory) rather than benchmark-specific.

### 4c. High-Noise Causal Advantage

The causal advantage on high-noise is directionally strong.  All tests use
the one-sided hypothesis H_a: causal regret < surrogate_only regret.
Two-sided p-values are also reported for completeness.

| Budget | Causal wins | MWU p (one-sided) | MWU p (two-sided) |
|--------|------------|-------------------|-------------------|
| B20 | 7/10 | 0.031 | 0.062 |
| B40 | 9/10 | 0.007 | 0.014 |
| B80 | 7/10 | 0.019 | 0.037 |

Under two-sided testing at alpha=0.05, B40 and B80 are significant but B20
is not (p=0.062).  Under one-sided testing all three budgets are significant.
The directional hypothesis is justified by the prior Sprint 19-23 evidence
that causal should outperform surrogate_only on positive-control benchmarks.
However, the bimodal mode still appears (seeds 7, 8 catastrophic).

## 5. Null Control Results

**Verdict: PASS.**

| Strategy | Budget | Test MAE |
|----------|--------|----------|
| random | 20 | 3260.48 +/- 2.15 |
| random | 40 | 3261.58 +/- 0.34 |
| surrogate_only | 20 | 3255.49 +/- 0.00 |
| surrogate_only | 40 | 3255.49 +/- 0.00 |
| causal | 20 | 3262.19 +/- 0.75 |
| causal | 40 | 3262.78 +/- 0.02 |

Per-budget breakdown:

| Budget | Max delta | Within threshold? |
|--------|-----------|-------------------|
| B20 | 0.2% (surrogate_only vs random) | yes |
| B40 | 0.2% (surrogate_only vs random) | yes |

Maximum strategy difference: 0.2%.  Well within the 2% safety threshold.
No strategy shows consistent improvement on permuted data.

### 5a. Null Control History

| Sprint | Max Diff | Verdict | Source |
|--------|----------|---------|--------|
| S18 | 0.15% | PASS | sprint-18-discovery-trust-scorecard |
| S19 | 0.15% | PASS | sprint-19-differentiation-scorecard |
| S20 | 0.20% | PASS | sprint-20-post-ax-rerun-report |
| S21 | 0.18% | PASS | sprint-21-attribution-scorecard |
| S22 | 0.23% | PASS | sprint-22-alignment-only-confirmation-report |
| S23 | 0.20% | PASS | sprint-23-stability-scorecard |
| **S24** | **0.20%** | **PASS** | this report |

## 6. Stability Gate Assessment

### 6a. Did the Fix Meet the Targets?

| Target | Result | Met? |
|--------|--------|------|
| Base B80 catastrophic seeds = 0/10 | 3/10 | **NO** |
| Base B80 mean regret < 2.0 | 5.30 | **NO** |
| Base B80 std < 3.0 | 6.82 | **NO** |
| High-noise remains directionally strong | 7/10 wins, two-sided p=0.037 | YES |
| Null control within 2% | 0.2% max | YES |

**The categorical diversity fix did not meet the stability gate.**
None of the three B80 stability targets were met.

### 6b. Why the Fix Had No Effect

The categorical diversity injection (PR #128) correctly ensures every
categorical value appears in the Ax reranking batch.  But the
alignment-only reranker selects candidates based on how closely their
parameters match the current best result on *causal ancestor* dimensions.
`treat_day_filter` is not a causal ancestor of the objective, so the
reranker is neutral on it.

The diversity candidates are created by copying `candidates[0]` (the first
Ax-generated candidate) and substituting the categorical value.  These
diversity candidates score identically to `candidates[0]` on ancestor
alignment — but `candidates[0]` may not be the best-aligned candidate in
the batch.  The reranker still picks whichever original Ax candidate has the
best alignment score, regardless of its categorical value.

The fix guarantees that "all" appears in the batch, but it does not
guarantee that "all" is *selected*.  With deterministic seed forwarding,
the same seeds produce the same Ax candidates, the same alignment scores,
and the same reranker choices — resulting in identical outcomes.

### 6c. What This Tells Us

The bimodal B80 failure is not in the *availability* of diverse categorical
candidates (which the fix addresses).  It is in the *selection* dynamics:

1. The GP model learns from early exploration that "weekday" is promising
2. The alignment-only reranker is blind to `treat_day_filter` (correct behavior)
3. But this blindness means the reranker cannot *prefer* the diversity
   candidate even when it's present
4. The exploitation phase (steps 50-79) perturbs parameters locally and
   rarely flips the categorical value

The fix addresses step 1's downstream effect (diverse candidates exist) but
not the selection mechanism that actually drives the lock-in.

## 7. Decision

### 7a. Stability Gate

**NOT MET.**  The categorical diversity fix had no measurable effect on the
bimodal B80 failure.  All three stability targets (catastrophic count, mean,
std) remain at Sprint 23 hardened levels.

### 7b. What Is Confirmed

1. **Null control**: PASS (0.2% max, 7 consecutive sprints clean)
2. **High-noise causal advantage**: significant at B40 and B80 (two-sided
   p=0.014, 0.037); B20 marginal (two-sided p=0.062)
3. **Alignment-only reranking**: correct default (Sprint 21 A/B confirmed,
   Sprint 22 revert confirmed)
4. **Seed forwarding**: working as intended (deterministic per-seed behavior)

### 7c. What Remains Open

The bimodal B80 mode is not in candidate availability — it is in the
selection and exploitation dynamics.  Potential next directions:

1. **Categorical-aware reranking**: add a small bonus for categorical
   diversity in the alignment score (without reopening balanced reranking)
2. **Exploitation-phase categorical sampling**: force categorical variation
   during exploitation perturbation (steps 50-79), not just optimization
3. **GP acquisition diversification**: modify the Ax acquisition function to
   encourage exploration across categorical dimensions
4. **Accept the bimodal mode**: treat 1-3/10 catastrophic seeds as an
   inherent property of the GP-based approach and focus on downstream
   robustness (e.g., multi-seed ensembling)

### 7d. Sprint 25 Recommendation

The project is not ready to broaden benchmark scope.  The bimodal B80
failure persists with a well-understood mechanism (categorical lock-in) but
no effective fix yet.

Sprint 25 should investigate **exploitation-phase categorical sampling** as
the most promising next intervention.  The optimization phase (steps 10-49)
already has diversity candidates in the batch; the exploitation phase (steps
50-79) perturbs 1-2 variables at a time and is the phase where categorical
lock-in becomes permanent.

If exploitation-phase intervention also fails, the project should consider
accepting the bimodal mode as irreducible and shifting focus to multi-seed
robustness strategies.
