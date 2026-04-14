# Sprint 30 ERCOT Reality Report

**Date**: 2026-04-12
**Sprint**: 30 (General Causal Autoresearch: Reality Check And Portability)
**Issue**: #162
**Branch**: `sprint-30/ercot-reality-gate`
**Base commit**: `8bafc26` (Sprint 30 portability brief merged)
**Optimizer path**: ax_botorch (ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0)

## Verdict

**PARTIAL REAL-WORLD SIGNAL** -- the Sprint 29 default change
(`causal_exploration_weight=0.0`) breaks the long-standing
"causal indistinguishable from surrogate-only on real ERCOT data" result.
Causal is now certified better than surrogate-only on COAST (p=0.008) and
trending better on NORTH_C (p=0.059).  However, causal still does not
statistically beat random on either dataset, and all strategies still
select ridge regression.  This is the first real-world evidence of a
causal vs surrogate-only differentiation, but it is not a full causal
advantage claim on ERCOT forecasting.

This is one reality-check lane within the broader general-causal
autoresearch roadmap (see Sprint 30 portability brief).  It does not
redefine the project around ERCOT.

## 1. Prior Reports and What Changed

The March 2026 ERCOT reports (NORTH_C and COAST) both found:

1. `causal` and `surrogate_only` produced **identical** test MAE at every
   seed, budget, and strategy combination
2. `random` was marginally better than both engine strategies
3. All strategies converged to `ridge` regression
4. Budget barely mattered (B20 ≈ B40 ≈ B80)

The Sprint 29 scorecard changed one optimizer-core default
(`causal_exploration_weight: 0.3 → 0.0`).  This PR tests whether that
change moves the real-world picture under the same locked benchmark
contract.

## 2. NORTH_C Results (Sprint 30 vs Sprint 16)

**Std convention:** all Sprint 30 std values use population std
(ddof=0).  The Sprint 16 comparison columns preserve the values from
the prior reports, which may have used sample std (ddof=1) — small
discrepancies in random std between the two columns (e.g., 0.36 vs
0.32) reflect this convention difference, not a re-run of random.

### 2a. Test MAE (mean ± std across 5 seeds)

| Strategy | Budget | Sprint 16 | Sprint 30 | Δ |
|----------|--------|-----------|-----------|---|
| random | 20 | 132.50 ± 0.22 | 132.50 ± 0.22 | 0.00 |
| random | 40 | 132.50 ± 0.22 | 132.50 ± 0.22 | 0.00 |
| random | 80 | 132.35 ± 0.10 | 132.35 ± 0.10 | 0.00 |
| surrogate_only | 20 | 133.14 ± 0.01 | **133.32** ± 0.03 | +0.18 |
| surrogate_only | 40 | 133.15 ± 0.01 | **133.32** ± 0.03 | +0.17 |
| surrogate_only | 80 | 132.72 ± 0.37 | **132.98** ± 0.30 | +0.26 |
| causal | 20 | 133.14 ± 0.01 | **133.06** ± 0.23 | -0.08 |
| causal | 40 | 133.15 ± 0.01 | **132.72** ± 0.39 | -0.43 |
| causal | 80 | 132.72 ± 0.37 | **132.48** ± 0.34 | -0.24 |

Causal and surrogate_only are **no longer identical**.  Causal improved
by 0.08-0.43 MAE across budgets while surrogate_only degraded slightly.

### 2b. NORTH_C B80 Head-to-Head (5 Seeds)

| Comparison | Delta | Cohen's d | MWU p (2-sided) | Wins | Verdict |
|-----------|-------|-----------|----------------|------|---------|
| causal vs surrogate_only | **-0.498** | -1.39 | **0.059** | 4/5 | Trending (causal better) |
| causal vs random | +0.129 | +0.46 | 0.402 | 1/5 | Random still better |
| surrogate_only vs random | +0.627 | +2.54 | **0.008** | 0/5 | Random beats s.o. |

### 2c. NORTH_C Val-Test Gap

| Strategy | Val MAE | Test MAE | Gap |
|----------|---------|----------|-----|
| random | 124.858 | 132.348 | 7.490 |
| causal | 124.830 | 132.477 | 7.647 |
| surrogate_only | 125.069 | 132.975 | 7.906 |

Causal has a slightly smaller val-test gap than surrogate_only (7.65 vs
7.91), suggesting less overfitting.  Random still has the smallest gap.

## 3. COAST Results (Sprint 30 vs Sprint 16)

### 3a. Test MAE (mean ± std across 5 seeds)

| Strategy | Budget | Sprint 16 | Sprint 30 | Δ |
|----------|--------|-----------|-----------|---|
| random | 20 | 105.14 ± 0.36 | 105.14 ± 0.32 | 0.00 |
| random | 40 | 105.20 ± 0.30 | 105.20 ± 0.27 | 0.00 |
| random | 80 | 105.21 ± 0.24 | 105.21 ± 0.21 | 0.00 |
| surrogate_only | 20 | 105.84 ± 0.00 | 105.84 ± 0.00 | 0.00 |
| surrogate_only | 40 | 105.84 ± 0.00 | 105.84 ± 0.00 | 0.00 |
| surrogate_only | 80 | 105.58 ± 0.15 | **105.72** ± 0.14 | +0.14 |
| causal | 20 | 105.84 ± 0.00 | **105.48** ± 0.26 | -0.36 |
| causal | 40 | 105.84 ± 0.00 | **105.03** ± 0.06 | -0.81 |
| causal | 80 | 105.58 ± 0.15 | **104.88** ± 0.54 | -0.70 |

Causal is now **distinct from surrogate_only** and improved by 0.36-0.81
MAE across budgets.  Causal also beats random in mean at every budget,
though the causal-vs-random MWU at B80 is p=0.690 (not significant).
B40 and B20 causal-vs-random were not formally tested.

### 3b. COAST B80 Head-to-Head (5 Seeds)

| Comparison | Delta | Cohen's d | MWU p (2-sided) | Wins | Verdict |
|-----------|-------|-----------|----------------|------|---------|
| causal vs surrogate_only | **-0.842** | -1.92 | **0.008** | 5/5 | **Certified** (causal better) |
| causal vs random | -0.332 | -0.72 | 0.690 | 3/5 | Trending, not significant |
| surrogate_only vs random | +0.511 | +2.55 | **0.008** | 0/5 | Random beats s.o. |

### 3c. COAST Val-Test Gap

| Strategy | Val MAE | Test MAE | Gap |
|----------|---------|----------|-----|
| causal | 90.124 | 104.876 | **14.753** |
| random | 90.200 | 105.208 | 15.008 |
| surrogate_only | 90.368 | 105.719 | 15.351 |

Causal has the **smallest** val-test gap on COAST.  This matters: it
means the causal improvement holds up on held-out data rather than
being validation-only movement.

## 4. Win/Loss Matrix (Test MAE, Head-to-Head, B80)

| Comparison | NORTH_C | COAST |
|-----------|---------|-------|
| causal < surrogate_only | 4/5 | **5/5** |
| causal < random | 1/5 | 3/5 |
| surrogate_only < random | 0/5 | 0/5 |

## 5. Model Selection Distribution

Both datasets still show 100% ridge selection across all 45 runs per
strategy per dataset.  The Sprint 29 default change did not move
strategies off ridge — it found **better ridge hyperparameters**.

## 6. Answers to the Reality Check Questions

### 6a. Does the current merged baseline improve test MAE on real ERCOT benchmarks?

**Partially, yes.** Causal test MAE improved on both datasets:
- NORTH_C B80: 132.72 → 132.48 (-0.24 MAE)
- COAST B80: 105.58 → 104.88 (-0.70 MAE)

Surrogate-only stayed flat or slightly degraded on both datasets.
Random is unchanged (same seed sequence, no graph).

### 6b. Do results move on held-out test performance rather than only validation metrics?

**Yes.** Causal has the smallest val-test gap on COAST (14.75 vs 15.00
for random and 15.35 for surrogate_only), indicating the improvement
is real rather than validation-only movement.  On NORTH_C, causal's
val-test gap (7.65) is smaller than surrogate_only's (7.91) but larger
than random's (7.49).

### 6c. Does causal now differ meaningfully from surrogate_only on real data?

**Yes — this is the most important finding.** The prior reports found
causal and surrogate_only produced identical test MAE at every seed,
budget, and strategy combination.  Under the new default:

- **COAST B80**: causal 104.88 vs surrogate_only 105.72, MWU p=0.008,
  5/5 head-to-head wins, Cohen's d=-1.92.  **Certified causal advantage
  over surrogate_only**.
- **NORTH_C B80**: causal 132.48 vs surrogate_only 132.98, MWU p=0.059,
  4/5 head-to-head wins, Cohen's d=-1.39.  **Trending causal advantage
  over surrogate_only**.

This is the first real-world evidence of the causal vs surrogate-only
differentiation that Sprint 16 identified as missing.

### 6d. Which parts of the current stack are specific to ERCOT forecasting?

Answered in the [Sprint 30 portability brief](sprint-30-general-causal-portability-brief.md).
Summary: the core engine is fully portable; 6 of 7 active regression
gate rows are ERCOT-tied; the `MarketingLogAdapter` is the next
recommended non-energy validation surface.

### 6e. What should the next non-energy benchmark contract be?

Also answered in the portability brief: a marketing offline policy
benchmark built on the shipped `MarketingLogAdapter`, with 10 seeds,
B20/B40/B80, a permuted-outcome null control, and the same evidence
standards used for ERCOT.

## 7. What This Does Not Prove

Important caveats to stay honest:

1. **Causal does not statistically beat random.** On COAST, causal's
   -0.33 MAE vs random is not statistically significant (p=0.69, 3/5
   seeds).  On NORTH_C, causal is worse than random in mean (+0.13 MAE).
2. **All strategies still select ridge.** The improvement is in ridge
   hyperparameters, not model class.
3. **The gaps are small in absolute terms.** Best-case improvement is
   0.70 MAE on COAST (out of ~105 MAE baseline, ~0.7% relative).
4. **Sample size is small.** Only 5 seeds per strategy-budget combination.
   A 10-seed rerun would be more convincing, especially for the NORTH_C
   p=0.059 result.
5. **Prior reports' observation that random is often marginally best
   still holds.** The change is that causal is now separable from
   surrogate-only, not that causal has become the clear winner.

## 8. Updated Classification

| Dataset | Sprint 16 Finding | Sprint 30 Finding |
|---------|-----------------|------------------|
| NORTH_C | causal == surrogate_only, random best | causal trending better than s.o. (p=0.059), random still best |
| COAST | causal == surrogate_only, random best | causal certified better than s.o. (p=0.008), causal trending better than random (not significant) |

## 9. Sprint 31 Recommendation

**The ERCOT reality check produced a partial signal, not a clean win.**
Three things should happen in parallel in Sprint 31:

1. **Rerun ERCOT with 10 seeds** to firm up the NORTH_C p=0.059 result
   and the COAST causal vs random question.  This is a low-cost
   experiment (45 runs per dataset).
2. **Start the marketing offline policy benchmark** recommended in the
   Sprint 30 portability brief.  This tests whether the Sprint 29
   default transfers to a non-energy domain with intervention-oriented
   semantics.
3. **Publish the Sprint 30 reality-and-generalization scorecard**
   synthesizing this ERCOT evidence with the portability brief and
   deciding the Sprint 31 direction.

The project should not pivot to "we now have causal advantage on
ERCOT."  The project should continue on the general-causal autoresearch
roadmap, with ERCOT as one validation lane that is now more
informative than it was before.

## 10. Provenance

### Environment

- Python 3.13.12
- ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0
- git SHA: 8bafc26

### Optimizer Path

All 90 runs (45 NORTH_C + 45 COAST) confirmed
`optimizer_path: "ax_botorch"` in provenance metadata.

### Artifacts

Local (not committed):
```
artifacts/sprint-30-ercot-reality-gate/north_c_results.json
artifacts/sprint-30-ercot-reality-gate/coast_results.json
```

### Commands

```bash
DATA_NORTH=/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet
DATA_COAST=/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet
OUT=/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-30-ercot-reality-gate

uv run python scripts/energy_predictive_benchmark.py \
  --data-path "$DATA_NORTH" \
  --budgets 20,40,80 --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output "$OUT/north_c_results.json"

uv run python scripts/energy_predictive_benchmark.py \
  --data-path "$DATA_COAST" \
  --budgets 20,40,80 --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output "$OUT/coast_results.json"
```

Runtime: roughly 60 min per dataset (45 runs each, mostly random-strategy model fitting).
