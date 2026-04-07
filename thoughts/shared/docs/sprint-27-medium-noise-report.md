# Sprint 27 Medium-Noise Crossover Report

## Metadata

- **Date**: 2026-04-07
- **Sprint**: 27 (Medium-Noise Crossover)
- **Issue**: #142
- **Branch**: `sprint-27/medium-noise-crossover`
- **Base commit**: `b5530aa` (Sprint 26 expansion scorecard merged to main)

## 1. Executive Summary

The medium-noise demand-response variant (9D: 3 real + 2 original noise +
4 nuisance) sits cleanly between the base (5D) and high-noise (15D)
benchmarks.  Causal wins 10/10 seeds at B80 (two-sided p=0.007), with mean
regret 1.87 — between the base (1.13) and high-noise (2.57).

The crossover story is now a smooth gradient, not a binary switch:

| Variant | Total dims | Noise dims | B80 causal mean | B80 causal wins | B80 two-sided p |
|---------|-----------|------------|-----------------|-----------------|-----------------|
| Base | 5 | 2 | 1.13 | 9/10 | 0.045 |
| **Medium** | **9** | **6** | **1.87** | **10/10** | **0.007** |
| High-noise | 15 | 12 | 2.57 | 8/10 | 0.014 |

As noise dimensionality increases, causal pruning provides an increasingly
decisive advantage over surrogate_only search.  The surrogate_only strategy
degrades steadily (B80 mean regret: 4.90 → 9.61 → 15.23) while causal
remains stable (1.13 → 1.87 → 2.57).

## 2. Medium-Noise Variant Design

### 2a. Shape

- **Total dimensions**: 9
- **Real parents**: 3 (treat_temp_threshold, treat_hour_start, treat_hour_end)
- **Original noise**: 2 (treat_humidity_threshold, treat_day_filter [categorical])
- **New nuisance**: 4 (noise_var_0 through noise_var_3, continuous [0, 1])
- **Categorical variables**: 1 (treat_day_filter — same as base)

### 2b. Rationale

The base has 5 variables (noise ratio 2/5 = 40%).  High-noise has 15
(noise ratio 12/15 = 80%).  Medium-noise has 9 (noise ratio 6/9 = 67%),
sitting between the two.  This tests whether the crossover from causal
advantage to surrogate parity happens at an intermediate noise burden.

### 2c. Oracle

Non-degenerate: uses the same treatment-effect function as the base
benchmark (sigmoid × Gaussian on temperature and hour).  Oracle treat rate
and oracle value are identical to the base variant because nuisance
variables have zero effect on outcomes.

## 3. Benchmark Results

### 3a. Medium-Noise Summary (mean regret +/- std, 10 seeds)

| Strategy | B20 | B40 | B80 |
|----------|-----|-----|-----|
| random | 21.61 +/- 13.45 | 15.15 +/- 8.87 | 9.36 +/- 4.30 |
| surrogate_only | 25.24 +/- 4.60 | 22.21 +/- 3.34 | 9.61 +/- 5.38 |
| causal | 23.20 +/- 12.98 | 5.86 +/- 5.72 | **1.87 +/- 1.74** |

### 3b. Medium-Noise B80 Causal Per-Seed

| Seed | Regret | Status |
|------|--------|--------|
| 0 | 1.44 | mediocre |
| 1 | 0.35 | good |
| 2 | 0.35 | good |
| 3 | 3.20 | mediocre |
| 4 | 3.52 | mediocre |
| 5 | 0.51 | good |
| 6 | 0.36 | good |
| 7 | 3.21 | mediocre |
| 8 | 0.35 | good |
| 9 | 5.43 | mediocre |

- Catastrophic (>10): **0/10**
- Good (<1.0): **5/10**

### 3c. Causal vs Surrogate-Only Statistical Tests

All tests use the directional hypothesis H_a: causal regret < surrogate_only.
Two-sided p-values reported as the conservative default.

| Budget | Causal wins | One-sided p | Two-sided p |
|--------|------------|-------------|-------------|
| B20 | 5/10 | 0.575 | 0.910 |
| B40 | 9/10 | 0.001 | 0.001 |
| B80 | 10/10 | 0.004 | 0.007 |

Causal advantage emerges at B40 and strengthens at B80.  At B20 the
advantage is not detectable (insufficient data for 9D exploration).

## 4. Crossover Analysis

### 4a. Noise-Dimension Gradient (B80 Causal)

| Variant | Dims | Noise dims | Causal mean | Causal std | S.O. mean | Causal wins | Two-sided p |
|---------|------|-----------|-------------|------------|-----------|-------------|-------------|
| Base | 5 | 2 | 1.13 | 1.40 | 4.90 | 9/10 | 0.045 |
| Medium | 9 | 6 | 1.87 | 1.74 | 9.61 | 10/10 | 0.007 |
| High | 15 | 12 | 2.57 | 2.28 | 15.23 | 8/10 | 0.014 |

### 4b. Key Observation

The gradient is smooth and monotonic in both directions:

- **Causal regret increases slowly**: 1.13 → 1.87 → 2.57 (mild degradation)
- **Surrogate-only regret increases sharply**: 4.90 → 9.61 → 15.23 (nearly linear)
- **Causal win rate remains high**: 9/10, 10/10, 8/10

This confirms the crossover story: causal pruning provides *stable*
performance across noise levels because it focuses on the 3 real parents.
Surrogate-only degrades proportionally with noise dimensionality because
the RF must model increasingly many irrelevant dimensions.

### 4c. Where Is the Crossover?

There is no crossover within the demand-response family.  Causal wins at
all tested noise levels (5D, 9D, 15D).  The crossover to surrogate_only
advantage appears to require a structurally different landscape (smooth,
all-continuous, no categoricals — as seen in the dose-response benchmark).

The medium-noise result strengthens the claim that within the
demand-response family, causal advantage is robust and scales with noise
burden.  The crossover is *across families*, not within the noise-dimension
gradient.

## 5. Null Control

**PASS.** Maximum strategy difference 0.2%, identical to Sprint 25.
9th consecutive sprint with clean null control (S18–S27).

## 6. Core Questions

### 6a. Is the oracle non-degenerate?

**Yes.** Same oracle as the base variant (nuisance variables have zero
effect on outcomes).

### 6b. Does medium-noise sit between base and high-noise?

**Yes, cleanly.** Mean regret 1.87 sits between base (1.13) and
high-noise (2.57).  Surrogate-only degradation is monotonic (4.90 → 9.61
→ 15.23).

### 6c. Does causal beat surrogate_only at B40/B80?

**Yes, decisively.** 9/10 wins at B40 (two-sided p=0.001), 10/10 wins at
B80 (two-sided p=0.007).

### 6d. Is the result consistent with the crossover story?

**Yes, and it sharpens it.** The crossover is not a within-family
noise-threshold phenomenon.  Causal advantage is robust across all
noise levels in the demand-response family.  The crossover to surrogate
advantage requires a structurally different landscape (smooth, no
categoricals).
