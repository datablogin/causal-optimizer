# Sprint 20 Stability Audit Report

## Metadata

- **Date**: 2026-04-01
- **Sprint**: 20 Step 1 (Stability Audit)
- **Issue**: #106
- **Branch**: `sprint-20/stability-audit`
- **Predecessor**: Sprint 19 Differentiation Scorecard

## 1. Purpose

Sprint 19 reported that causal guidance beats surrogate_only on the base
counterfactual benchmark (regret 0.98 vs 2.16 at B80) and on the
high-noise variant (regret 3.47 vs 8.74 at B80). Those results were
based on 5 seeds.

This audit answers: **does the Sprint 19 causal win survive a wider
seed sweep, or is it fragile?**

## 2. Methodology

### 2a. Versions Compared

| Label | Commit | Description |
|-------|--------|-------------|
| S18 baseline | `a0f8d5f` | Sprint 18 final (discovery trust scorecard) |
| Current main | `52f7aef` | Sprint 19 final (`126d0d8` Ax path fix is ancestor of this commit) |

### 2b. Benchmark Configuration

- **Benchmark**: Counterfactual demand-response (base variant)
- **Data**: `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet`
- **Seeds**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 seeds)
- **Budgets**: 20, 40, 80
- **Strategies**: random, surrogate_only, causal
- **Treatment cost**: 60.0
- **Total runs per version**: 90 (10 seeds x 3 budgets x 3 strategies)
- **Oracle test-set value**: 48.41

Additionally, the **high-noise variant** was run with the same 10-seed
sweep on current main only (S18 does not have the high-noise variant).

### 2c. Commands

**S18 baseline** (run from worktree at `a0f8d5f`):

```bash
uv run python scripts/counterfactual_benchmark.py \
    --data-path .../ercot_north_c_dfw_2022_2024.parquet \
    --budgets 20,40,80 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --strategies random,surrogate_only,causal \
    --output artifacts/stability_base_s18.json
```

**Current main** (run from worktree at `52f7aef`):

```bash
uv run python scripts/counterfactual_benchmark.py \
    --data-path .../ercot_north_c_dfw_2022_2024.parquet \
    --variant base \
    --budgets 20,40,80 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --strategies random,surrogate_only,causal \
    --output artifacts/stability_base_main.json
```

**High-noise** (main only):

```bash
uv run python scripts/counterfactual_benchmark.py \
    --data-path .../ercot_north_c_dfw_2022_2024.parquet \
    --variant high_noise \
    --budgets 20,40,80 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --strategies random,surrogate_only,causal \
    --output artifacts/stability_high_noise_main.json
```

### 2d. Runtime

| Run | Duration |
|-----|----------|
| S18 base (90 runs) | 423s (7.1 min) |
| Main base (90 runs) | 427s (7.1 min) |
| Main high-noise (90 runs) | 589s (9.8 min) |

## 3. Reproducibility Finding

Before analyzing performance, we discovered an important reproducibility
issue.

**S18 baseline is fully reproducible.** Seeds 0-4 from our fresh S18 run
match the Sprint 18 scorecard results exactly (within rounding). All
learned strategies produce identical outputs when re-run with the same
seed on the same commit.

**Current main does NOT reproduce the Sprint 19 scorecard results.** For
the causal strategy, seeds 0-4 on our fresh run produce different
regret values from the Sprint 19 scorecard, despite running the same
commit (`52f7aef`). For surrogate_only, seeds 0-4 also differ from
the Sprint 19 scorecard values.

However, **current main is deterministic within a single session.**
Running the same seed twice on main produces identical results. The
difference from the Sprint 19 scorecard is because the scorecard was
generated before the Ax path fix (`126d0d8`) was merged. The Sprint 19
scorecard ran benchmarks on pre-review code, then the Ax path fix was
applied during review, changing optimizer behavior. The published
scorecard numbers do not correspond to the merged code.

**Random strategy is identical** between S18 and main (expected, since
random search does not use the optimizer engine).

**Surrogate_only is identical** between S18 and main (expected, since
surrogate_only passes `causal_graph=None`, so the Sprint 19 causal
changes have no effect on this code path).

This means the Sprint 19 scorecard's comparison table (S18 vs S19) was
comparing S18 results against pre-review-fix S19 code. The numbers in
that table are not reproducible from the merged main.

## 4. Base Counterfactual Results (10 Seeds)

### 4a. Summary Table

| Strategy | Budget | S18 Mean | S18 Std | S18 Median | Main Mean | Main Std | Main Median | Delta |
|----------|--------|----------|---------|------------|-----------|----------|-------------|-------|
| causal | 20 | 24.68 | 7.91 | 26.11 | 15.59 | 9.29 | 14.08 | -9.09 |
| causal | 40 | 24.60 | 7.85 | 26.06 | 15.51 | 9.01 | 13.67 | -9.09 |
| causal | 80 | 4.58 | 4.64 | 2.76 | 11.10 | 10.19 | 12.53 | +6.52 |
| random | 20 | 20.58 | 11.34 | 18.79 | 20.58 | 11.34 | 18.79 | +0.00 |
| random | 40 | 12.75 | 9.27 | 9.40 | 12.75 | 9.27 | 9.40 | +0.00 |
| random | 80 | 7.77 | 2.83 | 8.19 | 7.77 | 2.83 | 8.19 | +0.00 |
| surrogate_only | 20 | 19.13 | 9.50 | 20.61 | 19.13 | 9.50 | 20.61 | +0.00 |
| surrogate_only | 40 | 18.58 | 10.05 | 20.28 | 18.58 | 10.05 | 20.28 | +0.00 |
| surrogate_only | 80 | 2.16 | 1.19 | 3.06 | 2.16 | 1.19 | 3.06 | +0.00 |

### 4b. Key Observations

1. **Random and surrogate_only are identical** across versions (delta =
   0.00 at all budgets). Only the causal strategy changed.

2. **Causal improved at B20 and B40** (mean regret dropped from 24.68 to
   15.59 at B20, from 24.60 to 15.51 at B40).

3. **Causal regressed at B80** (mean regret increased from 4.58 to 11.10,
   median from 2.76 to 12.53). The median is higher than the mean on main,
   indicating that the majority of seeds perform badly. This is the opposite
   of what the Sprint 19 scorecard reported (which claimed improvement from
   2.46 to 0.98 at B80).

4. **The B80 causal distribution is bimodal.** Sorted regret values:
   `[0.36, 0.58, 1.04, 1.70, 10.16, 14.89, 14.99, 15.64, 17.57, 34.10]`.
   Four seeds achieve near-oracle performance (regret < 3), while six
   seeds have catastrophically high regret (10-34). In contrast,
   surrogate_only at B80 is stable: all 10 seeds between 0.35 and 3.19.

5. **Surrogate_only at B80 is the most consistent strategy.**
   Mean = 2.16, std = 1.19, all seeds in a tight range.

### 4c. Head-to-Head Win Rates

**Current main (causal vs surrogate_only):**

| Budget | Causal Wins | Surrogate Wins | Ties |
|--------|-------------|----------------|------|
| B20 | 7/10 (70%) | 3/10 (30%) | 0 |
| B40 | 7/10 (70%) | 3/10 (30%) | 0 |
| B80 | 3/10 (30%) | 7/10 (70%) | 0 |

**S18 baseline (causal vs surrogate_only):**

| Budget | Causal Wins | Surrogate Wins | Ties |
|--------|-------------|----------------|------|
| B20 | 0/10 (0%) | 10/10 (100%) | 0 |
| B40 | 1/10 (10%) | 9/10 (90%) | 0 |
| B80 | 3/10 (30%) | 7/10 (70%) | 0 |

The Sprint 19 changes made causal more competitive at low budgets
(B20, B40) but did not change the B80 picture: surrogate_only still
wins 7/10 seeds at B80 on both versions.

### 4d. Statistical Significance (Mann-Whitney U)

| Comparison | Budget | U | p-value | Significant? |
|------------|--------|---|---------|--------------|
| Main: causal vs surrogate_only | 20 | 41 | 0.5202 | No |
| Main: causal vs surrogate_only | 40 | 39 | 0.4272 | No |
| Main: causal vs surrogate_only | 80 | 70 | 0.1402 | No |
| S18: causal vs surrogate_only | 20 | 72 | 0.1036 | No |
| S18: causal vs surrogate_only | 40 | 72 | 0.1037 | No |
| S18: causal vs surrogate_only | 80 | 60 | 0.4723 | No |

**No comparison reaches p < 0.05.** With 10 seeds, the variance in
both strategies is too high to reliably separate them.

### 4e. Rank Stability

**Current main:**

| Budget | Strategy | Mean Rank | #1 | #2 | #3 |
|--------|----------|-----------|----|----|-----|
| 20 | causal | 1.60 | 5 | 4 | 1 |
| 20 | random | 2.30 | 2 | 3 | 5 |
| 20 | surrogate_only | 2.10 | 3 | 3 | 4 |
| 40 | causal | 1.80 | 3 | 6 | 1 |
| 40 | random | 1.90 | 4 | 3 | 3 |
| 40 | surrogate_only | 2.30 | 3 | 1 | 6 |
| 80 | causal | 2.30 | 3 | 1 | 6 |
| 80 | random | 2.30 | 1 | 5 | 4 |
| 80 | surrogate_only | 1.40 | 6 | 4 | 0 |

**S18 baseline:**

| Budget | Strategy | Mean Rank | #1 | #2 | #3 |
|--------|----------|-----------|----|----|-----|
| 20 | causal | 2.60 | 0 | 4 | 6 |
| 20 | random | 2.00 | 4 | 2 | 4 |
| 20 | surrogate_only | 1.40 | 6 | 4 | 0 |
| 40 | causal | 2.60 | 0 | 4 | 6 |
| 40 | random | 1.70 | 6 | 1 | 3 |
| 40 | surrogate_only | 1.70 | 4 | 5 | 1 |
| 80 | causal | 1.90 | 3 | 5 | 2 |
| 80 | random | 2.70 | 1 | 1 | 8 |
| 80 | surrogate_only | 1.40 | 6 | 4 | 0 |

Rank stability is weak for all learned strategies. Causal is rank 1
on 5/10 seeds at B20 (main) but rank 3 on 6/10 seeds at B80 (main).
No strategy holds a consistent rank advantage across budgets.

### 4f. Per-Seed Comparison (Causal B80)

| Seed | S18 Causal | Main Causal | S18 Surrogate | Main Surrogate |
|------|-----------|-------------|--------------|----------------|
| 0 | 0.91 | 0.36 | 3.10 | 3.10 |
| 1 | 2.46 | 1.04 | 3.11 | 3.11 |
| 2 | 2.44 | 34.10 | 0.64 | 0.64 |
| 3 | 3.42 | 17.57 | 0.95 | 0.95 |
| 4 | 3.06 | 10.16 | 0.95 | 0.95 |
| 5 | 1.59 | 1.70 | 0.35 | 0.35 |
| 6 | 0.75 | 0.58 | 3.18 | 3.18 |
| 7 | 6.32 | 14.99 | 3.06 | 3.06 |
| 8 | 7.98 | 14.89 | 3.06 | 3.06 |
| 9 | 16.85 | 15.64 | 3.19 | 3.19 |

The Sprint 19 optimizer-core changes made causal better on some seeds
(0, 1, 6) and dramatically worse on others (2, 3, 4, 7, 8). The
bimodal failure pattern is new to the Sprint 19 code.

## 5. High-Noise Variant Results (Main Only, 10 Seeds)

The high-noise variant (15 dimensions: 3 causal + 2 noise + 10 nuisance)
cannot be run on S18 (variant was added in Sprint 19).

### 5a. Summary Table

| Strategy | Budget | Mean | Std | Median | Min | Max |
|----------|--------|------|-----|--------|-----|-----|
| causal | 20 | 21.52 | 8.44 | 19.98 | 10.07 | 39.62 |
| causal | 40 | 20.16 | 8.33 | 19.76 | 9.75 | 39.62 |
| causal | 80 | 10.49 | 6.34 | 14.84 | 1.69 | 16.65 |
| random | 20 | 23.99 | 8.95 | 21.71 | 10.03 | 43.51 |
| random | 40 | 18.14 | 7.90 | 18.12 | 4.25 | 31.22 |
| random | 80 | 10.71 | 3.53 | 9.93 | 4.25 | 16.29 |
| surrogate_only | 20 | 26.86 | 4.56 | 28.37 | 20.61 | 31.16 |
| surrogate_only | 40 | 26.37 | 4.98 | 28.29 | 20.59 | 31.16 |
| surrogate_only | 80 | 15.67 | 11.74 | 16.78 | 1.20 | 28.66 |

### 5b. Win Rates and Statistical Tests

| Budget | Causal Wins vs Surr | p-value (MWU) |
|--------|---------------------|---------------|
| B20 | 6/10 | 0.0746 (ns) |
| B40 | 6/10 | 0.0875 (ns) |
| B80 | 6/10 | 0.4723 (ns) |

No comparison reaches statistical significance. Causal has a slight
edge in win frequency (6/10 at every budget), but the variance is too
high for the advantage to be reliable.

### 5c. Sprint 19 Scorecard vs 10-Seed Reality

The Sprint 19 scorecard (5 seeds) reported causal B80 mean regret =
3.47 (std 5.66) vs surrogate_only = 8.74 (std 10.88), a "60% lower
regret" claim. Our 10-seed run shows causal B80 mean = 10.49 (std 6.34)
vs surrogate_only = 15.67 (std 11.74). The direction is the same
(causal < surrogate_only) but the gap is smaller (33% vs 60%) and not
statistically significant.

The Sprint 19 scorecard overestimated the causal advantage because:
1. It used only 5 seeds (too few for the observed variance).
2. The 5-seed sample happened to favor causal.

### 5d. High-Noise Assessment

On the high-noise variant, causal shows a **directional** advantage
over surrogate_only (lower mean regret at all budgets, 6/10 win rate),
but the advantage is not statistically significant with 10 seeds. The
15-dimensional search space does create a harder problem where causal
graph pruning plausibly helps, but the current optimizer is too
variable to produce a reliable signal.

## 6. Is the Surrogate_Only Regression Systematic or Stochastic?

The Sprint 19 scorecard reported surrogate_only regressing at B20 and
B40 relative to S18. This audit finds:

**Surrogate_only is identical between S18 and main.** The Sprint 19
code changes do not affect surrogate_only behavior when
`causal_graph=None`. The surrogate_only "regression" reported in the
Sprint 19 scorecard was an artifact of comparing against pre-review-fix
code (which produced different surrogate_only values). In the merged
code, surrogate_only is completely unchanged.

## 7. Stability Verdict

**FRAGILE.**

The evidence does not support a stable causal advantage:

1. **No comparison reaches statistical significance** (all p > 0.05,
   Mann-Whitney U, 10 seeds).

2. **Causal regressed at B80** compared to S18 (mean regret 11.10 vs
   4.58), the budget level where the Sprint 19 scorecard claimed the
   strongest win.

3. **Causal has a bimodal failure mode at B80** (4 seeds excellent,
   6 seeds catastrophic). Surrogate_only is much more consistent.

4. **The Sprint 19 scorecard numbers are not reproducible** from the
   merged main code. The scorecard was generated on pre-review-fix code,
   then the Ax path fix changed optimizer behavior.

5. **High-noise advantage is directional but not significant.** Causal
   wins 6/10 seeds at every budget, but p > 0.07 everywhere.

6. **Causal did improve at low budgets (B20, B40)** relative to S18.
   The Sprint 19 soft-influence changes moved the causal strategy from
   "always worse than surrogate_only" at low budgets to "competitive."
   This is genuine progress but not enough to claim a reliable advantage.

## 8. Implications for Ax Re-Ranking Workstream

The bimodal failure pattern at B80 is the most actionable finding.
When the Ax path generates multiple candidates and re-ranks by causal
alignment, it sometimes selects candidates that score well on alignment
but poorly on objective quality. This produces catastrophic regret on
seeds where the alignment signal is misleading.

Recommendations for the Ax re-ranking workstream:

1. **The balanced composite score (objective + alignment) is the right
   fix.** The current alignment-only re-ranking is the proximate cause
   of the bimodal failures. Blending GP-predicted objective value with
   alignment should prevent selecting poor-objective candidates.

2. **Test with 10+ seeds, not 5.** The 5-seed results that motivated
   Sprint 19's optimism were not representative. Any future optimizer
   change should be evaluated on at least 10 seeds.

3. **Track B80 variance specifically.** The B80 bimodal pattern is the
   clearest failure mode. Success criteria for the Ax fix should include
   reducing B80 causal std below 5.0 (currently 10.19).

4. **Do not weaken surrogate_only.** Surrogate_only at B80 (mean 2.16,
   std 1.19) is the reliability benchmark. Any optimizer change that
   makes surrogate_only worse is a regression.

## 9. Artifacts

| File | Description |
|------|-------------|
| `artifacts/stability_base_s18.json` | S18 baseline, base variant, 10 seeds |
| `artifacts/stability_base_main.json` | Current main, base variant, 10 seeds |
| `artifacts/stability_high_noise_main.json` | Current main, high-noise variant, 10 seeds |
| `artifacts/stability_audit_summary.csv` | Summary CSV of mean/std/delta |

Artifact files are stored locally at
`/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/`
(not committed to the repository) per project convention.

## 10. Summary

| Question | Answer |
|----------|--------|
| Does the causal win survive a wider seed sweep? | **No.** It reverses at B80. |
| Is the Sprint 19 causal advantage stable? | **No.** Fragile -- bimodal at B80. |
| Is the surrogate_only regression systematic? | **No.** It was an artifact of pre-review-fix code. |
| Is causal better than S18 causal? | **Mixed.** Better at B20/B40, worse at B80. |
| Is the high-noise advantage real? | **Directional but not significant.** |
| Should Sprint 20 proceed with Ax re-ranking? | **Yes.** The bimodal failure mode is addressable. |
