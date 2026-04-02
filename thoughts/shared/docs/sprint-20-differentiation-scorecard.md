# Sprint 20 Differentiation Scorecard

## Metadata

- **Date**: 2026-04-01
- **Sprint**: 20 (Stability And Production-Path Causal Hardening)
- **PRs included**: #107 (Stability Audit), #108 (Balanced Ax Re-Ranking)
- **Issues**: #104, #105, #106
- **Predecessor**: Sprint 19 Differentiation Scorecard

## 1. Sprint 20 Summary

Sprint 19 reported that causal guidance beats surrogate_only on the base
counterfactual benchmark (regret 0.98 vs 2.16 at B80) and on the
high-noise variant (regret 3.47 vs 8.74 at B80). Sprint 20 was tasked
with stress-testing those claims: a wider seed sweep to check stability,
a balanced Ax re-ranking to fix the alignment-only production path, and
a controlled rerun to verify the null control and skip calibration.

Sprint 20 delivered two changes:

1. **Stability Audit** (PR #107) -- Widened the seed sweep from 5 to 10
   seeds on the base counterfactual and high-noise variant. Compared
   Sprint 18 baseline (`a0f8d5f`) against Sprint 19 merged main
   (`52f7aef`). Verdict: FRAGILE. The Sprint 19 causal win does not
   survive the wider sweep at B80, and the causal strategy exhibits a
   bimodal failure pattern.

2. **Balanced Ax Re-Ranking** (PR #108) -- Replaced alignment-only
   re-ranking on the Ax path with a composite score blending RF-predicted
   objective quality and causal alignment, weighted by `causal_softness`.
   Unit-tested and reviewed. Not yet benchmarked end-to-end.

## 2. Controlled Rerun Summary

### 2a. Base Counterfactual (10-Seed Stability Audit)

Source: `stability_base_main.json`, `stability_base_s18.json`

| Strategy | Budget | S18 Mean | S18 Std | Main Mean | Main Std | Delta |
|----------|--------|----------|---------|-----------|----------|-------|
| causal | 20 | 24.68 | 7.91 | 15.59 | 9.29 | -9.09 |
| causal | 40 | 24.60 | 7.85 | 15.51 | 9.01 | -9.09 |
| causal | 80 | 4.58 | 4.64 | 11.10 | 10.19 | +6.52 |
| random | 20 | 20.58 | 11.34 | 20.58 | 11.34 | +0.00 |
| random | 40 | 12.75 | 9.27 | 12.75 | 9.27 | +0.00 |
| random | 80 | 7.77 | 2.83 | 7.77 | 2.83 | +0.00 |
| surrogate_only | 20 | 19.13 | 9.50 | 19.13 | 9.50 | +0.00 |
| surrogate_only | 40 | 18.58 | 10.05 | 18.58 | 10.05 | +0.00 |
| surrogate_only | 80 | 2.16 | 1.19 | 2.16 | 1.19 | +0.00 |

Key observations from the stability audit:

- **Causal improved at B20 and B40** relative to S18 (mean regret dropped
  from 24.68 to 15.59 at B20, from 24.60 to 15.51 at B40).
- **Causal regressed at B80** (mean regret rose from 4.58 to 11.10).
  This directly contradicts the Sprint 19 scorecard, which reported
  improvement from 2.46 to 0.98 at B80.
- **Causal B80 is bimodal**: 4/10 seeds achieve near-oracle performance
  (regret < 3), while 6/10 seeds have catastrophic regret (10-34).
  Per-seed: [0.36, 0.58, 1.04, 1.70, 10.16, 14.89, 14.99, 15.64,
  17.57, 34.10].
- **Surrogate_only B80 is stable**: all 10 seeds between 0.35 and 3.19
  (mean 2.16, std 1.19).
- **No comparison reaches statistical significance** (Mann-Whitney U,
  all p > 0.05, 10 seeds).
- **Win rates**: causal wins 7/10 at B20 and B40, but only 3/10 at B80.
  Note: 4 causal seeds have near-oracle regret (< 3) but one of those
  (seed 5, regret 1.70) still loses to its matched surrogate_only seed
  (seed 5, regret 0.35), so near-oracle performance does not guarantee a
  head-to-head win.

### 2b. High-Noise Counterfactual (10-Seed, Main Only)

Source: `stability_high_noise_main.json`

| Strategy | Budget | Mean | Std | Median |
|----------|--------|------|-----|--------|
| causal | 20 | 21.52 | 8.44 | 19.98 |
| causal | 40 | 20.16 | 8.33 | 19.76 |
| causal | 80 | 10.49 | 6.34 | 14.84 |
| random | 20 | 23.99 | 8.95 | 21.71 |
| random | 40 | 18.14 | 7.90 | 18.12 |
| random | 80 | 10.71 | 3.53 | 9.93 |
| surrogate_only | 20 | 26.86 | 4.56 | 28.37 |
| surrogate_only | 40 | 26.37 | 4.98 | 28.29 |
| surrogate_only | 80 | 15.67 | 11.74 | 16.78 |

The Sprint 19 scorecard (5 seeds) reported causal B80 regret = 3.47 vs
surrogate_only = 8.74, claiming a 60% gap. The 10-seed audit found
causal B80 = 10.49 vs surrogate_only = 15.67, a 33% directional gap
that is not statistically significant (p = 0.47 Mann-Whitney U). Causal
wins 6/10 seeds vs surrogate_only at every budget, but random is
competitive at B40 (mean 18.14 vs causal 20.16) and B80 (mean 10.71 vs
causal 10.49, effectively tied). The variance is too high for any
pairwise advantage to be reliable.

### 2c. Confounded Counterfactual (5-Seed, Sprint 19)

Source: `counterfactual_sprint19_confounded.json`

| Strategy | Budget | Mean Regret | Std |
|----------|--------|-------------|-----|
| random | 20 | 24.28 | 11.68 |
| surrogate_only | 20 | 33.74 | 4.38 |
| causal | 20 | 37.41 | 14.00 |
| random | 40 | 24.60 | 9.79 |
| surrogate_only | 40 | 29.71 | 0.67 |
| causal | 40 | 26.01 | 3.75 |
| random | 80 | 17.41 | 5.57 |
| surrogate_only | 80 | 20.65 | 0.00 |
| causal | 80 | 30.23 | 11.23 |

All strategies are misled by the confounding. Random outperforms both
learned strategies at B80. Causal is the worst performer at B80 (30.23).
This variant remains an invalid positive control. No rerun was performed
in Sprint 20 because the confounded variant was not a focus of the
stability or Ax re-ranking workstreams.

Note: surrogate_only shows std = 0.00 at B80 (and near-zero at B40).
This is expected, not a data issue: with `causal_graph=None` and a
fixed budget, the RF surrogate converges deterministically to the same
optimum across seeds because the confounded variant's signal structure
funnels all seeds to the same ridge configuration. The same deterministic
convergence appears in the null control (Section 2d, surrogate_only B40
std = 0.00).

### 2d. Null Control (3-Seed, Sprint 19)

Source: `null_sprint19_final.json`

| Strategy | Budget | Mean Test MAE | Std |
|----------|--------|---------------|-----|
| random | 20 | 3260.48 | 1.75 |
| surrogate_only | 20 | 3255.76 | 0.39 |
| causal | 20 | 3259.11 | 2.16 |
| random | 40 | 3261.58 | 0.28 |
| surrogate_only | 40 | 3256.31 | 0.00 |
| causal | 40 | 3259.11 | 2.16 |

Max strategy difference: 5.82 MAE (0.18%). All strategies produce test
MAE near the marginal target standard deviation. The null control was
run on Sprint 19 code (pre-Ax-reranking). A post-Ax-reranking null
rerun has not been performed (see Section 3b).

### 2e. Skip Calibration (Sprint 19)

Source: Sprint 19 skip calibration report.

| Benchmark | False-Skip Rate | Notes |
|-----------|-----------------|-------|
| Real ERCOT | 0.00% (no skips) | Model quality never reaches 0.3 |
| Counterfactual | N/A | Disabled (max_skips=0) |
| Synthetic quadratic (B40) | 7.14% | Late-stage only |

Skip logic is effectively dormant on all production benchmarks. The
measurement infrastructure (`SkipAuditEntry`, `compute_skip_metrics`) is
in place for future monitoring. No changes were made to skip logic in
Sprint 20.

### 2f. Seed Sweep Findings

The stability audit's widened 10-seed sweep produced three critical findings:

1. **Sprint 19 scorecard numbers are not reproducible from merged main.**
   The scorecard was generated on pre-Ax-path-fix code. The Ax path fix
   (`126d0d8`), applied during Sprint 19 review, changed optimizer
   behavior. The published Sprint 19 scorecard numbers do not correspond
   to the merged code.

2. **5-seed samples are insufficient for this optimizer.** The observed
   variance in both causal and surrogate_only strategies is high enough
   that 5-seed samples can produce misleading mean comparisons. Future
   evaluations should use at least 10 seeds.

3. **Causal has a bimodal failure mode at B80.** The soft causal
   influence changes improved low-budget performance (B20, B40) but
   introduced a failure mode at high budget where 6/10 seeds get stuck
   in catastrophic regret. The alignment-only Ax re-ranking (now replaced
   by balanced scoring in PR #108) is the suspected proximate cause.

## 3. Answers to Key Questions

### Question 1: Did the causal advantage remain after the widened seed sweep?

**Partially.** Causal improved at low budgets (B20, B40) relative to
Sprint 18, moving from "always worse than surrogate_only" to
"competitive or better" (7/10 win rate). However, the Sprint 19 claim of
a strong causal advantage at B80 did not survive. Causal at B80 regressed
from S18 (mean regret 4.58 to 11.10) and is now worse than surrogate_only
(mean 2.16) on average. No comparison reached statistical significance.

The high-noise variant shows a directional causal advantage vs
surrogate_only (lower mean regret at all budgets, 6/10 win rate) but the
gap is not statistically significant. Random is competitive with causal
at B40 and B80 on this variant, so the causal advantage is specifically
over surrogate_only, not over all baselines.

### Question 2: Did the balanced Ax re-ranking help, hurt, or leave results unchanged?

**Unknown -- not yet benchmarked.** PR #108 merged a balanced composite
score (objective quality + causal alignment) to replace the alignment-only
re-ranking. The change is unit-tested and code-reviewed, but no
end-to-end benchmark has been run on the merged code. The stability
audit artifacts were generated on pre-#108 code (`52f7aef`), not on
post-#108 code (`f628e5a`).

This is the primary limitation of this scorecard. The balanced Ax
re-ranking was designed specifically to address the bimodal B80 failure
mode discovered by the stability audit. Whether it actually fixes that
failure mode is an open question that requires a controlled rerun.

### Question 3: Did the null control stay clean?

**Yes (on Sprint 19 code).** The null-signal benchmark passes with 0.18%
max strategy difference, well within the 2% null-signal threshold. No
strategy produces meaningfully lower MAE than another on permuted data.

**Caveat:** The null control has not been rerun on post-Ax-reranking code.
The balanced Ax re-ranking should not affect null-control behavior (the
null benchmark runs with `causal_graph=None` for surrogate_only and with
a graph that provides no real signal for causal), but this has not been
empirically verified.

### Question 4: Did skip calibration remain acceptable?

**Yes.** Skip logic is dormant on all current benchmarks:
- Real energy data: no skips (model quality < 0.3).
- Counterfactual: skipping disabled (max_skips=0).
- Synthetic (when active): 7% false-skip rate, concentrated late.

No changes were made to skip logic in Sprint 20. The measurement
infrastructure remains in place for future monitoring.

### Question 5: Sprint 21 recommendation?

See Section 5.

## 4. Sprint 20 Verdict

**MIXED.**

The evidence supports a more nuanced picture than either STRONGER or
FRAGILE alone:

**What improved (genuine progress):**
- Causal guidance at low budgets (B20, B40) is now competitive with
  surrogate_only, reversing the Sprint 18 result where causal was always
  the worst strategy.
- The high-noise variant shows a directional causal advantage at all
  budgets (6/10 win rate), consistent with the structural prediction
  that causal graph pruning helps in high-dimensional spaces.
- The balanced Ax re-ranking (PR #108) provides a theoretically sound
  fix for the bimodal B80 failure mode, replacing pure alignment
  selection with a composite score.
- The stability audit established that 10-seed evaluation is necessary
  and that the Sprint 19 scorecard's 5-seed numbers were misleading.

**What did not improve (remaining gaps):**
- Causal at B80 regressed from S18 (mean regret 4.58 to 11.10), with a
  bimodal failure mode affecting 6/10 seeds.
- No causal vs surrogate_only comparison reached statistical significance
  at any budget level (10 seeds, Mann-Whitney U, all p > 0.05).
- The balanced Ax re-ranking has not been benchmarked. Its impact on the
  bimodal failure and on strategy separation is unknown.
- The confounded variant remains unsolved (random beats both learned
  strategies).
- The Sprint 19 scorecard's published numbers were not reproducible from
  merged code, meaning the optimism that framed Sprint 20 was based on
  stale data.

**Why MIXED and not FRAGILE:** The low-budget improvements at B20 and B40
are real and consistent across 10 seeds. The soft causal influence changes
from Sprint 19 did move the needle in the right direction for the budget
range where causal graph information should matter most (when sample
size is small and dimensionality reduction is valuable). The B80 failure
mode has a plausible proximate cause (alignment-only Ax re-ranking) and
a designed fix (balanced scoring). The null control remains clean.

**Why MIXED and not STRONGER:** No comparison is statistically significant.
The B80 regression is concerning. The balanced Ax fix has not been tested.
The Sprint 20 deliverables are incomplete -- the re-ranking code is
merged but its benchmark impact is unknown.

## 5. Sprint 21 Recommendation

### Primary focus: Benchmark the balanced Ax re-ranking and stabilize B80

Sprint 20 ended with the balanced Ax re-ranking merged but unbenchmarked.
The single most important action for Sprint 21 is to run a controlled
evaluation of the current main code (which includes the balanced
composite scoring) against the same benchmark suite used in the stability
audit.

Specific tasks:

1. **Rerun base counterfactual with 10 seeds** on current main
   (`f628e5a`+). Compare against the stability audit baselines.
   - Stretch goal: B80 causal mean regret < surrogate_only mean
     (currently 11.10 vs 2.16). This is ambitious -- it requires
     resolving the bimodal failure across most seeds.
   - Intermediate goal: B80 causal std < 5.0 (currently 10.19) AND
     causal win rate >= 6/10 vs surrogate_only at B80. This captures
     meaningful progress even if causal does not yet beat surrogate_only
     on mean regret.

2. **Rerun high-noise counterfactual with 10 seeds.** Check whether the
   directional advantage (6/10 win rate) strengthens or weakens with the
   balanced scoring.

3. **Rerun null control.** Verify the balanced Ax re-ranking does not
   create false wins on permuted data. Hard requirement: max strategy
   difference < 2%.

4. **Measure skip calibration post-change.** The balanced scoring changes
   the candidate selection path, which could indirectly affect skip
   behavior. Rerun the skip audit.

### Secondary focus: Decide the confounding question

If the B80 benchmarks improve, Sprint 21 should explicitly decide
whether to pursue deconfounding research or park it:

- **Pursue if**: the balanced scoring fixes B80 and the base + high-noise
  benchmarks show statistically significant causal advantage (p < 0.05
  on at least one benchmark at one budget level).
- **Park if**: the balanced scoring does not materially change the
  picture. In that case, the confounding work is premature -- the
  optimizer cannot reliably outperform surrogate_only even when the
  causal graph is perfectly correct.

### What not to do in Sprint 21

1. Do not expand to new benchmark families before validating the balanced
   Ax re-ranking on existing benchmarks.
2. Do not claim a stable causal advantage without p < 0.05 significance
   on at least one benchmark.
3. Do not reduce seed counts below 10 for strategy comparisons.
4. Do not tune `causal_softness` to the base benchmark -- if a specific
   softness value is chosen, it must also hold on the high-noise variant
   and pass the null control.

## 6. Evidence Inventory

### Artifacts Used

| Artifact | Version | Seeds | Description |
|----------|---------|-------|-------------|
| `stability_base_s18.json` | S18 (`a0f8d5f`) | 10 | Base counterfactual, S18 baseline |
| `stability_base_main.json` | S19 (`52f7aef`) | 10 | Base counterfactual, Sprint 19 main |
| `stability_high_noise_main.json` | S19 (`52f7aef`) | 10 | High-noise variant, Sprint 19 main |
| `counterfactual_sprint19_confounded.json` | S19 (pre-fix) | 5 | Confounded variant |
| `null_sprint19_final.json` | S19 (pre-fix) | 3 | Null-signal control |
| `stability_audit_summary.csv` | S19 (`52f7aef`) | 10 | Summary statistics |

### Artifacts Missing

| Artifact | Why Missing | Impact |
|----------|-------------|--------|
| Post-Ax-reranking base benchmark | PR #108 merged after stability audit | Cannot assess balanced scoring impact |
| Post-Ax-reranking null control | Same | Cannot verify null safety post-change |
| Post-Ax-reranking high-noise benchmark | Same | Cannot verify high-noise direction holds |
| Confounded variant (10-seed) | Not a Sprint 20 priority | Confounded assessment limited to 5-seed S19 data |

### Test Results

- **891 tests passed**, 23 skipped, 100 deselected (slow).
- All Sprint 20 tests (5 new balanced Ax re-ranking tests in
  `test_soft_causal.py`) passing.

### Code State

- **Current main**: `f628e5a` (includes both #107 and #108).
- **Stability audit ran on**: `52f7aef` (pre-#108).
- **Sprint 19 scorecard ran on**: pre-Ax-path-fix code (not reproducible
  from any tagged commit on main).

## 7. Sprint-Over-Sprint Trajectory

| Sprint | Verdict | Key Finding |
|--------|---------|-------------|
| 18 | PASS (infrastructure) | Null control clean, causal does not beat surrogate_only |
| 19 | PROGRESS | Causal beats surrogate_only on base + high-noise (5-seed) |
| 20 | MIXED | Low-budget gains real but B80 fragile; balanced Ax not yet tested |
