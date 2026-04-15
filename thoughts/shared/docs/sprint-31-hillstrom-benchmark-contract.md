# Sprint 31 Hillstrom Benchmark Contract

**Date:** 2026-04-14
**Sprint:** 31 (General Causal Autoresearch: First Non-Energy Real-Data Benchmark)
**Issue:** follow-on to [22-sprint-31-generalization-research-plan.md](../plans/22-sprint-31-generalization-research-plan.md)
**Primary source:** <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
**Status:** Contract and execution brief. Not an implementation PR.

## 0. Purpose Of This Doc

This document is the launch contract for the project's first non-energy
real-data benchmark. It does **not** implement the benchmark. It pins:

1. what dataset slice to run first
2. why that slice is the smallest credible answer
3. how to map Hillstrom columns into the existing `MarketingLogAdapter`
4. what reuses cleanly, what needs a small wrapper, and what should wait
5. the evidence contract the first run must satisfy
6. the follow-on dataset queue after Hillstrom

The binding constraint is the one stated in the Sprint 31 plan and the
Sprint 30 portability brief: the project should produce a credible
non-energy answer as quickly as possible, without committing to a
multi-action adapter rewrite on the first dataset.

## 1. Suitability Assessment

### 1a. Why Hillstrom Is The Right First Non-Energy Benchmark

1. **Intervention semantics, not passive forecasting.** Hillstrom is a
   randomized e-mail marketing experiment with explicit treatment
   assignment and observed response. It matches the engine's
   intervention framing rather than fighting it.
2. **Randomized, not observational.** Treatment was randomized across
   roughly one-third mens e-mail, one-third womens e-mail, and
   one-third no e-mail. Propensities are known and balanced, so the IPS
   stack is not load-bearing for identification on the first run.
3. **Small and self-contained.** 64,000 rows is trivial to cache
   locally, does not require batching, and lets the benchmark iterate
   fast. The contract can be debugged in minutes per run, not hours.
4. **Closest open fit to the existing adapter.** The binary-treatment
   / continuous-outcome / known-propensity / cost-constrained shape
   maps directly onto `MarketingLogAdapter` after a narrow column
   wrapper. No new architecture is required for a binary launch
   contract.
5. **Honest negative result is still useful.** If causal fails to
   separate from surrogate-only on Hillstrom, the failure will be
   diagnosable — dataset is open, randomized, and well studied — and
   will cleanly localize whether the gap is optimizer-side, adapter-side,
   or IPS variance-side.
6. **Creates a ramp, not a dead end.** Hillstrom is a stepping stone to
   Criteo Uplift (larger, same framing) and Open Bandit Dataset
   (multi-action, architecture-expansion). Starting here gives the
   project a disciplined path into harder marketing and bandit
   benchmarks without skipping the credibility step.

### 1b. Why Hillstrom Is Not Trivial

1. **Three arms, not two.** The original dataset is `Mens E-Mail`,
   `Womens E-Mail`, `No E-Mail`. The current `MarketingLogAdapter`
   enforces binary treatment (`{0, 1}`). Every launch choice has to
   collapse the three arms into a binary comparison or wait for a
   multi-action adapter.
2. **No per-customer cost column.** Hillstrom ships with `visit`,
   `conversion`, and `spend` outcome columns but no send cost. The
   adapter requires a cost column. A small wrapper has to assign a
   fixed per-send cost (marketing-realistic, but a modeling decision).
3. **No segmentation aligned with the adapter's segment scoring.** The
   adapter scores `"high_value" / "medium" / "low"` segments. Hillstrom
   ships `history_segment` (spend deciles) and `history` (continuous
   365-day spend). The wrapper has to synthesize a `segment` column.
4. **No channel heterogeneity on the treatment side.** All treatment is
   e-mail, while the adapter's search space spans `email_share /
   social_share / search_share`. For a single-channel launch, those
   variables are degenerate and should be fixed, not tuned.
5. **Known non-uniform effect across arms.** Published Hillstrom
   analyses find the womens arm has a stronger visit and conversion
   lift than the mens arm. Pooling the two arms dilutes the effect.
   That has to inform the launch slice decision.

### 1c. Overall Verdict

Hillstrom is **well suited** as the first open non-energy benchmark,
provided the launch contract is deliberately narrow and does not try to
exercise the full adapter surface on day one.

## 2. Launch Contract Recommendation

### 2a. Decision Space

Three launch shapes are credible:

1. `Womens E-Mail vs No E-Mail` (single arm, strongest effect)
2. `Mens E-Mail vs No E-Mail` (single arm, weaker effect)
3. Pooled `(Mens + Womens) E-Mail vs No E-Mail` (binary collapse of
   the full dataset)

A fourth shape — full three-arm multi-action policy — is deferred.

### 2b. Recommendation

**Launch with `Womens E-Mail vs No E-Mail` as the primary slice, and
carry pooled `Any E-Mail vs No E-Mail` as a secondary reference slice
inside the same run.**

One primary slice, one secondary slice, one null control, 10 seeds,
3 budgets. Nothing else in the first run.

### 2c. Why This Slice

1. **Strongest defensible signal on the first try.** The womens arm is
   the published strongest-effect subset of Hillstrom. Starting there
   maximizes the probability that a real causal advantage, if present,
   is visible at `B80` instead of being buried in sampling noise.
2. **Still a proper randomized comparison.** Dropping the mens arm
   leaves ~42,600 rows split roughly evenly between treated and
   control. That is more than enough for stable IPS weighting on the
   adapter's 6-variable search space.
3. **Minimizes false-negative risk without overclaiming.** A certified
   Womens-arm win will be reported as "causal found a better policy on
   the Hillstrom womens slice", not "causal solved marketing." The
   pooled shadow slice exists precisely to test whether the result
   survives averaging across both arms.
4. **Leaves mens arm available as a follow-on ablation.** If the
   womens slice is a clean win, rerunning the identical contract on
   the mens arm becomes a one-command generalization test inside
   Sprint 31 or Sprint 32.
5. **Does not require any multi-action adapter work.** Binary
   treatment, binary comparison, single e-mail channel. Zero
   architecture risk on the first non-energy answer.

### 2d. What The Secondary Pooled Slice Adds

The pooled `(Mens + Womens) E-Mail vs No E-Mail` slice is included
because it answers a different question:

1. Does the causal advantage on the womens slice survive when the
   second arm is folded in as "more data"?
2. Does pooling wash the effect out, or does it concentrate it?
3. Does the optimizer discover that the effective lever is customer
   targeting rather than blanket treatment?

The pooled slice is **not** a positive certification target — a
Sprint 31 "certified causal win" claim is only ever attached to the
primary womens slice. The pooled slice functions as a **one-way
veto gate**: it cannot upgrade a near-parity primary result to a
certified win, but a statistically significant surrogate-only
advantage on the pooled slice (see Section 5h criterion 4) *can*
block certification of the primary result. The primary claim
language is locked to the womens slice in all cases.

### 2e. Explicit Exclusions

The first run must not:

1. attempt a three-arm policy (needs multi-action adapter)
2. attempt cross-channel allocation (all treatment is e-mail)
3. attempt to tune cost assumptions as part of the search space
4. attempt to learn propensities (they are known and randomized)
5. attempt any customer-level uplift model (out of scope for the
   policy-evaluation adapter)

## 3. Hillstrom To MarketingLogAdapter Column Mapping

### 3a. Hillstrom Columns

The canonical Hillstrom CSV has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `recency` | int | Months since last purchase |
| `history_segment` | str | Spend bucket label (e.g., `"1) $0 - $100"`) |
| `history` | float | Dollar value of past-year purchases |
| `mens` | int (0/1) | Indicator: purchased mens merchandise |
| `womens` | int (0/1) | Indicator: purchased womens merchandise |
| `zip_code` | str | `"Urban" / "Suburban" / "Rural"` |
| `newbie` | int (0/1) | New customer in last twelve months |
| `channel` | str | Past purchase channel: `"Phone" / "Web" / "Multichannel"` |
| `segment` | str | **Treatment arm:** `"Mens E-Mail" / "Womens E-Mail" / "No E-Mail"` |
| `visit` | int (0/1) | Visited in 2-week window after send |
| `conversion` | int (0/1) | Converted in 2-week window after send |
| `spend` | float | Dollar spend in 2-week window after send |

### 3b. Adapter Required Columns And Hillstrom Source

| Adapter column | Hillstrom source | Transform |
|----------------|------------------|-----------|
| `treatment` (int 0/1) | `segment` | `1` if `"Womens E-Mail"`, `0` if `"No E-Mail"`, drop `"Mens E-Mail"` for the primary slice |
| `outcome` (float) | `spend` | Pass-through. Primary objective is IPS-weighted mean `spend`. |
| `cost` (float) | synthesized | Fixed per-send cost, e.g., `0.10` per treated observation, `0.0` per control. Document in the benchmark report. |

Note on cost: Hillstrom has no per-customer send cost. A fixed constant
is not "made up" — it represents a realistic marketing assumption and
is recorded in provenance. The cost value does not affect the
`policy_value` objective: verified against
`causal_optimizer/domain_adapters/marketing_logs.py` where
`policy_value` is computed purely from IPS-normalized weights and
the `outcome` column (see the `self-normalized IPS-weighted average
outcome` path in the adapter), while `cost` only appears in the
`total_cost` descriptor. Any fixed per-send cost shifts `total_cost`
but leaves `policy_value` unchanged across strategies. The benchmark
should fix the cost constant, not tune it.

### 3c. Adapter Optional Columns And Hillstrom Source

| Adapter column | Hillstrom source | Transform |
|----------------|------------------|-----------|
| `propensity` (float) | known | **Primary `Womens E-Mail vs No E-Mail` slice:** exactly `0.5` on every row. Each of the three Hillstrom arms has `P = 1/3`, so within the two-arm primary slice `P(treated) = (1/3) / (2/3) = 0.5`. **Secondary pooled `Any E-Mail vs No E-Mail` slice:** exactly `2/3` on every row (computed as `2.0 / 3.0`, not a rounded constant), derived from `P(any email) = 1/3 + 1/3 = 2/3` in the full three-arm RCT. Using `0.5` on the pooled slice would produce biased IPS weights (treated over-weighted, control under-weighted) and inflate pooled `policy_value`. The `HillstromLoader` must select the correct constant per slice, not a single global `0.5`. |
| `channel` (str) | constant | Set to `"email"` for all rows. Hillstrom has a `channel` column but it describes the customer's past purchase channel, not the marketing send channel. It must not be used as the adapter's `channel` field. |
| `segment` (str) | `history_segment` | Bucket map using the raw Hillstrom CSV strings (numeric prefix is part of the value): `"5) $500 - $750" / "6) $750 - $1,000" / "7) $1,000 +"` → `"high_value"`; `"3) $200 - $350" / "4) $350 - $500"` → `"medium"`; `"1) $0 - $100" / "2) $100 - $200"` → `"low"`. Document the mapping in the wrapper. The wrapper must match the raw string; stripping the numeric prefix first is also acceptable but must be explicit. |
| `timestamp` (datetime) | not used | Hillstrom does not provide send-time timestamps. Omit. |

### 3d. Features Dropped On First Run

1. `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`,
   `channel` (past purchase) are dropped from the adapter input on the
   first run because they are not referenced by the adapter's policy
   evaluation.
2. These should be preserved in the raw fixture so that a later sprint
   can add segment-conditioned policies without reloading the source.

### 3e. Search Space Treatment

The `MarketingLogAdapter` has 6 continuous variables. On Hillstrom:

| Variable | First-run treatment |
|----------|-------------------|
| `eligibility_threshold` | Tuned by the optimizer (in `[0.0, 1.0]`) |
| `email_share` | **Fixed at `1.0`**. All treatment is e-mail. Not tuned. |
| `social_share_of_remainder` | **Fixed at `0.0`**. Degenerate on Hillstrom. Not tuned. |
| `min_propensity_clip` | **Fixed at `0.01`**. Degenerate on Hillstrom: propensity is a per-slice constant (`0.5` on the primary slice, `2/3` on the pooled slice). Both are ≥ every value in the adapter range `[0.01, 0.5]`, so the floor never rewrites a propensity — the worst case is the boundary `clip = 0.5` on the primary slice, which clips `0.5` to `0.5` (a no-op). The optimizer therefore sees a flat response surface along this dimension on both slices. Not tuned. |
| `regularization` | Tuned (in `[0.001, 10.0]`) |
| `treatment_budget_pct` | Tuned (in `[0.1, 1.0]`) |

The effective search space for the first run is **3 tuned continuous
variables** (`eligibility_threshold`, `regularization`,
`treatment_budget_pct`). This is smaller than any current active
regression gate row and smaller than the dose-response row (6D). That
is intentional: the first non-energy benchmark should not also be the
first high-dimensional one, and dimensions that are degenerate on the
dataset should be pre-baked rather than wasted as search bandwidth.

Implementation note: the launch contract prefers a small
`HillstromLoader` wrapper that pre-bakes `email_share=1.0`,
`social_share_of_remainder=0.0`, and `min_propensity_clip=0.01` into
the parameter dict, rather than modifying `MarketingLogAdapter` to
support partial search spaces.

## 4. Gap Analysis

### 4a. Reuses Directly (No Code Change)

1. `MarketingLogAdapter` binary treatment validation
2. `MarketingLogAdapter` IPS / IPW policy evaluation math
3. `MarketingLogAdapter` zero-support fallback
4. `MarketingLogAdapter` 14-edge prior causal graph
5. `MarketingLogAdapter` objective definition (`policy_value`, maximize)
6. `ExperimentEngine` loop, phase transitions, screening
7. `suggest_parameters` strategy routing
8. Benchmark runner and provenance stack
9. `OffPolicyPredictor` and all optimizer-core defaults from Sprint 29
10. Per-seed reporting, MWU testing, Cohen's d, population-std
    conventions from the Sprint 30 ERCOT report

### 4b. Small Wrapper Required (In-Sprint Work, Doc + Script)

Single new module, nothing invasive:

1. A `HillstromLoader` or `scripts/hillstrom_benchmark.py` that:
   - downloads the CSV from the official MineThatData source (or
     consumes a locally cached copy)
   - filters to `{"Womens E-Mail", "No E-Mail"}` for the primary slice
     and to `{"Mens E-Mail", "Womens E-Mail", "No E-Mail"}` for the
     pooled secondary slice
   - remaps `segment` → `treatment` (pooled: `1` if either e-mail arm,
     `0` if no e-mail)
   - pass-through `spend` → `outcome`
   - assigns constant `cost` (e.g., `0.10` treated, `0.0` control)
   - assigns a **slice-specific** constant `propensity`: `0.5` on the
     primary womens slice and `≈ 0.667` on the pooled slice
   - assigns constant `channel = "email"`
   - maps `history_segment` → `segment` per the bucket map above
     (match the raw CSV strings including numeric prefix)
   - emits the reshaped DataFrame to an in-memory `MarketingLogAdapter`
     once per slice
2. A Hillstrom-specific benchmark scenario class modeled on
   `DoseResponseScenario`
3. A permuted-outcome null control that operates on the reshaped frame
4. A provenance record including the CSV SHA and the wrapper version

Nothing in this list requires a change to
`causal_optimizer/domain_adapters/marketing_logs.py`.

### 4c. Deferred To Later Sprints

1. **Multi-action policy** (true three-arm) — needs a new multi-action
   adapter or a multi-action extension of `MarketingLogAdapter`. Out of
   scope for Sprint 31. Candidate for Sprint 33+.
2. **Customer-level uplift models (CATE)** — the adapter parameterizes
   a policy, not a predictor. A learned uplift model would be a
   meaningful architecture change and should be its own sprint.
3. **Channel allocation search** — only makes sense on a dataset that
   actually contains more than one send channel. Defer to Criteo or a
   multi-channel dataset.
4. **Covariate-conditioned propensities** — Hillstrom propensities are
   constant by design. Defer to Criteo or Open Bandit.
5. **Segment-conditioned policies using raw Hillstrom covariates**
   (`recency`, `history`, `zip_code`, `newbie`) — interesting but
   out of the first-run scope. Sprint 32 follow-on candidate.
6. **Binary classification outcome (conversion) as primary** — the
   first run should optimize continuous `spend` (larger dynamic range,
   smoother IPS estimates). `conversion` and `visit` should be
   recorded as descriptors only. A future sprint can re-run with
   conversion as the primary to test whether the optimizer picks the
   same policy under a binary objective.

## 5. Evidence Contract

### 5a. Strategies

1. `random`
2. `surrogate_only`
3. `causal`

Mirror the Sprint 29 and Sprint 30 benchmark discipline exactly. Do
not add a fourth strategy on the first run.

### 5b. Seeds

10 seeds. This is the same seed budget used for the Sprint 29
optimizer-core regression gate and is the minimum needed for the
two-sided Mann-Whitney U test to resolve `p <= 0.05` claims without
being sample-size-limited.

5-seed exploratory runs are acceptable for smoke-testing the wrapper,
but must not be published as a Sprint 31 verdict.

### 5c. Budgets

`B20, B40, B80`, where the budget label is the exact number of
experiments per strategy per seed (20, 40, and 80 respectively),
matching the Sprint 29 optimizer-core regression gate and the
Sprint 30 ERCOT reality rerun. Per-slice per-budget experiment
count on Hillstrom: 3 strategies × 3 budgets × 10 seeds = 90
experiments per slice (primary or pooled); 2 budgets × 3 strategies
× 10 seeds = 60 experiments for the null control (B80 not
required). Total full-benchmark experiment count on Hillstrom:
`90 + 90 + 60 = 240 adapter evaluations per HillstromLoader build`.
Claim language is locked on `B80` by convention.

### 5d. Primary Objective

`policy_value` (IPS-weighted mean `spend` under the proposed policy).
Maximize.

### 5e. Diagnostics Per Seed

The benchmark report must include, for each seed and budget:

1. `policy_value` mean and 95% bootstrap confidence interval
2. `effective_sample_size` (Kish's ESS from IPS weights)
3. `max_ips_weight`
4. `weight_cv` (coefficient of variation of positive IPS weights)
5. `zero_support` indicator (0/1)
6. `propensity_clip_fraction`
7. `treated_fraction`
8. `total_cost` (for provenance; not the objective)

These match the metrics `MarketingLogAdapter` already returns. No new
metrics are required on the first run.

### 5f. Aggregate Diagnostics

1. per-strategy `policy_value` mean, population std (ddof=0), wins out
   of 10
2. per-comparison two-sided Mann-Whitney U p-value (causal vs s.o.,
   causal vs random, s.o. vs random)
3. Cohen's d using sample-pooled std (ddof=1), consistent with the
   Sprint 30 ERCOT report
4. explicit claim language:
   - `certified` if `p <= 0.05`
   - `trending` if `0.05 < p <= 0.10`
   - `near-parity` if `p > 0.10`
5. optimizer-path provenance (`ax_botorch` is the primary path; RF
   fallback is a secondary drift signal only)

### 5g. Null Control

Permuted-outcome null control:

1. Shuffle the `spend` column across rows while preserving treatment
   assignment and all other columns.
2. Rerun all three strategies on the shuffled frame at B20 and B40.
   `B80` is not required for the null control.
3. Expected behavior: all three strategies should return `policy_value`
   statistically indistinguishable from the shuffled baseline mean.
4. Null control fails if any strategy achieves `policy_value` more
   than `2%` above the **shuffled-frame IPS-weighted grand mean of
   `spend`** (i.e., the `policy_value` returned by the adapter on the
   shuffled frame under a uniform-treatment policy — not the raw
   mean of the `spend` column). The `2%` tolerance mirrors the
   existing ERCOT and synthetic null-control gates (Sprint 18
   onward), which have passed 11 clean runs under this threshold. Caveat: Hillstrom `spend` is right-skewed and
   zero-inflated (most customers spend `$0`), so a 2% relative
   threshold on IPS-weighted mean spend may be small in absolute
   dollars and noisier across permutation seeds than the energy-MAE
   gates. **Pre-committed fallback ladder** (so threshold selection
   cannot be negotiated mid-sprint after seeing results):
   (a) if more than 3 of 10 null-control seeds exceed the `2%`
   threshold purely from permutation noise, widen to `5%` and rerun
   the null control once;
   (b) if the widened `5%` threshold still fails to discriminate,
   switch the baseline statistic from IPS-weighted mean to
   IPS-weighted trimmed mean (trim top/bottom 1%) and rerun once
   more;
   (c) any outcome beyond step (b) constitutes a Sprint 31
   null-control failure and blocks the real-slice verdict.

If the null control fails, Sprint 31 must stop and diagnose before
reporting the real-slice verdict.

### 5h. Success And Failure Criteria

Success (Sprint 31 certified Hillstrom win):

1. causal `policy_value` at `B80` is strictly greater than
   surrogate_only `policy_value` at `B80` on the primary slice, with
   `p <= 0.05` under two-sided Mann-Whitney U
2. causal is not statistically worse than random at `B80` on the
   primary slice
3. the null control passes on the primary slice
4. the pooled secondary slice does not show a statistically
   significant reversal (i.e., s.o. does not beat causal at `p <=
   0.05` on the pooled slice)

Partial success (Sprint 31 trending Hillstrom signal):

1. causal mean beats surrogate-only in direction at `B80`
2. the comparison is `0.05 < p <= 0.10`
3. the null control passes

Failure (Sprint 31 negative Hillstrom result):

1. causal is statistically worse than random at `B80`, or
2. the null control fails, or
3. causal is statistically worse than surrogate-only at `p <= 0.05`

A negative result is not a project failure. It is a diagnosable
localization of the generality claim and feeds into Criteo and Open
Bandit planning.

### 5i. Claim Language Discipline

1. Do not label any Hillstrom finding as "causal advantage on
   marketing" without the pooled secondary slice result and the null
   control result in the same report.
2. Do not extrapolate to "general causal advantage" from a single
   randomized marketing dataset.
3. Always label MWU p-values as two-sided.
4. Always state std convention (population `ddof=0` in tables,
   sample-pooled `ddof=1` for Cohen's d).

## 6. Follow-On Dataset Queue

Sprint 31 covers Hillstrom only. This queue exists so that a Hillstrom
result — positive or negative — has a defined next step rather than
re-litigating dataset choice mid-sprint.

### 6a. Criteo Uplift Prediction Dataset (Priority 2)

**Official source:** <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>

**Role:** Scale-up test. Same binary treatment framing as the
Hillstrom primary slice, much larger sample, closer to production-scale
uplift evaluation.

**Why after Hillstrom, not before:**

1. 25,309,483 rows. Ingestion, caching, and per-run cost are real
   concerns.
2. License is CC-BY-NC-SA 4.0 — usable for research but not for
   arbitrary redistribution. Dataset access audit required before
   commit.
3. Feature set is 12 anonymized features plus `visit` and `conversion`
   outcomes. Binary treatment. No per-customer cost.
4. Worse first debugging surface because the benchmark contract is
   still being shaken out.

**Entry criteria from Hillstrom:**

1. Hillstrom contract is stable (no open adapter gaps)
2. The wrapper pattern (column map + synthesized cost + null control)
   is proven to work end-to-end
3. The Sprint 31 report establishes the evidence discipline for
   non-energy runs

**Sprint 32 candidate contract shape:**

1. Subsample to 1M rows for the first run with a fixed seed
2. Same 3 strategies, 10 seeds, B20/B40/B80
3. Same null control discipline
4. Primary outcome: `visit` (binary, treated as 0/1 float); secondary:
   `conversion`
5. Fixed per-visit cost, tuned only if the Hillstrom pattern
   translated cleanly

### 6b. Open Bandit Dataset / ZOZO (Priority 3)

**Official source:** <https://research.zozo.com/data.html>

**Role:** Architecture expansion. Multi-action offline policy
evaluation with logged propensities.

**Why after Criteo, not after Hillstrom:**

1. Multi-action, not binary. A binary collapse loses most of what
   makes the dataset interesting.
2. Requires either a new multi-action adapter or a meaningful
   extension of `MarketingLogAdapter`.
3. The accompanying `obp` library is a strong reference for offline
   policy evaluation. Any adapter extension should be measured against
   `obp`'s estimators rather than reinventing IPS math.

**Entry criteria:**

1. Binary causal advantage has been reproduced on at least one of
   Hillstrom or Criteo
2. Multi-action adapter design has its own planning doc
3. Sprint scope is explicitly architectural, not just "another
   benchmark"

### 6c. Not Queued Yet

1. **KuaiRand** — real recommender intervention data, much larger
   scale. Later stage. Only makes sense after Open Bandit has
   validated the multi-action stack.
2. **Generic Kaggle marketing tables without treatment/control** —
   explicitly excluded by the Sprint 31 plan. Not in the queue.
3. **Finance price-only adapters** — excluded by the real-data adapter
   requirements doc.

## 7. Execution Order For Sprint 31

This contract does not execute the benchmark. Implementation is a
follow-on sprint-31 issue. The recommended execution order for that
follow-on:

1. write the `HillstromLoader` wrapper (no changes to
   `MarketingLogAdapter`)
2. bake a small cached fixture slice for CI (e.g., 2,000 rows) and
   keep the full CSV local-only
3. stand up a Hillstrom benchmark scenario class modeled on
   `DoseResponseScenario`
4. run 5-seed smoke test (primary slice only, B40 only) to verify
   wrapper correctness end-to-end; in the same smoke test, assert
   both per-slice propensity invariants on small samples:
   (a) primary `Womens E-Mail vs No E-Mail` slice: every row has
   `propensity == 0.5` exactly;
   (b) pooled `Any E-Mail vs No E-Mail` slice: every row has
   `propensity == 2.0 / 3.0` exactly (no rounded constants like
   `0.667`).
   Swapping `0.5` onto the pooled slice is the single most likely
   implementation bug for a reader who skims section 3c
5. run full 10-seed, 3-budget primary + secondary + null control
   benchmark
6. publish the Sprint 31 Hillstrom report
7. decide Sprint 32 path (Criteo kickoff or Hillstrom extension)

## 8. What This Doc Commits The Project To

1. the first non-energy benchmark is Hillstrom, not ERCOT
2. the first slice is `Womens E-Mail vs No E-Mail`
3. the first run reuses `MarketingLogAdapter` unchanged
4. the first run uses the Sprint 30 evidence discipline exactly
5. the first run is followed by Criteo before any multi-action work
6. claim language stays scoped to the primary slice

## 9. What This Doc Does Not Commit The Project To

1. any change to `causal_optimizer/domain_adapters/marketing_logs.py`
2. any new adapter class in `causal_optimizer/domain_adapters/`
3. any tuning of `causal_exploration_weight` or other Sprint 29
   defaults on the first run
4. abandoning ERCOT as a validation lane
5. claiming generality from one benchmark result

## 10. References

1. [22-sprint-31-generalization-research-plan.md](../plans/22-sprint-31-generalization-research-plan.md)
2. [sprint-30-general-causal-portability-brief.md](sprint-30-general-causal-portability-brief.md)
3. [sprint-30-ercot-reality-report.md](sprint-30-ercot-reality-report.md)
4. [04-real-data-adapter-requirements.md](../plans/04-real-data-adapter-requirements.md)
5. [marketing-log-adapter.md](marketing-log-adapter.md)
6. [00-origin.md](../../00-origin.md)
7. MineThatData E-Mail Analytics And Data Mining Challenge — <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>
8. Criteo Uplift Prediction Dataset — <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
9. Open Bandit Dataset — <https://research.zozo.com/data.html>
