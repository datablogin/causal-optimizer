# Sprint 32 Criteo Uplift Benchmark Contract

**Date:** 2026-04-16
**Sprint:** 32 (Criteo Uplift Benchmark Contract)
**Issue:** #177
**Branch:** `sprint-32/criteo-benchmark-contract`
**Predecessor:** Sprint 31 Hillstrom benchmark (PR #176), Criteo audit (PR #174)
**Status:** Contract and execution brief. Not an implementation PR.

## 0. Purpose

This document is the benchmark contract for the project's first Criteo
Uplift run. It does **not** implement the benchmark. It pins:

1. what the first executable Criteo benchmark should optimize and report
2. what must be frozen versus tunable in the first pass
3. how to map Criteo columns into the existing `MarketingLogAdapter`
4. what diagnostics and null controls are mandatory before trusting any
   result
5. what a pass and a failure would each mean for the project

The binding constraint is that Criteo is the second marketing benchmark,
not the first. Hillstrom (Sprint 31) already proved the wrapper pattern,
exposed the RF-backend boundary, and established the evidence discipline.
This contract builds on those lessons rather than relitigating them.

## 1. Dataset Choice And Why Now

### 1a. Why Criteo Is The Right Second Marketing Benchmark

1. **Same problem class, harder stress test.** Criteo is a binary-
   treatment uplift benchmark like Hillstrom, so it keeps the
   intervention framing clean. But it is harder in every measurable
   dimension: 14M rows vs 64K, 85:15 treatment imbalance vs 50:50,
   binary outcomes vs continuous spend, anonymized features vs
   interpretable covariates.
2. **Tests whether the Hillstrom outcome was dataset-specific.** The
   Sprint 31 lessons-learned doc identifies three possible explanations
   for Hillstrom's surrogate-only advantage: (a) weak treatment effect
   on a narrow search space, (b) RF-backend artifact, (c) a broader
   sign that surrogate-only is stronger on binary uplift problems.
   Criteo can distinguish (a) from (c).
3. **Compatible with the proven wrapper pattern.** The Criteo audit
   (PR #174) confirmed that a `CriteoLoader` wrapper over
   `MarketingLogAdapter` is sufficient for the first run. No new
   adapter or policy-evaluation stack is required.
4. **Keeps the benchmark queue disciplined.** Open Bandit Dataset
   requires a multi-action adapter rewrite. Starting Criteo first
   means the next marketing benchmark tests scale and imbalance
   tolerance without entangling a new architecture project.

### 1b. Why Criteo Is Not Trivial

1. **85:15 treatment imbalance.** Control observations carry IPS weight
   `1 / (1 - 0.85) = 6.67`. This is 3.3x higher than Hillstrom's
   maximum weight of 2.0. The IPS stack will be under real stress.
2. **Binary outcomes only.** `visit` (4.70% base rate) and `conversion`
   (0.29%) are both binary. IPS-weighted means of rare binary events
   are noisy. There is no continuous outcome like Hillstrom's `spend`
   to smooth the signal.
3. **Anonymized features.** All 12 features are randomly projected
   floats. No interpretable causal graph can be constructed from domain
   knowledge. The causal strategy loses its primary advantage (graph-
   ancestor focus) unless a data-driven graph is learned.
4. **Non-uniform subsampling.** The dataset was deliberately subsampled
   so that absolute incrementality levels cannot be recovered. Relative
   signal (which features predict uplift) is preserved, but effect
   sizes cannot be compared to Hillstrom or energy benchmarks in
   absolute terms.
5. **Exposure noncompliance.** Not all treated users saw an ad. The ITT
   estimand (using `treatment`) dilutes the effect relative to a per-
   protocol analysis (using `exposure`). The first run uses ITT only.

## 2. What Hillstrom Changed About This Decision

Sprint 31 produced five lessons that directly shape the Criteo contract:

### 2a. Backend Differences Matter

Hillstrom ran on the RF fallback backend. The strongest synthetic causal
wins were Ax-primary. The Hillstrom result therefore does not tell us
whether the causal path would have performed differently under Ax/BoTorch
on the same data. **Criteo must attempt to run under the Ax/BoTorch
backend.** If Ax is unavailable, the report must flag this as an open
confound, exactly as the Hillstrom report did.

### 2b. Null-Control Interpretation Matters

Hillstrom's null-control pass showed that all three strategies can
produce policy values above the null baseline on permuted data (8/10
seeds exceeding baseline). This means high policy values alone are not
clean treatment-effect evidence. **Criteo's null control must follow the
same permuted-outcome discipline** and the report must separate
"optimizer found a high-value policy under the estimator" from "clean
treatment-effect evidence."

### 2c. Narrow Search Spaces Can Blunt Graph Value

Hillstrom's 3-variable active search space is much narrower than the
energy benchmarks where causal guidance had the clearest wins. The causal
graph may have more leverage when the search problem contains more
irrelevant or weakly relevant directions. **Criteo's first run uses the
same 3-variable active space as Hillstrom** (for comparability), but a
follow-on run should consider widening the search space if the first run
shows near-parity.

### 2d. A Wrapper That Runs Is Not A Benchmark That Proves Generality

The Hillstrom wrapper worked end to end. The benchmark still showed a
surrogate-only advantage. **Criteo must not declare victory because the
wrapper runs.** The verdict depends on strategy ordering, significance
tests, and null-control behavior.

### 2e. Marketing Benchmarks Need IPS-First Diagnostics

Hillstrom's null-control and tail behavior both pointed to the same
practical need: marketing benchmarks should make ESS, max weight, weight
CV, and support coverage first-class diagnostics. **Criteo makes this
even more important** because of its 85:15 imbalance and rare binary
outcomes. If ESS drops below 100 on any seed, that seed's result is
flagged as unreliable.

## 3. Official Source, License, And Local-Data Expectations

### 3a. Source And Access

| Item | Value |
|------|-------|
| Official page | Criteo AI Lab |
| Direct download (v2.1) | `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz` |
| Hugging Face mirror | `criteo/criteo-uplift` on Hugging Face Datasets |
| Reference paper | Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018, KDD London |
| Registration | None. Direct download, no authentication. |

### 3b. License

**CC-BY-NC-SA-4.0** (Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International).

Permitted: non-commercial research use, derivative works under the same
license, public sharing of results and reports.

Restricted: commercial use, redistribution under a different license,
attempts to recover original features or user identity.

**Required attribution:** Cite Diemert et al. 2018 in any report.

**Project impact:** Any CI fixture committed to the repo must carry the
same CC-BY-NC-SA-4.0 license and attribution. A `LICENSE-CRITEO` file
must be added alongside the fixture. This is stricter than Hillstrom
(public domain / unrestricted).

### 3c. Dataset Shape

| Metric | Value |
|--------|-------|
| Rows | 13,979,592 |
| Columns | 16 (12 features `f0`-`f11` + `treatment` + `exposure` + `visit` + `conversion`) |
| Compressed size | ~297 MB (gzip CSV) |
| Uncompressed size | ~311 MB |
| In-memory (float64) | ~1.7 GB (full), ~120 MB (1M subsample) |

### 3d. Local-Data Expectations

1. **Full dataset:** downloaded once to a local path outside the repo.
   Not committed. Not required for CI.
2. **1M-row subsample:** generated from the full dataset with a fixed
   seed. Used for all first-run benchmarks. Stored locally, not
   committed.
3. **CI fixture:** a 3,000-row deterministic subsample committed to
   `tests/fixtures/criteo_uplift_fixture.csv` with a `LICENSE-CRITEO`
   file. Must preserve the 85:15 treatment ratio. Used for smoke tests
   and adapter validation only, not benchmark verdicts.

## 4. First-Run Benchmark Shape

### 4a. Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 1 | `random` | Uniform random parameter sampling |
| 2 | `surrogate_only` | RF or Ax/BoTorch surrogate, no causal graph |
| 3 | `causal` | Surrogate + causal graph guidance |

Same three strategies as Hillstrom and the synthetic regression gate.

### 4b. Seeds

10 seeds (0-9). Same seed budget as Hillstrom and the Sprint 29
regression gate. Minimum needed for two-sided MWU at p <= 0.05.

### 4c. Budgets

B20, B40, B80. Same as Hillstrom. Claim language locked on B80.

### 4d. Primary Outcome

**`visit`** (binary 0/1, 4.70% base rate).

Rationale: `visit` has 16x the positive rate of `conversion` (4.70% vs
0.29%). On a 1M-row subsample, `visit` produces ~47,000 positive
outcomes (~7,050 in the control arm), which is adequate for IPS-weighted
estimation. `conversion` produces ~2,900 positives (~435 in control),
which is marginal for stable per-seed estimates.

IPS-weighted `policy_value` on binary 0/1 data becomes an IPS-weighted
visit rate. The optimizer searches for policies that maximize this rate.

### 4e. Secondary Outcome

**`conversion`** (binary 0/1, 0.29% base rate). Reported in the
benchmark report as a secondary aggregate, but not used as the
optimization target. If the first run shows that `visit`-based IPS
estimates are too noisy, `conversion` will not rescue the benchmark.

### 4f. Dataset Scale

**Fixed-seed 1M-row subsample** for the first run.

Implementation: `CriteoLoader` reads the full CSV/Parquet, samples 1M
rows with `DataFrame.sample(n=1_000_000, random_state=CRITEO_SAMPLE_SEED)`,
and passes the subsample to `MarketingLogAdapter`. The sample seed is a
named constant, not a per-run seed. Every strategy-budget-seed
combination operates on the same 1M rows.

Rationale: 1M rows keeps per-call latency negligible (~120 MB in memory)
while providing enough control-arm observations (~150,000) for stable
IPS estimation. The full 14M can be used for extended runs if the first
pass shows promise.

### 4g. Total Experiment Count

| Component | Count |
|-----------|-------|
| Real benchmark: 3 strategies x 3 budgets x 10 seeds | 90 runs |
| Null control: 3 strategies x 2 budgets x 10 seeds | 60 runs |
| **Total** | **150 runs** |

At B80, each run executes up to 80 adapter calls. Worst-case total
adapter calls: 150 x 80 = 12,000. At ~1M rows per call with vectorized
numpy, this should complete within 1-2 hours on a modern machine.

## 5. Wrapper Mapping Into MarketingLogAdapter

### 5a. Criteo Columns

| Column | Type | Description |
|--------|------|-------------|
| `f0` - `f11` | float | Anonymized, randomly projected features |
| `treatment` | int (0/1) | Binary treatment indicator |
| `exposure` | int (0/1) | Whether treated user was exposed to an ad |
| `visit` | int (0/1) | User visited after ad exposure window |
| `conversion` | int (0/1) | User converted after window |

### 5b. Adapter Required Columns

| Adapter column | Criteo source | Transform |
|----------------|---------------|-----------|
| `treatment` (int 0/1) | `treatment` | Pass-through. Already binary 0/1. |
| `outcome` (float) | `visit` | Pass-through. Binary 0/1 used as float. |
| `cost` (float) | synthesized | `0.01` per treated observation, `0.0` per control. Fixed constant, not tuned. Does not affect `policy_value`. |

### 5c. Adapter Optional Columns

| Adapter column | Criteo source | Transform |
|----------------|---------------|-----------|
| `propensity` (float) | synthesized | Constant `0.85` for all rows. Known from the randomization design. See Section 6b for details. |
| `channel` (str) | constant | `"email"` for all rows. Criteo is single-channel (ad display). Degenerate. |
| `segment` (str) | omitted | Criteo has no natural segment column. All 12 features are anonymized. The adapter assigns uniform `segment_score = 0.2` to all rows. See Section 5d. |

### 5d. Segment Column Decision

Criteo has no natural segment column. Three options were considered:

1. **Omit segment entirely.** Adapter assigns all rows
   `segment_score = 0.2`. Uplift scores depend only on channel weight
   and regularization. Less heterogeneity for the optimizer to exploit.
2. **Synthesize from feature quantiles.** Map tertiles of `f0` to
   `"high_value" / "medium" / "low"`. Creates heterogeneity but is
   arbitrary.
3. **Defer.** Accept reduced heterogeneity and diagnose from results.

**Decision: Option 1 (omit segment) for the first run.** Rationale:
synthesizing a segment from anonymized features would fabricate structure
that does not exist in the data. If the first run shows all policies
collapse to the same `policy_value`, the follow-on run should synthesize
a segment and compare.

### 5e. Features Dropped On First Run

`f0` - `f11` (anonymized features) and `exposure` are dropped from the
adapter input. The adapter's policy evaluation does not reference raw
features. `exposure` is reserved for a future per-protocol analysis.

The `CriteoLoader` should retain `conversion` on the subsample frame
for secondary reporting, but it is not passed to the adapter as an
input column.

### 5f. Degenerate Column Consequences

With segment omitted and channel constant, the adapter's uplift score
computation simplifies:

- `channel_weight` = constant (all rows get `email_share + 0.1`)
- `segment_score` = constant (all rows get `0.2`)
- `raw_score` = constant for all rows
- After regularization and normalization, `uplift_score` = `0.5` for
  all rows (uniform)

This means `eligibility_threshold` becomes the sole policy lever that
varies treatment assignment across observations. When
`eligibility_threshold < 0.5`, all rows are eligible; when
`eligibility_threshold > 0.5`, no rows are eligible. The optimizer
searches a step-function response surface.

**This is a known limitation of the first run.** It mirrors the
Hillstrom behavior where the optimal corner (`eligibility_threshold=0.0`,
`treatment_budget_pct=1.0`) dominated. On Criteo, the optimizer should
discover the same corner if the treat-everyone policy is optimal under
IPS weighting at 85:15 imbalance.

The consequence is explicit: the first Criteo run tests whether the
engine's phase transitions, surrogate modeling, and screening add value
on a degenerate surface under heavy IPS variance. It does not test
whether causal graph focus helps on a heterogeneous surface. That test
requires a follow-on run with a synthesized segment or a wider search
space.

## 6. Search-Space And Propensity Decisions

### 6a. Search Space

| Variable | First-run treatment | Range |
|----------|-------------------|-------|
| `eligibility_threshold` | **Tuned** | [0.0, 1.0] |
| `email_share` | **Fixed at `1.0`** | degenerate (single channel) |
| `social_share_of_remainder` | **Fixed at `0.0`** | degenerate (single channel) |
| `min_propensity_clip` | **Fixed at `0.01`** | see Section 6c |
| `regularization` | **Tuned** | [0.001, 10.0] |
| `treatment_budget_pct` | **Tuned** | [0.1, 1.0] |

Effective active search space: **3 tuned continuous variables**
(`eligibility_threshold`, `regularization`, `treatment_budget_pct`).
Same dimensionality as Hillstrom, for direct comparability.

### 6b. Propensity Decision

**Constant `0.85` for all rows.**

Justification: treatment was randomized at 85:15. The v2.1 release
rebalanced treatment ratios across incrementality tests, so within-test
propensity should be approximately constant conditional on features.
The `MarketingLogAdapter` falls back to marginal treatment rate when the
propensity column is absent, which would yield `0.85` automatically.
However, the `CriteoLoader` should explicitly add `propensity = 0.85`
for clarity and provenance.

Risk: if residual propensity heterogeneity remains after v2 rebalancing,
the constant assumption introduces bias. The first run should monitor
`weight_cv` and `max_ips_weight` to detect whether IPS weights are
suspiciously variable despite the constant-propensity assumption. Under
a true constant propensity of 0.85:

- Treated observations: IPS weight = `1 / 0.85 = 1.176`
- Control observations: IPS weight = `1 / 0.15 = 6.667`

Expected `weight_cv` under constant propensity and uniform policy
assignment: calculable from the 85:15 mix. If observed `weight_cv`
exceeds this expected value substantially, propensity heterogeneity may
be present.

### 6c. min_propensity_clip Decision

**Fixed at `0.01` (conservative, not tuned).**

The bias-variance tradeoff:

- **Tunable:** the optimizer could search `min_propensity_clip` in
  `[0.01, 0.5]`. But on Criteo, the propensity is a constant `0.85`,
  which is within the adapter's clip range. The clip floor only fires
  when `propensity < min_propensity_clip` or
  `propensity > 1 - min_propensity_clip`. At `clip = 0.01`,
  `1 - 0.01 = 0.99 > 0.85`, so no clipping occurs. At `clip = 0.15`,
  `1 - 0.15 = 0.85`, which clips the propensity to exactly the
  boundary — a no-op. At `clip > 0.15`, the control-arm propensity
  `0.15` would be clipped upward, reducing IPS variance but introducing
  bias.
- **Fixed at 0.01:** the clip never fires on either arm. This is the
  most conservative choice: no bias introduced, full IPS variance
  preserved. The diagnostics (ESS, weight_cv, max_ips_weight) will
  reveal whether that variance is tolerable.

The decision is to fix at `0.01` and let the first run's diagnostics
determine whether a wider clip is needed. If ESS is consistently below
100, the follow-on run should consider tuning `min_propensity_clip` or
fixing it at a higher value (e.g., `0.10`).

### 6d. Prior Causal Graph

The Hillstrom contract projected the full 14-edge adapter graph to a
7-edge subgraph over active variables. The same projection applies on
Criteo:

```text
eligibility_threshold  --> treated_fraction
treatment_budget_pct   --> treated_fraction
regularization         --> treated_fraction
regularization         --> policy_value
treated_fraction       --> total_cost
treated_fraction       --> policy_value
treated_fraction       --> effective_sample_size
```

Dropped edges (7 total): `email_share -> {total_cost, policy_value}`,
`social_share_of_remainder -> {total_cost, policy_value}`,
`min_propensity_clip -> {total_cost, policy_value,
effective_sample_size}`.

**Alternative: empty graph + auto-discovery.** Because Criteo features
are anonymized, no domain-knowledge prior graph over features is
possible. The projected policy-variable graph above encodes only the
mechanical relationship between policy levers and metrics. A follow-on
run could test `discovery_method="correlation"` at the exploration-to-
optimization phase transition to learn a data-driven graph. This would
be the first real-data test of the `GraphLearner` pathway.

**Decision for first run:** use the projected 7-edge policy-variable
graph, same as Hillstrom. This preserves comparability. Document the
auto-discovery alternative as a follow-on recommendation.

## 7. Diagnostics And Null-Control Requirements

### 7a. Per-Seed Diagnostics

The benchmark report must include, for each seed and budget:

| Diagnostic | Source | Purpose |
|------------|--------|---------|
| `policy_value` | adapter metric | Primary objective |
| `effective_sample_size` | adapter metric (Kish's ESS) | IPS reliability. **Flag if < 100.** |
| `max_ips_weight` | adapter metric | Variance risk. Expected: 6.667 under constant propensity. **Flag if > 20.** |
| `weight_cv` | adapter metric | Weight stability. **Flag if > 3.0.** |
| `zero_support` | adapter metric (0/1) | Whether any observations match the policy. **Flag if 1.0.** |
| `propensity_clip_fraction` | adapter metric | Should be 0.0 under `clip=0.01` with constant propensity 0.85. **Flag if > 0.** |
| `treated_fraction` | adapter metric | Policy selectivity |
| `total_cost` | adapter metric | Provenance only |

These match the metrics `MarketingLogAdapter` already returns. No new
metrics are required.

### 7b. Aggregate Diagnostics

1. Per-strategy `policy_value` mean, population std (ddof=0), wins out
   of 10
2. Per-comparison two-sided Mann-Whitney U p-value (causal vs s.o.,
   causal vs random, s.o. vs random)
3. Cohen's d using sample-pooled std (ddof=1)
4. Claim language: `certified` (p <= 0.05), `trending` (0.05 < p <=
   0.10), `near-parity` (p > 0.10)
5. Optimizer-path provenance (Ax/BoTorch primary or RF fallback)
6. Median ESS across seeds per strategy-budget cell. If median ESS <
   100 for any cell, the cell's verdict is flagged as "low-ESS
   unreliable"

### 7c. Provenance Requirements

The benchmark report must record:

1. Criteo dataset version (v2.1)
2. Subsample seed and row count (1M)
3. Treatment ratio in the subsample (should be ~85:15)
4. Visit rate in the subsample (should be ~4.70%)
5. Number of positive outcomes in the control arm
6. Optimizer backend (Ax/BoTorch or RF fallback)
7. Fixed parameter values (`email_share=1.0`,
   `social_share_of_remainder=0.0`, `min_propensity_clip=0.01`,
   `cost=0.01/0.0`, `propensity=0.85`)
8. Projected prior graph (7 edges)
9. Diemert et al. 2018 citation

### 7d. Null Control

Permuted-outcome null control, following the Hillstrom discipline:

1. **Permutation target:** shuffle the `visit` column across rows using
   a deterministic permutation seeded per null-control seed. Preserve
   treatment assignment, propensities, and all other columns. Call the
   shuffled frame `D_null`.
2. **Budgets:** B20 and B40 only. B80 is not required for the null
   control.
3. **Strategies:** all three (random, surrogate_only, causal).
4. **Seeds:** 10.
5. **Null baseline:** `mu = mean(visit)` on the unshuffled 1M-row
   subsample. This is the raw visit rate (~0.047). Because shuffling is
   a permutation of the same column values, `mu` is identical on `D`
   and `D_null`.

**Failure criterion:** the null control fails if any strategy's mean
`policy_value` across the 10 null-control seeds exceeds `1.05 * mu` at
any budget.

Note: the tolerance is `5%`, wider than Hillstrom's initial `2%`
threshold. Rationale: `visit` is a binary 0/1 column with a 4.70% base
rate. IPS-weighted means of sparse binary outcomes on permuted data are
inherently noisier than IPS-weighted means of continuous `spend`. The
Hillstrom null control already showed that the `2%` threshold can be
tight on right-skewed zero-inflated data. Starting at `5%` on a sparse
binary outcome avoids a predictable fallback-ladder invocation.

**Pre-committed fallback:** if more than 3 of 10 null-control seeds for
any strategy exceed `1.05 * mu`:
  (a) widen to `1.10 * mu` and re-evaluate the same seeds (no new runs);
  (b) if `10%` still fails, the null control fails and blocks the real
  verdict.

If the null control fails, the implementation sprint must stop and
diagnose before reporting any real-data verdict.

## 8. What A Pass Would Mean

### 8a. Success (Certified Criteo Win)

All four conditions must hold:

1. Causal `policy_value` at B80 is strictly greater than surrogate_only
   at B80, with `p <= 0.05` under two-sided Mann-Whitney U.
2. Causal is not statistically worse than random at B80.
3. The null control passes.
4. Median ESS at B80 is >= 100 for all strategies.

A certified Criteo win would mean: the engine's causal guidance provides
a measurable advantage on a large-scale, imbalanced, binary-treatment
uplift benchmark with anonymized features. Combined with the synthetic
wins (medium-noise, high-noise, dose-response), this would strengthen
the generality claim beyond energy-only evidence.

### 8b. Partial Success (Trending Signal)

1. Causal mean beats surrogate-only in direction at B80.
2. The comparison is `0.05 < p <= 0.10`.
3. The null control passes.
4. ESS is adequate (median >= 100).

A trending signal would motivate: (a) extending to the full 14M rows,
(b) tuning min_propensity_clip or synthesizing a segment column, (c) a
follow-on run with auto-discovered causal graph.

### 8c. Near-Parity

1. No statistically significant difference between causal and
   surrogate_only at any budget.
2. The null control passes.
3. ESS is adequate.

Near-parity with adequate ESS would mean: the degenerate search surface
and/or the lack of an informative prior graph prevents the causal path
from differentiating. This is diagnosable and motivates a wider search
space or synthesized heterogeneity in a follow-on run.

### 8d. IPS Variance Failure

1. Median ESS < 100 at B80 for any strategy.
2. Or: `weight_cv > 5.0` consistently across seeds.
3. Or: the null control fails.

This would mean: the IPS stack is under too much stress at 85:15
imbalance with binary outcomes. The project should consider: (a) using
the full 14M rows, (b) implementing doubly robust estimation, (c)
tuning `min_propensity_clip` to a bias-accepting value.

## 9. What A Failure Would Mean

### 9a. Surrogate-Only Advantage (Same As Hillstrom)

If surrogate_only beats causal at certified significance on Criteo, the
project has two consecutive marketing benchmarks where the causal path
does not add value. This would narrow the generality claim:

- Causal guidance helps on energy and synthetic benchmarks (confirmed).
- Causal guidance does not help on marketing uplift benchmarks under the
  current engine (two data points).

This is a productive result. It localizes the boundary to the marketing
uplift problem class and motivates investigation into whether the issue
is: (a) the degenerate search surface, (b) the lack of a meaningful
prior graph, (c) the IPS estimation variance, or (d) a real limit of
the causal approach on binary uplift.

### 9b. Causal Worse Than Random

If causal is statistically worse than random at B80, this would be a
regression signal. It would mean the causal graph is actively hurting
the optimizer, not just failing to help. This would require immediate
diagnosis before proceeding with any further marketing benchmarks.

### 9c. Null Control Failure

A null control failure would mean the IPS-weighted policy evaluation
can be gamed even on noise. This blocks all other verdicts and requires
diagnosing the IPS stack before continuing.

## 10. Recommendation For The Implementation Sprint

### 10a. Scope

The implementation sprint should:

1. Write a `CriteoLoader` wrapper that:
   - reads the Criteo v2.1 CSV or Parquet
   - subsamples to 1M rows with a fixed seed
   - maps columns per Section 5b/5c
   - pre-bakes frozen parameters per Section 6a
   - projects the prior graph per Section 6d
   - generates the permuted-outcome null control frame

2. Commit a 3,000-row CI fixture with `LICENSE-CRITEO` attribution.

3. Write a benchmark script modeled on `scripts/hillstrom_benchmark.py`.

4. Run the full 150-run benchmark (90 real + 60 null control).

5. Publish the Sprint 33 Criteo benchmark report following the evidence
   discipline established on Hillstrom.

### 10b. What Must Not Change

1. `MarketingLogAdapter` source code (no adapter changes).
2. Sprint 29 engine defaults (`causal_exploration_weight=0.0`).
3. The 3-strategy, 10-seed, 3-budget evidence discipline.
4. The claim language conventions (certified / trending / near-parity).

### 10c. Prerequisite Before Implementation

The Criteo v2.1 CSV must be downloaded locally. The implementation
sprint should document the download path and verify the file hash.
The raw file must not be committed to the repository.

### 10d. Follow-On Run Candidates (Not In First Implementation)

If the first run shows near-parity or IPS variance issues:

1. **Synthesized segment from feature quantiles.** Tertiles of `f0`
   mapped to `"high_value" / "medium" / "low"` to introduce uplift-
   score heterogeneity.
2. **Auto-discovered causal graph.** Enable `discovery_method=
   "correlation"` at the exploration-to-optimization phase transition.
   First real-data test of `GraphLearner`.
3. **Full 14M-row run.** If 1M rows produce marginal ESS, scaling to
   the full dataset increases control-arm count from ~150K to ~2.1M.
4. **Tunable `min_propensity_clip`.** If IPS variance is the bottleneck,
   tuning the clip to `[0.05, 0.20]` trades bias for variance reduction.
5. **Doubly robust estimation.** Adapter extension to combine IPS
   weighting with an outcome model for variance reduction. This is a
   Sprint 34+ consideration.
6. **Wider search space.** Add `min_propensity_clip` as a tuned variable
   to create a 4-dimensional search, or synthesize additional policy
   levers from Criteo features.

## 11. References

1. Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). "A Large Scale Benchmark for Uplift Modeling." AdKDD 2018, KDD London.
2. Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2021). "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling." arXiv:2111.10106.
3. [Sprint 31 Hillstrom Lessons Learned](sprint-31-hillstrom-lessons-learned.md)
4. [Sprint 31 Hillstrom Benchmark Report](sprint-31-hillstrom-benchmark-report.md)
5. [Sprint 31 Criteo Uplift Access and Adapter-Gap Audit](sprint-31-criteo-uplift-access-and-gap-audit.md)
6. [Sprint 31 Hillstrom Benchmark Contract](sprint-31-hillstrom-benchmark-contract.md)
7. [Sprint 31 Generalization Research Plan](../plans/22-sprint-31-generalization-research-plan.md)
8. [Real-Data Adapter Requirements](../plans/04-real-data-adapter-requirements.md)
9. [MarketingLogAdapter Documentation](marketing-log-adapter.md)
