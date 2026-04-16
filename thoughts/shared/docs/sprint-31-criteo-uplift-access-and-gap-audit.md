# Sprint 31 Criteo Uplift Access and Adapter-Gap Audit

**Date:** 2026-04-16
**Sprint:** 31 (General Causal Autoresearch: First Non-Energy Real-Data Benchmark)
**Issue:** [#171](https://github.com/datablogin/causal-optimizer/issues/171)
**Branch:** `sprint-31/criteo-uplift-audit`
**Status:** Audit document. No code changes.

## 0. Purpose

This document is the access, licensing, and adapter-gap audit for the
Criteo Uplift Prediction Dataset, the Priority 2 follow-on benchmark
after Hillstrom in the Sprint 31 generalization research plan.

It answers six questions:

1. Can we legally use the dataset for this project?
2. How big is it and what does local setup cost?
3. What is the treatment and outcome structure?
4. Are propensities explicit or do they need estimation?
5. How far is the dataset from the current adapter contract?
6. When should it enter the benchmark queue?

## 1. Official Source and Download Path

| Item | Value |
|------|-------|
| Official page | <https://ailab.criteo.com/criteo-uplift-prediction-dataset/> |
| Direct download (v2.1) | `https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz` |
| Hugging Face mirror | <https://huggingface.co/datasets/criteo/criteo-uplift> |
| Reference paper | Diemert et al., "A Large Scale Benchmark for Uplift Modeling", AdKDD 2018, KDD London |
| Extended paper | Diemert et al., "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling", 2021 (arXiv:2111.10106) |
| Reference code | <https://github.com/criteo-research/large-scale-ITE-UM-benchmark> |

**Registration requirements:** None. The direct download link works
without authentication. The Hugging Face mirror also provides
unauthenticated access and offers both CSV and Parquet formats.

**Dataset versions:** The current recommended version is v2.1. The
v2 release rebalanced the treatment ratio across multiple incrementality
tests that were pooled into the dataset, eliminating hidden confounding
from varying per-advertiser test designs. v2.1 is a minor update to v2 and retains the v2 rebalancing.

## 2. License and Usage Restrictions

**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC-BY-NC-SA-4.0).

**What is permitted:**

1. Non-commercial research use, including academic benchmarking
2. Derivative works (e.g., subsampled fixtures, adapted formats) under
   the same CC-BY-NC-SA-4.0 license
3. Public sharing of results, analyses, and benchmark reports derived
   from the data
4. Redistribution with proper attribution and under the same license

**What is restricted:**

1. Commercial use of the raw data or derivative datasets
2. Redistribution under a different license
3. Attempts to recover original features or user identity (features are
   anonymized and randomly projected)

**Required attribution:** Cite the Diemert et al. 2018 paper in any
publication or report using the dataset.

**Project impact:** The CC-BY-NC-SA-4.0 license is compatible with this
project's research benchmarking use case. However, any fixture subset
committed to the repository must carry the same license and attribution.
The project should include a `LICENSE-CRITEO` or `DATA-LICENSE` file
alongside any committed Criteo fixture data, and the benchmark report
must include the required citation.

**Comparison to Hillstrom:** Hillstrom is public domain / unrestricted.
Criteo has stricter licensing. This is manageable but requires explicit
license tracking that Hillstrom did not need.

## 3. Dataset Size and Local Setup Cost

| Metric | Value |
|--------|-------|
| Rows | 13,979,592 |
| Columns | 16 (12 features + 4 labels/indicators) |
| Compressed size (gzip CSV) | ~297 MB |
| Uncompressed CSV | ~311 MB (per Hugging Face metadata) |
| In-memory (pandas float64) | ~1.7 GB estimated (14M rows x 16 cols x 8 bytes) |

**Local setup cost:**

1. Download: ~300 MB, one-time, no authentication
2. Disk: ~300 MB compressed or ~311 MB uncompressed; both are small by
   modern standards
3. Memory: loading the full 14M-row CSV into a pandas DataFrame at
   default dtypes (float64) will consume approximately 1.7 GB. Using
   int8 for the 4 binary columns and float32 for features would cut
   this to ~850 MB, but the default estimate is the conservative
   planning number. Either way, this is manageable on any modern
   development machine but is a meaningful step up from Hillstrom
   (64K rows, ~5 MB) or the marketing fixture (300 rows, ~30 KB).
4. Per-run cost: a single `MarketingLogAdapter.run_experiment()` call
   iterates over the full DataFrame with numpy vectorized operations.
   On 14M rows this will be slower than the 300-row fixture but should
   remain under 1 second per call. The benchmark cost comes from
   repetition: 3 strategies x 3 budgets x 10 seeds x up to 80
   experiments = up to 7,200 adapter calls at the full dataset size.
5. Subsampling recommendation: the Hillstrom benchmark contract
   (Section 6a) already recommends subsampling to 1M rows for the
   first Criteo run. At 1M rows, in-memory cost drops to ~120 MB and
   per-call latency becomes negligible.

**CI fixture:** A small deterministic subsample (e.g., 2,000-5,000 rows)
should be committed to `tests/fixtures/` for CI, following the same
pattern as the marketing log fixture. The subsample must preserve the
treatment ratio (85:15) and carry the CC-BY-NC-SA-4.0 license notice.

## 4. Treatment Structure

**Treatment type:** Binary. Each row has `treatment = 1` (treated) or
`treatment = 0` (control).

**Assignment mechanism:** Randomized control trial (incrementality test).
At a predefined point in time, each user was randomly assigned to either
the treated or control population. The treatment variable indicates
whether the advertising platform participated in the real-time bidding
(RTB) auction for that user.

**Treatment ratio:** 85% treated, 15% control. This imbalance is
intentional and reflects real-world advertising economics: advertisers
maintain only a small control holdout because withholding ads from
potential customers costs revenue.

**Exposure column:** The dataset includes a separate `exposure` column
(binary) indicating whether the treated user was effectively exposed to
an advertisement. The logical constraint is `treatment = 0` implies
`exposure = 0` (control users are never exposed), but not all treated
users are exposed (treatment = 1 does not guarantee exposure = 1). This
creates an intent-to-treat (ITT) versus per-protocol distinction:

- **ITT analysis** uses the `treatment` column directly. This is the
  standard uplift modeling framing and is what the current adapter
  supports.
- **Per-protocol / exposure-based analysis** uses the `exposure` column.
  This introduces noncompliance and requires instrumental variable
  methods or sensitivity analysis that the current adapter does not
  support.

**Adapter compatibility:** The `treatment` column maps directly to the
`MarketingLogAdapter`'s binary treatment requirement. The `exposure`
column is a bonus for advanced analysis but is not needed for the
first-pass benchmark. The first run should use `treatment` (ITT), not
`exposure`.

## 5. Outcome Structure

| Column | Type | Description | Rate |
|--------|------|-------------|------|
| `visit` | int (0/1) | User visited after ad exposure window | 4.70% |
| `conversion` | int (0/1) | User converted (purchased) after window | 0.29% |

**No continuous outcome.** Unlike Hillstrom (which has `spend` as a
continuous outcome), Criteo provides only binary outcomes. This is the
single most important structural difference from the Hillstrom contract.

**Implications for the adapter:**

1. The `MarketingLogAdapter` computes `policy_value` as IPS-weighted
   mean outcome. On binary 0/1 data, this becomes an IPS-weighted
   conversion or visit rate.
2. IPS-weighted means of rare binary events (0.29% conversion rate) are
   noisy. The effective sample size (ESS) diagnostic already in the
   adapter will be critical for monitoring this.
3. The adapter does not enforce continuous outcomes, so binary outcomes
   will work mechanically. But the optimizer's ability to separate
   strategies on a 0.29% base-rate binary signal is an open empirical
   question.

**No cost column.** Like Hillstrom, Criteo does not ship a per-user
cost. A fixed synthetic cost must be assigned (e.g., `0.01` per treated
observation, `0.0` per control). The same pattern proven on Hillstrom
applies.

**Primary objective recommendation:** Use `visit` (4.70% base rate) as
the primary outcome for the first run. `conversion` (0.29%) is too rare
for stable per-seed IPS-weighted estimates at the subsampled scale
(1M rows x 0.29% = ~2,900 positive outcomes; of those,
~15% are control = ~435 control-arm conversions). Track `conversion` as a secondary reported outcome.

## 6. Propensity Availability

**Propensities are not explicit in the dataset.** The Criteo CSV does
not include a per-user propensity score column.

However, because treatment was randomized:

1. The marginal propensity is known: `P(treatment = 1) = 0.85`.
2. Whether propensity varies with covariates is not documented. The
   v2 rebalancing equalized treatment ratios across incrementality
   tests, so within-test randomization should yield approximately
   constant propensity conditional on features.
3. The safest first-pass assumption is **uniform propensity at 0.85**
   for all treated users, analogous to the constant propensity used on
   Hillstrom (0.5 on the primary slice, 2/3 on the pooled slice).

**Adapter handling:** The `MarketingLogAdapter` falls back to marginal
treatment rate as uniform propensity when the `propensity` column is
absent. This fallback will produce `propensity = 0.85` on Criteo, which
is correct under the uniform-propensity assumption. Alternatively, the
wrapper can explicitly add a `propensity = 0.85` column for clarity.

**Risk:** If propensity actually varies with covariates (e.g., different
advertisers had different control holdout rates before the v2
rebalancing), the uniform assumption introduces bias. The v2 rebalancing
was designed to mitigate this, but the degree of residual heterogeneity
is not documented. The first run should report `weight_cv` and
`max_ips_weight` diagnostics to detect whether IPS weights are
suspiciously variable despite the constant-propensity assumption.

## 7. Adapter Compatibility Assessment

### 7a. Can It Work with the Current MarketingLogAdapter via a Small Wrapper?

**Yes, with caveats.** The same wrapper pattern designed for Hillstrom
(a `CriteoLoader` analogous to `HillstromLoader`) can map Criteo columns
onto the adapter contract:

| Adapter column | Criteo source | Transform |
|----------------|---------------|-----------|
| `treatment` | `treatment` | Pass-through (already binary 0/1) |
| `outcome` | `visit` (primary) or `conversion` (secondary) | Pass-through (binary 0/1, used as float) |
| `cost` | synthesized | Fixed constant (e.g., 0.01 treated, 0.0 control) |
| `propensity` | synthesized | Constant 0.85 (from known randomization ratio) |
| `channel` | constant | `"ad"` (degenerate, single channel; Criteo is display/RTB, not e-mail) |
| `segment` | synthesized | Must be derived from f0-f11; no natural segment column exists |

**What reuses cleanly:**

1. Binary treatment validation (already 0/1)
2. IPS/IPW policy evaluation math
3. Zero-support fallback
4. Objective definition (`policy_value`, maximize)
5. `ExperimentEngine` loop, phases, screening
6. Benchmark runner and provenance stack
7. Per-seed reporting, MWU testing, Cohen's d conventions

**What requires wrapper logic (no adapter changes):**

1. Subsampling to 1M rows with a fixed seed
2. Synthesizing `cost` and `propensity` columns
3. Synthesizing a `segment` column from anonymized features (or omitting
   it and accepting the adapter's uniform segment scoring)
4. Fixing degenerate search variables (`email_share = 1.0`,
   `social_share_of_remainder = 0.0`, `min_propensity_clip = 0.01`)
   as was done for Hillstrom
5. Projecting the prior causal graph to active variables

### 7b. Does It Need a New Adapter?

**Not for the first run.** The `MarketingLogAdapter` via wrapper is
sufficient for an ITT-based binary-treatment binary-outcome benchmark.

A new adapter would be needed only if the project wants to:

1. Use the `exposure` column for per-protocol analysis (requires
   instrumental variable or noncompliance handling)
2. Train a CATE/uplift model rather than evaluate a fixed policy
3. Incorporate feature-dependent propensity estimation
4. Support the 85:15 imbalance with specialized IPS variance reduction
   (e.g., doubly robust estimation with outcome modeling)

These are Sprint 33+ concerns, not first-run requirements.

### 7c. Does It Need a New Policy-Evaluation Stack?

**No for the first run. Possibly for a mature benchmark.**

The current IPS/IPW stack will work but faces two stress tests that
Hillstrom does not impose:

1. **Extreme treatment imbalance (85:15).** Control observations have
   IPS weight `1 / (1 - 0.85) = 6.67`. This is higher than Hillstrom's
   maximum weight of 2.0 (from `1 / 0.5`). The `min_propensity_clip`
   parameter does not help here. The adapter's `min_propensity_clip`
   parameter range is `[0.01, 0.5]`, and the adapter clips propensities
   symmetrically to `[clip, 1 - clip]`. At the default `clip = 0.01`,
   propensities are clipped to `[0.01, 0.99]`. Since Criteo's constant
   propensity of `0.85` falls inside this range, clipping never fires
   and the control-arm weight remains `1 / (1 - 0.85) = 6.67`. Even at
   the maximum `clip = 0.5`, the range becomes `[0.5, 0.5]`, which
   would collapse all propensities to 0.5 — a distortion, not a fix.
   The 6.67x weight on 15% of
   observations means control-arm observations dominate IPS estimates.
   This is not wrong, but it increases variance. Note that this is
   structurally different from the typical clipping use case (estimated
   propensities near 0 or 1): Criteo's propensity of 0.85 is moderate,
   but its complement (0.15) creates high control-arm weights.

2. **Rare binary outcomes (0.29% conversion, 4.70% visit).** IPS
   weighting on a sparse binary outcome amplifies noise. A single
   control-arm conversion with weight 6.67 can swing the IPS estimate
   materially. This is a known challenge in the uplift modeling
   literature and is one reason Criteo is considered a harder benchmark
   than Hillstrom.

If the first run shows unacceptable IPS variance (ESS consistently
below 100, `weight_cv` above 5.0, or null control failure), the project
should consider:

1. Self-normalized IPS (already implemented in the adapter)
2. Doubly robust (DR) estimation with an outcome model — requires
   adapter extension
3. Trimmed IPS with explicit bias-variance tradeoff — requires adapter
   extension

These are diagnosable from the first run's diagnostics and do not need
to be solved before starting.

### 7d. Segment Column Gap

The Hillstrom wrapper maps `history_segment` to the adapter's
`"high_value" / "medium" / "low"` segment scoring. Criteo has no
natural segment column — all 12 features are anonymized floats.

Options for the first run:

1. **Omit the segment column entirely.** The adapter assigns all
   observations a default `segment_score = 0.2` (the `"low"` weight).
   This makes the uplift score depend only on channel weight and
   regularization, reducing effective heterogeneity.
2. **Synthesize a segment from feature quantiles.** For example,
   tertiles of `f0` mapped to `"high_value" / "medium" / "low"`. This
   is arbitrary but creates heterogeneity for the optimizer to exploit.
3. **Defer segment scoring.** Accept that the first Criteo run has less
   policy heterogeneity than Hillstrom. This is honest and avoids
   fabricating structure.

**Recommendation:** Option 1 (omit segment) for the first run. The
optimizer can still search over `eligibility_threshold`,
`regularization`, and `treatment_budget_pct`. If the first run shows
that all policies collapse to the same `policy_value` due to lack of
heterogeneity, synthesize a segment column on the follow-up run.

## 8. Biggest Positivity and Support Risks

### 8a. Treatment Imbalance (85:15)

The 85% treatment ratio means the control arm is small. On a 1M-row
subsample: ~150,000 control observations. This is still large in
absolute terms, but IPS weights on those observations are 6.67x, which
amplifies any noise in control-arm outcomes.

**Mitigation:** The adapter's self-normalized IPS already helps. Monitor
`effective_sample_size` and `weight_cv` per seed. If ESS drops below
~500 on the 1M subsample, the IPS estimates are unreliable.

### 8b. Rare Outcomes

Conversion at 0.29% means ~2,900 positive outcomes per 1M rows, of
which ~435 are in the control arm. Visit at 4.70% means ~47,000
positives, ~7,050 in control. The visit outcome is borderline adequate
for IPS estimation; conversion is marginal.

**Mitigation:** Use `visit` as primary outcome. Report `conversion` as
secondary. If `visit`-based IPS estimates are still too noisy, consider
using the full 14M rows instead of the 1M subsample.

### 8c. Non-Uniform Sub-Sampling

The Criteo documentation states the dataset was "non-uniformly
sub-sampled so that the original incrementality level cannot be deduced."
This means the observed treatment effects in the dataset are
deliberately distorted from the true population effects. The dataset
preserves *relative* signal (which features predict uplift) but not
*absolute* incrementality levels.

**Impact on the project:** The optimizer searches for policies that
maximize `policy_value` relative to alternatives. It does not need to
recover absolute incrementality. The non-uniform subsampling therefore
does not invalidate the benchmark use case, but it means the project
cannot compare Criteo effect sizes to Hillstrom or energy effect sizes
in absolute terms. The benchmark report's limitations section must
state that absolute effect sizes are not comparable across datasets
due to Criteo's non-uniform subsampling.

### 8d. Anonymized Features

All 12 features are anonymized and randomly projected. This means:

1. No interpretable causal graph can be constructed from domain
   knowledge. Any prior graph must be either empty or data-driven.
2. The adapter's segment scoring (which relies on named segments like
   `"high_value"`) has no natural mapping.
3. The optimizer cannot benefit from domain-specific variable focus.

**Impact on causal guidance:** Without an informative prior graph, the
causal strategy loses its primary advantage (focus on graph ancestors of
the outcome). The first run will test whether the engine's other
mechanisms (screening, phase transitions, exploitation) provide value
even without a strong prior. This is a valid and interesting test of the
engine's robustness — if causal matches surrogate-only without a prior,
it demonstrates graceful degradation.

Alternatively, the wrapper could use `GraphLearner` (auto-discovery)
to learn a data-driven graph at the exploration-to-optimization phase
transition. This would test the `discovery_method` pathway on a real
dataset for the first time.

### 8e. Exposure Noncompliance

Not all treated users were exposed to ads. If the first run uses
`treatment` (ITT), this dilutes the treatment effect estimate because
some "treated" users never saw the ad. The dilution is not a bug — ITT
is the standard causal estimand — but it reduces the signal the
optimizer can exploit.

If the dilution is severe enough to wash out any strategy differences,
a follow-up run could filter to `exposure = 1` rows only. However, this
introduces selection bias (exposure is not randomized) and would require
either an IV approach or a sensitivity analysis, neither of which the
current adapter supports.

**Recommendation:** Start with ITT (use `treatment` column). Diagnose
dilution from the first run's per-seed diagnostics before considering
exposure-based analysis.

## 9. Recommendation: When Criteo Should Enter the Benchmark Queue

### Entry Sprint: Sprint 33 (earliest), contingent on Hillstrom outcome

**Dependency chain:** Sprint 31 proves the wrapper pattern on Hillstrom.
Sprint 32 publishes the Hillstrom report and calibrates IPS diagnostics.
Sprint 33 starts Criteo.

**Prerequisites before starting Criteo:**

1. **Hillstrom benchmark is stable.** The wrapper pattern (column
   mapping, synthesized cost, null control, projected causal graph,
   pre-baked degenerate variables) must be proven end-to-end on
   Hillstrom before being replicated for Criteo. This is the Sprint 31
   Hillstrom contract.

2. **Hillstrom report is published.** The evidence discipline (per-seed
   diagnostics, MWU tests, null control, claim language) must be
   exercised on Hillstrom before being applied to a harder dataset.

3. **IPS variance diagnostics are understood.** The Hillstrom run
   should establish baseline expectations for `weight_cv`,
   `max_ips_weight`, and `effective_sample_size` on a dataset where
   propensities
   are balanced (0.5). Criteo's 85:15 imbalance will stress these
   diagnostics, and the team needs calibrated expectations.

4. **License tracking is in place.** A mechanism for attaching
   CC-BY-NC-SA-4.0 attribution to committed fixture data must exist
   before Criteo data enters the repo. Concrete proposal: add a
   `tests/fixtures/DATA-LICENSES.md` file listing each fixture's
   dataset name, license, required attribution, and row count.

### Recommended Sprint 33 Criteo Contract Shape

| Element | Specification |
|---------|---------------|
| Data | Subsample to 1M rows with fixed seed; full 14M for extended runs |
| CI fixture | 2,000-5,000 rows committed with CC-BY-NC-SA-4.0 notice |
| Treatment column | `treatment` (ITT, not exposure-based) |
| Primary outcome | `visit` (4.70% base rate) |
| Secondary outcome | `conversion` (0.29%, reported but not optimized) |
| Cost | Fixed synthetic (e.g., 0.01 treated, 0.0 control) |
| Propensity | Constant 0.85 (from known randomization ratio) |
| Segment | Omitted (uniform scoring) on first run |
| Search space | 3 tuned variables (same as Hillstrom: `eligibility_threshold`, `regularization`, `treatment_budget_pct`) |
| Strategies | random, surrogate_only, causal |
| Seeds | 10 |
| Budgets | B20, B40, B80 |
| Null control | Permuted `visit` column, B20 and B40 |
| Prior graph | Either empty (test engine without prior) or data-driven via `GraphLearner` |
| Success criterion | Causal >= surrogate_only at B80, p <= 0.05 |

### Why Not Sprint 32

Sprint 32 should be reserved for one of:

1. Hillstrom follow-on (mens arm, pooled ablation, or covariate-
   conditioned policies)
2. Diagnosing any Hillstrom failures or near-parity results
3. Completing the marketing offline policy benchmark with the existing
   fixture data (if Hillstrom shows the wrapper pattern works)

Starting Criteo before fully digesting Hillstrom would repeat the
pattern the Sprint 31 plan warns against: moving to a harder dataset
before proving the approach on a simpler one.

### What Criteo Would Prove That Hillstrom Cannot

1. **Scale tolerance.** 1M+ rows vs 42K rows tests whether the engine
   and adapter perform acceptably at production-relevant data sizes.
2. **Robustness to treatment imbalance.** 85:15 vs Hillstrom's 50:50
   tests whether the IPS stack degrades gracefully under realistic
   control-holdout ratios.
3. **Engine value without a prior graph.** Anonymized features force the
   engine to rely on screening and data-driven structure rather than
   domain knowledge. This is a purer test of the engine's algorithmic
   contribution.
4. **Rare-outcome signal detection.** 4.70% visit rate (and especially
   0.29% conversion) tests whether the optimizer can separate strategies
   on sparse binary signals.

### What Criteo Cannot Prove

1. **Multi-action policy evaluation.** Criteo is binary treatment.
   Multi-action requires Open Bandit Dataset (Priority 3).
2. **Continuous-outcome optimization.** Both Criteo outcomes are binary.
   Only Hillstrom `spend` and the energy benchmarks test continuous
   objectives.
3. **Domain-interpretable causal reasoning.** Anonymized features mean
   any learned graph is opaque. Only Hillstrom and energy provide
   interpretable variable names.

## 10. Summary Decision Table

| Question | Answer |
|----------|--------|
| Can we use it legally? | Yes, under CC-BY-NC-SA-4.0. Non-commercial research is permitted. |
| Registration required? | No. Direct download, no authentication. |
| Local setup cost? | ~300 MB download, ~1.7 GB memory for full load, ~120 MB for 1M subsample. Manageable. |
| Treatment structure? | Binary (treated/control), randomized, 85:15 ratio. |
| Outcomes? | Binary: `visit` (4.70%) and `conversion` (0.29%). No continuous outcome. |
| Propensities explicit? | No, but known from randomization (uniform 0.85). Adapter fallback handles this. |
| Works with current adapter? | Yes, via wrapper. Same pattern as Hillstrom. |
| Needs new adapter? | No, for first run. Possibly for exposure-based or DR analysis later. |
| Needs new policy-eval stack? | No, for first run. DR estimation is a Sprint 34+ consideration. |
| Biggest risks? | 85:15 imbalance amplifies IPS variance; rare outcomes reduce signal; anonymized features prevent interpretable causal graph. |
| When should it enter the queue? | Sprint 33, after Hillstrom is stable and the wrapper pattern is proven. |

## 11. References

1. Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). "A Large Scale Benchmark for Uplift Modeling." AdKDD 2018 Workshop, KDD London.
2. Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2021). "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling." arXiv:2111.10106.
3. [Criteo AI Lab — Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
4. [Hugging Face — criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift)
5. [GitHub — criteo-research/large-scale-ITE-UM-benchmark](https://github.com/criteo-research/large-scale-ITE-UM-benchmark)
6. [scikit-uplift — fetch_criteo documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_criteo.html)
7. [Sprint 31 Generalization Research Plan](../plans/22-sprint-31-generalization-research-plan.md)
8. [Sprint 31 Hillstrom Benchmark Contract](sprint-31-hillstrom-benchmark-contract.md)
9. [Sprint 30 General Causal Portability Brief](sprint-30-general-causal-portability-brief.md)
10. [Real-Data Adapter Requirements](../plans/04-real-data-adapter-requirements.md)
11. [MarketingLogAdapter Documentation](marketing-log-adapter.md)
