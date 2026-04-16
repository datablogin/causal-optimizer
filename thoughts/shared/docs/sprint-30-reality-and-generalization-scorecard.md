# Sprint 30 Reality-and-Generalization Scorecard

**Date**: 2026-04-16
**Sprint**: 30 (General Causal Autoresearch: Reality Check And Portability)
**Issue**: #164
**Predecessors**:
- PR #165 -- Portability brief (re-anchored project as domain-agnostic)
- PR #166 -- ERCOT reality report (PARTIAL REAL-WORLD SIGNAL)
- PR #167 -- Hillstrom benchmark contract (Sprint 31 non-energy benchmark)
- PR #169 -- Hillstrom benchmark harness (Sprint 31 non-energy benchmark code)

## Verdict

**REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC** -- the Sprint 29 default
change (`causal_exploration_weight=0.0`) produced the first real-world
differentiation between causal and surrogate-only on ERCOT data.
COAST is certified (p=0.008, two-sided MWU, 5/5 wins).  NORTH_C is
trending (p=0.059, two-sided MWU, 4/5 wins).  However, this signal
is (a) limited to 5 seeds, (b) confined to ERCOT energy forecasting,
and (c) does not yet show causal beating random.  The portability brief
and Hillstrom benchmark harness define a credible non-energy path, but
no non-energy empirical evidence exists yet.  The project's real-world
signal improved; its generality claim remains unproven.

## 1. Five Required Questions

### 1a. Did Sprint 29 improve real ERCOT outcomes?

**Yes, partially.**

The Sprint 29 default change broke the long-standing "causal identical
to surrogate-only on real ERCOT data" result from Sprint 16.

| Dataset | Comparison | B80 Delta (MAE) | Cohen's d | MWU p (2-sided) | Wins | Verdict |
|---------|-----------|-----------------|-----------|----------------|------|---------|
| COAST | causal vs s.o. | **-0.842** | -1.92 | **0.008** | 5/5 | Certified |
| COAST | causal vs random | -0.332 | -0.72 | 0.690 | 3/5 | Not significant |
| NORTH_C | causal vs s.o. | **-0.498** | -1.39 | **0.059** | 4/5 | Trending |
| NORTH_C | causal vs random | +0.129 | +0.46 | 0.402 | 1/5 | Random still better |

This is the first real-world evidence of causal vs surrogate-only
differentiation.  It does not constitute a full causal advantage claim
on ERCOT forecasting because:

1. Causal does not statistically beat random on either dataset
2. All strategies still select ridge regression (improvement is in
   hyperparameters, not model class)
3. Absolute gaps are small (~0.7% relative improvement on COAST)

### 1b. How trustworthy is that ERCOT signal at 5 seeds?

**Informative but insufficient for firm certification.**

The COAST result (p=0.008) is robust at 5 seeds -- even a standard
Bonferroni correction for two datasets would leave it significant.
The NORTH_C result (p=0.059) is marginal and could easily move to
either side of 0.05 with additional seeds.

Sprint 31 should rerun both datasets with 10 seeds (reusing the
existing 5 seeds plus 5 new ones) to:

1. Firm up or reject the NORTH_C trending result
2. Test whether the COAST causal-vs-random gap (mean -0.33, p=0.690)
   tightens with more power
3. Reduce the risk that the 5-seed COAST certification is a
   small-sample artifact

### 1c. Does the project now have a credible non-energy path?

**Yes, structurally.**

Sprint 30 produced two artifacts that define the path:

1. **Portability brief** (PR #165): catalogued the engine as fully
   domain-portable with 4 shipped adapters; identified that 6 of 7
   active regression gate rows are ERCOT-tied; recommended marketing
   offline policy as the next non-energy benchmark
2. **Hillstrom benchmark harness** (PR #169): shipped code for the
   first non-energy benchmark using the Hillstrom MineThatData email
   campaign dataset; this is a real marketing uplift benchmark, not
   a synthetic control

Additionally, the portability brief identified Criteo and Open Bandit
as future non-energy benchmark candidates.

However, no non-energy benchmark results exist yet.  The path is
defined but not walked.  Sprint 31 must produce Hillstrom results to
convert this structural claim into empirical evidence.

### 1d. Does Sprint 30 move the repo toward a general causal research assistant?

**Yes, but the movement is preparatory, not evidential.**

The portability brief re-anchored the project identity: the
causal-optimizer is a domain-agnostic automated research organization,
not an energy forecasting tool.  ERCOT is the first and most exercised
validation surface, not the product identity.

The ERCOT reality report is informative (first real-world signal) but
does not redefine the project as ERCOT-specific.  The Hillstrom harness
sets up the first non-energy empirical test.

What Sprint 30 did:
- Defined the generalization thesis (portability brief)
- Produced real-world evidence on the existing domain (ERCOT report)
- Built infrastructure for the first non-energy test (Hillstrom harness)

What Sprint 30 did not do:
- Produce non-energy empirical results
- Prove the engine's causal advantage transfers across domains

### 1e. What should Sprint 31 prioritize?

**Run Hillstrom as the first non-energy benchmark AND continue ERCOT
validation in parallel.**

Specifically:

1. **Hillstrom benchmark execution** -- run the Hillstrom harness with
   10 seeds at B20/B40/B80, comparing causal vs surrogate_only vs
   random.  This is the highest-priority item because it is the first
   empirical test of domain portability.

2. **ERCOT 10-seed rerun** -- extend NORTH_C and COAST to 10 seeds
   (5 incremental runs per strategy-budget combination per dataset).
   This firms up the p=0.059 NORTH_C result and tests whether the
   COAST signal holds with more statistical power.

3. **Publish Sprint 31 scorecard** -- synthesize Hillstrom results
   with the extended ERCOT evidence into a generalization verdict.

The project should not pivot to "we now have causal advantage on
ERCOT."  ERCOT is one validation lane.  Hillstrom is the generality
test.

## 2. Evidence Summary

### 2a. Synthetic Benchmark State (from Sprint 29, unchanged)

| # | Benchmark | B80 Causal Mean | B80 p (2-sided MWU) | Classification |
|---|-----------|----------------|--------------------|--------------------|
| 1 | Base energy | 1.01 | 0.112 | Trending (mean improved, lost significance) |
| 2 | Medium-noise | 1.19 | 0.002 | Certified causal win |
| 3 | High-noise | 1.08 | 0.001 | Certified causal win |
| 4 | Confounded | -- | -- | All strategies misled |
| 5 | Null control | 0.2% max delta | PASS | 11th clean run |
| 6 | Interaction | 1.90 | 0.225 | Near-parity (was s.o. advantage) |
| 7 | Dose-response | 0.22 | 0.003 | Certified causal win (Ax-primary) |

No synthetic benchmarks were rerun in Sprint 30.  These results carry
forward from Sprint 29.

### 2b. Real ERCOT State (Sprint 30, new)

| Dataset | Causal vs S.O. B80 p (2-sided MWU) | Causal vs Random B80 p | Seeds | Classification |
|---------|-----------------------------------|----------------------|-------|----------------|
| COAST | **0.008** | 0.690 | 5 | Causal certified > s.o.; not > random |
| NORTH_C | **0.059** | 0.402 | 5 | Causal trending > s.o.; not > random |

### 2c. Non-Energy Benchmark State (Sprint 30, new)

| Benchmark | Status | Domain | Results |
|-----------|--------|--------|---------|
| Hillstrom email campaign | Harness merged, not yet run | Marketing uplift | None yet |
| Criteo uplift | Identified, not started | Advertising uplift | None |
| Open Bandit Pipeline | Identified, not started | Recommendation policy | None |

## 3. Verdict Justification

The verdict **REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC** rests on three
observations:

1. **Real-world improved**: Sprint 30 produced the first ERCOT causal vs
   surrogate-only differentiation.  COAST is certified (p=0.008).
   NORTH_C is trending (p=0.059).  This is a genuine improvement over
   the Sprint 16 baseline where causal and surrogate-only were
   identical on real data.

2. **But domain-specific**: all real-world evidence is confined to ERCOT
   energy forecasting.  No non-energy benchmark has produced results.
   The synthetic benchmark suite is heavily energy-weighted (6 of 7
   rows use ERCOT covariates).  Dose-response is the only non-energy
   certified win, and it is synthetic.

3. **Generalization path is defined but not walked**: the portability
   brief, Hillstrom harness, and benchmark portfolio target provide the
   structural path to test generality.  But structural readiness is not
   empirical evidence.

Alternative verdicts considered:

- **REAL-WORLD IMPROVED AND GENERALIZING** -- rejected because no
  non-energy empirical evidence exists.  The Hillstrom harness is
  infrastructure, not results.
- **REAL-WORLD FLAT, GENERALIZATION PATH DEFINED** -- rejected because
  the ERCOT signal is real (p=0.008 on COAST) and represents a genuine
  improvement over Sprint 16.
- **INCONCLUSIVE** -- rejected because the COAST result is
  statistically significant and the NORTH_C result is directionally
  consistent.

## 4. Sprint 30 Deliverables

| Deliverable | PR | Status | What It Proved |
|-------------|-----|--------|---------------|
| Portability brief | #165 | Merged | Engine is domain-portable; benchmark portfolio is energy-heavy |
| ERCOT reality report | #166 | Merged | First causal vs s.o. differentiation on real data |
| Hillstrom contract | #167 | Merged | Non-energy benchmark design specified |
| Hillstrom harness | #169 | Merged | Non-energy benchmark code ready to run |
| This scorecard | #164 | This PR | Sprint 30 closure and Sprint 31 direction |

## 5. What Changed Between Sprint 29 and Sprint 30

| Dimension | Sprint 29 State | Sprint 30 State |
|-----------|----------------|----------------|
| Real ERCOT | causal == s.o. (Sprint 16) | causal > s.o. on COAST (p=0.008), trending on NORTH_C (p=0.059) |
| Non-energy benchmarks | dose-response only (synthetic) | dose-response + Hillstrom harness (no results yet) |
| Project identity | implicit energy-first | explicit domain-agnostic (portability brief) |
| Benchmark portfolio balance | 6/7 rows energy-tied | same, but non-energy path defined |
| Production default | causal_exploration_weight=0.0 (new in S29) | unchanged, validated on real data |

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| NORTH_C p=0.059 fails to certify at 10 seeds | Medium | Low -- COAST is already certified | Rerun with 10 seeds in Sprint 31 |
| COAST p=0.008 does not survive 10-seed rerun | Low | High -- would undermine the real-world claim | 10-seed rerun is the test |
| Hillstrom shows no causal advantage | Medium | Medium -- would narrow the generality claim | Failure is informative and diagnosable |
| Hillstrom shows causal advantage on marketing but not transferable | Low | Low -- still proves domain portability | Document the boundary conditions |
| 5-seed ERCOT results are a small-sample artifact | Low-Medium | High | 10-seed rerun addresses this directly |

## 7. Updated Project Position

The project is a **trustworthy automated research harness with a first
real-world signal and a defined generalization path**.

What is now established:

1. Causal guidance wins on 3 of 7 synthetic benchmarks under Ax (medium,
   high, dose-response); base is trending; interaction is near-parity
2. The Sprint 29 default change produced the first real-world causal vs
   surrogate-only differentiation on ERCOT (COAST certified, NORTH_C
   trending)
3. Causal still does not beat random on real ERCOT data
4. The engine is architecturally domain-portable; the benchmark
   portfolio is empirically energy-dominated
5. The Hillstrom harness is the first non-energy benchmark ready to run

What is not yet established:

1. Whether the causal advantage transfers to non-energy domains
2. Whether causal can beat random on any real-world task
3. Whether the ERCOT signal holds at 10 seeds

## Provenance

### Source Documents

1. [Sprint 30 portability brief](sprint-30-general-causal-portability-brief.md) -- PR #165
2. [Sprint 30 ERCOT reality report](sprint-30-ercot-reality-report.md) -- PR #166
3. [Sprint 29 optimizer-core scorecard](sprint-29-optimizer-core-scorecard.md) -- PR #161
4. [Benchmark state](../plans/07-benchmark-state.md)
5. [Handoff document](handoff.md)

### Statistical Conventions

- All p-values are two-sided Mann-Whitney U tests unless otherwise noted
- Population std (ddof=0) in tables
- "Certified" = p <= 0.05; "Trending" = 0.05 < p <= 0.15; "Not significant" = p > 0.15
- "Winner" reserved for statistically significant results only
