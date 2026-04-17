# Sprint 33 Generalization Scorecard

**Date:** 2026-04-17
**Sprint:** 33 (closure / synthesis)
**Predecessors:**
- PR #165 -- Sprint 30 portability brief
- PR #166 -- Sprint 30 ERCOT reality report
- PR #169 -- Sprint 30 Hillstrom benchmark harness
- PR #176 -- Sprint 31 Hillstrom benchmark report
- PR #178 -- Sprint 32 Criteo benchmark contract
- PR #180 -- Sprint 33 Criteo benchmark report (Run 1 + Run 2)
- Plan: [24-sprint-34-recommendation.md](../plans/24-sprint-34-recommendation.md)

## Verdict

**GENERALITY IS REAL BUT CONDITIONAL.** The Sprint 29 default change
(`causal_exploration_weight=0.0`) produced a first real-world causal vs
`surrogate_only` differentiation on ERCOT that has held up in every
rerun since. The project then ran two independent non-energy real-data
benchmarks. Hillstrom returned a clean `surrogate_only` advantage on
the pooled slice under RF fallback. Criteo returned near-parity under
Ax/BoTorch on both a degenerate 2-variable surface and a heterogeneous
f0-tertile follow-up. The correct summary is that the causal-optimizer
is still a general causal research harness, but current causal advantage
over `surrogate_only` is conditional on landscape structure, noise
burden, and search-space breadth, not universal across domains.

## 1. Five Required Conclusions

### 1a. ERCOT remains the strongest real-world positive signal

ERCOT is still the one real-data domain where the Sprint 29 default
change produced a durable separation between `causal` and
`surrogate_only`:

| Dataset | Comparison | B80 Delta (MAE) | Two-sided MWU p | Wins | Seeds | Classification |
|---------|------------|-----------------|-----------------|------|-------|----------------|
| COAST | causal vs s.o. | -0.842 | **0.008** | 5/5 | 5 | Certified |
| COAST | causal vs random | -0.332 | 0.690 | 3/5 | 5 | Not significant |
| NORTH_C | causal vs s.o. | -0.498 | **0.059** | 4/5 | 5 | Trending |
| NORTH_C | causal vs random | +0.129 | 0.402 | 1/5 | 5 | Not significant (random mean lower) |

Source: [Sprint 30 ERCOT reality report](sprint-30-ercot-reality-report.md) (PR #166).

What that supports:

1. COAST B80 is the only certified real-world causal vs `surrogate_only`
   separation in the current portfolio (p=0.008, two-sided MWU).
2. NORTH_C B80 is directionally consistent but not yet certified
   (p=0.059, two-sided MWU).
3. Both results are at 5 seeds and still do not show `causal` beating
   `random`; all strategies converge to ridge.
4. The improvement is in hyperparameters, not model class.

What that does not support:

1. A claim that causal beats random on any real-world task.
2. A claim that ERCOT certifies the engine as generally advantage-bearing
   across domains.
3. The 10-seed rerun has not been abandoned; it was deprioritized relative
   to non-energy expansion (Hillstrom, Criteo, Open Bandit) and remains
   on the backlog. It is not the Sprint 34 critical path.

ERCOT is the strongest real-world positive signal the project currently
has, but it is one domain, at 5 seeds, with a `causal`-vs-`random`
gap that has not closed.

### 1b. Hillstrom is a real non-energy boundary result favoring `surrogate_only`

Hillstrom was the first non-energy real-data benchmark. The pooled slice
returned a certified `surrogate_only` advantage at all three budgets, and
the primary slice was mostly `surrogate_only` or near-parity:

| Slice | Budget | Comparison | Two-sided MWU p | Wins (s.o.) | Classification |
|-------|--------|------------|-----------------|-------------|----------------|
| Primary | B20 | causal vs s.o. | 0.060 | 8/10 | Trending (s.o.) |
| Primary | B40 | causal vs s.o. | **0.0001** | 10/10 | Certified (s.o.) |
| Primary | B80 | causal vs s.o. | 0.817 | 7/10 | Near-parity (bimodal causal tail) |
| Pooled | B20 | causal vs s.o. | **0.017** | 8/10 | Certified (s.o.) |
| Pooled | B40 | causal vs s.o. | **0.002** | 9/10 | Certified (s.o.) |
| Pooled | B80 | causal vs s.o. | **0.019** | 7/10 | Certified (s.o.) |

Source: [Sprint 31 Hillstrom benchmark report](sprint-31-hillstrom-benchmark-report.md) (PR #176)
and [Sprint 31 Hillstrom lessons learned](sprint-31-hillstrom-lessons-learned.md).

Important constraints the headline must carry:

1. Hillstrom ran on the RF fallback backend, not Ax/BoTorch. The
   strongest synthetic causal wins in the project are Ax-primary. The
   Hillstrom verdict is therefore a real non-energy boundary result, not
   a backend-matched comparison to the certified Ax-primary wins.
2. The active search space is three variables wide
   (`eligibility_threshold`, `regularization`, `treatment_budget_pct`).
   That is narrower than the rows where causal guidance had its clearest
   synthetic wins and is a plausible boundary condition on graph leverage.
3. Causal still beats `random` at B80 on the primary slice (two-sided
   MWU p=0.0004, 9/10 wins). The causal path is not inert on Hillstrom;
   it is just not the best of the three paths under this configuration.
4. The null-control pass showed that policy values above the simple
   baseline can arise on permuted outcomes for some strategies, meaning
   the primary-slice numbers should not be read as clean treatment-effect
   evidence on their own.

The right framing is: **Hillstrom is a specific, diagnosable non-energy
boundary result in which `surrogate_only` was stronger under the current
RF-backed setup on this dataset.** It is not a general claim that causal
guidance fails on marketing data, and it is not a backend-matched refutation.

### 1c. Criteo is near-parity under Ax/BoTorch even after the heterogeneous follow-up

Criteo was the second non-energy real-data benchmark and the first
Ax-primary marketing benchmark. The Sprint 32 contract required a
mandatory heterogeneous Run 2 if Run 1 returned near-parity. Both runs
returned near-parity:

**Run 1 (degenerate 2-variable surface, 10 seeds, Ax/BoTorch):**

| Budget | causal vs s.o. | Two-sided MWU p | Classification |
|--------|----------------|------------------|----------------|
| B20 | identical all 10 seeds | 1.000 | Near-parity (exact tie) |
| B40 | identical all 10 seeds | 1.000 | Near-parity (exact tie) |
| B80 | identical all 10 seeds | 1.000 | Near-parity (exact tie) |

**Run 2 (synthesized f0-tertile segments, 10 seeds, Ax/BoTorch):**

| Budget | causal vs s.o. | Two-sided MWU p | Classification |
|--------|----------------|------------------|----------------|
| B20 | causal 10/10 corner; s.o. misses on some | 0.168 | Near-parity |
| B40 | identical all 10 seeds | 1.000 | Near-parity |
| B80 | identical all 10 seeds | 0.368 | Near-parity |

Source: [Sprint 33 Criteo benchmark report](sprint-33-criteo-benchmark-report.md) (PR #180).

Combined Criteo verdict per the Sprint 32 contract interpretation table:
**near-parity**.

What that supports:

1. The engine runs cleanly on a 1M-row, 85:15 treatment-imbalanced,
   binary-outcome marketing log under Ax/BoTorch.
2. The IPS stack is stable at 85:15 imbalance (ESS ~850K for the
   optimized strategies, no zero-support events, no variance
   pathologies).
3. The propensity gate and null control both passed.
4. `surrogate_only` and `causal` both converge to the same
   treat-everyone corner even after heterogeneity is injected via f0
   tertiles. The heterogeneous follow-up did not unlock a causal
   advantage.

What that does not support:

1. A claim that the marketing domain is generally causal-flat. Criteo
   was run with a 2-variable active surface and synthesized-label
   heterogeneity, not a rich heterogeneous treatment-effect structure.
2. A claim that Criteo refutes ERCOT. ERCOT and Criteo are different
   problems; the Criteo result is about whether causal guidance adds
   value over a pure surrogate on this specific uplift surface.

The correct framing is: **under the first executable Criteo contract,
causal and `surrogate_only` are empirically interchangeable, and the
heterogeneous follow-up did not change that.** The right next test is
not a third binary marketing rerun; it is a problem class where the
causal path has something structurally different to do.

### 1d. The project is still a general causal research harness, but current causal advantage is conditional rather than universal

The project identity as a domain-agnostic automated causal research
harness is intact. The synthetic boundary map has not moved since
Sprint 29. What has moved is the real-data evidence:

**Ax-primary synthetic evidence (unchanged since Sprint 29):**

| Benchmark | B80 causal mean | Two-sided MWU p | Classification |
|-----------|-----------------|------------------|----------------|
| Base energy | 1.01 | 0.112 | Trending (mean improved, lost significance) |
| Medium-noise | 1.19 | 0.002 | Certified causal win |
| High-noise | 1.08 | 0.001 | Certified causal win |
| Interaction policy | 1.90 | 0.225 | Near-parity (was s.o. advantage) |
| Dose-response | 0.22 | 0.003 | Certified causal win |
| Null control | 0.2% max delta | PASS | 11th clean run |
| Confounded demand-response | -- | -- | All strategies misled |

**Real-data evidence after Sprint 33:**

| Domain | Backend | Result |
|--------|---------|--------|
| ERCOT COAST B80 | Ax | causal > s.o. certified (p=0.008, 5 seeds) |
| ERCOT NORTH_C B80 | Ax | causal > s.o. trending (p=0.059, 5 seeds) |
| Hillstrom pooled B20/B40/B80 | RF | s.o. > causal certified at all three |
| Hillstrom primary B40 | RF | s.o. > causal certified |
| Hillstrom primary B80 | RF | near-parity (bimodal causal tail) |
| Criteo Run 1 (B20/B40/B80) | Ax | near-parity (exact tie) |
| Criteo Run 2 (B20/B40/B80) | Ax | near-parity |

The honest reading is:

1. The engine is architecturally domain-portable; adapters exist for
   marketing, ML training, and energy load.
2. Causal advantage over `surrogate_only` currently depends on noise
   burden, landscape structure, and search-space breadth. That is why
   medium and high-noise demand-response show certified causal wins,
   why dose-response needs Ax specifically to show a causal win, and why
   interaction flipped from a `surrogate_only` advantage to near-parity
   after Sprint 29.
3. On real data, the conditional advantage has shown up once (ERCOT),
   was absent once (Hillstrom, under RF fallback, narrow space), and
   was absent once under Ax (Criteo).
4. No merged real-data row currently shows `causal` statistically
   beating `random`. ERCOT is the best separation from `surrogate_only`
   but still not a full causal-advantage claim relative to random.

The project framing therefore stays: **trustworthy automated causal
research harness, one conditional real-world positive (ERCOT), one
clean non-energy boundary (Hillstrom), one near-parity under a larger
Ax-primary marketing surface (Criteo).**

### 1e. Open Bandit is the next frontier, not another immediate binary marketing rerun

The Sprint 34 recommendation is to move the project to multi-action /
logged-policy data via Open Bandit rather than run a third binary
marketing benchmark. This scorecard endorses that direction on the
evidence:

1. Hillstrom and Criteo together already answer the narrow question
   "does causal guidance automatically win on binary uplift marketing
   data?" -- the honest answer is "not under the current adapter and
   not on the first two real tests." Another binary rerun would mostly
   re-test the same question.
2. Criteo's near-parity held even after the contractual heterogeneous
   follow-up, which removes the cheapest remaining explanation that the
   result was a degenerate-surface artifact.
3. The next useful empirical question is whether causal guidance helps
   on a structurally different intervention format. Open Bandit provides
   logged multi-action policy data with real propensity structure. That
   is the right next surface.
4. A binary-marketing rerun would also risk re-entangling the adapter
   discussion and the optimizer-core discussion, which Sprint 28 already
   separated into Ax-primary and RF-secondary tracks.

Open Bandit is therefore the Sprint 34 critical path. Sprint 33 is a
documentation / memory-sync sprint whose job is to put the correct
post-Hillstrom, post-Criteo project state on the record so the Open
Bandit contract work does not start from a stale snapshot.

## 2. Evidence Summary

### 2a. Synthetic benchmark state (carried forward from Sprint 29)

No synthetic benchmark was rerun in Sprint 30, Sprint 31, Sprint 32, or
Sprint 33. The Sprint 29 Ax-primary table still holds:

| # | Benchmark | B80 causal mean | Two-sided MWU p | Classification |
|---|-----------|-----------------|------------------|----------------|
| 1 | Base energy | 1.01 | 0.112 | Trending |
| 2 | Medium-noise | 1.19 | 0.002 | Certified causal win |
| 3 | High-noise | 1.08 | 0.001 | Certified causal win |
| 4 | Confounded | -- | -- | All strategies misled |
| 5 | Null control | 0.2% max delta | PASS | 11th clean synthetic run (through S29, S26 did not rerun) |
| 6 | Interaction policy | 1.90 | 0.225 | Near-parity (was s.o.) |
| 7 | Dose-response | 0.22 | 0.003 | Certified causal win (Ax-primary) |

### 2b. Real-data state after Sprint 33

| Domain | Dataset / slice | Backend | Seeds | Comparison | Two-sided MWU p | Classification |
|--------|-----------------|---------|-------|------------|------------------|----------------|
| Energy | ERCOT COAST B80 | Ax | 5 | causal vs s.o. | 0.008 | Certified causal |
| Energy | ERCOT COAST B80 | Ax | 5 | causal vs random | 0.690 | Not significant |
| Energy | ERCOT NORTH_C B80 | Ax | 5 | causal vs s.o. | 0.059 | Trending causal |
| Energy | ERCOT NORTH_C B80 | Ax | 5 | causal vs random | 0.402 | Not significant (random mean lower) |
| Marketing | Hillstrom primary B20 | RF | 10 | causal vs s.o. | 0.060 | Trending s.o. |
| Marketing | Hillstrom primary B40 | RF | 10 | causal vs s.o. | 0.0001 | Certified s.o. |
| Marketing | Hillstrom primary B80 | RF | 10 | causal vs s.o. | 0.817 | Near-parity (bimodal causal tail) |
| Marketing | Hillstrom primary B80 | RF | 10 | causal vs random | 0.0004 | Certified causal > random |
| Marketing | Hillstrom pooled B20 | RF | 10 | causal vs s.o. | 0.017 | Certified s.o. |
| Marketing | Hillstrom pooled B40 | RF | 10 | causal vs s.o. | 0.002 | Certified s.o. |
| Marketing | Hillstrom pooled B80 | RF | 10 | causal vs s.o. | 0.019 | Certified s.o. |
| Marketing | Criteo Run 1 B20/B40/B80 | Ax | 10 | causal vs s.o. | 1.000 | Near-parity (exact tie) |
| Marketing | Criteo Run 2 B20 | Ax | 10 | causal vs s.o. | 0.168 | Near-parity |
| Marketing | Criteo Run 2 B40 | Ax | 10 | causal vs s.o. | 1.000 | Near-parity |
| Marketing | Criteo Run 2 B80 | Ax | 10 | causal vs s.o. | 0.368 | Near-parity |

All p-values are two-sided Mann-Whitney U. Population std (ddof=0) is
the table convention; sample-pooled std (ddof=1) is used in the source
reports when Cohen's d is quoted.

### 2c. Guardrails that held through Sprint 33

1. Null control passed on Hillstrom and Criteo (Run 1 and Run 2). The
   Criteo null band was tight; the Hillstrom null-control pass had the
   known caveat that permuted-outcome policy values can still exceed
   the simple baseline.
2. IPS diagnostics on Criteo were stable at 85:15 imbalance (ESS
   ~849,982 on the optimized strategies, zero zero-support events).
3. The propensity gate on the Criteo 1M subsample passed (max deviation
   0.79pp vs 2pp threshold).
4. Backend provenance was preserved. Hillstrom is explicitly an
   RF-backend result. Criteo is explicitly an Ax/BoTorch-backend result.
   This scorecard does not mix them in the same verdict row.

## 3. Verdict Justification

The verdict **GENERALITY IS REAL BUT CONDITIONAL** rests on four
independent observations:

1. **ERCOT is real and durable at 5 seeds.** The COAST signal
   (p=0.008, two-sided MWU) is the only certified real-world causal vs
   `surrogate_only` separation in the portfolio. It broke the long-standing
   Sprint 16 "causal identical to `surrogate_only`" result on real data.
2. **Hillstrom is a specific, diagnosable non-energy boundary.** Pooled
   B20/B40/B80 are all certified `surrogate_only` advantages. The result
   was produced under RF fallback on a three-variable search space, so
   the boundary is a conditional one rather than a general marketing-domain
   claim. Causal still beats `random` at primary B80, which is informative
   about where the causal path is not inert.
3. **Criteo, under Ax/BoTorch and after the mandatory heterogeneous
   follow-up, came back near-parity.** That removes the cheapest
   "degenerate-surface artifact" explanation for the tie. It does not
   flip the project's synthetic Ax wins, but it does mean a larger
   Ax-primary binary uplift benchmark did not produce a causal advantage.
4. **The synthetic Ax boundary is unchanged.** Medium-noise, high-noise,
   and dose-response remain certified causal wins. Base is trending.
   Interaction is near-parity. The null control is clean. That is the
   evidence that the engine is still a general causal research harness
   rather than only an ERCOT tool.

Alternative verdicts considered and rejected:

- **GENERALIZATION CONFIRMED** -- rejected because Hillstrom and Criteo
  both failed to return a causal advantage and Criteo is the
  backend-matched non-energy test.
- **GENERALIZATION REFUTED** -- rejected because the Criteo surface and
  Hillstrom adapter configuration are both narrow, the synthetic Ax
  boundary still holds, and ERCOT is a real and durable signal.
- **ENERGY-ONLY TOOL** -- rejected because the engine shares the same
  Sprint 29 default across all rows, dose-response is synthetic but
  non-energy, and the engine ran cleanly end-to-end on Criteo.

## 4. Sprint 33 Deliverables

| Deliverable | PR | Status | Role |
|-------------|-----|--------|------|
| Criteo benchmark contract (Sprint 32) | #178 | Merged | Pinned the executable Criteo shape |
| Criteo benchmark report (Run 1 + Run 2) | #180 | Merged | First Ax-primary marketing benchmark; near-parity verdict |
| Sprint 31 Hillstrom lessons learned | local doc | Not tracked by this PR (local note authored alongside PR #176) | Captures the specific Hillstrom boundary |
| This scorecard | (this PR) | This PR | Sprint 33 closure and Sprint 34 bridge |

## 5. What Changed Between Sprint 30 And Sprint 33

| Dimension | Sprint 30 state | Sprint 33 state |
|-----------|-----------------|-----------------|
| Real ERCOT | causal > s.o. on COAST (p=0.008), trending on NORTH_C (p=0.059) | Unchanged; still the strongest real-world positive |
| Non-energy empirical evidence | Hillstrom harness merged, no results | Hillstrom certified s.o. pooled slice; Criteo near-parity Run 1 + Run 2 |
| Backend coverage on real marketing data | None | Hillstrom (RF), Criteo (Ax/BoTorch) |
| Marketing-generalization question | Unanswered | Answered as conditional near-parity / s.o. advantage under current adapter |
| IPS / null-control discipline on marketing data | Untested at scale | Passed on Criteo 1M, passed on Hillstrom |
| Next frontier | Hillstrom + Criteo | Open Bandit / multi-action |

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ERCOT 5-seed signal loosens at 10 seeds | Medium | High -- would weaken the one real-world positive | Keep the 10-seed rerun on the backlog; do not overclaim from 5-seed numbers today |
| Future agents misread Criteo near-parity as "marketing is causal-flat" | Medium | Medium -- would distort the public framing | Scorecard and handoff must state this is a specific surface / adapter result, not a general marketing claim |
| Sprint 34 reopens binary marketing instead of moving to multi-action | Low | Medium -- would slow the generality program | Sprint 34 recommendation and this scorecard both call this out as out of scope |
| Open Bandit contract mis-scopes multi-action requirements | Medium | Medium -- adapter / evaluator work may be heavier than expected | Sprint 34 is explicitly a contract sprint, not an implementation sprint |

## 7. What Sprint 34 Should Do

Per the [Sprint 34 recommendation](../plans/24-sprint-34-recommendation.md):

1. Draft the Open Bandit contract and multi-action architecture brief in
   a single authoritative document.
2. Choose a narrow first scope: one campaign-policy slice, offline
   evaluation only, one primary reward metric, one
   random / logging-policy baseline.
3. Define the minimum multi-action adapter interface and the minimum
   OPE stack required for an honest first run.
4. Specify null-control, support, and estimator-stability gates for the
   multi-action setting.
5. Do not start coding a multi-action adapter before the contract is
   merged.
6. Do not reopen Hillstrom or Criteo as the main lane.

## 8. What Not To Do

1. Do not frame ERCOT as a general causal advantage on real data. The
   `causal` vs `random` gap on ERCOT has not closed.
2. Do not frame Hillstrom as proving marketing is surrogate-only. It is
   one dataset, under RF fallback, with a narrow three-variable space.
3. Do not frame Criteo as refuting causal guidance. It is one larger
   dataset on a two-variable active surface under Ax/BoTorch, with
   synthesized-label heterogeneity, not a real heterogeneous
   treatment-effect structure.
4. Do not fold Hillstrom and Criteo into a single "marketing result" row.
   They are backend-distinct (RF vs Ax), dataset-distinct, and
   search-space-distinct. They should be reported separately.
5. Do not treat this scorecard as reopening the Sprint 29 synthetic
   verdicts. Those rows did not re-run.

## Provenance

### Source documents

1. [Sprint 30 reality-and-generalization scorecard](sprint-30-reality-and-generalization-scorecard.md) -- PR #164
2. [Sprint 30 ERCOT reality report](sprint-30-ercot-reality-report.md) -- PR #166
3. [Sprint 30 portability brief](sprint-30-general-causal-portability-brief.md) -- PR #165
4. [Sprint 31 Hillstrom benchmark report](sprint-31-hillstrom-benchmark-report.md) -- PR #176
5. [Sprint 31 Hillstrom lessons learned](sprint-31-hillstrom-lessons-learned.md) -- local note authored alongside PR #176, not separately merged
6. [Sprint 32 Criteo benchmark contract](sprint-32-criteo-benchmark-contract.md) -- PR #178
7. [Sprint 33 Criteo benchmark report](sprint-33-criteo-benchmark-report.md) -- PR #180
8. [Sprint 29 optimizer-core scorecard](sprint-29-optimizer-core-scorecard.md) -- PR #161
9. [Sprint 34 recommendation](../plans/24-sprint-34-recommendation.md)
10. [Benchmark state](../plans/07-benchmark-state.md)
11. [Handoff document](handoff.md)

### Statistical conventions

- All p-values are two-sided Mann-Whitney U tests unless otherwise noted.
- Population std (ddof=0) in tables; sample-pooled std (ddof=1) is used
  in source reports when Cohen's d is quoted.
- "Certified" = p <= 0.05; "Trending" = 0.05 < p <= 0.15;
  "Not significant" = p > 0.15.
- "Winner" is reserved for statistically significant rows. When a row
  is directionally better but p > 0.05, it is called "trending" or
  "mean improved" rather than "won."
- Backend provenance is preserved: RF fallback and Ax/BoTorch results
  are not mixed in the same verdict row.
