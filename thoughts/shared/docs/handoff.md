# Causal Optimizer Handoff Document

**Date:** 2026-04-16  
**Current sprint:** 30  
**Current state:** Sprint 30 complete — REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC  
**Main repo status:** safe restart point is this doc + benchmark state file

## What The Next Agent Needs To Know

Sprint 30 is complete.  The verdict is **REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC**.

What Sprint 30 established:

1. The Sprint 29 default change (`causal_exploration_weight=0.0`) produced the first real-world causal vs surrogate-only differentiation on ERCOT data
2. COAST: causal certified better than surrogate-only (p=0.008, two-sided MWU, 5/5 wins at B80)
3. NORTH_C: causal trending better than surrogate-only (p=0.059, two-sided MWU, 4/5 wins at B80)
4. Causal still does not statistically beat random on either ERCOT dataset
5. The portability brief re-anchored the project as domain-agnostic; ERCOT is one validation lane, not the project identity
6. The Hillstrom marketing email benchmark harness is merged and ready to run (first non-energy benchmark)
7. No non-energy benchmark results exist yet — the generality claim is structural, not empirical
8. All Sprint 29 synthetic benchmark results carry forward unchanged
9. The ERCOT signal is based on 5 seeds; 10-seed rerun recommended for Sprint 31

The next task is Sprint 31: run Hillstrom as the first non-energy benchmark AND continue ERCOT 10-seed validation in parallel.

## Current GitHub Status

### Sprint 28 Done

1. [PR #149](https://github.com/datablogin/causal-optimizer/pull/149) merged
   - issue: [#146](https://github.com/datablogin/causal-optimizer/issues/146)
   - result: optimizer-path provenance added to benchmark artifacts
2. [PR #150](https://github.com/datablogin/causal-optimizer/pull/150) merged
   - issue: [#147](https://github.com/datablogin/causal-optimizer/issues/147)
   - result: Ax/BoTorch seven-benchmark regression gate passed
3. [PR #151](https://github.com/datablogin/causal-optimizer/pull/151) merged
   - issue: [#148](https://github.com/datablogin/causal-optimizer/issues/148)
   - result: backend baseline scorecard published with verdict **AX PRIMARY, RF SECONDARY**

### Sprint 29 Done

1. [PR #155](https://github.com/datablogin/causal-optimizer/pull/155) merged
   - issue: [#152](https://github.com/datablogin/causal-optimizer/issues/152)
   - result: trajectory diagnosis found that interaction failure is early causal pressure and dose-response looked real but underpowered at 5 seeds
2. [PR #158](https://github.com/datablogin/causal-optimizer/pull/158) merged
   - follow-up certification on the same Sprint 29 line
   - result: dose-response is now a certified Ax-primary causal win at B80 (`0.19 / 0.03` vs `0.92 / 0.66`, two-sided `p=0.002`, `9/10` wins)
3. [PR #159](https://github.com/datablogin/causal-optimizer/pull/159) merged
   - result: interaction ablation showed exploration weighting is the primary cause of the B20 catastrophe; alignment bonus adds damage but is not the primary cause
4. [PR #160](https://github.com/datablogin/causal-optimizer/pull/160) merged
   - issue: [#153](https://github.com/datablogin/causal-optimizer/issues/153)
   - result: production default `causal_exploration_weight` changed from `0.3` to `0.0`; `causal_softness` left unchanged
5. [PR #161](https://github.com/datablogin/causal-optimizer/pull/161) merged
   - issue: [#154](https://github.com/datablogin/causal-optimizer/issues/154)
   - result: Ax-primary regression gate passed with verdict **GENERALITY IMPROVED**

### Sprint 30 Done

1. [PR #165](https://github.com/datablogin/causal-optimizer/pull/165) merged
   - issue: [#163](https://github.com/datablogin/causal-optimizer/issues/163)
   - result: portability brief re-anchored project as domain-agnostic; engine fully portable, benchmark portfolio energy-heavy
2. [PR #166](https://github.com/datablogin/causal-optimizer/pull/166) merged
   - issue: [#162](https://github.com/datablogin/causal-optimizer/issues/162)
   - result: ERCOT reality report — first causal vs s.o. differentiation on real data (COAST p=0.008, NORTH_C p=0.059)
3. [PR #167](https://github.com/datablogin/causal-optimizer/pull/167) merged
   - result: Hillstrom benchmark contract for first non-energy benchmark
4. [PR #169](https://github.com/datablogin/causal-optimizer/pull/169) merged
   - issue: [#168](https://github.com/datablogin/causal-optimizer/issues/168)
   - result: Hillstrom benchmark harness code merged, ready to run

## Current Best Evidence

### Synthetic Benchmarks (Sprint 29, unchanged in Sprint 30)

Certified Ax-primary causal wins:

1. medium-noise B80: causal mean/std `1.19 / 1.52`, causal wins `10/10`, two-sided `p=0.002`
2. high-noise B80: causal mean/std `1.08 / 1.72`, causal wins `10/10`, two-sided `p=0.001`
3. dose-response B80: causal mean/std `0.22 / 0.03`, causal wins `9/10`, two-sided `p=0.003`

Trending (mean improved but p > 0.05):

1. base B80: causal mean/std `1.01 / 1.10`, catastrophic seeds `0/10`, causal wins `7/10`, two-sided `p=0.112`

Near-parity (improved from s.o. advantage):

1. interaction policy B80: causal mean `1.90` vs s.o. `2.18`, `p=0.225`

Remaining boundary rows:

1. confounded demand-response: all strategies can still be misled
2. null control: 11 clean runs across 12 sprint slots

### Real ERCOT Benchmarks (Sprint 30, new)

1. COAST B80: causal certified better than s.o. (MAE 104.88 vs 105.72, two-sided `p=0.008`, 5/5 wins); causal vs random not significant (p=0.690)
2. NORTH_C B80: causal trending better than s.o. (MAE 132.48 vs 132.98, two-sided `p=0.059`, 4/5 wins); causal vs random not significant (p=0.402)
3. Both results at 5 seeds only; 10-seed rerun recommended

### Non-Energy Benchmarks

1. Hillstrom email campaign: harness merged (PR #169), no results yet
2. Criteo uplift: identified, not started
3. Open Bandit Pipeline: identified, not started

## What Sprint 31 Should Do

1. Run Hillstrom as the first non-energy benchmark (highest priority — tests generality claim)
2. Rerun ERCOT with 10 seeds to firm up NORTH_C p=0.059 and COAST signal
3. Publish Sprint 31 generalization scorecard synthesizing both results

## Files To Read First

1. [07-benchmark-state.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/07-benchmark-state.md)
2. [sprint-30-reality-and-generalization-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-30-reality-and-generalization-scorecard.md)
3. [sprint-30-ercot-reality-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-30-ercot-reality-report.md)
4. [sprint-30-general-causal-portability-brief.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-30-general-causal-portability-brief.md)
5. [sprint-29-optimizer-core-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-optimizer-core-scorecard.md)

## Immediate Instructions For The Next Agent

Sprint 30 is complete.  Execute Sprint 31:

1. read the [Sprint 30 reality-and-generalization scorecard](thoughts/shared/docs/sprint-30-reality-and-generalization-scorecard.md) for the full verdict
2. run the Hillstrom benchmark harness (PR #169) with 10 seeds at B20/B40/B80 -- this is the first non-energy empirical test
3. rerun ERCOT NORTH_C and COAST with 10 seeds (5 incremental per strategy-budget-dataset) to firm up the real-world signal
4. publish a Sprint 31 generalization scorecard synthesizing Hillstrom + extended ERCOT results

## One-Line Situation Summary

Sprint 30 is complete with verdict REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC: first ERCOT causal vs s.o. differentiation (COAST p=0.008, NORTH_C p=0.059) but no non-energy results yet -- Sprint 31 should run Hillstrom and extend ERCOT to 10 seeds.
