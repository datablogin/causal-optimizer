# Causal Optimizer Handoff Document

**Date:** 2026-04-11  
**Current sprint:** 29  
**Current state:** Sprint 29 complete — GENERALITY IMPROVED  
**Main repo status:** safe restart point is this doc + benchmark state file

## What The Next Agent Needs To Know

Sprint 29 is complete.  The verdict is **GENERALITY IMPROVED**.

What Sprint 29 established:

1. Ax/BoTorch remains the primary backend for row-level causal-advantage claims
2. RF fallback remains a secondary drift-detection signal, not a substitute baseline
3. dose-response is a certified Ax-primary causal win (p=0.003)
4. medium-noise and high-noise are certified and improved (p=0.002, p=0.001)
5. base mean improved (1.13→1.01) but lost significance (p=0.112, was 0.045) — now trending
6. interaction flipped from surrogate-only advantage to near-parity (1.90 vs 2.18, p=0.225)
7. production default `causal_exploration_weight` is now `0.0`
8. `causal_softness` is intentionally still `0.5`
9. null control: 11th clean run (0.2%)
10. no benchmark row has a statistically significant surrogate-only advantage under Ax

The next task is Sprint 30 planning.

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

### Sprint 29 Merged So Far

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

## Current Best Evidence

Certified Ax-primary causal wins (Sprint 29, after `causal_exploration_weight=0.0`):

1. medium-noise B80: causal mean/std `1.19 / 1.52`, causal wins `10/10`, two-sided `p=0.002`
2. high-noise B80: causal mean/std `1.08 / 1.72`, causal wins `10/10`, two-sided `p=0.001`
3. dose-response B80: causal mean/std `0.22 / 0.03`, causal wins `9/10`, two-sided `p=0.003`

Trending (mean improved but p > 0.05):

1. base B80: causal mean/std `1.01 / 1.10`, catastrophic seeds `0/10`, causal wins `7/10`, two-sided `p=0.112` (was `p=0.045` under old default)

Near-parity (improved from s.o. advantage):

1. interaction policy B80: causal mean `1.90` vs s.o. `2.18`, `p=0.225` (was s.o. winning at `p=0.014`)

Remaining boundary rows:

1. confounded demand-response: all strategies can still be misled
2. null control: 11 clean runs across 12 sprint slots; Sprint 26 intentionally did not rerun it

Notes:

1. `graph_only` was empirically best on interaction in PR #159, but the exact Ax-path mechanism is not isolated by that ablation alone
2. the regression gate (PR #161) confirmed the default change improved generality — interaction flipped to near-parity, all rows improved in mean regret

## What Sprint 30 Should Decide

1. whether to pursue a certified interaction win (reduce `causal_softness` to 0.0, increase seeds to 20)
2. whether to investigate the base row's loss of significance (p=0.112)
3. whether to return to real-world ERCOT benchmarks instead of further synthetic tuning

## Files To Read First

1. [07-benchmark-state.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/07-benchmark-state.md)
2. [sprint-29-optimizer-core-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-optimizer-core-scorecard.md)
3. [sprint-29-optimizer-core-regression-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-optimizer-core-regression-report.md)
4. [sprint-28-backend-baseline-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-28-backend-baseline-scorecard.md)
5. [20-sprint-29-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/20-sprint-29-recommendation.md)

## Immediate Instructions For The Next Agent

Sprint 29 is complete.  Plan Sprint 30:

1. read the [Sprint 29 optimizer-core scorecard](thoughts/shared/docs/sprint-29-optimizer-core-scorecard.md) for the full verdict
2. decide whether to pursue a certified interaction win (reduce `causal_softness`, increase seeds) or return to real-world ERCOT benchmarks
3. the benchmark suite is ready to evaluate further optimizer changes
4. the base row's loss of significance (p=0.112) may warrant investigation in Sprint 30

## One-Line Situation Summary

Sprint 29 is complete with verdict GENERALITY IMPROVED: all rows improved in mean regret, interaction flipped to near-parity, but base lost significance — Sprint 30 should decide the next direction.
