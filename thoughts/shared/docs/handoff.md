# Causal Optimizer Handoff Document

**Date:** 2026-04-11  
**Current sprint:** 29  
**Current state:** Sprint 29 diagnosis and intervention merged; regression gate and scorecard pending  
**Main repo status:** safe restart point is this doc + benchmark state file

## What The Next Agent Needs To Know

The project is no longer deciding whether Sprint 29 should start.
Sprint 29 is already in its final verification phase.

What is established now:

1. Ax/BoTorch remains the primary backend for row-level causal-advantage claims
2. RF fallback remains a secondary drift-detection signal, not a substitute baseline
3. dose-response is now a certified Ax-primary causal win after the 10-seed rerun
4. interaction is the last remaining surrogate-only advantage
5. the interaction failure mode is early causal pressure, with exploration weighting as the primary culprit
6. production default `causal_exploration_weight` is now `0.0`
7. `causal_softness` is intentionally still `0.5`

The next task is no longer diagnosis.
The next task is the full Ax-primary regression gate and Sprint 29 scorecard.

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
5. [Issue #154](https://github.com/datablogin/causal-optimizer/issues/154)
   - still open
   - this is the current critical path: Ax-primary regression gate + Sprint 29 scorecard

Notes:

1. `#152` and `#153` are complete in substance, though GitHub issue closure may still need cleanup
2. the README is behind the latest merged Sprint 29 evidence and should not be treated as the source of truth until `#154` lands

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

Important caution:

1. `graph_only` was empirically best on interaction in PR #159, but the exact Ax-path mechanism is not isolated by that ablation alone
2. Sprint 29 should not claim the interaction row is solved until `#154` reruns the full gate after the default change

## What Sprint 29 Must Do Now

Sprint 29 should proceed in this order:

1. rerun the Ax-primary regression gate after the `causal_exploration_weight=0.0` default change
2. verify that the demand-response wins are preserved
3. verify that dose-response remains a certified causal win
4. measure whether interaction materially improves without breaking the rest of the suite
5. confirm `optimizer_path: "ax_botorch"` on the compared artifacts
6. confirm null control remains clean if it is rerun
7. publish the Sprint 29 scorecard and update public-facing docs

The first agent in a fresh session should start with [Issue #154](https://github.com/datablogin/causal-optimizer/issues/154) and use:

1. [sprint-29-optimizer-core-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-29-optimizer-core-scorecard.md)

## Files To Read First

1. [07-benchmark-state.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/07-benchmark-state.md)
2. [20-sprint-29-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/20-sprint-29-recommendation.md)
3. [sprint-28-backend-baseline-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-28-backend-baseline-scorecard.md)
4. [sprint-29-trajectory-diagnosis-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-trajectory-diagnosis-report.md)
5. [sprint-29-dose-response-10seed-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-dose-response-10seed-report.md)
6. [sprint-29-interaction-ablation-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-interaction-ablation-report.md)
7. [sprint-29-adaptive-causal-guidance-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-adaptive-causal-guidance-report.md)

## Immediate Instructions For The Next Agent

Start [Issue #154](https://github.com/datablogin/causal-optimizer/issues/154):

1. use [sprint-29-optimizer-core-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-29-optimizer-core-scorecard.md) as the contract
2. run the full Ax-primary gate after the merged `causal_exploration_weight=0.0` default change
3. preserve the demand-response wins if they are still real
4. check whether interaction improves enough to change the cross-family boundary story
5. update the README and benchmark-state documents only after the gate result is established
6. publish the Sprint 29 scorecard with explicit backend and statistical language

## One-Line Situation Summary

Sprint 29 has already diagnosed the failure mode, certified dose-response, and merged the narrow default change; the only remaining step is the full Ax-primary regression gate and scorecard.
