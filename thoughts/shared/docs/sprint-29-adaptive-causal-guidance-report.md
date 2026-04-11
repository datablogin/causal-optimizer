# Sprint 29 Adaptive Causal Guidance Report

**Date**: 2026-04-11
**Sprint**: 29 (Adaptive Causal Guidance)
**Issue**: #153
**Branch**: `sprint-29/adaptive-causal-guidance`
**Base commit**: `76aeb8f` (Sprint 29 interaction ablation merged to main)

## Verdict

**DEFAULT CHANGED** -- `causal_exploration_weight` default changed from
`0.3` to `0.0`.  This is the narrowest evidence-supported intervention
from the Sprint 29 ablation (PR #159).

## 1. What Exact Default Changed?

| Parameter | Old Default | New Default | Location |
|-----------|------------|------------|----------|
| `causal_exploration_weight` | 0.3 | **0.0** | `engine/loop.py:123`, `optimizer/suggest.py:118` |

The parameter controls the strength of causal bias during exploration
(experiments 1-10).  At 0.0, exploration uses pure LHS with no ancestor
weighting.  The causal graph is still passed to the engine and used
during the optimization phase (alignment bonus, targeted perturbation).

## 2. Where Is the Default Defined in Code?

Two source-of-truth locations:

1. **`causal_optimizer/engine/loop.py`** line 123:
   `ExperimentEngine.__init__()` constructor parameter default
2. **`causal_optimizer/optimizer/suggest.py`** line 118:
   `suggest_parameters()` function parameter default

Both were changed from `0.3` to `0.0`.

Additionally updated:
- **`engine/loop.py`** docstring (line 166-170): updated default
  documentation from "Default 0.3" to "Default 0.0" with Sprint 29
  rationale
- **`tests/unit/test_soft_causal.py`** line 421: test assertion updated
  from `== 0.3` to `== 0.0` with Sprint 29 rationale in docstring

## 3. Why Is This the Narrowest Evidence-Supported Intervention?

The Sprint 29 ablation (PR #159) showed:

| Arm | Weight | Softness | B20 Mean | B80 Mean |
|-----|--------|----------|----------|----------|
| surrogate_only | — | — | 5.44 | 2.18 |
| causal_default | 0.3 | 0.5 | **13.83** | 3.17 |
| no_exploration | 0.0 | 0.5 | 4.76 | 1.90 |
| no_alignment | 0.3 | 0.0 | 7.77 | 2.79 |
| graph_only | 0.0 | 0.0 | 4.13 | **1.80** |

Removing exploration weighting is sufficient to eliminate the B20
catastrophe (13.83 → 4.76).  While graph_only (also removing alignment
bonus) is empirically the best arm, the ablation did not isolate the
exact Ax-path mechanism for the alignment-bonus contribution.  Changing
one parameter at a time is the safer approach:

1. This PR changes `causal_exploration_weight` only
2. Issue #154 runs the full regression gate to verify no regressions
3. If the regression gate passes and interaction improves, the project
   can consider whether to also change `causal_softness` in a separate
   evidence-supported step

## 4. What Did We Intentionally Leave Unchanged?

| Item | Status | Reason |
|------|--------|--------|
| `causal_softness` default (0.5) | **Unchanged** | Alignment bonus effect is not independently harmful enough to justify a second default change in the same PR |
| README public claims | **Unchanged** | Claims should only update after #154 regression gate confirms results |
| Benchmark harness | **Unchanged** | No benchmark code changes needed |
| Ablation script | **Unchanged** | Already uses explicit parameter values, not defaults |
| Other optimizer behavior | **Unchanged** | No heuristic gates, no adaptive logic |

## 5. What Must #154 Verify in the Full Ax-Primary Regression Gate?

Issue #154 must run the Ax-primary regression gate and confirm:

1. **Demand-response family preserved:**
   - Base B80: 0/10 catastrophic, mean < 2.0, causal wins (p <= 0.05)
   - Medium-noise B80: causal wins (p <= 0.01)
   - High-noise B80: causal wins (p <= 0.02)
2. **Dose-response preserved:** causal wins at B80 (p <= 0.05 at 10 seeds)
3. **Interaction improved:** B80 mean regret < 2.0 (currently 3.17 with
   old default; expected ~1.90 based on ablation no_exploration arm)
4. **Null control clean:** max delta < 2%
5. **Confounded unchanged:** all strategies converge to same wrong optimum

**Risks:**
- The demand-response family has always been tested with the old default
  (weight=0.3).  While the ablation suggests exploration weighting is
  harmful on interaction, it was not tested on demand-response.  This
  is the primary risk: if demand-response relied on causal-weighted
  exploration for its wins, the change could regress those rows.
- If demand-response regresses, the fix is to revert this PR and
  instead gate exploration weight behind a landscape-type heuristic
  (e.g., enable only when categoricals are present).

## 6. Test Results

```
$ uv run pytest tests/unit/ -x -q --tb=short
1001 passed, 12 skipped in 255.31s
```

Key tests:
- `test_engine_default_causal_config`: asserts `weight == 0.0`, `softness == 0.5`
- `test_engine_accepts_causal_config_params`: asserts explicit override
  still works (`weight=0.5`, `softness=1.0`)
- All 1001 tests pass with no regressions

## 7. Change Summary

| File | Change |
|------|--------|
| `causal_optimizer/engine/loop.py:123` | Default `0.3` → `0.0` |
| `causal_optimizer/engine/loop.py:166-170` | Docstring updated |
| `causal_optimizer/optimizer/suggest.py:118` | Default `0.3` → `0.0` |
| `tests/unit/test_soft_causal.py:411-422` | Test + docstring updated |
