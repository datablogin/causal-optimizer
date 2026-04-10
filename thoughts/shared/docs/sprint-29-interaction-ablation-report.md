# Sprint 29 Interaction Ablation Report

**Date**: 2026-04-10
**Sprint**: 29 (Adaptive Causal Guidance)
**Issue**: #152
**Branch**: `sprint-29/interaction-ablation`
**Base commit**: `92d3a47` (Sprint 29 dose-response 10-seed merged to main)
**Optimizer path**: ax_botorch (ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0)

## Verdict

**BOTH COMPONENTS CONTRIBUTE, BUT EXPLORATION WEIGHTING IS THE PRIMARY
PROBLEM.** Disabling causal-weighted exploration alone eliminates the B20
catastrophe and recovers competitive performance.  Disabling both
exploration weighting and alignment bonus produces the best arm overall,
beating surrogate-only at every budget.

## 1. Executive Summary

The Sprint 29 trajectory diagnosis (PR #155) hypothesized that early
causal pressure causes the interaction row's B20 failure, but could not
separate the contributions of causal-weighted exploration
(experiments 1-10) from the optimization-phase alignment bonus
(experiment 11+) without an ablation.

This report runs a 5-arm ablation study on the interaction policy
benchmark under the Ax/BoTorch primary path:

| Arm | Exploration Weight | Alignment Bonus | Graph? |
|-----|-------------------|-----------------|--------|
| surrogate_only | 0.3 (irrelevant) | 0.5 (irrelevant) | No |
| causal_default | 0.3 | 0.5 | Yes |
| no_exploration | 0.0 | 0.5 | Yes |
| no_alignment | 0.3 | 0.0 | Yes |
| graph_only | 0.0 | 0.0 | Yes |

**Key findings:**

1. **graph_only is the best arm** at B80 (1.80, std 0.02), beating
   surrogate_only (2.18, std 0.75) and all other causal variants
2. **Exploration weighting is the primary cause of the B20 catastrophe:**
   removing it (no_exploration, graph_only) eliminates the failure;
   keeping it (causal_default, no_alignment) preserves it
3. **Alignment bonus adds modest harm on top of exploration weighting:**
   no_alignment (7.77 at B20) is better than causal_default (13.83) but
   still worse than surrogate_only (5.44)
4. **Graph-only causal guidance (no early pressure) outperforms
   surrogate-only** -- the causal graph's variable pruning helps when
   it does not distort exploration or optimization

## 2. Results by Arm and Budget

**Std convention:** all std values use population std (ddof=0).

### 2a. Mean Regret

| Arm | B20 Mean (Std) | B40 Mean (Std) | B80 Mean (Std) |
|-----|---------------|---------------|---------------|
| surrogate_only | 5.44 (1.83) | 4.20 (1.97) | 2.18 (0.75) |
| causal_default | **13.83** (5.06) | 5.09 (2.07) | 3.17 (1.61) |
| no_exploration | 4.76 (1.77) | 2.12 (0.60) | 1.90 (0.23) |
| no_alignment | **7.77** (2.02) | 5.26 (1.37) | 2.79 (1.72) |
| graph_only | **4.13** (2.07) | **1.81** (0.12) | **1.80** (0.02) |

### 2b. MWU vs Surrogate-Only (Two-Sided)

| Arm | B20 p | B40 p | B80 p | B80 Direction |
|-----|-------|-------|-------|---------------|
| causal_default | **0.002** (worse) | 0.469 | **0.014** (worse) | Worse |
| no_exploration | 0.487 | 0.057 | 0.225 | Better (not significant) |
| no_alignment | **0.020** (worse) | 0.118 | 0.677 | Worse (not significant) |
| graph_only | 0.304 | **0.018** (better) | 0.303 | Better (not significant) |

### 2c. Win Counts vs Surrogate-Only at B80

| Arm | Causal Wins | S.O. Wins | Ties |
|-----|------------|-----------|------|
| causal_default | 1/10 | 9/10 | 0 |
| no_exploration | 6/10 | 4/10 | 0 |
| no_alignment | 3/10 | 7/10 | 0 |
| graph_only | 6/10 | 4/10 | 0 |

## 3. B20 Per-Seed Detail

The B20 catastrophe is where the mechanism is clearest.

| Seed | S.O. | Causal Default | No Exploration | No Alignment | Graph Only |
|------|------|---------------|----------------|-------------|-----------|
| 0 | 5.55 | **13.02** | 2.73 | 7.40 | 1.84 |
| 1 | 5.55 | **18.28** | 3.34 | **13.27** | 3.29 |
| 2 | 5.55 | **17.82** | 5.55 | 7.19 | 2.18 |
| 3 | 5.55 | **19.67** | 5.55 | 6.27 | 2.59 |
| 4 | 2.14 | 6.77 | 2.20 | 5.90 | 3.16 |
| 5 | 5.55 | **16.40** | 3.43 | 6.22 | 5.74 |
| 6 | 6.93 | **19.88** | 6.33 | 7.40 | 2.77 |
| 7 | 2.22 | 6.85 | 6.93 | 7.54 | 4.34 |
| 8 | 7.65 | **12.20** | 3.93 | 9.01 | 7.65 |
| 9 | 7.72 | 7.45 | 7.57 | 7.56 | 7.72 |

**Pattern:** causal_default has 7/10 seeds above 12.0 at B20.
no_alignment has 1/10 above 12.0 (seed 1: 13.27).
no_exploration and graph_only have 0/10 above 12.0.

The exploration weighting (weight=0.3) is the necessary condition for
the B20 catastrophe.  The alignment bonus amplifies it but does not
cause it independently.

## 4. Mechanism Analysis

### 4a. Isolating the Exploration Weighting Effect

Comparing arms that differ only in exploration weighting:

| Comparison | B20 | B40 | B80 |
|-----------|-----|-----|-----|
| causal_default (w=0.3, s=0.5) | 13.83 | 5.09 | 3.17 |
| no_exploration (w=0.0, s=0.5) | 4.76 | 2.12 | 1.90 |
| **Difference** | **-9.07** | **-2.97** | **-1.27** |

Removing exploration weighting reduces B20 regret by 9.07 (66%).  The
effect persists at B40 and B80, suggesting the early damage accumulates.

### 4b. Isolating the Alignment Bonus Effect

Comparing arms that differ only in alignment bonus:

| Comparison | B20 | B40 | B80 |
|-----------|-----|-----|-----|
| no_alignment (w=0.3, s=0.0) | 7.77 | 5.26 | 2.79 |
| causal_default (w=0.3, s=0.5) | 13.83 | 5.09 | 3.17 |
| **Difference (s=0.5 - s=0.0)** | **+6.06** | **-0.17** | **+0.38** |

Adding alignment bonus on top of exploration weighting makes B20
worse by 6.06.  But without exploration weighting:

| Comparison | B20 | B40 | B80 |
|-----------|-----|-----|-----|
| graph_only (w=0.0, s=0.0) | 4.13 | 1.81 | 1.80 |
| no_exploration (w=0.0, s=0.5) | 4.76 | 2.12 | 1.90 |
| **Difference (s=0.5 - s=0.0)** | **+0.63** | **+0.31** | **+0.10** |

Without exploration weighting, alignment bonus adds only 0.63 at B20
(modest, not catastrophic) and 0.10 at B80.

### 4c. The Graph-Only Surprise

The most striking result is that **graph_only** (causal graph present,
but zero exploration weighting and zero alignment bonus) is the best arm
at every budget:

| Budget | graph_only | surrogate_only | Gap |
|--------|-----------|---------------|-----|
| B20 | 4.13 | 5.44 | -1.31 (24% better) |
| B40 | 1.81 | 4.20 | -2.39 (57% better) |
| B80 | 1.80 | 2.18 | -0.38 (17% better) |

Graph-only at B80 has std 0.02 — near-zero variance across all 10
seeds.  This means the presence of the causal graph is empirically
beneficial on this surface when early pressure is removed.

The exact Ax-path mechanism is not isolated by this ablation.  On the
Ax path with `causal_softness=0.0`, `_suggest_bayesian()` runs in
soft mode with `ax_focus=None`, so Ax optimizes all 7 variables during
the optimization phase.  The graph's presence may influence candidate
generation or phase-transition logic in ways this study does not
measure.  What we can say: graph_only empirically outperforms
surrogate_only at every budget, and the benefit is not due to
exploration weighting or alignment bonus (both are zero).

## 5. Mechanism Verdict

| Question | Answer |
|----------|--------|
| Is exploration weighting the main problem? | **Yes** — removing it eliminates the B20 catastrophe (13.83 → 4.76) |
| Is alignment bonus the main problem? | **No** — it amplifies the damage when exploration weighting is present, but alone adds only 0.63 at B20 |
| Do both contribute? | **Yes** — both add harm, but exploration weighting is 9.07 vs alignment's 0.63 when isolated |
| Is the causal graph itself harmful? | **No** — graph_only outperforms surrogate_only at every budget |

**Mechanism:** causal-weighted exploration (weight=0.3) biases the
initial LHS design toward ancestor-dimension coverage, which produces a
worse exploration history on the 3-way super-additive interaction
surface.  The alignment bonus (softness=0.5) further amplifies this by
rewarding ancestor-displacement during early optimization when the GP
cannot guide it.  Removing exploration weighting is sufficient to
eliminate the catastrophe; removing both produces the best overall arm.

## 6. Evidence vs Speculation

### Evidence-supported

1. Exploration weighting causes the B20 catastrophe: removing it drops
   B20 from 13.83 to 4.76 (ablation data)
2. Alignment bonus amplifies but does not independently cause the failure:
   no_alignment B20 is 7.77, better than default but still worse than
   surrogate_only (ablation data)
3. Graph-only outperforms surrogate-only at every budget (ablation data)
4. Graph-only at B80 has std 0.02 (near-zero variance, ablation data)

### Plausible but not directly measured

1. The exploration weighting harms this surface specifically because of
   the 3-way interaction structure (plausible from surface analysis, but
   not tested on other interaction surfaces)
2. The improvement from graph-only comes from some graph-aware code path
   in the Ax optimization pipeline (plausible, but the exact mechanism
   is not isolated by this ablation — on the Ax path with softness=0.0,
   Ax optimizes all 7 variables with ax_focus=None)

## 7. Recommendation for Issue #153

**Set `causal_exploration_weight=0.0` as the default** for the causal
strategy, and consider reducing `causal_softness` to 0.0 as well.

Justification:
1. graph_only (w=0.0, s=0.0) is the best arm on the interaction
   benchmark, and previous sprints showed the demand-response family
   runs with the default (w=0.3, s=0.5) — a regression gate is needed
   to verify that demand-response wins are preserved under the new
   defaults
2. The causal graph's presence is empirically beneficial on the
   interaction row when early pressure is removed (exact Ax-path
   mechanism not isolated by this ablation)
3. This is a narrow, testable change: modify two defaults, rerun the
   Ax-primary regression gate

**Success criteria for #153:**
- Interaction B80: regret <= 2.0 (currently 3.17 → expect ~1.80)
- Demand-response B80: all three variants preserve their certified wins
- Null control: remains clean
- Dose-response: remains at or near 0.19

**Alternative if demand-response regresses:** gate the exploration
weight behind a landscape-complexity heuristic (e.g., reduce weight
when the search space has no categoricals or when dimensionality is
below a threshold).

## 8. Provenance

### Environment

- Python 3.13.12
- ax-platform 1.2.4, botorch 0.17.2, torch 2.10.0
- git SHA: 92d3a47

### Optimizer Path

All runs confirmed `optimizer_path: "ax_botorch"` in provenance.

### Artifacts

Local (not committed):
```
artifacts/sprint-29-interaction-ablation/ablation_results.json
```

### Exact Commands Run

```bash
uv run python3 scripts/interaction_ablation.py \
  --data-path /path/to/ercot_north_c_dfw_2022_2024.parquet \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --budgets 20,40,80 \
  --output ablation_results.json
```

Runtime: 3469 seconds (~58 minutes), 150 runs (5 arms × 3 budgets × 10 seeds).

## 9. Summary Table

| Arm | Weight | Softness | Graph | B20 | B40 | B80 | B80 vs S.O. |
|-----|--------|----------|-------|-----|-----|-----|-------------|
| surrogate_only | — | — | No | 5.44 | 4.20 | 2.18 | baseline |
| causal_default | 0.3 | 0.5 | Yes | **13.83** | 5.09 | 3.17 | worse (p=0.014) |
| no_exploration | 0.0 | 0.5 | Yes | 4.76 | 2.12 | 1.90 | better (n.s.) |
| no_alignment | 0.3 | 0.0 | Yes | 7.77 | 5.26 | 2.79 | worse (n.s.) |
| graph_only | 0.0 | 0.0 | Yes | **4.13** | **1.81** | **1.80** | **better** (n.s.) |
