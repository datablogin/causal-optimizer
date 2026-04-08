# Sprint 27 Crossover Scorecard

## Metadata

- **Date**: 2026-04-08
- **Sprint**: 27 (Crossover Scorecard)
- **Issue**: #141
- **Branch**: `sprint-27/crossover-scorecard`
- **Base commit**: `70367f8` (Sprint 27 combined regression gate merged to main)
- **Predecessors**:
  - PR #143 -- Medium-noise crossover variant (Ax/BoTorch path)
  - PR #144 -- Combined regression gate across all 7 benchmarks (RF fallback)

## Verdict

**BOUNDARY CLEARER** -- the medium-noise variant fills the gap between base
and high-noise, producing a smooth causal-advantage gradient.  The
crossover is across benchmark families (landscape structure), not within
the noise-dimension axis.  The combined regression gate confirms
qualitative consistency across all 7 benchmarks under the RF fallback
path.

## 1. Executive Summary

Sprint 27 set out to answer one question: where exactly does causal
advantage end and surrogate advantage begin?  The medium-noise variant
(PR #143) and the 7-benchmark regression gate (PR #144) together produce
the clearest picture yet.

**Within the demand-response family**, causal wins at every tested noise
level.  The advantage is not a threshold effect -- it is a smooth gradient
where causal regret degrades mildly while surrogate-only regret degrades
sharply as noise dimensionality increases.

**Across benchmark families**, the crossover depends on landscape
structure, not noise volume alone.  Smooth, all-continuous landscapes
(dose-response) and interaction-dominated surfaces (interaction policy)
favor surrogate-only.  Noisy, categorical-barrier landscapes
(demand-response variants) favor causal.

The null control remains clean for the 9th run across S18--S27 (S26 did
not re-run).  The combined regression gate passes with the
expected caveat that RF-fallback absolute numbers are not directly
comparable to Sprint 25 Ax/BoTorch priors.

## 2. Core Questions

### 2a. Where does causal currently win?

Causal wins on the demand-response family at all noise levels under
Ax/BoTorch:

| Variant | Dims | Noise | B80 Causal Mean | B80 S.O. Mean | Catastrophic | Causal Wins | Two-Sided p |
|---------|------|-------|-----------------|---------------|--------------|-------------|-------------|
| Base | 5 | 2 | 1.13 | 4.90 | 0/10 | 9/10 | 0.045 |
| Medium | 9 | 6 | 1.87 | 9.61 | 0/10 | 10/10 | 0.007 |
| High | 15 | 12 | 2.57 | 15.23 | 0/10 | 8/10 | 0.014 |

All three results are statistically significant (two-sided MWU).  The
pattern is consistent: causal pruning focuses on the 3 real parents,
providing stable performance regardless of how many noise dimensions are
present.  Surrogate-only degrades roughly linearly with noise
dimensionality.

The high-noise result is additionally confirmed under RF fallback: 9/10
causal wins at B80 (two-sided p=0.017), demonstrating robustness across
optimizer backends.

### 2b. Where does it tie?

Interaction policy tied under Ax/BoTorch (Sprint 26: B80 causal 2.85 vs
s.o. 2.19, two-sided p=0.68).  Both guided strategies beat random
decisively (p=0.0003).  The causal graph prunes noise dimensions but does
not encode the three-way interaction structure, so causal and
surrogate-only converge once enough data is available.

Under RF fallback, the tie shifted to a surrogate-only win (B80 causal
4.31 vs s.o. 1.76, two-sided p=0.0006).  This shift is attributable to
the optimizer backend difference, not a code regression.

### 2c. Where does it lose?

- **Dose-response** (smooth Emax curve, 6D, no categoricals):
  surrogate-only wins decisively under both Ax/BoTorch (S26: s.o. 1.32 vs
  causal 6.51) and RF fallback (S27: s.o. 2.80 vs causal 7.99).

- **Confounded energy** (Simpson's paradox with bidirected edges): all
  strategies are misled.  This is a known limitation -- bidirected edges
  alone are insufficient to deconfound.

### 2d. Does the medium-noise variant clarify the crossover boundary?

**Yes.**  The medium-noise result eliminates two alternative hypotheses:

1. **"Causal advantage is a threshold effect."**  Rejected.  Causal
   regret increases smoothly (1.13 -> 1.87 -> 2.57) rather than jumping
   at a particular noise level.

2. **"The crossover happens within the demand-response family at some
   intermediate noise burden."**  Rejected.  Causal wins at all three
   points (5D, 9D, 15D) with statistical significance.  The crossover to
   surrogate advantage requires a structurally different landscape.

The boundary is now characterized as landscape-structural, not
noise-dimensional:

| Landscape Feature | Causal Advantage | Surrogate Advantage |
|-------------------|------------------|---------------------|
| High noise-to-signal ratio | Yes (all 3 demand-response variants) | No |
| Categorical barriers | Yes (weekday lock-in) | No |
| Smooth continuous surface | No | Yes (dose-response) |
| Interaction-dominated | Tie (Ax) / No (RF) | Tie (Ax) / Yes (RF) |
| Confounded | No (all misled) | No (all misled) |

### 2e. Did the combined regression gate keep trusted results intact?

**Yes, qualitatively.**  The Sprint 27 regression gate ran all 7
benchmarks under the RF fallback path (Ax/BoTorch not installed).
Results:

- **Qualitative family-level conclusions remain usable**, with documented
  backend-sensitive shifts in base (causal win → tie) and interaction
  (tie → surrogate-only win)
- **Null control PASS** (0.2% max delta, 9th clean run)
- **High-noise robust** across backends (9/10 causal wins, two-sided
  p=0.017 under RF)
- **Medium-noise confirmed** (8/10 causal wins, two-sided p=0.026 under
  RF)

The quantitative caveat: the Sprint 25 stability gate targets (0/10
catastrophic, mean < 2.0, std < 3.0 on base B80) were calibrated for
the Ax/BoTorch path.  Under RF fallback, base B80 shows 3/10 catastrophic
seeds (mean 7.32, std 10.14).  This is not a code regression -- the
exploitation-phase categorical sweep is less effective without Ax/BoTorch.
RF-fallback-specific gate targets should be established in Sprint 28.

### 2f. Should Sprint 28 expand further or return to optimizer-core work?

**Return to optimizer-core work.**  The benchmark suite (7 benchmarks
across 4 families) provides sufficient coverage to evaluate optimizer
changes.  The most productive next step is to re-run the combined
regression gate with Ax/BoTorch installed, establishing directly
comparable baselines against the Sprint 25 priors and RF-fallback-specific
gate targets.  Adding more benchmark families would increase coverage
without sharpening the causal-advantage story further.

## 3. Stability Gate Status

### 3a. Sprint 25 Gate (Ax/BoTorch Path -- Trusted Priors)

| Metric | S25 Value | Target | Status |
|--------|-----------|--------|--------|
| Base B80 catastrophic | 0/10 | 0/10 | MET |
| Base B80 mean regret | 1.13 | < 2.0 | MET |
| Base B80 std | 1.40 | < 3.0 | MET |
| High-noise B80 wins | 8/10 (two-sided p=0.014) | directionally strong | MET |
| Null control max delta | 0.2% | < 2% | MET |

These remain the trusted priors.  The Sprint 27 regression gate did not
re-validate them under Ax/BoTorch; it validated the RF fallback path.

### 3b. Sprint 27 Regression Gate (RF Fallback Path)

| Metric | S27 RF Value | Notes |
|--------|-------------|-------|
| Base B80 catastrophic | 3/10 | Not comparable to S25 Ax gate |
| Base B80 mean regret | 7.32 | Not comparable to S25 Ax gate |
| High-noise B80 wins | 9/10 (two-sided p=0.017) | Consistent with S25 |
| Medium-noise B80 wins | 8/10 (two-sided p=0.026) | New baseline |
| Null control max delta | 0.2% | PASS, 9th clean run |
| All 7 benchmarks ran | Yes | First-ever full-suite gate |

### 3c. Null Control Streak

| Sprint | Max Diff | Verdict |
|--------|----------|---------|
| S18 | 0.15% | PASS |
| S19 | 0.15% | PASS |
| S20 | 0.20% | PASS |
| S21 | 0.18% | PASS |
| S22 | 0.23% | PASS |
| S23 | 0.20% | PASS |
| S24 | 0.20% | PASS |
| S25 | 0.20% | PASS |
| S26 | (not re-run) | -- |
| **S27** | **0.20%** | **PASS** |

9 clean null runs across 10 sprints.

## 4. Benchmark Coverage Summary (7 Benchmarks)

| # | Benchmark | Domain | Dims | Cat. | B80 Winner (Ax) | B80 Winner (RF) | Sprint |
|---|-----------|--------|------|------|-----------------|-----------------|--------|
| 1 | Base energy | Demand-response | 5 | 1 | Causal (p=0.045) | Tie (p=0.97) | S18 |
| 2 | Medium-noise | Demand-response | 9 | 1 | Causal (p=0.007) | Causal (p=0.026) | S27 |
| 3 | High-noise | Demand-response | 15 | 1 | Causal (p=0.014) | Causal (p=0.017) | S19 |
| 4 | Confounded | Demand-response | 5 | 1 | None (all misled) | None (all misled) | S19 |
| 5 | Null control | Permuted targets | 5 | 1 | PASS (0.2%) | PASS (0.2%) | S18 |
| 6 | Interaction | Energy policy | 7 | 0 | Tie (p=0.68) | S.O. (p=0.0006) | S26 |
| 7 | Dose-response | Clinical trial | 6 | 0 | S.O. wins | S.O. wins (p=0.008) | S26 |

All p-values are two-sided Mann-Whitney U.

## 5. The Crossover Story

### 5a. Within-Family Gradient (Demand-Response, Ax/BoTorch)

The noise-dimension gradient is smooth and monotonic.  Causal pruning
provides stable performance while surrogate-only degrades proportionally.

```
B80 Mean Regret vs Noise Dimensions (Ax/BoTorch path)

Causal:         1.13 ---- 1.87 ---- 2.57        (slow rise)
Surrogate-only: 4.90 ---- 9.61 ---- 15.23       (steep rise)
                 5D        9D        15D
```

There is no crossover within this family.  Causal wins at every point.

### 5b. Across-Family Picture

The crossover is structural.  On the demand-response family (categorical
barriers, multiple noise dimensions, treatment-effect oracle), causal
wins.  On smooth continuous landscapes (dose-response Emax curve, no
categoricals), surrogate-only wins.  On interaction-dominated surfaces,
the result depends on the optimizer backend (tie under Ax, surrogate-only
under RF).

**What makes causal win:**
- Higher noise burden in the demand-response family, especially when
  paired with categorical barriers
- Treatment-effect structure that aligns with the causal graph

**What makes surrogate-only win:**
- Smooth, low-dimensional, all-continuous landscapes
- Surface structure that is directly learnable by the surrogate

**What neither strategy handles:**
- Confounded data (bidirected edges alone are insufficient)

## 6. Evidence Provenance

All claims trace to one of the following source documents:

| Claim | Source |
|-------|--------|
| Base B80: 0/10 catastrophic, mean 1.13, 9/10 wins (two-sided p=0.045) | Sprint 25 stability scorecard (PR #136) |
| Medium B80: 0/10 catastrophic, mean 1.87, 10/10 wins (two-sided p=0.007) | Sprint 27 medium-noise report (PR #143) |
| High B80: 0/10 catastrophic, mean 2.57, 8/10 wins (two-sided p=0.014) | Sprint 25 stability scorecard (PR #136) |
| Surrogate-only B80 gradient: 4.90 -> 9.61 -> 15.23 | Sprint 25 scorecard + Sprint 27 medium-noise report |
| Interaction: tie at B80 under Ax (two-sided p=0.68) | Sprint 26 expansion scorecard (PR #139) |
| Dose-response: s.o. regret 1.32 vs causal 6.51 at B80 | Sprint 26 expansion scorecard (PR #139) |
| RF regression gate: 7/7 benchmarks passed, family-level conclusions usable with backend-sensitive shifts | Sprint 27 combined regression report (PR #144) |
| RF high-noise: 9/10 causal wins (two-sided p=0.017) | Sprint 27 combined regression report (PR #144) |
| RF medium-noise: 8/10 causal wins (two-sided p=0.026) | Sprint 27 combined regression report (PR #144) |
| Null control: 0.2%, 9th clean run | Sprint 27 combined regression report (PR #144) |

## 7. Continue / Pivot Checklist

### 7a. Coverage Check

1. Did the medium-noise variant fill the gradient gap? **YES** -- smooth
   monotonic gradient across 5D, 9D, 15D.
2. Did the regression gate cover all benchmarks? **YES** -- first-ever
   7-benchmark combined gate.
3. Did null control remain clean? **YES** -- 0.2%, 9th clean null run
   (S26 did not re-run).
4. Did any prior result regress? **NO code regressions** -- family-level
   conclusions hold under RF fallback, with documented backend-sensitive
   shifts on base and interaction benchmarks.

### 7b. What We Learned

1. The crossover boundary is structural (landscape family), not
   dimensional (noise count threshold).
2. Causal advantage scales smoothly with noise burden within the
   demand-response family -- no cliff or threshold detected.
3. The RF fallback path preserves qualitative strategy ordering but has
   different absolute performance characteristics, especially on the base
   benchmark where the exploitation-phase sweep depends on the Ax/BoTorch
   path.
4. The 7-benchmark suite provides sufficient coverage for routine
   regression testing.

## 8. Sprint 28 Recommendation

Sprint 28 should return to optimizer-core work rather than expand
benchmark coverage further:

1. **Re-run the combined regression gate with Ax/BoTorch installed** to
   produce directly comparable numbers against the Sprint 25 priors.
   This is the highest-priority action.
2. **Establish RF-fallback-specific gate targets** as a separate baseline
   so that future regression gates on either path have appropriate
   thresholds.
3. **Add `--optimizer-path` provenance** to benchmark artifacts so that
   results are automatically tagged by backend.
4. **Consider whether the interaction benchmark warrants a harder variant**
   (more noise dimensions or categorical interactions) to separate causal
   from surrogate-only, or whether the current tie/loss result is the
   correct steady-state finding.

The benchmark suite is mature enough to evaluate optimizer changes.  The
next productive step is infrastructure (backend-tagged regression gates),
not more benchmark families.
