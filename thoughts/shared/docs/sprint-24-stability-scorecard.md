# Sprint 24 Stability Scorecard

## Metadata

- **Date**: 2026-04-06
- **Sprint**: 24 (Stability Scorecard)
- **Issue**: #125
- **Branch**: `sprint-24/stability-scorecard`
- **Base commit**: `0945697` (Sprint 24 post-fix report merged to main)
- **Predecessors**: PR #128 (categorical diversity fix), PR #129 (post-fix stability report)

## Verdict

**STILL TOO FRAGILE.**

## 1. Executive Summary

Sprint 24 attempted to fix the known B80 bimodal catastrophic-seed failure
by injecting categorical diversity into the Ax candidate batch (PR #128).
The post-fix stability sweep (PR #129) showed the fix had zero measurable
effect: base B80 results are numerically identical to Sprint 23 hardened
(3/10 catastrophic, mean 5.30, std 6.82).

The fix was correctly implemented, well-tested, and narrowly scoped.  It
failed because the bimodal failure is not in candidate *availability* — it
is in candidate *selection*.  The alignment-only reranker is blind to
`treat_day_filter` (which is not a causal ancestor), so it cannot prefer
diversity candidates even when they are present.  With deterministic seed
forwarding, the same seeds produce identical outcomes regardless of whether
diversity candidates exist in the batch.

Alignment-only remains the correct production default.  Null control
remains clean (7 consecutive sprints, 0.2% max delta).  High-noise causal
advantage is confirmed at B40 and B80 (two-sided p=0.014, 0.037).  But
the project is not ready to broaden benchmark scope while 1-3/10 seeds
produce catastrophic regret on the base positive-control benchmark.

## 2. What Sprint 24 Did

### 2a. Categorical Diversity Fix (PR #128)

`inject_categorical_diversity()` was added to `suggest.py` and called in
`_suggest_bayesian()` after Ax candidate generation and before
alignment-only reranking.  For each categorical variable, the function
checks which values are missing from the candidate batch and creates
diversity candidates by copying the first Ax candidate and substituting
the missing value.

- No mutation of the caller's list (returns a new list when injection occurs)
- Placed after the early-return guard (diversity only when reranking will consume the pool)
- All 956 fast tests pass; full CI green on 3.10-3.13 + Bayesian

### 2b. Post-Fix Stability Sweep (PR #129)

Full benchmark grid:

| Benchmark | Budgets | Seeds | Strategies | Runs |
|-----------|---------|-------|------------|------|
| Base counterfactual | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 90 |
| High-noise counterfactual | 20, 40, 80 | 0-9 | random, surrogate_only, causal | 90 |
| Null control | 20, 40 | 0, 1, 2 | random, surrogate_only, causal | 18 |

Total: 198 benchmark runs on commit `dacd8fe` (PR #128 merged).

## 3. Core Questions

### 3a. Is alignment-only still the correct production default?

**Yes.**  This has been stable since Sprint 21's locked A/B test showed
alignment-only matched or beat balanced reranking.  Sprint 22 confirmed
the revert.  Sprint 23 diagnostics showed the reranker is neutral on the
noise dimension.  Sprint 24 adds no reason to change this.

### 3b. Did the categorical-diversity fix reduce or eliminate the B80 failure?

**No.**  The fix had zero effect.  The mechanism is explained in the
post-fix report (PR #129, section 6b): the alignment-only reranker scores
candidates on causal-ancestor alignment only.  `treat_day_filter` is not a
causal ancestor, so the reranker is neutral on it.  Diversity candidates
score identically to the Ax candidate they were cloned from on the
ancestor dimensions, so the reranker's choice is unchanged.  With
deterministic seed forwarding, the identical seeds fail identically.

### 3c. Did high-noise remain strong?

**Yes, at B40 and B80.**  Causal wins 9/10 at B40 (two-sided p=0.014) and
7/10 at B80 (two-sided p=0.037).  B20 is marginal (7/10 wins, two-sided
p=0.062).  Surrogate-only is catastrophic on 7/10 seeds at B80 (mean
15.30) while causal is catastrophic on only 2/10 (mean 4.56).

### 3d. Did null control remain clean?

**Yes.**  Maximum strategy difference 0.2%, well within the 2% threshold.
This is the 7th consecutive sprint with a clean null control:

| Sprint | Max Diff | Verdict |
|--------|----------|---------|
| S18 | 0.15% | PASS |
| S19 | 0.15% | PASS |
| S20 | 0.20% | PASS |
| S21 | 0.18% | PASS |
| S22 | 0.23% | PASS |
| S23 | 0.20% | PASS |
| S24 | 0.20% | PASS |

### 3e. Is the project ready to broaden benchmark scope?

**No.**  The bimodal B80 failure persists at 1-3/10 catastrophic seeds per
session.  Expanding to new benchmark families before resolving it would
mean running new benchmarks with a known ~10-30% catastrophic-seed rate,
making it impossible to distinguish new-benchmark instability from the
existing categorical lock-in issue.

## 4. Evidence Summary

### 4a. Base B80 Trajectory (Causal Strategy)

| Session | Mean | Std | Catastrophic | Seeds < 1.0 | Fix applied |
|---------|------|-----|-------------|-------------|-------------|
| S20 (pre-Ax) | 11.10 | 10.19 | 6/10 | 4/10 | none |
| S20 (post-Ax) | 3.85 | 5.59 | 2/10 | 6/10 | balanced reranking |
| S21 (balanced) | 3.57 | 5.69 | 2/10 | 7/10 | balanced reranking |
| S21 (align-only) | 0.52 | 0.16 | 0/10 | 10/10 | alignment-only |
| S22 | 3.26 | 5.78 | 2/10 | 8/10 | alignment-only (revert) |
| S23 benchmark | 1.81 | 4.35 | 1/10 | 9/10 | seed forwarding |
| S23 hardened | 5.30 | 6.82 | 3/10 | 6/10 | PyTorch determinism (reverted) |
| **S24** | **5.30** | **6.82** | **3/10** | **6/10** | **categorical diversity** |

The S24 row is numerically identical to S23 hardened because seed
forwarding makes Ax deterministic per seed and the diversity fix does
not change which candidate the reranker selects.

The S21 alignment-only result (0/10 catastrophic) remains an unreproduced
outlier.  Typical sessions show 1-3 catastrophic seeds.

### 4b. High-Noise B80 (Causal vs Surrogate-Only)

| Metric | Causal | Surrogate-Only |
|--------|--------|----------------|
| Mean regret | 4.56 | 15.30 |
| Std | 5.92 | 11.47 |
| Catastrophic | 2/10 | 7/10 |
| Wins | 7/10 | 3/10 |
| MWU two-sided p | 0.037 | — |

High-noise is the strongest evidence for causal advantage.  The causal
strategy avoids the catastrophic mode on 8/10 seeds where surrogate_only
suffers from it.

### 4c. What Fixes Have Been Tried

| Sprint | Fix | Effect on B80 catastrophic rate |
|--------|-----|-------------------------------|
| S20 | Balanced Ax reranking | 6/10 -> 2/10 (but not attributed) |
| S21 | Locked A/B: balanced vs alignment-only | Alignment-only: 0/10 (unreproduced) |
| S22 | Revert to alignment-only | 2/10 (typical range) |
| S23 | PyTorch determinism | 3/10 (worse, reverted) |
| S23 | Seed forwarding | No reduction, but reproducible |
| **S24** | **Categorical diversity in Ax batch** | **3/10 (no change)** |

Three targeted fixes have now failed to reduce the catastrophic rate:
PyTorch determinism, seed forwarding (hygiene only), and categorical
diversity injection.

## 5. Diagnosis Update

### 5a. What We Know

The B80 bimodal failure is fully diagnosed (Sprint 23 report #122):

1. Bad seeds get trapped on `treat_day_filter = "weekday"` during early
   optimization because the GP model learns it is promising
2. The alignment-only reranker is blind to this variable (correct behavior:
   it is not a causal ancestor)
3. The exploitation phase (steps 50-79) perturbs 1-2 variables locally and
   rarely flips the categorical value
4. Good seeds escape during the optimization phase (steps 10-49) when Ax
   happens to generate an "all" candidate with strong continuous parameters

### 5b. What Sprint 24 Proved

Sprint 24 proved that the failure is not in candidate *availability*.
Diversity candidates are present in the batch after injection, but the
reranker does not select them because they score identically to the Ax
candidate they were cloned from on the ancestor dimensions that the
reranker scores on.

This narrows the remaining intervention surface to:

1. **Selection**: change how the reranker scores or breaks ties on
   non-ancestor dimensions
2. **Exploitation**: change how the exploitation phase handles categorical
   variables (currently: perturb 1-2 variables, which rarely flips
   categoricals)
3. **Acceptance**: treat the bimodal mode as irreducible and focus on
   robustness

### 5c. Why This Is Hard

The reranker's blindness to `treat_day_filter` is *correct* behavior.
The variable is not a causal ancestor, and making the reranker aware of
non-ancestor dimensions re-opens the design space that Sprint 21
conclusively resolved (balanced reranking was not the cause of improvement).

Any fix must thread a narrow needle: improve categorical exploration
without reintroducing the balanced-reranking design that Sprint 21 rejected.

## 6. Decision

### 6a. Verdict: STILL TOO FRAGILE

The project has now spent two sprints (23-24) on the B80 bimodal failure.
Sprint 23 diagnosed it conclusively.  Sprint 24's targeted fix had no
effect.  The catastrophic rate (1-3/10) is unchanged from Sprint 22.

This is not "IMPROVED BUT FRAGILE" (the Sprint 23 verdict) because
Sprint 24 delivered no improvement — only confirmation that candidate
availability is not the bottleneck.

This is not "INCONCLUSIVE" because the evidence is clear: the fix had
zero effect for well-understood reasons.

### 6b. What Remains Solid

1. **Null control**: 7 consecutive PASS verdicts, 0.15-0.23% range
2. **Causal advantage on high-noise**: statistically significant at B40
   and B80 under two-sided testing
3. **Alignment-only default**: confirmed across Sprints 21-24
4. **Benchmark infrastructure**: provenance, controlled comparisons, and
   statistical rigor are all working
5. **Root cause understanding**: the failure mechanism is fully diagnosed

### 6c. What Is Not Working

1. **Base B80 stability**: 1-3/10 catastrophic seeds in every session
   except the unreproduced S21 alignment-only outlier
2. **Fix attempts**: three targeted fixes have failed to reduce the
   catastrophic rate
3. **Base B80 statistical significance**: causal vs surrogate_only at B80
   is not significant (two-sided p=0.307) because catastrophic seeds
   inflate variance

## 7. Sprint 25 Recommendation

### 7a. Primary Option: Exploitation-Phase Categorical Sampling

The most promising remaining intervention targets the exploitation phase
(steps 50-79), where categorical lock-in becomes permanent.  Currently,
exploitation perturbs 1-2 continuous variables around the best-so-far
configuration.  Categorical variables are rarely flipped because they are
only one of many possible perturbation targets.

A narrow fix: during exploitation, periodically force a categorical
variable flip (e.g., every 5th exploitation step generates a variant with
each categorical value) to prevent permanent lock-in.

This targets the right phase and mechanism without reopening the reranking
design.

### 7b. Alternative: Accept and Robustify

If the exploitation-phase fix also fails, the project should consider
accepting the bimodal mode as an inherent property of GP-based Bayesian
optimization with categorical variables and shifting focus to:

1. **Multi-seed ensembling**: run 3-5 seeds and take the best result
2. **Seed screening**: detect catastrophic trajectories early (e.g., by
   step 40) and restart with a new seed
3. **Broader benchmark expansion**: proceed despite the 10-30% catastrophic
   rate, using multi-seed ensembling to mitigate it

### 7c. What Not To Do

1. Do not reopen balanced reranking (Sprint 21 rejected it)
2. Do not add categorical awareness to the alignment-only reranker (this
   is equivalent to balanced reranking by another name)
3. Do not broaden benchmark scope until the catastrophic rate is either
   reduced or explicitly accepted with a mitigation strategy

### 7d. Exit Criterion for Sprint 25

Sprint 25 is successful if either:

1. The exploitation-phase fix reduces catastrophic seeds to 0/10 and meets
   the stability gate (mean < 2.0, std < 3.0), OR
2. The project explicitly accepts the bimodal mode, implements multi-seed
   ensembling, and demonstrates that the ensemble approach eliminates
   catastrophic outcomes at the cost of additional compute
