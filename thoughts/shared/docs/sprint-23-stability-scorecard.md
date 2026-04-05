# Sprint 23 Stability Scorecard

## Metadata

- **Date**: 2026-04-05
- **Sprint**: 23 Step 3 (Stability Scorecard)
- **Issue**: #121
- **Branch**: `sprint-23/stability-scorecard`
- **Base commit**: `ec34737` (Sprint 23 hardening merged to main)
- **Predecessors**: Sprint 23 B80 seed diagnostics (#122), Ax stability hardening (#123)

## 1. Executive Summary

**Verdict: IMPROVED BUT FRAGILE.**

Sprint 23 materially advanced our understanding of the bimodal B80
catastrophic-seed failure without eliminating it.  The diagnostics
work (#122) identified a single, clear root cause -- categorical lock-in
on `treat_day_filter = "weekday"` during Ax candidate generation -- that
accounts for the entire ~14.5 regret gap between good and bad seeds.
The hardening work (#123) showed that PyTorch-level determinism does not
address the failure (it made things worse) and that the root cause is
in the GP model's exploration behavior, not floating-point non-determinism.
The seed-forwarding bug fix was retained as correct hygiene.  Null control
remains clean (0.2% max diff, well within the 2% threshold).

Alignment-only re-ranking is confirmed as the correct production default.
The causal advantage is real and statistically significant on both
positive-control benchmarks.  However, the bimodal B80 mode (1-3
catastrophic seeds per 10-seed session) persists across every session
since Sprint 20 except for one outlier (Sprint 21 alignment-only: 0/10).
The project now understands the failure well enough to design a targeted
fix (forced categorical diversity in Ax candidate generation), but that
fix has not been implemented or tested.

Sprint 24 should implement the categorical diversity fix and rerun
the stability sweep before broadening to new benchmark families.

## 2. Evidence from Sprint 23 Diagnostics (#122)

### 2a. Root Cause Identified

The B80 seed diagnostics report conclusively identified the failure
mechanism.  Every catastrophic seed shares a single signature:

- The optimizer's best-so-far parameters have `treat_day_filter = "weekday"`
- The optimizer never generates an `all`-filter candidate that displaces it
- The entire ~14.5 regret gap comes from this one categorical dimension
- Continuous parameters (temp threshold, hour start/end) are nearly identical
  between good and bad seeds

### 2b. Why Bad Seeds Stay Trapped

The self-reinforcing trap operates as follows:

1. Early exploration may land on `weekday` as initial best (by chance)
2. The GP model, trained on exploration data, learns `weekday` is promising
3. Ax generates more `weekday`-heavy candidates, reducing `all` exploration
4. The alignment-only re-ranker is blind to `treat_day_filter` (it is not
   a causal ancestor), so it cannot correct the categorical bias
5. The exploitation phase (steps 50-79) perturbs 1-2 variables at a time
   and rarely flips the categorical value successfully

### 2c. Good Seeds Escape During Optimization

Good seeds escape the trap when Ax happens to generate an `all` candidate
with good continuous parameters during the optimization phase (steps
10-49).  In the diagnostic run, seeds 3, 4, and 9 started with `weekday`
lock-in but escaped at steps 36-43.  The escape is probabilistic, not
guaranteed.

### 2d. Session Non-Determinism

Because Ax was unseeded (prior to the Sprint 23 fix), different sessions
produced different catastrophic seeds:

| Session | Catastrophic Seeds | Count |
|---------|-------------------|-------|
| Sprint 20 (pre-Ax fix) | 0,1,2,4,5,8 | 6/10 |
| Sprint 20 (post-Ax fix) | 1,4 | 2/10 |
| Sprint 21 (balanced) | 8,9 | 2/10 |
| Sprint 21 (align-only) | -- | 0/10 |
| Sprint 22 | 3,9 | 2/10 |
| S23 diagnostic | 2,8 | 2/10 |
| S23 benchmark | 9 | 1/10 |
| S23 hardened | 3,8,9 | 3/10 |

The which-seeds-fail is random; the failure mechanism is constant.

## 3. Evidence from Sprint 23 Hardening (#123)

### 3a. PyTorch Determinism: No Improvement (Reverted)

Sprint 23 tested `torch.use_deterministic_algorithms(True)`,
`torch.manual_seed(seed)`, and `torch.set_num_threads(1)` before each
GP model fit.  Results on both benchmarks were worse:

| Metric | S22 Baseline | S23 Hardened | Delta |
|--------|-------------|-------------|-------|
| Base B80 mean | 3.26 | 5.30 | +2.04 (worse) |
| Base B80 std | 5.78 | 6.82 | +1.04 (worse) |
| Base B80 catastrophic | 2/10 | 3/10 | +1 (worse) |
| High-noise B80 mean | 2.45 | 4.56 | +2.11 (worse) |
| High-noise B80 catastrophic | 1/10 | 2/10 | +1 (worse) |

The determinism settings also imposed a significant runtime penalty
from single-threaded BLAS.  They were reverted.

### 3b. Seed Forwarding: Bug Fix Retained

The `_suggest_bayesian()` function was not forwarding the engine's
step-derived seed to `AxBayesianOptimizer`.  This was a legitimate bug
-- every other randomness path in the codebase receives a seed.  The
fix ensures Ax candidate generation is deterministic with respect to
the top-level seed, simplifying future stability analysis.

The seed-forwarding fix does not eliminate the bimodal failure.  It makes
the failure reproducible within a session (same seed always fails) rather
than random across sessions.

### 3c. What the Hardening Proved

The hardening proved a negative result: the B80 bimodal failure is not
caused by thread-scheduling non-determinism or unseeded PyTorch
randomness.  The root cause is in the GP model's sensitivity to the
specific sequence of training observations -- certain observation
histories lead the GP to a categorical local optimum that it cannot
escape.

## 4. Null Control Status

**PASS.**

Sprint 22 null control: max strategy difference 0.23% (7.36 MAE),
well within the 2% threshold.  Sprint 23 hardening report independently
confirmed on current HEAD with a reduced null slice (budgets 20,40;
seeds 0,1,2): max strategy difference 0.2%.

The null control has passed in every session since Sprint 18:

| Sprint | Max Diff | Pct | Verdict |
|--------|----------|-----|---------|
| S18 | 4.99 | 0.15% | PASS |
| S19 | 4.99 | 0.15% | PASS |
| S20 | 6.50 | 0.20% | PASS |
| S21 A/B | 5.81 | 0.18% | PASS |
| S22 | 7.36 | 0.23% | PASS |
| S23 | -- | 0.2% | PASS |

No strategy shows consistent improvement on permuted data.  The null
control safety gate remains solid.

## 5. Stability Assessment

### 5a. Is the Bimodal Mode Reduced?

**No.**  Sprint 23 did not reduce the bimodal B80 failure rate.  Across
all sessions since Sprint 20 (excluding the Sprint 21 alignment-only
outlier), the catastrophic rate is 1-3 per 10 seeds.  Sprint 23's
hardening attempt produced the worst single-session result (3/10).

### 5b. Is the Bimodal Mode Understood?

**Yes, conclusively.**  The diagnostics work identified a single root
cause (categorical `treat_day_filter` lock-in), traced it to Ax
candidate generation, verified the re-ranker is neutral on the noise
dimension, confirmed that good seeds escape the same trap during
optimization phase, and showed the failure is entirely in one
categorical dimension with continuous parameters nearly identical
between good and bad seeds.

### 5c. Is the Bimodal Mode Unchanged?

**Operationally, yes.**  The failure rate, mechanism, and impact are the
same as Sprint 22.  What changed is our understanding: we now know
exactly what to fix (forced categorical diversity in candidate
generation) and exactly what does not work (PyTorch determinism).

### 5d. Sprint-Over-Sprint B80 Trajectory

| Session | Mean | Std | Catastrophic | Seeds < 1.0 |
|---------|------|-----|-------------|-------------|
| S20 (pre-Ax) | 11.10 | 10.19 | 6/10 | 4/10 |
| S20 (post-Ax) | 3.85 | 5.59 | 2/10 | 6/10 |
| S21 (balanced) | 3.57 | 5.69 | 2/10 | 7/10 |
| S21 (align-only) | 0.52 | 0.16 | 0/10 | 10/10 |
| S22 | 3.26 | 5.78 | 2/10 | 8/10 |
| S23 benchmark | 1.81 | 4.35 | 1/10 | 9/10 |
| S23 hardened | 5.30 | 6.82 | 3/10 | 6/10 |

The S21 alignment-only result (0/10 catastrophic) has not been
reproduced.  Typical sessions show 1-3 catastrophic seeds.  The best
recent session (S23 benchmark: 1/10, mean 1.81) is encouraging but
represents session-level luck, not a code improvement.

## 6. Decision

### 6a. Alignment-Only Status

**Confirmed as the correct production default.**  This has been stable
since Sprint 21's A/B test showed alignment-only matched or beat
balanced re-ranking.  Sprint 22 confirmed the revert was clean.
Sprint 23's diagnostics showed the re-ranker is neutral on the noise
dimension (neither helping nor hurting the categorical lock-in).

### 6b. Readiness for Broader Benchmark Scope

**Not yet.**  The bimodal B80 mode remains a known open failure with a
clear, untested fix.  Expanding to new benchmark families before
implementing the categorical diversity fix would mean:

1. Running new benchmarks with a known ~15-30% catastrophic-seed rate
2. Being unable to distinguish new-benchmark instability from the
   existing categorical lock-in issue
3. Potentially attributing benchmark failures to wrong causes

The project should close the known failure mode first, then expand.

### 6c. What Sprint 23 Actually Delivered

Sprint 23 delivered **diagnosis and reproducibility hygiene**, not
stability improvement.  Concretely:

- A conclusive root-cause analysis of the B80 bimodal failure
- Proof that PyTorch-level determinism is not the solution
- A seed-forwarding bug fix that makes Ax sessions reproducible
- A clear, specific fix target (forced categorical diversity)
- No reduction in the catastrophic-seed rate

This is valuable work.  Understanding a failure is a prerequisite to
fixing it.  But it is important not to conflate understanding the
failure with having fixed it.

## 7. Sprint 24 Recommendation

**Stay focused on the categorical diversity fix, then rerun the
stability sweep.**

Concrete plan:

1. **Implement forced categorical diversity in Ax candidate generation.**
   When generating the 5 candidates for re-ranking, ensure at least one
   candidate uses each value of each categorical variable.  This
   guarantees the optimizer always considers `all` vs `weekday` vs
   `weekend` and prevents the GP model's categorical preference from
   excluding alternatives.  This is the change identified by the
   diagnostics report as most likely to eliminate the catastrophic mode.

2. **Rerun the 10-seed base counterfactual benchmark.**  Success
   criterion: 0/10 catastrophic seeds (regret > 10), mean regret < 2.0,
   std < 3.0.  If met, rerun high-noise and null control to confirm no
   regressions.

3. **If the fix works:** declare STABLE ENOUGH and expand to new
   benchmark families in Sprint 25.

4. **If the fix does not work:** investigate GP model hyperparameter
   restarts or acquisition function diversification as deeper
   interventions before expanding scope.

Do not broaden benchmark scope until the categorical lock-in failure
is resolved or demonstrated to be irreducible.
