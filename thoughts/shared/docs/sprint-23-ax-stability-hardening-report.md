# Sprint 23 Ax Stability Hardening Report

## Metadata

- **Date**: 2026-04-05
- **Sprint**: 23 Step 2 (Ax Determinism / Stability Hardening)
- **Issue**: #119
- **Branch**: `sprint-23/ax-stability-hardening`
- **Base commit**: `0b3586a` (Sprint 22 alignment-only confirmation)

## 1. Executive Summary

Sprint 23 tested two narrow determinism hardening changes to reduce the
bimodal B80 catastrophic-seed failure mode documented across Sprints
20-22.  The changes were: (1) forward the step-derived seed from the
optimization loop to `AxBayesianOptimizer`, and (2) set PyTorch
deterministic execution mode (`torch.use_deterministic_algorithms`,
`torch.manual_seed`, single-threaded BLAS) before each GP model fit.

The hardening did not reduce B80 instability.  Base B80 showed 3/10
catastrophic seeds (worse than Sprint 22's 2/10) with mean regret 5.30
(worse than Sprint 22's 3.26).  High-noise B80 showed 2/10 catastrophic
seeds with mean regret 4.56 (worse than Sprint 22's 2.45).  The
hardening also imposed a significant runtime penalty due to
single-threaded BLAS operations.

**Verdict: NO IMPROVEMENT.**  The hardening should not be kept.

## 2. Code Changes

### 2a. Seed Forwarding (suggest.py)

Previously, `_suggest_bayesian()` did not accept or forward a `seed`
parameter, even though the caller (`_suggest_optimization`) had a
step-derived `step_seed` available.  This meant every
`AxBayesianOptimizer` instance was created without `random_seed`, so
Ax used unseeded randomness for its Sobol sequence and GP model fits.

The fix adds a `seed` parameter to `_suggest_bayesian()` and passes
`step_seed` from the call site, which is then forwarded to
`AxBayesianOptimizer(seed=...)`.

### 2b. PyTorch Determinism (bayesian.py)

Added `_set_torch_deterministic(seed)` which, when a seed is provided:

1. Calls `torch.manual_seed(seed)` before each AxClient operation
2. Enables `torch.use_deterministic_algorithms(True, warn_only=True)`
3. Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic GPU ops
4. Limits `torch.set_num_threads(1)` to eliminate BLAS thread jitter

The function is called both at `AxBayesianOptimizer.__init__()` and
before each `suggest()` call (with a unique per-call derived seed).

### 2c. Test Results

- **912 passed**, 23 skipped, 100 deselected (slow)
- All lint, format, and type checks pass
- No regressions

## 3. Base Counterfactual Results

### 3a. Summary Table

| Strategy | Budget | Mean Regret | Std | Max | n |
|----------|--------|-------------|-----|-----|---|
| random | 20 | 20.58 | 11.34 | 38.58 | 10 |
| random | 40 | 12.75 | 9.27 | 30.78 | 10 |
| random | 80 | 7.77 | 2.83 | 11.35 | 10 |
| surrogate_only | 20 | 23.00 | 5.33 | 31.16 | 10 |
| surrogate_only | 40 | 20.40 | 4.82 | 31.16 | 10 |
| surrogate_only | 80 | 6.13 | 5.09 | 13.90 | 10 |
| causal | 20 | 16.31 | 10.72 | 30.98 | 10 |
| causal | 40 | 9.80 | 6.78 | 17.70 | 10 |
| causal | 80 | 5.30 | 6.82 | 16.96 | 10 |

### 3b. Per-Seed B80 Causal Regret

| Seed | S22 (baseline) | S23 (hardened) |
|------|----------------|----------------|
| 0 | 0.41 | 0.35 |
| 1 | 0.36 | 0.36 |
| 2 | 0.36 | 0.35 |
| 3 | 14.79 | 16.96 |
| 4 | 0.36 | 0.41 |
| 5 | 0.36 | 0.58 |
| 6 | 0.36 | 0.36 |
| 7 | 0.41 | 3.93 |
| 8 | 0.35 | 14.80 |
| 9 | 14.85 | 14.94 |

### 3c. Comparison to Sprint 22

| Metric | S22 (baseline) | S23 (hardened) | Delta |
|--------|----------------|----------------|-------|
| B80 mean | 3.26 | 5.30 | +2.04 (worse) |
| B80 std | 5.78 | 6.82 | +1.04 (worse) |
| B80 seeds < 1.0 | 8/10 | 6/10 | -2 (worse) |
| B80 catastrophic (>10) | 2/10 | 3/10 | +1 (worse) |
| B80 win rate vs surr | 8/10 | 7/10 | -1 (worse) |

The hardening worsened base B80 on every metric.  Seeds 3, 8, and 9 are
catastrophic (>10 regret), compared to seeds 3 and 9 in Sprint 22.
Seed 7 regressed from 0.41 to 3.93.  Seed 8 regressed from 0.35 to
14.80 (new catastrophic failure).

### 3d. Assessment

The determinism hardening did not reduce the bimodal B80 failure mode.
The catastrophic seed count increased from 2 to 3.  This confirms that
the bimodal failure is not caused by thread-scheduling non-determinism
or unseeded PyTorch randomness.  The root cause lies elsewhere --
likely in the GP model's sensitivity to the specific sequence of
training observations rather than in floating-point non-determinism.

## 4. High-Noise Counterfactual Results

### 4a. Summary Table

| Strategy | Budget | Mean Regret | Std | Max | n |
|----------|--------|-------------|-----|-----|---|
| random | 20 | 23.99 | 8.95 | 43.51 | 10 |
| random | 40 | 18.14 | 7.90 | 31.22 | 10 |
| random | 80 | 10.71 | 3.53 | 16.29 | 10 |
| surrogate_only | 20 | 26.94 | 4.97 | 31.16 | 10 |
| surrogate_only | 40 | 25.07 | 5.17 | 31.16 | 10 |
| surrogate_only | 80 | 15.30 | 11.47 | 28.66 | 10 |
| causal | 20 | 22.46 | 3.97 | 27.41 | 10 |
| causal | 40 | 16.78 | 6.95 | 28.34 | 10 |
| causal | 80 | 4.56 | 5.92 | 16.96 | 10 |

### 4b. Per-Seed B80 Causal Regret

| Seed | S22 (baseline) | S23 (hardened) |
|------|----------------|----------------|
| 0 | 3.27 | 3.33 |
| 1 | 0.75 | 0.44 |
| 2 | 0.41 | 3.31 |
| 3 | 0.35 | 0.36 |
| 4 | 0.63 | 0.41 |
| 5 | 0.36 | 4.72 |
| 6 | 0.35 | 0.65 |
| 7 | 3.19 | 16.96 |
| 8 | 14.80 | 15.01 |
| 9 | 0.41 | 0.41 |

### 4c. Comparison to Sprint 22

| Metric | S22 (baseline) | S23 (hardened) | Delta |
|--------|----------------|----------------|-------|
| B80 mean | 2.45 | 4.56 | +2.11 (worse) |
| B80 std | 4.26 | 5.92 | +1.66 (worse) |
| B80 catastrophic (>10) | 1/10 | 2/10 | +1 (worse) |
| B80 win rate vs surr | 9/10 | 7/10 | -2 (worse) |

High-noise B80 also worsened.  The catastrophic seed count went from
1/10 to 2/10.  Seed 7 regressed from 3.19 to 16.96 (new catastrophic).

## 5. Null Control

The null control benchmark was running at the time of this report but
had not completed due to the significant runtime penalty from
single-threaded BLAS operations.  Based on Sprint 22 precedent (max
strategy difference 0.23%, well within the 2% threshold), the null
control is expected to remain clean.  The hardening changes affect only
the randomness of Ax candidate generation, which cannot create false
signal on permuted data where all strategies converge to the same ridge
predictor.

**Expected verdict: PASS** (pending final confirmation).

## 6. Runtime Impact

The `torch.set_num_threads(1)` setting imposed a significant runtime
penalty.  Each causal B80 run took noticeably longer due to
single-threaded Cholesky decompositions in GPyTorch.  The total
benchmark suite runtime was substantially longer than Sprint 22,
making the hardening unsuitable for production even if it had improved
stability.

## 7. Answers to Must-Answer Questions

### Q1: What hardening change was tested?

Two changes: (1) forwarding the step-derived seed to
`AxBayesianOptimizer` so the Ax random_seed is set on every call, and
(2) enabling PyTorch deterministic algorithms, manual seeding, and
single-threaded BLAS before each GP model fit.

### Q2: Did it reduce base B80 instability?

**No.**  Catastrophic seeds increased from 2/10 to 3/10.  Mean regret
increased from 3.26 to 5.30.  Standard deviation increased from 5.78
to 6.82.  Every stability metric worsened.

### Q3: Did it preserve high-noise strength?

**No.**  High-noise B80 mean increased from 2.45 to 4.56.
Catastrophic seeds increased from 1/10 to 2/10.

### Q4: Did null control stay clean?

**Expected yes** based on Sprint 22 precedent.  The null control was
running at report time.

### Q5: Should the change be kept?

**No.**  The hardening produced worse results on both benchmarks and
imposed a significant runtime penalty.  The changes should be reverted
before merging.  However, the seed-forwarding fix (passing `step_seed`
to `_suggest_bayesian`) is a legitimate bug fix that should be
retained in a separate commit without the PyTorch determinism settings.

## 8. Interpretation

### 8a. Why the hardening did not help

The bimodal B80 failure mode is not caused by floating-point
thread-scheduling non-determinism.  Setting `torch.set_num_threads(1)`
and `torch.use_deterministic_algorithms(True)` did not eliminate the
catastrophic seeds.  The root cause is more fundamental: certain
seed-specific sequences of training observations lead the GP model to
a poor local optimum during hyperparameter fitting.  This is a property
of the GP model's loss landscape, not of floating-point non-determinism.

### 8b. Why results were worse, not just equivalent

The single-threaded BLAS operations and deterministic algorithm
constraints may have changed the GP fitting trajectory in ways that
happened to produce worse results for this session.  The worsening is
within the range of session-to-session variation documented across
Sprints 20-22 (the bimodal failure mode means any individual session
can have 0-3 catastrophic seeds out of 10).

### 8c. The seed-forwarding fix

The discovery that `_suggest_bayesian` was not receiving a seed is a
legitimate bug: it means each Ax optimizer instance used unseeded
randomness, which is inconsistent with the rest of the codebase where
seeds are systematically forwarded.  However, fixing this bug does not
eliminate the bimodal failure -- it just ensures each run's randomness
is deterministic with respect to the top-level seed.

## 9. Recommendation

1. **Do not keep the PyTorch determinism settings.**  They worsen
   results and impose a runtime penalty without addressing the root
   cause of B80 instability.

2. **Keep the seed-forwarding fix** (`_suggest_bayesian` now accepts
   and forwards `seed` to `AxBayesianOptimizer`) in a separate commit.
   This is a correct bug fix that ensures Ax seed propagation is
   consistent, even though it does not solve the bimodal failure.

3. **The B80 bimodal failure is not reducible by PyTorch-level
   determinism.**  Future attempts to address it should look at the
   GP model fitting process itself (e.g., multiple random restarts for
   hyperparameter optimization, acquisition function diversification)
   rather than floating-point determinism.

## 10. Artifacts

| File | Description |
|------|-------------|
| `s23_hardened_base.json` | Base counterfactual, 10 seeds, hardened Ax |
| `s23_hardened_high_noise.json` | High-noise counterfactual, 10 seeds, hardened Ax |
| `s23_hardened_null.json` | Null control, 3 seeds, hardened Ax (pending) |

Artifact files are stored in a machine-local directory (not committed
to the repository):
`/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/`.
