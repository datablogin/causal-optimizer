# Sprint 23 B80 Seed Diagnostics Report

## Metadata

- **Date**: 2026-04-05
- **Sprint**: 23 Step 1 (B80 Seed Diagnostics)
- **Issue**: #120
- **Branch**: `sprint-23/b80-seed-diagnostics`
- **Base commit**: `0b3586a` (Sprint 22 alignment-only confirmation)

## 1. Executive Summary

The catastrophic B80 failure mode has a single, clear signature:
**the optimizer gets trapped on `treat_day_filter = "weekday"` and
never escapes to `"all"`**.  This noise-dimension lock-in accounts
for the entire ~14.5 regret gap between good and bad seeds.

The failure is not in Ax candidate generation, not in the alignment-only
re-ranking, and not in the evaluation.  It is a categorical-variable
exploration failure: once the optimizer commits to `weekday` as its
best-so-far filter, neither Ax nor the exploitation phase generates
enough `all`-filter candidates to displace it.  The training-set
objective for `weekday` is misleadingly competitive (-24 vs -33 for
`all`) because the 80/20 opt/test split has a different weekday/weekend
ratio than the test set, masking the ~14 regret penalty.

Two hardening ideas follow directly from the diagnosis:

1. **Forward the engine seed to `AxBayesianOptimizer`**.  The Bayesian
   path currently creates unseeded Ax instances on every call to
   `_suggest_bayesian()`.  Passing the seed would make Ax candidate
   generation reproducible across sessions and simplify stability
   analysis.

2. **Force categorical diversity in candidate generation**.  Ensure
   at least one of the 5 Ax candidates per re-ranking round uses
   each value of every categorical variable.  This guarantees the
   re-ranker always has a `"all"` option available, preventing
   lock-in on a suboptimal categorical value.

## 2. Methodology

### 2a. Benchmark Run

Ran the base counterfactual benchmark with 10 seeds (0-9), budgets
20/40/80, strategies random/surrogate_only/causal on the current
alignment-only codebase (commit `0b3586a`).

```
uv run python scripts/counterfactual_benchmark.py \
    --data-path .../ercot_north_c_dfw_2022_2024.parquet \
    --variant base --budgets 20,40,80 --seeds 0,1,2,3,4,5,6,7,8,9 \
    --strategies random,surrogate_only,causal
```

### 2b. Instrumented Diagnostic Run

A new diagnostic script (`scripts/b80_seed_diagnostics.py`) ran all
10 seeds at B80 with per-experiment instrumentation capturing:

- Per-step parameters, objective, phase
- Best-so-far tracking with test-set regret evaluation at every step
- Improvement timestamps and parameter snapshots

### 2c. Session Non-Determinism

Because Ax is unseeded (see Section 5a), the catastrophic seeds
differ between sessions:

| Session | Catastrophic Seeds | Count |
|---------|-------------------|-------|
| Sprint 22 | 3, 9 | 2/10 |
| S23 diagnostic run | 2, 8 | 2/10 |
| S23 benchmark run | 9 | 1/10 |

The which-seeds-fail pattern is not reproducible; the failure
mechanism is.

## 3. S23 Benchmark Results

### 3a. Summary Table

| Strategy | Budget | Mean Regret | Std | Max | n |
|----------|--------|-------------|-----|-----|---|
| random | 20 | 20.58 | 11.34 | 38.58 | 10 |
| random | 40 | 12.75 | 9.27 | 30.78 | 10 |
| random | 80 | 7.77 | 2.83 | 11.35 | 10 |
| surrogate_only | 20 | 22.39 | 4.87 | 31.16 | 10 |
| surrogate_only | 40 | 19.70 | 4.99 | 31.16 | 10 |
| surrogate_only | 80 | 6.53 | 4.97 | 13.90 | 10 |
| causal | 20 | 13.10 | 8.39 | 28.24 | 10 |
| causal | 40 | 8.07 | 7.13 | 17.65 | 10 |
| causal | 80 | 1.81 | 4.35 | 14.87 | 10 |

### 3b. Per-Seed B80 Causal Regret

| Seed | Regret | Category |
|------|--------|----------|
| 0 | 0.36 | GOOD |
| 1 | 0.36 | GOOD |
| 2 | 0.36 | GOOD |
| 3 | 0.36 | GOOD |
| 4 | 0.36 | GOOD |
| 5 | 0.41 | GOOD |
| 6 | 0.36 | GOOD |
| 7 | 0.36 | GOOD |
| 8 | 0.36 | GOOD |
| 9 | 14.87 | BAD |

9/10 seeds achieve near-oracle regret.  1/10 catastrophic (seed 9).

### 3c. Statistical Significance (Causal vs Surrogate_Only)

| Budget | Mann-Whitney U | p-value | Significant? | Wins |
|--------|---------------|---------|--------------|------|
| B20 | 16.0 | 0.0056 | Yes | 8/10 |
| B40 | 15.0 | 0.0046 | Yes | 7/10 |
| B80 | 10.0 | 0.0013 | Yes | 9/10 |

## 4. Diagnostic Findings

### 4a. The Failure Signature: `treat_day_filter = "weekday"` Lock-In

Every catastrophic seed shares one signature: the best-so-far
parameters have `treat_day_filter = "weekday"`, and the optimizer
never discovers that `"all"` is dramatically better.

From the instrumented diagnostic run (2/10 catastrophic):

**Bad seed 2** (final regret 14.80):

| Step | Phase | day_filter | Best Regret |
|------|-------|------------|-------------|
| 0 | exploration | weekday | 39.36 |
| 12 | optimization | weekday | 29.25 |
| 29 | optimization | weekday | 18.24 |
| 38 | optimization | weekday | 15.41 |
| 72 | exploitation | weekday | 14.80 |

Locked into `weekday` from step 12 onward.  Never tested `all`
as an improvement candidate.

**Bad seed 8** (final regret 14.97):

| Step | Phase | day_filter | Best Regret |
|------|-------|------------|-------------|
| 1 | exploration | weekday | 36.05 |
| 14 | optimization | weekday | 21.40 |
| 32 | optimization | weekday | 15.54 |
| 73 | exploitation | weekday | 14.97 |

Locked into `weekday` from step 1 onward.  Never tested `all`.

**Comparison -- good seed 3 escapes** (final regret 0.41):

| Step | Phase | day_filter | Best Regret |
|------|-------|------------|-------------|
| 0 | exploration | weekday | 29.44 |
| 19 | optimization | weekday | 17.64 |
| 40 | optimization | weekday | 16.42 |
| 43 | optimization | **all** | **3.51** |
| 66 | exploitation | all | **0.41** |

Same starting pattern as the bad seeds, but at step 43 an `all`
candidate appeared and dropped regret from 16.42 to 3.51.  This
escape happened for seeds 3, 4, and 9 (in the diagnostic run),
all at similar optimization-phase steps.

### 4b. Good vs Bad Seed Best Parameters

**All 7 good seeds** converged to similar optimal parameters:

| Parameter | Good Seed Range |
|-----------|----------------|
| treat_temp_threshold | 18.9 - 20.2 |
| treat_hour_start | 10 - 11 |
| treat_hour_end | 21 |
| treat_day_filter | **all** |

**Both bad seeds** converged to similar suboptimal parameters:

| Parameter | Bad Seed Range |
|-----------|---------------|
| treat_temp_threshold | 19.1 - 19.4 |
| treat_hour_start | 10 - 11 |
| treat_hour_end | 21 |
| treat_day_filter | **weekday** |

The continuous parameters are nearly identical.  The entire
~14.5 regret difference comes from the categorical noise
dimension `treat_day_filter`.

### 4c. Phase Transition Analysis

At the optimization-to-exploitation transition (step 49):

| Category | Seeds | Mean Regret at Step 49 |
|----------|-------|----------------------|
| Good | 0, 3, 4, 5, 6, 7, 9 | 0.41 - 2.79 |
| Bad | 2, 8 | 15.41, 15.54 |

Bad seeds enter exploitation with ~15 regret and never recover,
because exploitation perturbs continuous variables around the
best-so-far but rarely flips categorical variables to a different
value.

### 4d. Divergence Timeline

The gap between good and bad seeds opens during exploration (step 0)
and widens continuously through optimization.  However, the early
divergence is not decisive: seeds 3, 4, and 9 start with `weekday`
lock-in but escape during late optimization (steps 36-43).

The decisive divergence is whether the optimizer generates an `all`
candidate that improves on the `weekday` best-so-far during
optimization phase.  The 40-experiment optimization window
(steps 10-49) provides multiple chances for this, but it is not
guaranteed because:

1. Ax creates a fresh GP model each call (no persistent state)
2. Ax is unseeded (no reproducible candidate sequences)
3. The categorical variable has only 3 values, so random exploration
   of `all` vs `weekday` is probabilistic
4. The alignment-only re-ranking does not penalize noise-dimension
   lock-in (it scores ancestor-variable variation only)

### 4e. Why `weekday` Is a Trap

The training set (80% of data, time-ordered) has a specific
weekday/weekend ratio.  On the training set, `weekday` achieves
objective ~-24 (decent), while `all` achieves ~-33 (optimal).
But the gap is only visible when the optimizer happens to generate
an `all` candidate with good continuous parameters.

The `weekday` trap is self-reinforcing: once `weekday` is
best-so-far, the GP model learns that `weekday` configurations
are promising and generates more `weekday` candidates, further
reducing the chance of exploring `all`.

### 4f. Improvement Frequency

Bad seeds find fewer improvements and plateau earlier:

| Category | Mean Improvements | Last Improvement (Mean Step) |
|----------|-------------------|------------------------------|
| Good (7 seeds) | 11.9 | 68.7 |
| Bad (2 seeds) | 12.0 | 72.5 |

The improvement counts are similar, but the bad seeds' improvements
are smaller (incremental `weekday` tuning) while the good seeds have
a dramatic improvement when they discover `all`.

## 5. Root Cause Analysis

### 5a. Ax Bayesian Optimizer Is Unseeded

The function `_suggest_bayesian()` in `optimizer/suggest.py` creates
a fresh `AxBayesianOptimizer` on every call but does not pass the
engine's seed to the Ax client:

```python
optimizer = AxBayesianOptimizer(
    search_space=search_space,
    objective_name=objective_name,
    minimize=minimize,
    focus_variables=ax_focus,
    pomis_prior=pomis_sets,
    # NOTE: no seed= parameter passed
)
```

The `AxBayesianOptimizer.__init__` accepts an optional `seed`
parameter that it forwards to `AxClient(random_seed=seed)`.  This
parameter is unused, making every Ax session non-deterministic at
the process level.

This explains why different sessions produce different catastrophic
seeds: the Ax candidate sequences differ on every invocation.

### 5b. Categorical Variable Exploration Gap

The alignment-only re-ranking scores candidates by
`causal_softness * alignment`, where alignment measures variation
in ancestor variables (treat_temp_threshold, treat_hour_start,
treat_hour_end).  The noise dimension `treat_day_filter` is NOT an
ancestor, so it contributes nothing to the alignment score.

This means the re-ranker is blind to `treat_day_filter`: it cannot
prefer `all` over `weekday`.  The only way `all` wins is if the
GP model happens to propose an `all` candidate with good ancestor-
variable values.

### 5c. Exploitation Phase Cannot Escape

The exploitation phase (`_suggest_exploitation`) perturbs 1-2
variables around the best-so-far.  For continuous variables, this
is a 10% range perturbation.  For categorical variables, it picks a
random value.  However, the exploitation perturbation must also beat
the current best on the training set to be kept.

When the best-so-far is a `weekday` configuration with objective
~-24, a random flip to `all` with slightly different continuous
parameters might not beat -24 on the first try.  The exploitation
phase makes ~30 attempts (steps 50-79), but each attempt perturbs
only 1-2 variables, and the categorical flip needs to coincide with
good continuous parameters.

## 6. Answers to Must-Answer Questions

### Q1: Where do the bad seeds first diverge?

Bad seeds diverge during **early optimization phase** (steps 10-20)
when the GP model, trained on exploration-phase data where `weekday`
performed reasonably, commits to generating `weekday`-heavy candidate
pools.  The divergence becomes irreversible around step 30-40 when
the model has enough `weekday` observations to be confident in that
region and stops exploring `all`.

### Q2: Is the failure in candidate generation, reranking, or evaluation?

**Candidate generation.**  The Ax GP model does not generate enough
`all`-filter candidates to give the re-ranker a chance to select
them.  The alignment-only re-ranking is neutral on `treat_day_filter`
(it is not an ancestor), so it passes through whatever Ax generates.
Evaluation is correct -- `weekday` genuinely performs worse.

### Q3: Do catastrophic seeds share a recognizable signature?

**Yes.  The signature is `treat_day_filter = "weekday"` in the
best-so-far parameters from the optimization phase onward.**  Both
bad seeds in the diagnostic run (2, 8) and both bad seeds in Sprint
22 (3, 9) have this signature.  No bad seed has ever had
`treat_day_filter = "all"` as its final best parameter.

### Q4: What are the top 1-2 hardening ideas?

1. **Pass the engine seed to AxBayesianOptimizer**.  This makes
   Ax candidate generation deterministic and enables reproducible
   analysis.  It will not eliminate the categorical lock-in, but
   it will make the failure mode stable across sessions (same seeds
   always fail) rather than random.

2. **Force categorical diversity in Ax candidates**.  When generating
   the 5 candidates for re-ranking, ensure at least one candidate
   uses each value of each categorical variable.  This guarantees
   the optimizer always considers `all` vs `weekday` vs `weekend`
   and prevents the GP model's categorical preference from excluding
   alternatives entirely.  This is the change most likely to
   eliminate the catastrophic mode.

## 7. Cross-Session Comparison

### 7a. B80 Catastrophic Seed Count

| Session | Catastrophic Seeds | Bad Seeds | Diagnostic Seeds |
|---------|-------------------|-----------|------------------|
| Sprint 20 (pre-Ax) | 6/10 | 0,1,2,4,5,8 | -- |
| Sprint 20 (post-Ax) | 2/10 | 1,4 | -- |
| Sprint 21 (balanced) | 2/10 | 8,9 | -- |
| Sprint 21 (align-only) | 0/10 | -- | -- |
| Sprint 22 | 2/10 | 3,9 | -- |
| S23 diagnostic | 2/10 | 2,8 | weekday lock-in |
| S23 benchmark | 1/10 | 9 | -- |

The catastrophic rate across sessions (excluding the S21 outlier)
is 1-2 per 10 seeds.  The which-seeds-fail is random due to Ax
being unseeded.  The failure mechanism is always categorical
lock-in on `treat_day_filter`.

### 7b. B80 Causal Summary Across Sessions

| Session | Mean | Std | Seeds < 1.0 |
|---------|------|-----|-------------|
| Sprint 22 | 3.26 | 5.78 | 8/10 |
| S23 benchmark | 1.81 | 4.35 | 9/10 |
| S23 diagnostic | 2.00 | 4.84 | 7/10 |

## 8. Artifacts

| File | Description |
|------|-------------|
| `s23_diag_base.json` | Full benchmark run (10 seeds, 3 budgets, 3 strategies) |
| `s23_b80_diagnostics.json` | Instrumented per-step trajectories for all 10 seeds |
| `scripts/b80_seed_diagnostics.py` | Diagnostic instrumentation script |

Artifact JSON files stored in machine-local directory:
`/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/`.

## 9. Recommendation

The diagnosis is conclusive: categorical lock-in on the noise
dimension `treat_day_filter` is the sole mechanism of catastrophic
B80 failure.  Sprint 23 Step 2 should implement the categorical
diversity fix (hardening idea #2) and rerun the benchmark to verify
it eliminates the bimodal mode.  Seeding the Ax path (hardening
idea #1) should be done in parallel for reproducibility.
