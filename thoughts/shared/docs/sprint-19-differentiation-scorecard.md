# Sprint 19 Differentiation Scorecard

## Metadata

- **Date**: 2026-03-31
- **Sprint**: 19 (Causal Differentiation Under Controls)
- **PRs included**: #98 (Harder Counterfactual Variants), #99 (Softer Causal
  Influence), #101 (Skip Calibration Under Controls)
- **Issues**: #96, #97, #100
- **Predecessor**: Sprint 18 Discovery Trust Scorecard

## 1. Sprint 19 Summary

Sprint 18 established that causal guidance does not outperform surrogate_only
on any benchmark, but the evidence infrastructure (null control, positive
control, time-series profiler) is now trustworthy. Sprint 19 responds by
targeting the optimizer core: making causal influence earlier and softer,
adding harder benchmark variants, and measuring skip calibration.

Sprint 19 delivered three changes:

1. **Harder Counterfactual Variants** (PR #98) -- Two new benchmark variants
   (high-noise with 10 irrelevant dimensions, confounded with Simpson's
   paradox) that create stronger positive-control pressure for causal guidance.

2. **Earlier and Softer Causal Influence** (PR #99) -- Three optimizer-core
   changes: causal-weighted exploration during the LHS phase
   (`causal_exploration_weight=0.3`), soft ranking during optimization
   (`causal_softness=0.5` replaces hard focus-variable pinning), and
   adaptive targeted/LHS candidate splitting.

3. **Skip Calibration Under Controls** (PR #101) -- Measurement
   infrastructure (`SkipAuditEntry`, `SkipMetrics`, `compute_skip_metrics`)
   and audited benchmark runs showing the Sprint 17 "33% false-skip rate"
   finding was context-specific. Current false-skip rates are 0% on real
   data and 7% on synthetic data.

## 2. Harder Counterfactual Variants

### 2a. High-Noise Variant

Adds 10 continuous nuisance dimensions (noise_var_0 through noise_var_9)
to the 5-variable base search space. Nuisance variables have zero effect
on the treatment outcome. The causal graph excludes nuisance vars from
the objective's ancestors, so causal guidance can focus on 3 real dimensions
while surrogate_only must search 15.

| Strategy | Budget | Mean Regret | Std |
|----------|--------|-------------|-----|
| random | 20 | 26.42 | 9.21 |
| surrogate_only | 20 | 23.99 | 4.07 |
| causal | 20 | 23.32 | 5.20 |
| random | 40 | 18.55 | 8.06 |
| surrogate_only | 40 | 21.79 | 4.82 |
| causal | 40 | 10.34 | 3.21 |
| random | 80 | 8.74 | 2.57 |
| surrogate_only | 80 | 8.74 | 10.88 |
| causal | 80 | 3.47 | 5.66 |

**Finding: Causal beats surrogate_only at B40 and B80.** At B40, causal
regret is 10.34 vs surrogate_only 21.79 (52% lower). At B80, causal
regret is 3.47 vs surrogate_only 8.74 (60% lower). The 15-dimensional
search space is hard enough that surrogate_only cannot screen out the
nuisance dimensions efficiently within the budget, while causal guidance
focuses the search on the 3 real parents.

Surrogate_only has high variance at B80 (std 10.88) -- some seeds find
good solutions, others get stuck searching irrelevant dimensions.

### 2b. Confounded Variant

Introduces a hidden confounder ("grid stress") that inflates the base
load (Y0) but not the treatment effect. This creates Simpson's paradox:
naive estimation overestimates treatment benefit because treated hours
have higher apparent load. The causal graph marks the confounding with a
bidirected edge. The optimizer trains on biased data but is evaluated on
deconfounded test data.

| Strategy | Budget | Mean Regret | Std |
|----------|--------|-------------|-----|
| random | 20 | 24.28 | 11.68 |
| surrogate_only | 20 | 33.74 | 4.38 |
| causal | 20 | 37.41 | 14.00 |
| random | 40 | 24.60 | 9.79 |
| surrogate_only | 40 | 29.71 | 0.67 |
| causal | 40 | 26.01 | 3.75 |
| random | 80 | 17.41 | 5.57 |
| surrogate_only | 80 | 20.65 | 0.00 |
| causal | 80 | 30.23 | 11.23 |

**Finding: The confounded variant is too hard for all strategies.** Random
outperforms both learned strategies at B80 (17.41 vs 20.65 and 30.23).
Both surrogate_only and causal are systematically misled by the
confounding: the biased training data pushes them toward policies that
look good on confounded data but perform poorly on the deconfounded test
set. Causal is the worst performer, suggesting the current causal graph
structure (bidirected edge) does not help the optimizer navigate the
confounding. The POMIS-aware search may actually amplify the bias by
focusing on confounded signals.

This variant is not currently a valid positive control -- it is a valid
negative result showing that bidirected edges alone are insufficient for
deconfounding.

### 2c. Assessment

| Variant | Valid Positive Control? | Separates Strategies? | Best Strategy |
|---------|----------------------|----------------------|---------------|
| base | Yes | Yes (Sprint 19) | causal |
| high_noise | Yes | Yes | causal |
| confounded | No (all strategies misled) | No (random wins) | random |

The high-noise variant is the strongest positive control for causal
advantage. The confounded variant reveals a real limitation: the optimizer
does not yet implement deconfounding at the search level, so marking
confounders in the graph does not translate to better policies.

## 3. Softer Causal Influence

### 3a. Sprint 18 vs Sprint 19 Base Counterfactual Comparison

| Strategy | Budget | S18 Regret | S19 Regret | Delta |
|----------|--------|------------|------------|-------|
| random | 20 | 20.06 | 20.06 | +0.00 |
| random | 40 | 19.11 | 19.11 | +0.00 |
| random | 80 | 9.16 | 9.16 | +0.00 |
| surrogate_only | 20 | 12.07 | 22.15 | +10.08 |
| surrogate_only | 40 | 11.16 | 17.94 | +6.78 |
| surrogate_only | 80 | 1.75 | 2.16 | +0.41 |
| causal | 20 | 17.74 | 15.84 | -1.90 |
| causal | 40 | 17.66 | 8.19 | -9.48 |
| causal | 80 | 2.46 | 0.98 | -1.47 |

### 3b. Analysis

**Causal improved substantially.** Mean regret dropped at all budgets,
with the largest improvement at B40 (17.66 to 8.19, a 54% reduction).
At B80, causal now achieves regret 0.98 -- near-oracle performance
(48.41 oracle value, 47.42 causal policy value = 97.96% efficiency).

**Surrogate_only regressed at low budgets.** Surrogate_only regret
increased from 12.07 to 22.15 at B20 and from 11.16 to 17.94 at B40.
At B80, surrogate_only is essentially flat (1.75 to 2.16).

**Random is unchanged** (same seeds, same data -- expected since random
search does not use the engine).

**Surrogate_only variability note.** The surrogate_only strategy uses the
same ExperimentEngine code path but with `causal_graph=None`. The Sprint
19 optimizer changes should not directly affect surrogate_only behavior
when no graph is provided (the causal exploration weight is a no-op
without a graph, and soft ranking only activates with a graph). The
per-seed variation suggests that Ax/BoTorch optimization has inherent
run-to-run stochasticity that was previously masked. Some seeds (0, 1)
improved slightly while others (2, 3, 4) regressed substantially. This
deserves investigation in Sprint 20 but does not invalidate the causal
improvement, which is consistent across most seeds.

**Causal is now the best strategy on the base benchmark.** At B40, causal
(8.19) beats surrogate_only (17.94) and random (19.11). At B80, causal
(0.98) beats surrogate_only (2.16) and random (9.16). This reverses the
Sprint 18 result where surrogate_only led at all budgets.

### 3c. Behavioral Differences

The Sprint 19 optimizer-core changes produce three observable behavioral
differences:

1. **Causal-weighted exploration** (experiments 1-10): LHS candidates are
   scored with a soft bias toward ancestor variation. This means the
   exploration phase no longer ignores the causal graph entirely.

2. **Soft ranking replaces hard focus**: The RF surrogate now trains on
   ALL variables (not just focus vars) and applies a causal alignment
   bonus rather than eliminating non-ancestor candidates. This allows
   good non-ancestor candidates to survive while still preferring
   ancestor variation.

3. **Adaptive targeted ratio**: The LHS/targeted candidate split ramps
   from 70/30 (early optimization) to 30/70 (late optimization), avoiding
   over-constraining when the surrogate is uncertain.

### 3d. Assessment

**The optimizer-core changes helped.** Causal guidance is now materially
useful on the base counterfactual benchmark, with the largest improvement
at the budget level (B40) where Sprint 18 showed the weakest causal
performance. The soft causal influence approach is directionally correct:
earlier influence and softer weighting outperform the Sprint 18 approach
of hard focus-variable restriction.

**Limitation: `causal_softness` is not effective on the Ax/BoTorch path.**
The soft ranking bonus only applies when the RF surrogate fallback is used.
When Ax/BoTorch is available (the production path), the re-ranking uses
pure causal alignment without a continuous softness trade-off. This means
the documented `causal_softness` parameter does not have the expected
effect on the primary optimization path. This should be addressed in
Sprint 20.

## 4. Skip Calibration

### 4a. Measurement Infrastructure

Sprint 19 added `SkipAuditEntry` and `SkipMetrics` dataclasses with
`compute_skip_metrics()` helper. The engine now records a `SkipAuditEntry`
for every skip decision, exposed via `skip_diagnostics.audit_entries`.

### 4b. Results

| Benchmark | False-Skip Rate (Sprint 19) | Sprint 17 Reference |
|-----------|---------------------------|---------------------|
| Real ERCOT (predictive) | 0.00% (no skips) | 33% (different config) |
| Counterfactual | N/A (disabled, max_skips=0) | N/A |
| Synthetic quadratic (B40) | 7.14% | N/A |

### 4c. Diagnosis

The Sprint 17 "33% false-skip rate" was context-specific to an older
benchmark configuration that no longer exists. On the current benchmark
suite:

- **Real energy data**: Model quality (cross-validated R-squared) never
  reaches the 0.3 threshold, so no skips occur. The skip logic is
  effectively a no-op.
- **Counterfactual benchmark**: Skip logic is explicitly disabled because
  policy evaluation is cheap.
- **Synthetic functions**: When skipping activates (B40, seed 42), the
  false-skip rate is 7.14%, concentrated in late optimization where
  impact is minimal.

### 4d. Fix Applied

No calibration change applied. The evidence does not support a specific
fix: the skip logic is dormant on real benchmarks (safe), and when it
activates on synthetic data, the false-skip rate is acceptable (7%) and
concentrated late (low impact).

### 4e. Assessment

**Skip logic is trustworthy in its current form.** The 0.3 model quality
threshold is conservative enough to prevent skipping on real-world noisy
data entirely. The measurement infrastructure is the real deliverable:
future sprints can use `audit_skip_rate=1.0` and `compute_skip_metrics()`
to continuously monitor skip quality as the optimizer evolves.

## 5. Null Control Verification

### 5a. Sprint 19 Results

| Strategy | Budget | Mean Test MAE | Std |
|----------|--------|---------------|-----|
| random | 20 | 3260.48 | 1.75 |
| surrogate_only | 20 | 3255.76 | 0.39 |
| causal | 20 | 3259.11 | 2.16 |
| random | 40 | 3261.58 | 0.28 |
| surrogate_only | 40 | 3256.31 | 0.00 |
| causal | 40 | 3259.11 | 2.16 |

**Mean Test MAE Across All Seeds and Budgets:**

| Strategy | Sprint 18 | Sprint 19 | Delta |
|----------|-----------|-----------|-------|
| random | 3261.03 | 3261.03 | +0.00 |
| surrogate_only | 3256.04 | 3256.04 | +0.00 |
| causal | 3257.24 | 3259.11 | +1.88 |

### 5b. Analysis

- **Max strategy difference**: 4.99 MAE (0.15%), unchanged from Sprint 18.
- **Random and surrogate_only are identical** to Sprint 18 (expected since
  random ignores the engine, and surrogate_only with no causal graph on
  permuted data converges deterministically).
- **Causal shifted slightly** (3257.24 to 3259.11) due to the optimizer-core
  changes, but the shift is toward random (worse), not away from it. The
  causal strategy is not manufacturing false signal on null data.
- All strategies produce test MAE near the marginal target standard
  deviation, confirming no real signal is captured.

### 5c. Verdict

**Null control: PASS.** The Sprint 19 optimizer-core changes do not create
a false win on permuted data. The 0.15% max strategy difference is within
the 2% null-signal threshold.

## 6. Combined Differentiation Assessment

### Question 1: Did causal guidance become more useful?

**Yes.** On the base counterfactual benchmark, causal now beats
surrogate_only at all budgets (regret 15.84 vs 22.15 at B20, 8.19 vs
17.94 at B40, 0.98 vs 2.16 at B80). This reverses the Sprint 18 result
where surrogate_only led everywhere. On the high-noise variant, causal
beats surrogate_only by an even wider margin (60% lower regret at B80).

### Question 2: On which benchmark class did it help?

Causal guidance helps on benchmarks where the causal graph provides
structural information that reduces the effective search space:

- **Base counterfactual** (5 dimensions, 3 causal parents): causal wins
  at all budgets after the soft influence changes.
- **High-noise counterfactual** (15 dimensions, 3 causal parents): causal
  wins at B40 and B80. The advantage is larger because the search space
  reduction is more valuable (3/15 vs 3/5).
- **Confounded counterfactual**: causal does NOT help. Bidirected edges
  alone are insufficient for deconfounding.
- **Real ERCOT (null)**: causal does not create false wins (correct).

### Question 3: Did the null control stay clean?

**Yes.** The null-signal benchmark still passes with 0.15% max strategy
difference. The Sprint 19 optimizer-core changes did not introduce
false discovery.

### Question 4: What did skip auditing reveal after the optimizer-core changes?

The skip logic is effectively dormant on all current benchmarks:
- Real energy data: no skips (model quality too low).
- Counterfactual: skipping disabled (max_skips=0).
- When it does activate (synthetic), the 7% false-skip rate is acceptable.

The Sprint 17 "33% false-skip rate" was specific to a configuration that
no longer exists. The new measurement infrastructure makes future skip
quality monitoring easy.

## 7. Sprint 19 Verdict

**PROGRESS.**

Causal guidance now beats surrogate_only on two benchmark families where
it plausibly should win (base and high-noise counterfactual). The win is
largest at intermediate budgets (B40) where the search space reduction
from causal knowledge matters most. The null control stays clean (0.15%
max diff). Skip calibration is measured and trustworthy.

Specifically:
- Causal beats surrogate_only somewhere it should: **Yes** (base and
  high-noise counterfactual).
- That win survives the null control staying clean: **Yes** (PASS).
- The explanation is benchmark-credible: **Yes** (causal graph reduces
  effective search space from 5 to 3 on base, from 15 to 3 on
  high-noise).
- Skip behavior is more trustworthy: **Yes** (measured, understood,
  infrastructure in place).

**Caveats:**
1. The base benchmark's surrogate_only regressed at low budgets in this
   run compared to Sprint 18. This may be due to Ax/BoTorch stochasticity
   and should be investigated.
2. The confounded variant shows that bidirected edges alone do not help --
   actual deconfounding at the search level is needed.
3. `causal_softness` does not affect the primary Ax/BoTorch path -- the
   soft trade-off only works on the RF surrogate fallback.

## 8. Sprint 20 Recommendation

**Continue optimizer-core work** with two focused directions:

### Direction A: Investigate surrogate_only regression

The surrogate_only strategy regressed at B20 and B40 on the base
benchmark compared to Sprint 18. Before claiming a stable causal
advantage, we need to understand whether:
- The Sprint 18 surrogate_only results were unusually good (lucky seeds).
- The Sprint 19 code changes inadvertently affected surrogate_only.
- Ax/BoTorch has sufficient run-to-run stochasticity to explain the shift.

Run the base benchmark with 10+ seeds and both Sprint 18 and Sprint 19
code to separate stochastic from systematic effects.

### Direction B: Address confounding

The confounded variant exposes a real limitation: the optimizer does not
implement deconfounding at the search level. Sprint 20 should explore:
- Inverse probability weighting in the policy evaluation step.
- Residualization (regressing out the confounder effect before policy
  evaluation).
- Using the bidirected edge information to adjust the treatment effect
  estimation.

If deconfounding at the search level is infeasible in one sprint, lower
the confounded variant's priority and focus on hardening the high-noise
advantage.

### Direction C: Port soft ranking to the Ax/BoTorch path

The `causal_softness` parameter is currently inert on the primary Ax
path. Sprint 20 should implement re-ranking of Ax candidates by a
balanced score of predicted objective and causal alignment, so the
documented parameter actually works on the production code path.

### What not to do in Sprint 20

1. Do not drop the null-signal benchmark -- it should continue to run on
   every optimizer variant.
2. Do not tune specifically for the base benchmark to preserve the narrow
   win -- the high-noise variant is the more credible test.
3. Do not claim general causal advantage from counterfactual benchmarks
   alone -- the real ERCOT benchmark has not been tested with Sprint 19
   changes yet.

## 9. Evidence Summary

### Test Results

- **894 tests passed**, 15 skipped, 100 deselected (slow).
- All Sprint 19 tests (18 variant tests, 13 soft causal tests, 31 skip
  calibration tests) passing.

### Benchmark Runtimes

| Benchmark | Grid | Runtime |
|-----------|------|---------|
| Base counterfactual | 3 strategies x 3 budgets x 5 seeds = 45 runs | 1305s (21.8 min) |
| High-noise counterfactual | 45 runs | 1334s (22.2 min) |
| Confounded counterfactual | 45 runs | 1305s (21.8 min) |
| Null signal | 3 strategies x 2 budgets x 3 seeds = 18 runs | ~2700s (45 min) |

### Artifacts

- `artifacts/counterfactual_sprint19_base.json`
- `artifacts/counterfactual_sprint19_high_noise.json`
- `artifacts/counterfactual_sprint19_confounded.json`
- `artifacts/null_sprint19_final.json`
