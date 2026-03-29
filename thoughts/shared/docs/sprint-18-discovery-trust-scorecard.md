# Sprint 18 Discovery Trust Scorecard

## Metadata

- **Date**: 2026-03-29
- **Sprint**: 18 (Discovery Trust)
- **PRs included**: #86 (Time-Series Calendar Profiler), #87 (Counterfactual
  Benchmark Repair), #88 (Null-Signal Control Benchmark)
- **Issue**: #90

## 1. Sprint 18 Summary

Sprint 17 established that causal guidance does not yet outperform random search
on real ERCOT benchmarks, that the off-policy skip predictor has a 33% false-skip
rate, and that the original counterfactual benchmark was degenerate (oracle =
"never treat"). Sprint 17's verdict was **INVESTIGATE**: the harness works, but
the evidence does not yet support a causal-advantage claim.

Sprint 18 responds by strengthening the evidence base rather than tuning the
optimizer. It delivered three tools for evaluating discovery trust:

1. **Time-Series Calendar Profiler** (PR #86) -- A rule-based diagnostic that
   catches timezone, DST, cadence, and interval-convention errors before they
   inject false signal into benchmarks.

2. **Counterfactual Benchmark Repair** (PR #87) -- Re-parameterized the
   demand-response benchmark so the oracle treats 32% of rows (previously 0%),
   creating a non-trivial positive control where learned strategies have
   measurable regret.

3. **Null-Signal Control Benchmark** (PR #88) -- A permuted-target negative
   control that verifies the optimizer does not manufacture false wins from noise.

Together these answer the central question: is `causal-optimizer` a trustworthy
research system, or is it automating noise chasing?

## 2. Time-Series Profiler Validation

### Profiler Output on ERCOT NORTH_C (DFW)

| Field | Value |
|-------|-------|
| Rows | 26,291 |
| Range | 2022-01-01 06:00 UTC -- 2024-12-31 23:00 UTC |
| Cadence | hourly (regularity: 0.9998) |
| Monotonic | Yes |
| Duplicates | 0 |
| DST suspected | Yes |
| Recommended calendar TZ | **US/Central** |
| Recommended storage TZ | UTC |
| Interval convention | interval_start (confidence: 0.8) |
| Holiday calendar | US_FEDERAL |
| Gaps detected | 4 |
| Stop | No |

Key recommendations:

1. **[P1] calendar-local-tz**: Derive calendar features (hour_of_day,
   day_of_week, is_holiday) in US/Central. Reason: calendar features explain
   more target variance in US/Central than UTC.
2. **[P2] store-utc**: Store timestamps in UTC for unambiguous serialization.
3. **[P2] holiday-calendar**: Use US_FEDERAL holiday calendar derived in
   US/Central.

### Profiler Output on ERCOT COAST (Houston)

| Field | Value |
|-------|-------|
| Rows | 26,297 |
| Range | 2022-01-01 06:00 UTC -- 2024-12-31 22:00 UTC |
| Cadence | hourly (regularity: 1.0) |
| Monotonic | Yes |
| Duplicates | 0 |
| DST suspected | Yes |
| Recommended calendar TZ | **US/Central** |
| Recommended storage TZ | UTC |
| Interval convention | interval_start (confidence: 0.8) |
| Holiday calendar | US_FEDERAL |
| Gaps detected | 0 |
| Stop | No |

Recommendations are identical to NORTH_C: derive calendar features in
US/Central, store in UTC, use US_FEDERAL holidays.

### Assessment

**Does it correctly recommend US/Central for calendar features?** Yes. On both
datasets, the profiler selects US/Central over UTC based on higher eta-squared
scores for hour-of-day and a midnight-alignment bonus (data starts at local
midnight when interpreted in US/Central). This matches the known ERCOT market
convention.

**Does it detect any issues in the existing data?** The NORTH_C dataset has 4
gaps (intervals > 1.5x median), flagged as a warning. The COAST dataset has
perfect regularity. Neither dataset has duplicates, non-monotonic timestamps,
or stop-level issues.

**Does this reduce false-signal risk?** Yes. If a user derived hour_of_day in
UTC instead of US/Central, the resulting calendar features would be shifted by
5-6 hours relative to actual local demand patterns. The profiler catches this
and emits an actionable P1 recommendation. DST detection also flags the risk of
ambiguous fall-back hours if timestamps were stored in local time.

**Profiler verdict: PASS.** The profiler is rule-based, explainable, and
produces the correct recommendation on both ERCOT datasets. It reduces the risk
of silent false signal from timestamp misinterpretation.

## 3. Positive Control: Counterfactual Benchmark

### Oracle Statistics

| Metric | Value |
|--------|-------|
| Oracle treat rate | 31.9% (8,399 / 26,291 rows) |
| Oracle policy value (full dataset) | 36.80 |
| Oracle policy value (test set) | 48.41 |
| Always-treat policy value | 3.17 |
| Never-treat policy value | 0.00 |

The oracle treats roughly one-third of hours -- those with high temperature and
midday timing where demand-response benefit exceeds the $60 cost. This is a
meaningful decision boundary: always-treat is 10x worse than the oracle, and
never-treat scores zero.

### Regret by Strategy

| Strategy | Budget | Mean Regret | Std Regret | Decision Error |
|----------|--------|-------------|------------|----------------|
| random | 20 | 20.06 | 10.39 | 25.2% |
| random | 40 | 19.11 | 9.21 | 27.2% |
| random | 80 | 9.16 | 2.45 | 21.0% |
| surrogate_only | 20 | 12.07 | 7.17 | 19.3% |
| surrogate_only | 40 | 11.16 | 7.91 | 17.9% |
| surrogate_only | 80 | 1.75 | 1.11 | 7.3% |
| causal | 20 | 17.74 | 4.99 | 25.2% |
| causal | 40 | 17.66 | 4.89 | 25.1% |
| causal | 80 | 2.46 | 0.86 | 9.3% |

### Key Findings

1. **Benchmark is non-degenerate.** Oracle treat rate = 32%, policy value =
   36.80, random has substantial regret. The Sprint 17 degeneracy (oracle =
   "never treat") is fully resolved.

2. **Learned strategies dramatically outperform random at B80.** Surrogate_only
   regret = 1.75 and causal regret = 2.46 vs random regret = 9.16. Both
   learned strategies find near-optimal policies.

3. **Surrogate_only slightly edges causal.** Surrogate_only has lower regret
   than causal at all budgets (12.07 vs 17.74 at B20, 1.75 vs 2.46 at B80).
   The gap narrows with budget.

4. **Causal has lower variance.** Causal's cross-seed standard deviation is
   consistently lower than surrogate_only and random (e.g., B20: causal
   std=4.99 vs surrogate_only std=7.17 vs random std=10.39). Causal is more
   consistent, even when its mean is slightly worse.

5. **Performance is fixed.** Causal and surrogate_only have identical runtimes
   (~13s at B80). The Sprint 17 100x slowdown is fully resolved.

### Assessment

**Can the system exploit known signal?** Yes. Both surrogate_only and causal
find near-optimal policies on a benchmark with non-trivial structure. At B80,
both achieve regret under 2.5 against an oracle value of 48.41 -- over 95%
policy efficiency.

**Does causal beat random when the graph has non-trivial structure?** Yes, at
B80 (regret 2.46 vs 9.16). However, causal does not yet beat surrogate_only.
The benchmark's noise structure (2 noise dimensions out of 5 total) is simple
enough that surrogate_only can screen it out without causal knowledge. This is
an honest result: causal advantage may require graphs with more confounders or
harder noise patterns.

**Positive control verdict: PASS.** The benchmark is valid, discriminating, and
reproducible. It provides a credible positive-control test bed.

## 4. Negative Control: Null-Signal Benchmark

### Design

The benchmark permutes the target column (`target_load`) with a fixed seed,
destroying all temporal and covariate-to-target signal while preserving the
marginal distribution and full covariate structure. The permuted data uses the
same 60/20/20 blocked time split.

### Results

| Strategy | Budget | Test MAE (mean +/- std) |
|----------|--------|-------------------------|
| random | 20 | 3260.48 +/- 1.75 |
| random | 40 | 3261.58 +/- 0.28 |
| surrogate_only | 20 | 3255.76 +/- 0.39 |
| surrogate_only | 40 | 3256.31 +/- 0.00 |
| causal | 20 | 3256.96 +/- 2.08 |
| causal | 40 | 3257.51 +/- 2.86 |

**Mean Test MAE across all seeds and budgets:**

| Strategy | Mean Test MAE | Std |
|----------|---------------|-----|
| random | 3261.03 | 1.50 |
| surrogate_only | 3256.04 | 0.42 |
| causal | 3257.24 | 2.76 |

### Key Findings

1. **No strategy shows consistent held-out improvement beyond noise.** The
   difference between best (surrogate_only, 3256.04) and worst (random,
   3261.03) is 4.99 MAE units, a 0.15% relative difference. Well within the
   2% null-signal threshold.

2. **Apparent differences are noise artifacts.** Random search achieves slightly
   lower validation MAE but slightly higher test MAE -- the classic pattern of
   overfitting to validation noise.

3. **Surrogate_only converges deterministically on null data.** At B40 its std
   is ~0.00, confirming it locks in quickly when nothing predicts the target.

4. **All strategies produce test MAE near the marginal target standard
   deviation**, confirming no real signal is captured.

### Assessment

**Is the system resisting false discoveries on null data?** Yes. The null-signal
check returns PASS. No strategy beats random by more than 2% on held-out test
data. The small observed differences (0.15%) are consistent with random
variation and are not stable across seeds.

**Negative control verdict: PASS.** The optimizer correctly fails to find signal
where none exists. This provides confidence that wins on real data reflect
genuine signal.

## 5. Combined Discovery Trust Assessment

### Question 1: Is the system resisting false discoveries on null data?

**Yes.** The permuted-target negative control shows no stable promotable wins.
All three strategies converge to essentially the same test MAE (~3257), with
differences under 0.15%. The system does not manufacture false discovery.

### Question 2: Is the system capable of exploiting true signal in a benchmark designed to contain it?

**Yes, partially.** On the repaired counterfactual benchmark:

- Both learned strategies (surrogate_only and causal) dramatically outperform
  random at B80, achieving over 95% policy efficiency.
- Causal does not yet outperform surrogate_only. This is honest: the current
  benchmark's noise structure is simple enough to screen without causal
  knowledge.
- Causal shows notably lower cross-seed variance, suggesting causal graph
  guidance provides stability even when it does not improve the mean.

### Question 3: Should Sprint 19 focus on optimizer-core changes or more benchmark/diagnostic work?

**Optimizer-core changes, guided by the new benchmark infrastructure.**

Sprint 18 has established three things:

1. The system does not fabricate signal (null control PASS).
2. The system can exploit signal when it exists (positive control PASS).
3. The time-series profiler catches common data-semantics errors.

This means the evidence infrastructure is now strong enough to evaluate
optimizer-core changes with confidence. The project is no longer at risk of
confusing noise with signal.

The remaining gap is that **causal guidance does not yet outperform
surrogate_only** on any benchmark. The evidence suggests this is because:

- On the real ERCOT benchmark, the causal graph is a star (all variables are
  direct parents of MAE), providing no selective information.
- On the counterfactual benchmark, the noise dimensions are easy to screen
  without causal knowledge.
- At low budgets (B20, B40), causal guidance has limited room to help because
  it only activates after the exploration phase (experiment 11+). On the
  counterfactual benchmark, causal actually underperforms surrogate_only at
  these budgets, likely because the focus-variable restriction over-constrains
  early optimization steps.

These are addressable through optimizer-core improvements, and the benchmark
suite can now reliably evaluate whether those improvements help.

## 6. Sprint 19 Recommendation

### Primary focus: Optimizer-core changes to realize causal advantage

The evidence base is now strong enough to support targeted optimizer
experimentation. Recommended priorities:

**Priority 1: Earlier causal influence.** At B20 and B40, causal guidance has
limited impact because it only activates in the optimization phase (experiment
11+). With a budget of 20, only 10 experiments benefit from causal guidance,
and the focus-variable restriction may over-constrain those few steps. Sprint
19 should explore:

- Causal-informed exploration (use graph structure to bias LHS sampling)
- Earlier phase transition (start optimization before experiment 11)
- Causal filtering during exploration (prune unpromising regions early)

**Priority 2: Harder counterfactual variants.** The current counterfactual
benchmark has 2 noise dimensions out of 5 total, which is easy enough for
any surrogate to handle. Sprint 19 should add variants with:

- Higher-dimensional noise (10+ irrelevant variables)
- Confounding structure that makes surrogate-only estimation biased
- Interaction effects that require causal knowledge to resolve

**Priority 3: Skip calibration.** Sprint 17 showed a 33% false-skip rate.
While Sprint 18 sidestepped this by setting `max_skips=0` on the
counterfactual benchmark, production use requires trustworthy skip decisions.
Options include calibrated confidence thresholds, Platt scaling, or a burn-in
period.

**Priority 4: Real benchmark differentiation.** The real ERCOT benchmark
currently uses a star graph where causal guidance provides no advantage over
exhaustive surrogate search. Sprint 19 should either:

- Design a causal graph for ERCOT that encodes domain knowledge about indirect
  effects and confounders
- Identify a different real-data domain where causal structure matters

### What not to do in Sprint 19

1. Do not tune the optimizer to overfit ERCOT -- the profiler and controls exist
   to catch this.
2. Do not remove the null-signal benchmark -- it should run on every new
   optimizer variant.
3. Do not claim causal advantage without demonstrating it on both the positive
   and negative controls.

## 7. Overall Sprint 18 Verdict

| Deliverable | Status | Key Evidence |
|------------|--------|--------------|
| Time-Series Profiler | PASS | Correctly recommends US/Central for both ERCOT datasets; detects DST; catches gaps |
| Counterfactual Benchmark | PASS | Oracle treat rate = 32%; random has measurable regret; learned strategies find near-optimal policies |
| Null-Signal Control | PASS | No strategy wins on permuted data; 0.15% max difference; classic noise pattern |
| Discovery Trust | PASS | System resists false discovery and exploits true signal |

**Sprint 18 is successful.** The project now has:

1. A rule-based profiler that catches timestamp-semantics errors before they
   inject false signal.
2. A non-degenerate positive control that rewards intelligent search and has
   a non-trivial causal structure.
3. A clean negative control that verifies the optimizer does not manufacture
   false wins.

The system is not yet discovering causal advantage over surrogate-only search,
but we now know this with confidence rather than uncertainty. The controls are
strong, the evidence is honest, and the next sprint can focus on optimizer-core
improvements with trustworthy evaluation infrastructure in place.
