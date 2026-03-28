# Optimizer Improvement Briefs

Updated: 2026-03-26

## Purpose

Turn the next optimizer-improvement ideas into agent-ready briefs with a clear
implementation order that improves `causal-optimizer` as a system rather than
teaching it one benchmark.

This plan is explicitly designed to reduce the risk of overfitting to the first
real benchmark:

1. `ERCOT NCENT/NORTH_C` + DFW weather
2. locked `60 / 20 / 20` split
3. strategies: `random`, `surrogate_only`, `causal`

## Current Evidence

From the first real benchmark run:

1. `random` is marginally better than `causal` on held-out test MAE
2. `causal` and `surrogate_only` are effectively identical
3. all strategies converge to `ridge`
4. results are stable across seeds

Interpretation:

1. the benchmark harness is working
2. the causal path is not meaningfully differentiated under the current fallback path
3. benchmark-driven optimizer changes must now be judged across more than one task

Related preprocessing diagnostics follow-up:

1. `thoughts/shared/plans/09-time-series-calendar-profiler.md`

## Core Principle

Improve the optimizer in a way that should help across tasks:

1. better candidate ranking
2. real causal differentiation
3. more trustworthy skip decisions
4. stronger multi-benchmark evaluation

Do **not** optimize directly for:

1. `NORTH_C` specifically
2. one split window
3. `ridge` specifically
4. one hand-picked metric delta on one dataset

## Recommended Second Real Benchmark

### Choice

Use a second energy benchmark with the same harness:

1. **Load series:** `ERCOT COAST`
2. **Weather series:** Houston-area NOAA hourly weather
3. **Date range:** `2022-01-01` through `2024-12-31`
4. **Saved timezone:** `UTC`

### Why This Benchmark

1. It reuses the same benchmark contract and prep pattern, so it is cheap to add.
2. It gives a materially different load/weather regime from `NORTH_C`.
3. It tests whether optimizer improvements generalize across zones, not just across seeds.
4. It is still close enough to the first benchmark that differences are interpretable.

### Recommended Weather Station

Use a Houston-area NOAA hourly station. The exact station can be finalized in the
prep PR, but prefer a large airport station with strong continuity over 2022-2024.

Official source families:

1. ERCOT hourly load archives:
   [https://www.ercot.com/gridinfo/load/load_hist](https://www.ercot.com/gridinfo/load/load_hist)
2. ERCOT load/weather-zone reference:
   [https://www.ercot.com/gridinfo/load](https://www.ercot.com/gridinfo/load)
3. NOAA NCEI hourly weather product:
   [https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)

## Recommended Semi-Synthetic Counterfactual Benchmark

### Choice

Build a **semi-synthetic demand-response benchmark** using real ERCOT covariates
with known treatment effects.

### Concept

1. Start from real hourly/daily covariates from the prepared ERCOT benchmark data:
   - temperature
   - humidity
   - hour_of_day
   - day_of_week
   - is_holiday
   - lagged load features
2. Simulate a binary treatment such as:
   - `demand_response_event`
3. Generate potential outcomes with a known structural rule:
   - treatment lowers peak load more on very hot afternoons
   - treatment has near-zero effect on mild nights
   - treatment has a cost penalty
4. Keep the covariates real, but make the treatment assignment and treatment
   effect known so we have counterfactual ground truth.

### Why This Benchmark

1. It is much more realistic than a toy SCM.
2. It gives us known counterfactual truth, which real observational data does not.
3. It lets us test whether causal guidance improves intervention choice, not just
   predictive-model selection.

### What It Should Measure

1. policy value under known counterfactual outcomes
2. regret versus the oracle policy
3. treatment-effect ranking quality
4. whether `causal` beats `surrogate_only` and `random` when causal structure matters

## Agent Briefs

### Agent 1: Causal Fallback Differentiation

Goal:

Make `causal` behave materially differently from `surrogate_only` when the
system is using the RF-surrogate fallback path.

Why:

1. The first real benchmark showed `causal == surrogate_only`.
2. Until that changes, we are not really testing a causal optimizer.

Deliverables:

1. one design note explaining exactly where the fallback path loses causal influence
2. one implementation that injects causal guidance into candidate selection,
   scoring, or prioritization under the fallback path
3. one focused test showing that `causal` and `surrogate_only` now produce
   different candidate behavior on a controlled fixture
4. one benchmark-facing note describing expected behavior changes

Acceptance:

1. `causal` and `surrogate_only` are observably different in at least one
   deterministic test
2. the change is generic, not ERCOT-specific
3. existing benchmark tests remain green

### Agent 2: Second Real Benchmark

Goal:

Add a second real benchmark dataset using the same predictive-energy harness.

Choice:

1. `ERCOT COAST` + Houston-area NOAA weather

Deliverables:

1. local prep script or parameterized extension of the first prep script
2. prepared local Parquet dataset
3. smoke artifact
4. full artifact
5. summary CSV
6. benchmark report markdown using the shared template

Acceptance:

1. same benchmark command shape works without code changes
2. dataset passes the same QA gates as the first run
3. report includes the same split-boundary and provenance detail

### Agent 3: Multi-Benchmark Evaluation Harness

Goal:

Judge optimizer changes across a small benchmark suite instead of one dataset.

Deliverables:

1. a simple suite runner that can execute the real predictive benchmark across:
   - `ercot_north_c_dfw_2022_2024`
   - `ercot_coast_houston_2022_2024`
2. a suite summary artifact with per-benchmark and aggregate results
3. an acceptance rule for optimizer changes:
   - improve at least one benchmark
   - no material regression on the others
   - stable across seeds

Acceptance:

1. one command can run both real benchmarks
2. one report can summarize benchmark-by-benchmark and aggregate outcomes
3. the suite makes it easy to reject overfit changes

### Agent 4: Anytime And Skip Calibration

Goal:

Measure whether the optimizer is actually learning faster and whether skip
decisions are trustworthy.

Deliverables:

1. anytime metrics at budgets:
   - `5`
   - `10`
   - `20`
   - `40`
   - `80`
2. skip diagnostics:
   - candidate count considered
   - candidate count evaluated
   - skip ratio
   - skip confidence summary
3. optional audit mode:
   - occasionally force-evaluate a skipped candidate to estimate false-skip rate

Acceptance:

1. engine-based speedups can be explained with diagnostics
2. we can tell whether `causal` is winning because it is smarter or just skipping more

### Agent 5: Semi-Synthetic Counterfactual Benchmark

Goal:

Create the first benchmark that can answer a counterfactual question with known truth.

Choice:

1. semi-synthetic ERCOT demand-response benchmark

Deliverables:

1. benchmark design note describing:
   - covariates
   - treatment
   - potential outcomes
   - oracle policy
2. a data-generation script using real ERCOT covariates plus synthetic treatment effects
3. a benchmark harness and score definition
4. one smoke test and one reproducibility test

Acceptance:

1. counterfactual ground truth is explicit
2. the benchmark is not tied to one exact real dataset row order
3. `causal` has a plausible path to outperform weaker baselines if causal reasoning is real

## Implementation Order

This order is chosen to improve the optimizer without overfitting to one problem.

### Phase 1: Strengthen Measurement First

1. **Agent 2: Second Real Benchmark**
2. **Agent 3: Multi-Benchmark Evaluation Harness**

Reason:

1. We need a suite before trusting optimizer changes.
2. Otherwise we risk teaching the optimizer the quirks of `NORTH_C`.

### Phase 2: Make The Optimizer Actually Different

3. **Agent 1: Causal Fallback Differentiation**

Reason:

1. There is little value in tuning the causal path until it actually differs
   from `surrogate_only`.
2. This is the most important system-level improvement.

### Phase 3: Make Speed Claims Trustworthy

4. **Agent 4: Anytime And Skip Calibration**

Reason:

1. Engine-based strategies are currently faster.
2. We need to know whether that speedup is intelligent or just aggressive skipping.

### Phase 4: Add Counterfactual Ground Truth

5. **Agent 5: Semi-Synthetic Counterfactual Benchmark**

Reason:

1. This is the right place to test causal validity more directly.
2. It should come after the optimizer is differentiated and the real benchmark suite exists.

## Promotion Rule For Optimizer Changes

Do not call an optimizer change a real improvement unless:

1. `causal` becomes materially different from `surrogate_only`
2. the change improves held-out performance on at least one real benchmark
3. it does not materially regress the other real benchmarks
4. results remain stable across seeds
5. any speedup can be explained by trustworthy skip diagnostics

## Suggested Immediate Next Sprint

If we only do one sprint next, do this:

1. implement the second real benchmark
2. add a minimal two-benchmark suite report
3. then land one causal-fallback differentiation change

That is the fastest path to learning something real without overfitting to the
first benchmark.
