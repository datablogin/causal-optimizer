# Sprint 33 Recommendation

Updated: 2026-04-17

## Sprint Theme

**General Causal Autoresearch: First Large-Scale Marketing Uplift Execution**

Sprint 32 closed the planning loop for Criteo:

1. Hillstrom already gave the project its first non-energy real-data result
2. the Criteo audit established that the dataset is accessible and fits the
   current `MarketingLogAdapter` through a wrapper
3. the Sprint 32 contract pinned the first executable Criteo benchmark shape,
   including IPS diagnostics, null-control rules, backend discipline, and
   verdict semantics

That means the next sprint should stop planning and start executing.

## Goal

Run the first Criteo benchmark under the merged contract and end the sprint with
either:

1. a trustworthy Criteo benchmark report, or
2. a trustworthy blocked-state diagnosis if the propensity gate, IPS stack, or
   null control fails before a real verdict can be trusted

Sprint 33 should answer:

1. can the current engine run a large-scale binary uplift benchmark on real
   marketing data with the existing adapter contract?
2. does `causal` separate from `surrogate_only` on Criteo at all?
3. does the result look like Hillstrom repeated, or does the larger dataset
   change the story?
4. is the current IPS-based evaluation stable enough to support future
   marketing-domain claims?

## Why This Sprint Now

The biggest remaining ambiguity is no longer contractual. It is empirical.

After Sprint 32:

1. Criteo is the next best test of whether Hillstrom's result was dataset-
   specific or a broader marketing-uplift boundary
2. the contract already defines:
   - primary outcome
   - benchmark scale
   - fixed versus tunable variables
   - mandatory diagnostics
   - null-control behavior
   - what different outcomes would mean
3. delaying execution further would create planning churn without adding much
   real information

Sprint 33 should therefore execute the locked Criteo line rather than reopen the
contract unless the data itself forces a blocker.

## Strategy

Sprint 33 should proceed in one main lane with one conditional branch:

1. implement the `CriteoLoader`, fixture, benchmark script, and provenance path
2. run the fixed 1M-row benchmark and null control under the contract
3. if Run 1 is near-parity on the degenerate surface, run the mandatory second
   heterogeneous-surface batch before publishing the final report

## Main Workstream

### 1. Criteo Benchmark Implementation And First Report

This is the critical path.

Implement the contract exactly as merged in the Sprint 32 benchmark document:

1. write a `CriteoLoader` wrapper
2. keep `MarketingLogAdapter` unchanged
3. generate the 1M-row fixed-seed local subsample
4. commit a small CC-BY-NC-SA fixture for CI
5. build a benchmark runner modeled on the Hillstrom harness
6. run:
   - 3 strategies
   - budgets `20,40,80`
   - seeds `0-9`
   - 90 real runs
   - 60 null-control runs
7. publish one Criteo report that states either:
   - certified / trending / near-parity / surrogate-only advantage
   - or IPS/null-control failure if the evidence discipline breaks

The report should be the authoritative Sprint 33 artifact.

## Mandatory Gates

Sprint 33 must not force a benchmark verdict through failed guardrails.

The following are hard gates:

1. **Propensity heterogeneity gate**
   - if treatment rate by `f0` decile deviates by more than 2 percentage points
     from `0.85`, stop and switch to the fallback propensity path defined in the
     contract before running the benchmark
2. **Backend consistency gate**
   - do not mix Ax and RF within one verdict table
3. **Null-control gate**
   - if the null control fails even after the pre-committed fallback band, the
     sprint ends in diagnosis, not in a performance claim
4. **ESS / IPS-variance gate**
   - if median ESS is below 100 or weight behavior is consistently pathological,
     the sprint should say so plainly and treat the result as unreliable

## Conditional Branch

### 2. Mandatory Run 2 If Run 1 Is Near-Parity

The first Criteo run is intentionally degenerate:

1. `segment` is omitted
2. `channel` is constant
3. `regularization` is frozen because it is inert under the current adapter math
4. the effective first-pass space is only:
   - `eligibility_threshold`
   - `treatment_budget_pct`

That means Run 1 is a useful baseline, but it is not the whole story if all
three strategies collapse to the same corner.

If Run 1 is near-parity, Sprint 33 must also run the heterogeneous follow-up:

1. synthesize `segment` from `f0` tertiles
2. rerun the benchmark under the same evidence rules
3. report the combined Run 1 + Run 2 verdict using the contract language

This is not optional if the near-parity condition is met.

## Success Criteria

Sprint 33 is successful if:

1. the first Criteo benchmark is executed under the merged Sprint 32 contract
2. the project ends the sprint with a real report rather than only code
3. the report makes a clean distinction between:
   - performance ordering
   - IPS reliability
   - null-control behavior
4. any blocked state is surfaced honestly instead of being smoothed over
5. the sprint clarifies whether marketing uplift remains a surrogate-only domain
   or whether Criteo changes the story

## What Would Count As A Good Outcome

Best case:

1. Ax runs cleanly
2. the null control passes
3. ESS is adequate
4. `causal` beats `surrogate_only` or at least shows a credible trending signal
5. the project leaves Sprint 33 with stronger non-energy generalization evidence

Still valuable:

1. Criteo reproduces a surrogate-only advantage cleanly
2. the project learns that the Hillstrom result was not a one-off
3. the null-control and IPS diagnostics remain trustworthy
4. the result narrows the generality claim honestly instead of by implication

Also valuable:

1. the benchmark is blocked by the propensity gate, ESS, or null control
2. the failure is diagnosed precisely enough to determine the next technical move
   (`full 14M`, `clip tuning`, DR estimation, or adapter evolution)

## What Not To Do

1. do not reopen the benchmark contract unless the data forces a blocker the
   contract did not anticipate
2. do not modify `MarketingLogAdapter` in Sprint 33 unless a true blocker makes
   the contract impossible to execute
3. do not mix Ax and RF results in the same verdict table
4. do not skip the null control because the real run looks promising
5. do not report a degenerate-surface near-parity result as evidence that the
   causal path is generally flat on marketing data
6. do not publish a real-data win if the null-control or ESS gates fail

## Recommended Order

1. implement `CriteoLoader`, fixture, and benchmark runner
2. run the propensity gate on the 1M-row subsample
3. run the 90 real benchmark cells
4. run the 60 null-control cells
5. if Run 1 is near-parity, run the mandatory heterogeneous second batch
6. publish the Sprint 33 Criteo benchmark report

## Suggested Sprint 33 Issues

1. implement the Criteo benchmark harness and fixture
2. run the first Criteo benchmark and publish the report

These can be one issue if the team wants a single critical path, but the report
deliverable should not be dropped.

## Exit Criterion

At the end of Sprint 33, we should know:

1. whether the current engine can produce a trustworthy first Criteo result
2. whether marketing uplift still looks like a `surrogate_only`-favored domain
3. whether the current IPS stack is stable enough for larger real-data marketing
   claims
4. whether Sprint 34 should focus on:
   - follow-up Criteo diagnosis
   - Open Bandit / multi-action expansion
   - or adapter / evaluator improvements such as DR or clip tuning
