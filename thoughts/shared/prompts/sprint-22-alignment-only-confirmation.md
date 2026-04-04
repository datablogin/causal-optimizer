# Sprint 22 — Alignment-Only Confirmation

## Goal

Act on the Sprint 21 attribution result.

Sprint 21 concluded that balanced Ax reranking is **not attributed** as the cause of Sprint 20's observed improvement. The immediate job for Sprint 22 is not to invent another optimizer feature. It is to:

1. revert or disable balanced reranking back to alignment-only
2. confirm that the alignment-only path actually preserves the stronger positive-control behavior
3. verify that null-control safety remains intact after the revert
4. decide whether the project is ready for new benchmark families or still needs optimizer-core hardening

## Read Before Starting

1. `README.md`
2. `thoughts/shared/plans/07-benchmark-state.md`
3. `thoughts/shared/docs/sprint-20-post-ax-rerun-report.md`
4. `thoughts/shared/docs/sprint-21-controlled-ab-rerun-report.md`
5. `thoughts/shared/docs/sprint-21-attribution-scorecard.md`

## Working Assumption

Unless the evidence changes materially, treat Sprint 21's conclusion as the current project truth:

1. balanced reranking is not the supported path
2. alignment-only is the better current default
3. the next job is confirmation and stabilization, not fresh feature exploration

## What To Do

### Step 1 — Revert Balanced Reranking

Make the production path alignment-only again.

Requirements:

1. remove or disable the balanced reranking behavior introduced in PR `#108`
2. keep any useful test scaffolding only if it remains honest and non-confusing
3. do not leave hidden experiment toggles in production code unless there is a very strong reason and it is documented cleanly

### Step 2 — Rerun the Benchmark Controls

Rerun the minimum evidence stack on the reverted code:

1. base counterfactual
   - budgets: `20,40,80`
   - seeds: `0-9`
2. high-noise counterfactual
   - budgets: `20,40,80`
   - seeds: `0-9`
3. null control
   - budgets: `20,40`
   - seeds: `0,1,2`

If runtime permits, include one small ERCOT smoke/regression check to ensure there is no obvious breakage on the real benchmark path.

## Artifact Expectations

Write machine-local artifacts with clear names under:

- `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/`

Use names that clearly identify:

1. reverted alignment-only run
2. benchmark family
3. whether the artifact is base, high-noise, or null

## Report

Write:

- `thoughts/shared/docs/sprint-22-alignment-only-confirmation-report.md`

The report must answer:

1. after reverting balanced reranking, does alignment-only still deliver the strong base B80 behavior seen in Sprint 21's locked A/B comparison?
2. does high-noise remain at least directionally strong for causal vs surrogate_only?
3. does null control still pass cleanly?
4. is the project now ready to expand benchmark families, or should Sprint 22 stay focused on optimizer hardening?

## Required Verdicts

Choose one:

1. `CONFIRMED`
2. `MIXED`
3. `REGRESSED`
4. `INCONCLUSIVE`

## Recommendation Standard

1. If alignment-only confirms the Sprint 21 A/B result, say so plainly.
2. If the revert does not reproduce the expected base B80 strength, stop treating Sprint 21's A/B result as operationally settled.
3. Do not recommend new benchmark families unless:
   - base and high-noise remain healthy
   - null control stays clean
   - attribution remains coherent

## Acceptance

1. the reverted production path is clear and reviewable
2. every major claim in the report traces to Sprint 22 artifacts
3. null-control safety is explicit
4. the recommendation follows from the evidence rather than from optimizer preference

## Stop Conditions

Stop and report if:

1. the revert is not as narrow as expected
2. the reverted behavior does not match the Sprint 21 locked A/B evidence directionally
3. null control regresses
4. provenance capture is broken on the new artifacts

## Final Report Back

Report:

1. PR URL
2. branch
3. head commit
4. verdict
5. one-paragraph case for that verdict
6. one-sentence recommendation for what Sprint 23 should be
