# Benchmark State

Updated: 2026-04-09 (Sprint 28 complete -- AX PRIMARY, RF SECONDARY)

## Purpose

This file is the restart point for benchmark, attribution, and sprint-direction work in
`causal-optimizer`.

Use it when:

1. starting a new chat
2. handing context to another reviewer or agent
3. checking what is merged vs still pending
4. recovering after an interruption or crash

## Current Position

The project is now strongest as a **trustworthy research harness with a clearer causal boundary claim**.

What is true today:

1. the benchmark stack is disciplined and reusable
2. positive controls, negative controls, provenance, locked A/B reruns, stability gates, and boundary-mapping artifacts are all in place
3. the project can demonstrate causal advantage on controlled benchmark families where noise burden and categorical barriers matter
4. the project still has **not** demonstrated a reliable real-world causal advantage on the ERCOT forecasting benchmarks
5. alignment-only remains the correct production default after Sprint 21 attribution and Sprint 22 confirmation
6. Sprint 25 delivered the first mechanism-matched stability fix that actually met the base-B80 gate
7. Sprint 26 and Sprint 27 expanded benchmark coverage to seven rows and clarified the causal-advantage boundary
8. Sprint 28 is now tightening backend comparability so Ax/BoTorch and RF-fallback claims stop getting conflated

## Current Goal

Sprint 28 is complete.  The backend baseline scorecard verdict is
**AX PRIMARY, RF SECONDARY**.

Sprint 29 should return to optimizer-core work.  The benchmark contract is
clean enough to evaluate code changes with confidence.  The suite has explicit
provenance, directly comparable Ax baselines, defined RF-fallback gates, and
10 clean null-control runs.

## Current Sprint Status

### Sprint 28

Plan:

1. [19-sprint-28-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/19-sprint-28-recommendation.md)

Issues:

1. [#146](https://github.com/datablogin/causal-optimizer/issues/146) optimizer-path provenance
2. [#147](https://github.com/datablogin/causal-optimizer/issues/147) Ax/BoTorch seven-benchmark regression gate
3. [#148](https://github.com/datablogin/causal-optimizer/issues/148) backend baseline scorecard

Prompts:

1. [sprint-28-optimizer-path-provenance.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-28-optimizer-path-provenance.md)
2. [sprint-28-ax-regression-gate.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-28-ax-regression-gate.md)
3. [sprint-28-backend-baseline-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-28-backend-baseline-scorecard.md)

Merged PRs:

1. `#149` merged
   - PR: [#149](https://github.com/datablogin/causal-optimizer/pull/149)
   - merge commit: `f715173`
   - issue: `#146` closed
   - result: every benchmark artifact now records optimizer-path provenance
2. `#150` merged
   - PR: [#150](https://github.com/datablogin/causal-optimizer/pull/150)
   - merge commit: `f765d31`
   - issue: `#147` closed
   - result: Ax/BoTorch 7-benchmark gate PASS, trusted priors reproduced exactly
3. `#148` scorecard
   - issue: `#148`
   - result: AX PRIMARY, RF SECONDARY verdict

## Immediate Next Step

Sprint 28 is complete.  If resuming:

1. read the [Sprint 28 backend baseline scorecard](thoughts/shared/docs/sprint-28-backend-baseline-scorecard.md)
2. plan Sprint 29 -- return to optimizer-core work
3. use the characterized boundary to guide optimizer improvements

## Canonical Docs

Core references:

1. [README.md](/Users/robertwelborn/Projects/causal-optimizer/README.md)
2. [ercot-north-c-dfw-2022-2024-benchmark-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md)
3. [ercot-coast-houston-2022-2024-benchmark-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-coast-houston-2022-2024-benchmark-report.md)
4. [sprint-18-discovery-trust-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-18-discovery-trust-scorecard.md)
5. [sprint-19-differentiation-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-19-differentiation-scorecard.md)
6. [sprint-20-post-ax-rerun-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-20-post-ax-rerun-report.md)
7. [sprint-21-controlled-ab-rerun-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-21-controlled-ab-rerun-report.md)
8. [sprint-21-attribution-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-21-attribution-scorecard.md)
9. [sprint-22-alignment-only-confirmation-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-22-alignment-only-confirmation-report.md)
10. [sprint-23-stability-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-23-stability-scorecard.md)
11. [sprint-24-stability-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-24-stability-scorecard.md)
12. [sprint-25-stability-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-25-stability-scorecard.md)
13. [sprint-26-expansion-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-26-expansion-scorecard.md)
14. [sprint-27-medium-noise-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-27-medium-noise-report.md)
15. merged GitHub artifacts:
   - [PR #144](https://github.com/datablogin/causal-optimizer/pull/144) Sprint 27 combined regression gate
   - [PR #145](https://github.com/datablogin/causal-optimizer/pull/145) Sprint 27 crossover scorecard + README update
   - [PR #149](https://github.com/datablogin/causal-optimizer/pull/149) Sprint 28 optimizer-path provenance
   - [PR #150](https://github.com/datablogin/causal-optimizer/pull/150) Sprint 28 Ax/BoTorch regression gate (merged)
16. [19-sprint-28-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/19-sprint-28-recommendation.md)

## Benchmark / Evidence Rules

These should not drift:

1. held-out evaluation must remain genuinely held out
2. time-series semantics must be handled explicitly when relevant
3. null-signal controls must stay clean before we trust optimizer wins
4. provenance must be captured whenever attribution or backend-comparability claims matter
5. benchmark reports must separate:
   - observed performance improvement
   - attribution to a code change
6. attractive optimizer stories do not count if they fail the locked comparison
7. stability reports must separate:
   - understanding the failure
   - fixing the failure
8. backend path must be recorded explicitly; RF fallback and Ax/BoTorch are not interchangeable baselines

## What We Learned By Sprint

### Sprint 18

Merged PRs:

1. `#86` time-series calendar profiler
2. `#87` counterfactual benchmark repair
3. `#88` null-signal control benchmark
4. `#91` discovery trust scorecard

What we learned:

1. time-series semantics are a first-class source of false signal
2. repaired counterfactual benchmarks can serve as real positive controls
3. null controls are essential and the project can pass them
4. the system can now distinguish “real signal exists here” from “this is noise” much more reliably

### Sprint 19

Merged PRs:

1. `#98` harder counterfactual variants
2. `#99` earlier / softer causal influence
3. `#101` skip calibration under controls
4. `#103` differentiation scorecard

What we learned:

1. soft-causal optimizer changes produced the first meaningful causal progress
2. causal improved on base and high-noise positive controls
3. confounded variants remained hard / negative
4. skip calibration became measurable
5. final Sprint 19 verdict: `PROGRESS`

### Sprint 20

Merged PRs:

1. `#107` stability audit
2. `#108` balanced Ax reranking
3. `#109` differentiation scorecard
4. `#110` post-Ax controlled rerun report

What we learned:

1. the Sprint 19 gains were more fragile than the early read suggested
2. balanced Ax reranking looked promising in an observed rerun
3. null control stayed clean
4. attribution remained unresolved because the comparison was not locked tightly enough
5. final Sprint 20 read: observed picture `BETTER`, attribution unresolved

### Sprint 21

Merged PRs:

1. `#114` provenance hardening
2. `#115` controlled A/B rerun
3. `#116` attribution scorecard

What we learned:

1. provenance now records enough information to support real attribution claims
2. a locked A/B rerun compared balanced reranking to alignment-only in the same environment
3. on the base counterfactual at B80, alignment-only was clearly better
4. on high-noise, the two approaches were mostly indistinguishable
5. null control was identical on both sides
6. Sprint 20's observed improvement was real as a benchmark snapshot, but **not attributable** to balanced reranking

### Sprint 22

Merged PRs:

1. `#118` alignment-only confirmation

What we learned:

1. reverting to alignment-only was the right production decision
2. the stronger Sprint 21 alignment-only session was **not** cleanly reproduced
3. the project should stay on stability hardening before widening benchmark scope
4. final Sprint 22 verdict: `MIXED`

### Sprint 23

Merged PRs:

1. `#122` B80 seed diagnostics
2. `#123` Ax stability hardening
3. `#124` stability scorecard

What we learned:

1. the base-B80 failure is a categorical lock-in on `treat_day_filter = "weekday"`
2. the failure lives in Ax candidate generation / exploration behavior, not in the reranker or evaluation
3. PyTorch determinism hardening was the wrong fix and was reverted
4. forwarding the step seed into Ax was a valid reproducibility fix, but not a stability fix
5. final Sprint 23 verdict: `IMPROVED BUT FRAGILE`

### Sprint 24

Merged PRs:

1. `#128` categorical diversity fix
2. `#129` post-fix stability sweep
3. `#130` stability scorecard

What we learned:

1. the categorical-diversity fix was implemented correctly and kept production behavior narrow
2. the post-fix stability sweep was numerically identical to Sprint 23 hardened on the base-B80 slice
3. candidate availability was therefore a tested negative result
4. high-noise remained directionally strong
5. null control stayed clean
6. final Sprint 24 verdict: `STILL TOO FRAGILE`

### Sprint 25

Merged PRs:

1. `#131` exploitation-phase categorical sampling
2. `#136` stability scorecard

What we learned:

1. the failure was not candidate availability; it was exploitation-phase categorical selection
2. the exploitation-phase categorical sweep changed bad-seed trajectories directly
3. base B80 stability finally met the gate for the first time:
   - catastrophic seeds `0/10`
   - mean regret `1.13`
   - std `1.40`
4. high-noise stayed strong
5. null control passed again
6. final Sprint 25 verdict: `STABLE ENOUGH TO EXPAND`

### Sprint 26

Merged PRs:

1. `#137` dose-response clinical benchmark
2. `#138` interaction-policy positive-control family
3. `#139` expansion scorecard

What we learned:

1. benchmark coverage widened successfully
2. causal advantage is domain-dependent, not universal
3. interaction-heavy policy surfaces produced a tie or near-tie under the trusted path
4. smooth all-continuous dose-response surfaces favored `surrogate_only`
5. null control was intentionally not rerun in Sprint 26
6. final Sprint 26 verdict: `EXPANDED BUT MIXED`

### Sprint 27

Merged PRs:

1. `#143` medium-noise crossover variant
2. `#144` combined regression gate
3. `#145` crossover scorecard + README update

What we learned:

1. medium-noise filled the gap between base and high-noise cleanly within the demand-response family
2. the demand-response family now shows a smooth monotonic gradient where causal remains strong as noise burden rises
3. the crossover to `surrogate_only` advantage is structural across benchmark families, not simply a noise-dimension threshold within demand-response
4. the seven-benchmark combined gate passed under RF fallback with a caveat: family-level conclusions survived, but backend comparability remained imperfect
5. null control produced its ninth clean run across ten sprints (`S26` intentionally did not rerun it)
6. final Sprint 27 verdict: `BOUNDARY CLEARER`

### Sprint 28 (Backend-Comparable Baselines)

Merged PRs:

1. `#149` optimizer-path provenance (merged)
2. `#150` Ax/BoTorch regression gate (merged)

What we learned:

1. every benchmark artifact now records its optimizer path explicitly
2. the Ax/BoTorch seven-benchmark rerun restores direct comparability to the trusted Ax baselines
3. all demand-response trusted priors reproduced exactly (base matches S25, medium matches S27, high matches S25)
4. null control: 10th clean run (0.2% max delta)
5. two rows change winner by backend: base energy (tie under RF, causal win under Ax) and dose-response (s.o. win under RF, causal trend under Ax)
6. one row is backend-sensitive in magnitude: interaction (s.o. wins under both, gap narrows under Ax)
7. four rows are backend-invariant: medium-noise, high-noise, confounded, null control
8. verdict: **AX PRIMARY, RF SECONDARY**
9. RF fallback is useful for family-level drift detection but should not be used to certify or refute row-level causal-advantage claims
10. Sprint 29 should return to optimizer-core work

## Current Best Evidence

### Real Predictive Benchmarks

ERCOT NORTH_C and COAST still show:

1. `random` marginally better than engine-based strategies
2. `causal` effectively identical to `surrogate_only`
3. all strategies converging to `ridge`
4. highly stable results across seeds

Interpretation:

1. the real-task harness is functioning correctly
2. but the project still lacks a convincing real-world causal advantage on these forecasting tasks

### Positive / Negative Benchmark Boundary

Trusted Ax-primary demand-response references:

1. base B80 (Sprint 25 / reproduced in Sprint 28): causal mean/std `1.13 / 1.40`, catastrophic seeds `0/10`
2. medium-noise B80 (Sprint 27 / reproduced in Sprint 28): causal mean/std `1.87 / 1.74`, causal wins `10/10`, two-sided `p=0.007`
3. high-noise B80 (Sprint 25 / reproduced in Sprint 28): causal mean/std `2.57 / 2.28`, causal wins `8/10`, two-sided `p=0.014`

Cross-family rows:

1. confounded demand-response: all strategies can be misled; not a clean causal win row (backend-invariant)
2. interaction policy: `surrogate_only` wins under both backends, magnitude is backend-sensitive (backend-sensitive)
3. dose-response clinical: causal trends toward winning under Ax (regret 0.20 vs 1.19), s.o. wins under RF (Ax-primary)

Negative control:

1. null control clean across `10` runs over `11` sprint slots (backend-invariant)
2. Sprint 26 intentionally did not rerun null control
3. latest clean reruns stay within the `2%` tolerance

## Current Conclusion

The project is a **trustworthy automated research harness with a characterized causal boundary and clean backend separation**.

What is now established:

1. causal guidance clearly helps in the demand-response family under the trusted Ax path (3/3 variants, all statistically significant)
2. four of seven benchmark rows are backend-invariant; two are Ax-primary (winner changes by backend); one is backend-sensitive in magnitude
3. RF fallback is a secondary family-level regression signal, not a drop-in substitute for Ax-primary baselines
4. the benchmark contract is clean enough to evaluate optimizer-core changes

## Sprint 28 Exit Condition

Sprint 28 is **complete**.  The scorecard answered all five questions:

1. backend-invariant: medium-noise, high-noise, confounded, null control
2. Ax-primary: base energy, dose-response
3. backend-sensitive: interaction (magnitude)
4. RF fallback role: family-level drift detection, not row-level certification
5. Sprint 29: return to optimizer-core work

## Practical Next Step

If resuming from here:

1. read the [Sprint 28 backend baseline scorecard](thoughts/shared/docs/sprint-28-backend-baseline-scorecard.md) for the full verdict
2. plan Sprint 29 -- the scorecard recommends returning to optimizer-core work
3. the benchmark suite is ready to evaluate optimizer changes with confidence
