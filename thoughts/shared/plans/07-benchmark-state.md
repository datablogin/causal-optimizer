# Benchmark State

Updated: 2026-04-04 (Sprint 21 attribution complete, Sprint 22 framed)

## Purpose

This file is the restart point for the benchmark and discovery-trust work in `causal-optimizer`.

Use it when:

1. starting a new chat
2. handing context to another reviewer or agent
3. checking what is merged vs still pending
4. recovering after an interruption or crash

## Current Position

The project is now strongest as a **research and attribution system**.

What is true today:

1. the benchmark stack is disciplined and reusable
2. positive controls, negative controls, provenance, and locked A/B reruns are all in place
3. the project can demonstrate causal advantage on controlled benchmark tasks
4. the project has **not** yet demonstrated a reliable real-world causal advantage on the ERCOT forecasting benchmarks
5. Sprint 21 concluded that Sprint 20's observed post-merge improvement was **not attributable** to balanced Ax reranking

## Current Goal

The immediate goal is no longer to add another optimizer idea quickly.

It is to:

1. act on the Sprint 21 attribution result
2. revert or disable balanced reranking back to alignment-only
3. confirm that the reverted path still preserves the strong positive-control behavior
4. only then decide whether the system is ready for broader benchmark expansion

## Canonical Docs

Core references:

1. `README.md`
2. `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`
3. `thoughts/shared/docs/ercot-coast-houston-2022-2024-benchmark-report.md`
4. `thoughts/shared/docs/sprint-18-discovery-trust-scorecard.md`
5. `thoughts/shared/docs/sprint-19-differentiation-scorecard.md`
6. `thoughts/shared/docs/sprint-20-differentiation-scorecard.md`
7. `thoughts/shared/docs/sprint-20-post-ax-rerun-report.md`
8. `thoughts/shared/docs/sprint-21-controlled-ab-rerun-report.md`
9. `thoughts/shared/docs/sprint-21-attribution-scorecard.md`
10. `thoughts/shared/prompts/sprint-22-alignment-only-confirmation.md`

## Benchmark / Evidence Rules

These should not drift:

1. held-out evaluation must remain genuinely held out
2. time-series semantics must be handled explicitly when relevant
3. null-signal controls must stay clean before we trust optimizer wins
4. provenance must be captured whenever attribution claims matter
5. benchmark reports must separate:
   - observed performance improvement
   - attribution to a code change
6. attractive optimizer stories do not count if they fail the locked comparison

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
4. but attribution remained unresolved because the comparison was not locked tightly enough
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

## Current Best Evidence

### Real Predictive Benchmarks

ERCOT NORTH_C and COAST both show:

1. `random` marginally better than engine-based strategies
2. `causal` effectively identical to `surrogate_only`
3. all strategies converging to `ridge`
4. highly stable results across seeds

Interpretation:

1. the real-task harness is functioning correctly
2. but the project still lacks a convincing real-world causal advantage on these forecasting tasks

### Positive Controls

Locked A/B, base counterfactual:

1. balanced B80 causal mean/std: `3.57 / 5.69`
2. alignment-only B80 causal mean/std: `0.52 / 0.16`
3. balanced B80 wins vs `surrogate_only`: `8/10`
4. alignment-only B80 wins vs `surrogate_only`: `10/10`

Locked A/B, high-noise:

1. balanced B80 causal mean/std: `2.58 / 4.29`
2. alignment-only B80 causal mean/std: `3.27 / 4.21`
3. differences are small relative to variance
4. no clean attribution win for balanced reranking

### Negative Control

Locked A/B null control:

1. balanced and alignment-only artifacts are row-for-row identical
2. max strategy difference remains `0.18%`
3. null control remains safely below the `2%` threshold

## Current Conclusion

The project is succeeding at becoming a **trustworthy automated research harness**.

It is not yet succeeding at becoming a **reliably winning causal researcher on real predictive tasks**.

That distinction matters. It means:

1. the system is increasingly good at rejecting false wins
2. the benchmark program is producing useful evidence
3. more optimizer work should now be guided by attribution discipline, not just directional benchmark gains

## Sprint 22

Primary prompt:

1. `thoughts/shared/prompts/sprint-22-alignment-only-confirmation.md`

Sprint 22 should:

1. revert or disable balanced reranking back to alignment-only
2. rerun base, high-noise, and null controls on the reverted path
3. confirm whether the stronger alignment-only behavior is stable in production code
4. only then decide whether to expand to new benchmark families or continue optimizer-core hardening

## Practical Next Step

If resuming from here:

1. start Sprint 22 from the alignment-only confirmation prompt
2. do not widen scope into new optimizer features first
3. keep provenance and null-control checks as hard gates
4. treat the Sprint 21 attribution result as the current project truth until new evidence overturns it
