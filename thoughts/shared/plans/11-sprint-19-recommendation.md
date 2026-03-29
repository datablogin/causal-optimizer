# Sprint 19 Recommendation

Updated: 2026-03-29

## Sprint Theme

**Causal Differentiation Under Controls**

Sprint 18 improved the trustworthiness of our research process:

1. the time-series profiler can catch timestamp and calendar mistakes before they pollute evidence
2. the repaired counterfactual benchmark is now a real positive control
3. the null-signal benchmark gives us a real negative control
4. the combined scorecard can now tell us whether we are learning signal or just fitting noise

That leaves Sprint 19 with a narrower, better-defined problem:

1. `causal` still does not beat `surrogate_only`
2. we now have enough controls to test optimizer-core changes honestly
3. the next sprint should target differentiation, not more generic infrastructure

## Goal

Make causal guidance materially useful without breaking the new trust controls.

Sprint 19 should try to answer:

1. can we make `causal` outperform `surrogate_only` on at least one meaningful positive-control benchmark?
2. can we do that while the null-signal control still passes?
3. can we improve skip behavior without reintroducing noisy or brittle wins?

## Why This Sprint Now

The project is in a better position than it was two sprints ago:

1. the real ERCOT benchmark suite is stable and auditable
2. the counterfactual benchmark is now informative
3. the null-signal benchmark can catch false discovery
4. the profiler reduces a major source of time-series false signal

That means we no longer need to guess whether a change helped.

We can now make optimizer-core changes and judge them against:

1. a real-task suite
2. a positive control
3. a negative control
4. explicit skip-calibration metrics

## Sprint 19 Deliverables

### 1. Earlier / Softer Causal Influence In The Optimizer

Target the main observed gap from Sprint 18: causal guidance arrives too late and may be too restrictive when it does arrive.

Recommended focus:

1. introduce causal influence earlier than the current optimization phase
2. prefer soft causal weighting over hard candidate elimination where possible
3. preserve existing behavior behind a flag or clearly scoped config so we can compare old vs new directly

Candidate approaches:

1. causal-weighted exploration during the early search phase
2. earlier transition out of pure exploration
3. soft ranking bonus for causal-parent-aligned variables instead of hard focus-only search
4. targeted candidate generation that expands around causal variables without collapsing the search diversity too early

Success criteria:

1. `causal` becomes behaviorally distinct from `surrogate_only` in logs and results
2. the change improves at least one positive-control benchmark at fixed budget
3. the change does not create a promotable win on the null-signal benchmark

### 2. Harder Counterfactual Variants

The current repaired benchmark is valid, but still easy enough for `surrogate_only` to solve well without causal knowledge.

Sprint 19 should add harder variants that create a more realistic chance for causal advantage.

Recommended variants:

1. a high-noise variant with 10+ irrelevant dimensions
2. a confounded variant where naive surrogate search is more likely to chase the wrong policy surface
3. an interaction-heavy variant where the effect depends on combinations of variables rather than one obvious threshold family

Rules:

1. keep the oracle non-degenerate
2. document exactly what structure causal knowledge is supposed to help with
3. keep variants reproducible and cheap enough for routine reruns

Success criteria:

1. at least one variant remains a valid positive control with non-trivial oracle behavior
2. random remains clearly worse than learned strategies
3. the benchmark meaningfully separates optimizer strategies instead of collapsing into ties

### 3. Skip Calibration Under Controls

Sprint 17 showed that skip logic is measurable but not yet trustworthy.

Sprint 19 should move skip behavior from an observed weakness to a controlled target.

Recommended scope:

1. rerun the benchmark suite with auditing enabled on engine-backed strategies
2. measure false-skip rate, audit coverage, and skip-confidence behavior across:
   - real ERCOT suite
   - repaired counterfactual benchmark
   - null-signal benchmark
3. make one narrow calibration change only if the measured failure mode is clear

Calibration candidates:

1. stricter confidence threshold before skipping
2. burn-in period before skip logic activates
3. simple calibration layer for predicted improvement probabilities

Success criteria:

1. false-skip rate improves materially from Sprint 17 levels
2. runtime remains acceptable for the benchmark suite
3. no benchmark starts failing because skip logic becomes too aggressive or too conservative

### 4. Sprint 19 Differentiation Scorecard

End the sprint with a combined report that answers a narrower question than Sprint 18:

1. did causal guidance become more useful?
2. on which benchmark class did it help?
3. did the null control stay clean?
4. what did skip auditing reveal after the optimizer-core changes?

The report should not claim general causal advantage unless the evidence supports it.

## Proposed Workstreams

### Workstream A: Optimizer-Core Causal Differentiation

Scope:

1. implement one primary causal-core change
2. add focused tests showing the new behavior differs from `surrogate_only`
3. keep comparison against the old path possible

Owner profile:

1. optimizer internals
2. controlled experimentation
3. comfort with search heuristics and failure analysis

### Workstream B: Harder Positive Controls

Scope:

1. add at least one harder counterfactual variant
2. validate oracle behavior and benchmark discriminativeness
3. document the intended causal advantage mechanism

Owner profile:

1. benchmark design
2. causal reasoning
3. skepticism about misleading benchmarks

### Workstream C: Skip Calibration Measurement

Scope:

1. run audited benchmarks
2. summarize false-skip behavior across benchmark classes
3. propose or implement one narrow calibration change if justified by the evidence

Owner profile:

1. measurement discipline
2. experimentation hygiene
3. runtime-aware evaluation

### Workstream D: Final Sprint Report

Scope:

1. combine positive-control, negative-control, real-task, and skip-audit evidence
2. state clearly whether Sprint 19 improved causal differentiation
3. recommend whether Sprint 20 should continue optimizer-core work or return to benchmark design

## Recommended Execution Order

### Phase 1: Positive-Control Pressure First

1. add harder counterfactual variants
2. validate that they remain non-degenerate and discriminating

Reason:

1. if we want causal differentiation, we need a benchmark where causal knowledge has room to matter

### Phase 2: One Narrow Optimizer-Core Change

3. implement one causal-core change
4. rerun the positive control and null control before broadening scope

Reason:

1. one well-measured change is more useful than several overlapping tweaks
2. the null control should guard against accidental noise-chasing

### Phase 3: Skip Calibration Evaluation

5. enable audited skip measurement across the suite
6. adjust calibration only if the measured failure mode is clear

Reason:

1. skip logic should be tuned against evidence, not intuition

### Phase 4: Final Synthesis

7. produce a Sprint 19 differentiation scorecard

Reason:

1. the sprint should end with a decision about causal usefulness, not just more runs

## Acceptance Criteria

Sprint 19 is successful if all of the following are true:

1. at least one harder counterfactual variant is a valid positive control with a non-trivial oracle and clear separation from random
2. at least one optimizer-core change makes `causal` measurably different from `surrogate_only`
3. the null-signal benchmark still does not produce a promotable false win
4. audited skip metrics are reported and interpreted, not left as raw telemetry
5. the final report can state clearly whether causal differentiation improved and under what conditions

## What Would Count As Real Progress

I would count Sprint 19 as meaningful progress if it shows:

1. `causal` wins somewhere it plausibly should win
2. that win survives the null control staying clean
3. the explanation for the win is benchmark-credible, not post-hoc storytelling
4. skip behavior becomes more trustworthy rather than merely more complicated

## What Not To Do

1. do not tune specifically for ERCOT to manufacture a narrow win
2. do not weaken the null-signal benchmark to make optimizer changes look better
3. do not stack multiple optimizer-core changes before we can attribute effects
4. do not claim general causal advantage from one benchmark family alone
