# Sprint 31 Hillstrom Lessons Learned

**Date:** 2026-04-16
**Updated:** 2026-04-17 (Sprint 33 closure status note added to Sprint 32 Recommendation)
**Sprint:** 31
**Predecessors:**
- PR #169 -- Hillstrom benchmark harness
- PR #176 -- Hillstrom benchmark report
- PR #174 -- Criteo uplift access and adapter-gap audit

## Summary

Sprint 31 produced the first non-energy real-data benchmark result for the
project. That result was not a clean causal win. On Hillstrom, the
`surrogate_only` path beat `causal` at low and medium budgets on both slices,
and it held a certified advantage on the pooled slice at all three budgets.
At the same time, the benchmark was still valuable: it proved the engine can
run a real marketing benchmark end to end, it exposed a real boundary outside
ERCOT, and it gave the project a more honest generalization story.

The right conclusion is not "marketing failed." The right conclusion is that
**domain transfer is conditional, backend-sensitive, and easier to overclaim
than to prove**. Hillstrom was the first real test of that.

## What Hillstrom Established

1. **The first non-energy real-data lane is now real, not hypothetical.**
   The project no longer depends only on synthetic non-energy evidence or
   energy-backed real data. A real marketing benchmark now exists and has been
   run under the same per-seed, multi-budget discipline used elsewhere.

2. **Causal guidance does not automatically beat surrogate-only in a new
   domain.** The pooled Hillstrom slice showed a certified `surrogate_only`
   advantage at B20, B40, and B80. The primary slice showed a trending
   `surrogate_only` advantage at B20, a certified one at B40, and only
   near-parity at B80.

3. **The causal path still showed a real high-budget effect relative to random
   on the primary slice.** At primary B80, `causal` beat `random`
   significantly even though it did not beat `surrogate_only`. That means the
   causal path is not inert on Hillstrom; it is just not the best of the three
   paths under this benchmark configuration.

4. **Null-control interpretation must stay disciplined.** Hillstrom reinforced
   that policy values above a simple baseline can arise even after permuting
   outcomes. That does not invalidate the benchmark, but it does mean the
   project should keep separating "optimizer found a high-value policy under the
   estimator" from "clean treatment-effect evidence."

5. **Backend parity is unresolved.** Hillstrom ran on the RF fallback backend,
   while the strongest synthetic causal wins were Ax-primary. The project now
   has a legitimate cross-domain result, but not a backend-matched one.

## Core Lessons

### 1. Generality is not a slogan

Hillstrom is the first real reminder that a domain-agnostic research assistant
has to survive contact with new intervention data. The right bar is not
"adapter exists" or "harness runs." The right bar is whether the strategy
ordering still supports the causal thesis after the move.

### 2. Negative or mixed results are productive if they are specific

Hillstrom did not produce an ambiguous failure. It produced a specific one:

1. `surrogate_only` is stronger on the pooled slice
2. the primary slice is mostly flat, with a small causal B80 tail
3. the treatment effect on spend is weak enough that the no-treatment baseline
   remains hard to beat consistently

That is useful. It tells us where the current engine struggles and keeps the
project from mistaking portability of code for portability of advantage.

### 3. Narrow search spaces may blunt the value of graph focus

Hillstrom's active search space is only three variables wide. That is much
narrower than the energy rows where causal guidance had the clearest wins.
This does not prove that dimensionality is the whole story, but it is now a
plausible boundary condition: the graph may have more leverage when the search
problem contains more irrelevant or weakly relevant directions.

### 4. Marketing benchmarks need stronger IPS-first diagnostics

The Hillstrom null-control and tail behavior both point to the same practical
need: marketing benchmarks should make effective sample size, maximum weight,
weight variation, and support coverage first-class diagnostics rather than
afterthoughts. Criteo will make this even more important because of its 85:15
treatment imbalance.

### 5. We should separate "backend question" from "domain question"

Hillstrom leaves two distinct open questions:

1. Would the same benchmark behave differently under Ax/BoTorch?
2. Does the current causal approach transfer better on a larger, more
   feature-rich binary uplift dataset?

Those are not the same question. Hillstrom answered the first non-energy
question under RF fallback. Criteo is the cleaner next answer to the second.

## What We Should Not Do

1. Do not overreact by abandoning the generalization program because Hillstrom
   was not a causal win.
2. Do not overclaim by saying Hillstrom disproves cross-domain causal value in
   general.
3. Do not spend the next sprint only retuning Hillstrom without learning
   whether the pattern survives on a second marketing dataset.

## Decision

The project should treat Hillstrom as a **boundary-mapping result** and move
forward.

Recommended next move:

1. Publish this lessons-learned interpretation as the Sprint 31 bridge note.
2. Open the Criteo uplift line as the next marketing benchmark.
3. Use Criteo to test whether the Hillstrom outcome was mainly:
   - a weak-treatment / narrow-search-space artifact
   - an RF-backend artifact
   - or a broader sign that surrogate-only remains stronger on binary uplift
     policy problems

## Why Criteo Is The Right Next Dataset

Criteo is the right follow-on because it is:

1. still a binary-treatment marketing benchmark, so it keeps the intervention
   framing clean
2. much larger and more feature-rich than Hillstrom, so it is a stronger test
   of whether graph-guided focus helps in a noisier setting
3. compatible with the current `MarketingLogAdapter` via a wrapper rather than
   a new multi-action architecture project

Open Bandit remains important, but it is the wrong immediate follow-up. It
would entangle the marketing-generalization question with a multi-action OPE
stack change. Criteo keeps the next experiment difficult without changing the
problem class.

## Sprint 32 Recommendation

> **Status note (added during Sprint 33 closure, 2026-04-17):** Sprint 32
> executed this recommendation; the Criteo contract was merged as
> [PR #178](https://github.com/datablogin/causal-optimizer/pull/178) and
> Sprint 33 executed it in
> [PR #180](https://github.com/datablogin/causal-optimizer/pull/180).
> The verdict came back as near-parity. The recommendation below is
> preserved as the Sprint 31 lesson-learned snapshot.

Sprint 32 should open the Criteo line with a **benchmark contract sprint**,
not a full implementation sprint.

That contract should settle:

1. primary outcome: `visit`
2. secondary outcome: `conversion`
3. first-run scale: fixed-seed 1M-row subsample
4. wrapper contract over `MarketingLogAdapter`
5. IPS-variance diagnostics and null-control requirements
6. whether `min_propensity_clip` is fixed or tunable in the first run

## One-Line Takeaway

Hillstrom gave the project its first honest non-energy boundary result:
`surrogate_only` was stronger under the current RF-backed setup, but the
benchmark was still a success because it proved the generalization program is
now empirical, and it gave Criteo a clear job to do next.
