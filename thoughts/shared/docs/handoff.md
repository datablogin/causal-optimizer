# Causal Optimizer Handoff Document

**Date:** 2026-04-21
**Current sprint:** 36 planning (multi-action prior graph for Open Bandit) -- Sprint 35 complete
**Current state:** Sprint 35 complete (Open Bandit adapter, OPE stack, bridge, and first Men/Random benchmark report all merged; issues #185, #186, #187, #190 closed; PRs #188, #189, #191, #192 merged). Sprint 33 verdict GENERALITY IS REAL BUT CONDITIONAL carries forward unchanged.
**Main repo status:** safe restart point is this doc + benchmark state file + Sprint 33 generalization scorecard + Sprint 34 Open Bandit contract + Sprint 35 Open Bandit benchmark report

## What The Next Agent Needs To Know

Sprint 35 is complete. The first Open Bandit Men/Random benchmark ran
end-to-end on the full 452,949-row slice under Ax/BoTorch primary with all
five Sprint 34 Section 7 support gates green. Both `causal` and
`surrogate_only` beat `random` at certified significance (`p = 0.0002`,
two-sided MWU, 10/10 seeds, every budget) but `causal == surrogate_only` as
an exact bit-identical tie on every seed at every budget. The tie is
mechanical: `BanditLogAdapter.get_prior_graph() -> None` per Sprint 34
contract Section 4e, so the `causal` path has no extra information over
`surrogate_only` on this slice.

Do **not** open new Sprint 35 issues. The Sprint 35 lane is closed. The
empirical work that matters for a fresh agent is:

1. Sprint 30 produced the first real-world causal vs `surrogate_only`
   differentiation on ERCOT (COAST p=0.008, NORTH_C p=0.059; 5 seeds;
   Ax-primary). `causal` still does not beat `random`.
2. Sprint 31 ran Hillstrom as the first non-energy real-data benchmark
   under RF fallback. The pooled slice returned a certified
   `surrogate_only` advantage at B20/B40/B80 (p=0.017 / 0.002 / 0.019).
   The primary slice was trending s.o. at B20, certified s.o. at B40,
   near-parity at B80 with a bimodal causal tail. Causal beat `random`
   at primary B80 (p=0.0004).
3. Sprint 32 merged the Criteo contract. Sprint 33 executed it.
4. Sprint 33 Criteo Run 1 (degenerate 2-variable surface, 10 seeds,
   Ax/BoTorch) returned exact-tie near-parity on all three budgets.
   The mandatory Run 2 (synthesized f0-tertile segments) also returned
   near-parity on all three budgets. Combined Criteo verdict: near-parity.
5. The Sprint 33 closure verdict is **GENERALITY IS REAL BUT
   CONDITIONAL**. That verdict carries forward unchanged; Sprint 34 and
   Sprint 35 did not rerun any synthetic, energy, or binary-marketing
   benchmark.
6. Sprint 34 pinned the seven Open Bandit contract decisions (see
   [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md)):
   Men/Random first slice, new `DomainAdapter` subclass, SNIPW-primary
   OPE stack, verdict at B80, five support gates, OBP as an optional
   extra, and the three-issue Sprint 35 implementation shape.
7. Sprint 35 executed that contract in four merged PRs:
   - PR [#189](https://github.com/datablogin/causal-optimizer/pull/189) (issue [#185](https://github.com/datablogin/causal-optimizer/issues/185)) -- `BanditLogAdapter` and Men/Random smoke test
   - PR [#188](https://github.com/datablogin/causal-optimizer/pull/188) (issue [#186](https://github.com/datablogin/causal-optimizer/issues/186)) -- SNIPW / DM / DR OPE stack and Section 7 gate logic
   - PR [#191](https://github.com/datablogin/causal-optimizer/pull/191) (issue [#190](https://github.com/datablogin/causal-optimizer/issues/190)) -- three bridge seams between the adapter and the OPE stack (position normalization, conditional propensity schema, OBP version provenance)
   - PR [#192](https://github.com/datablogin/causal-optimizer/pull/192) (issue [#187](https://github.com/datablogin/causal-optimizer/issues/187)) -- first Men/Random benchmark report

## Current GitHub Status

### Sprint 28 Done

1. [PR #149](https://github.com/datablogin/causal-optimizer/pull/149) merged -- optimizer-path provenance
2. [PR #150](https://github.com/datablogin/causal-optimizer/pull/150) merged -- Ax/BoTorch seven-benchmark regression gate passed
3. [PR #151](https://github.com/datablogin/causal-optimizer/pull/151) merged -- backend baseline scorecard, verdict **AX PRIMARY, RF SECONDARY**

### Sprint 29 Done

1. [PR #155](https://github.com/datablogin/causal-optimizer/pull/155) merged -- trajectory diagnosis
2. [PR #158](https://github.com/datablogin/causal-optimizer/pull/158) merged -- dose-response 10-seed certification (p=0.002, 9/10 wins)
3. [PR #159](https://github.com/datablogin/causal-optimizer/pull/159) merged -- interaction ablation
4. [PR #160](https://github.com/datablogin/causal-optimizer/pull/160) merged -- production default `causal_exploration_weight` 0.3 to 0.0
5. [PR #161](https://github.com/datablogin/causal-optimizer/pull/161) merged -- Ax-primary regression gate, verdict **GENERALITY IMPROVED**

### Sprint 30 Done

1. [PR #165](https://github.com/datablogin/causal-optimizer/pull/165) merged -- portability brief
2. [PR #166](https://github.com/datablogin/causal-optimizer/pull/166) merged -- ERCOT reality report (COAST p=0.008, NORTH_C p=0.059)
3. [PR #167](https://github.com/datablogin/causal-optimizer/pull/167) merged -- Hillstrom benchmark contract
4. [PR #169](https://github.com/datablogin/causal-optimizer/pull/169) merged -- Hillstrom benchmark harness

### Sprint 31 Done

1. [PR #176](https://github.com/datablogin/causal-optimizer/pull/176) merged -- Hillstrom benchmark report, verdict **SURROGATE-ONLY ADVANTAGE** on pooled slice, near-parity with causal B80 tail on primary

### Sprint 32 Done

1. [PR #178](https://github.com/datablogin/causal-optimizer/pull/178) merged -- Criteo uplift benchmark contract

### Sprint 33 Done

1. [PR #174](https://github.com/datablogin/causal-optimizer/pull/174) merged -- Criteo access and adapter-gap audit (Sprint 31/32 handoff)
2. [PR #180](https://github.com/datablogin/causal-optimizer/pull/180) merged -- Criteo benchmark Run 1 + Run 2 report, combined verdict **NEAR-PARITY**
3. [PR #183](https://github.com/datablogin/causal-optimizer/pull/183) merged -- Sprint 33 generalization scorecard and restart-doc sync

### Sprint 34 Done

1. [PR #184](https://github.com/datablogin/causal-optimizer/pull/184) merged -- Open Bandit contract / multi-action architecture brief (closes #182)

### Sprint 35 Done

1. [PR #189](https://github.com/datablogin/causal-optimizer/pull/189) merged -- Sprint 35.A `BanditLogAdapter` and Men/Random smoke test (closes [#185](https://github.com/datablogin/causal-optimizer/issues/185)); smoke test confirms the OBD `action_prob` schema is conditional `P(item | position) = 1 / n_items`
2. [PR #188](https://github.com/datablogin/causal-optimizer/pull/188) merged -- Sprint 35.B SNIPW / DM / DR OPE stack and Section 7 gate logic (closes [#186](https://github.com/datablogin/causal-optimizer/issues/186))
3. [PR #191](https://github.com/datablogin/causal-optimizer/pull/191) merged -- Sprint 35 bridge between adapter and OPE path (closes [#190](https://github.com/datablogin/causal-optimizer/issues/190)): position normalization helper, conditional propensity schema pin, OBP version provenance helper, `BanditLogAdapter.to_bandit_feedback()`
4. [PR #192](https://github.com/datablogin/causal-optimizer/pull/192) merged -- Sprint 35.C first Men/Random Open Bandit benchmark report (closes [#187](https://github.com/datablogin/causal-optimizer/issues/187)); verdict **exact tie between `causal` and `surrogate_only`, both CERTIFIED over `random`**; all five Section 7 gates PASS

## Current Best Evidence

### Synthetic Ax-primary Benchmarks (from Sprint 29, unchanged)

Certified causal wins:

1. medium-noise B80: causal mean/std `1.19 / 1.52`, wins `10/10`, two-sided `p=0.002`
2. high-noise B80: causal mean/std `1.08 / 1.72`, wins `10/10`, two-sided `p=0.001`
3. dose-response B80: causal mean/std `0.22 / 0.03`, wins `9/10`, two-sided `p=0.003`

Trending:

1. base B80: causal mean/std `1.01 / 1.10`, catastrophic seeds `0/10`, wins `7/10`, two-sided `p=0.112`

Near-parity:

1. interaction policy B80: causal `1.90` vs s.o. `2.18`, `p=0.225` (flipped from s.o. advantage after Sprint 29)

Remaining boundary rows:

1. confounded demand-response: all strategies misled
2. null control: 11 clean runs across 12 sprint slots (through S29, S26 did not rerun); Hillstrom, Criteo, and Open Bandit Men/Random also passed their null controls

### Real Energy Benchmarks (Sprint 30)

1. ERCOT COAST B80: causal certified > s.o. (MAE 104.88 vs 105.72, two-sided `p=0.008`, 5/5 wins); causal vs random not significant (p=0.690)
2. ERCOT NORTH_C B80: causal trending > s.o. (MAE 132.48 vs 132.98, two-sided `p=0.059`, 4/5 wins); causal vs random not significant (p=0.402)
3. Both results at 5 seeds. 10-seed rerun remains on the backlog but was not executed.

### Real Non-Energy Benchmarks (Sprint 31 + Sprint 33)

Hillstrom (RF fallback, 10 seeds, 3-variable active search space):

1. primary B20: trending s.o. (p=0.060, 8/10 wins s.o.)
2. primary B40: certified s.o. (p=0.0001, 10/10 wins s.o.)
3. primary B80: near-parity with bimodal causal tail (p=0.817, causal mean higher but driven by 3 seeds)
4. primary B80 causal vs random: certified causal > random (p=0.0004, 9/10 wins)
5. pooled B20: certified s.o. (p=0.017)
6. pooled B40: certified s.o. (p=0.002)
7. pooled B80: certified s.o. (p=0.019)
8. null-control pass had the caveat that policy values above the simple baseline can arise even after permuting outcomes

Criteo (Ax/BoTorch, 10 seeds, 1M-row subsample of 13,979,592-row dataset):

1. Run 1 (degenerate 2-variable surface) B20/B40/B80: exact-tie near-parity (p=1.000 on all three)
2. Run 2 (synthesized f0-tertile segments) B20/B40/B80: near-parity (p=0.168 / 1.000 / 0.368)
3. Propensity gate passed (max deviation 0.79pp vs 2pp threshold)
4. IPS stack stable at 85:15 treatment imbalance (ESS ~849,982 on optimized strategies, no zero-support events)
5. Null control passed within 5% band on all cells
6. Combined Criteo verdict per Sprint 32 contract: **near-parity**

### Real Multi-Action Benchmark (Sprint 35)

Open Bandit Men/Random (Ax/BoTorch, 10 seeds, full 452,949-row slice, 34
actions, 3 positions; see
[sprint-35-open-bandit-benchmark-report.md](sprint-35-open-bandit-benchmark-report.md)):

1. `causal` vs `surrogate_only` B20/B40/B80: exact tie on every seed (p=1.000 at every budget). The tie is mechanical: `BanditLogAdapter.get_prior_graph() -> None`, so `causal` reduces to `surrogate_only`.
2. `causal` vs `random` B20/B40/B80: certified (p=0.0002, 10/10 wins at every budget); absolute lift at B80 is 0.000751 (~14% relative over random, ~20% relative over logged-policy μ=0.005124).
3. `surrogate_only` vs `random` B20/B40/B80: certified (p=0.0002, 10/10 wins at every budget), identical to the causal-vs-random row.
4. Propensity schema confirmed (Section 7d) as conditional `P(item | position) = 1 / 34`; empirical mean 0.029411764706, relative deviation ~2e-15 vs target, within the 10% relative band.
5. Section 7 gates all PASS: null-control (max ratio 1.0154 vs 1.05 band), ESS floor (median 49,867 vs floor 4,530), zero-support (0% best-of-seed), propensity sanity (above), DR/SNIPW cross-check (max divergence 0.48% vs 25% tolerance).
6. Verdict per Sprint 34 contract Section 12: the "least valuable but acceptable" first-run outcome (clean diagnostics, no `causal` vs `surrogate_only` separation). Carries the same weight as the Criteo near-parity row.

Important: the exact tie is **not** evidence that causal guidance is inert
on multi-action data in general. It is the expected null result under a
null prior graph. The next ingredient required to answer the causal vs
surrogate-only question on Open Bandit is a bandit-log-compatible
multi-action prior graph, not more plumbing.

## What Sprint 36 Should Do

Sprint 35 is complete. Planning for Sprint 36 is **pending in companion
PR #195** (recommendation and prompt docs owned by a separate track) and
is **not yet on `main`** as of this PR. Once PR #195 merges, Sprint 36
planning will live in `thoughts/shared/plans/26-sprint-36-recommendation.md`
and `thoughts/shared/prompts/sprint-36-open-bandit-prior-graph.md`; until
then, treat those paths as forthcoming rather than available. The
high-level framing the next agent should start from:

1. The Open Bandit lane is now real: adapter, OPE stack, Section 7 gates, and a published Men/Random benchmark all exist on `main`.
2. The causal-vs-surrogate differentiation question on multi-action data is still unanswered. The Sprint 35 tie is mechanical, not empirical.
3. The next missing ingredient is causal structure on Open Bandit, not more plumbing. Authoring a defensible multi-action prior graph (or a principled way to discover one for bandit logs) is the most direct way to answer the question Sprint 35 could not.
4. Hillstrom and Criteo should not be reopened as the main lane.
5. The ERCOT 10-seed rerun remains on the backlog but is not the Sprint 36 critical path.

## Sprint 34 Contract Decisions (Merged)

The Sprint 34 Open Bandit contract merged as PR #184, closing issue [#182](https://github.com/datablogin/causal-optimizer/issues/182). See the full document at [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md). Summary of the contract decisions:

1. **First slice:** ZOZOTOWN Men campaign, uniform-random logger (~453K rows, 34 actions, 3 positions, binary click reward). Sprint 35.C confirmed 452,949 rows on the full slice.
2. **Adapter:** a new `DomainAdapter` subclass (not a subclass of `MarketingLogAdapter`). Parameterizes an item-scoring policy in a 6-to-8 variable search space (softmax temperature, exploration epsilon, a small set of context-feature weights, a position-handling flag). Sprint 35.A landed this as `BanditLogAdapter` with six variables.
3. **OPE stack:** SNIPW primary, DM and DR secondary, DRos deferred to Sprint 36+.
4. **Objective:** maximize SNIPW-estimated CTR (`policy_value`). No revenue, no cost column, no multi-objective.
5. **Support gates:** null control (5% relative band above permuted baseline mean), ESS floor (`max(1000, n_rows/100)`), zero-support fraction `<= 10%`, propensity-mean sanity band (10% relative to schema-dependent target), and DR/SNIPW cross-check within 25% relative. All five passed in Sprint 35.C.
6. **OBP dependency:** accepted as an optional extra. OBP powers the data loader and estimators; OBP types are hidden behind the adapter boundary and the adapter fails fast if OBP is missing.
7. **Sprint 35 shape:** three ordered issues -- adapter, OPE stack + gates, and first Men/Random benchmark report. Delivered plus one planned bridge PR (#191) for the seams between tracks A and B.

## Files To Read First

1. [07-benchmark-state.md](../plans/07-benchmark-state.md)
2. [sprint-35-open-bandit-benchmark-report.md](sprint-35-open-bandit-benchmark-report.md)
3. [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md)
4. [sprint-33-generalization-scorecard.md](sprint-33-generalization-scorecard.md)
5. [sprint-33-criteo-benchmark-report.md](sprint-33-criteo-benchmark-report.md)
6. [sprint-31-hillstrom-benchmark-report.md](sprint-31-hillstrom-benchmark-report.md)
7. [sprint-31-hillstrom-lessons-learned.md](sprint-31-hillstrom-lessons-learned.md)
8. [sprint-31-open-bandit-access-and-gap-audit.md](sprint-31-open-bandit-access-and-gap-audit.md)
9. [sprint-30-reality-and-generalization-scorecard.md](sprint-30-reality-and-generalization-scorecard.md)
10. [sprint-30-general-causal-portability-brief.md](sprint-30-general-causal-portability-brief.md)
11. [24-sprint-34-recommendation.md](../plans/24-sprint-34-recommendation.md)

## Immediate Instructions For The Next Agent

Sprint 35 is complete. Do not open new Sprint 35 issues. The active work
is Sprint 36 planning:

1. Read the [Sprint 35 Open Bandit benchmark report](sprint-35-open-bandit-benchmark-report.md) first -- it is the new evidence since the Sprint 33 scorecard and pins the exact shape of the Open Bandit result (clean diagnostics, exact `causal == surrogate_only` tie explained by `get_prior_graph() -> None`, both beat `random` at certified significance).
2. Read the [Sprint 34 Open Bandit contract](sprint-34-open-bandit-contract.md) for the Men/Random slice, adapter interface, SNIPW-primary OPE stack, Section 7 support gates, and OBP-as-optional-extra dependency decision.
3. Sprint 36 planning itself is **pending in companion PR #195** (separate prompt and recommendation docs owned by that track) and is not yet on `main`; once PR #195 merges, those docs will be at `thoughts/shared/plans/26-sprint-36-recommendation.md` and `thoughts/shared/prompts/sprint-36-open-bandit-prior-graph.md`. Let that track own it. Do not start new Sprint 36 implementation scope from this doc.
4. Do not describe the Sprint 35 tie as evidence that causal guidance is inert on multi-action data in general -- the tie is mechanical (null prior graph), not empirical.
5. Do not reopen Hillstrom or Criteo as the main lane.
6. Do not claim `causal` beats `random` on real data as a general statement; ERCOT has not closed that gap. Note that on Open Bandit Men/Random both optimized strategies do beat `random` at certified significance, but both optimized strategies are bit-identical on that slice, so it is not a `causal`-specific claim.
7. Treat multi-action prior graph authoring (or principled discovery on bandit logs) as the most direct way to turn the Sprint 35 clean row into a real causal-vs-surrogate evidence row.

## One-Line Situation Summary

Sprint 33 closed with verdict **GENERALITY IS REAL BUT CONDITIONAL** (ERCOT COAST p=0.008 certified, Hillstrom pooled slice certified surrogate-only under RF, Criteo near-parity under Ax/BoTorch after the heterogeneous follow-up). Sprint 34 merged the Open Bandit contract as PR #184. Sprint 35 executed that contract in four merged PRs (#188, #189, #191, #192) and produced the first Men/Random benchmark on the full 452,949-row slice under Ax/BoTorch with all five Section 7 gates PASS: `causal` and `surrogate_only` tie bit-identically on every seed (exact tie is mechanical, driven by `BanditLogAdapter.get_prior_graph() -> None`) and both beat `random` at certified significance (p=0.0002, 10/10 wins, every budget); the Open Bandit lane is now real, but causal-vs-surrogate differentiation on multi-action data remains unanswered and the next missing ingredient is causal structure, not plumbing.
