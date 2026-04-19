# Causal Optimizer Handoff Document

**Date:** 2026-04-19
**Current sprint:** 35 planning (Open Bandit multi-action implementation) -- Sprint 34 complete
**Current state:** Sprint 34 complete (Open Bandit contract merged as PR #184, issue #182 closed). Sprint 33 verdict GENERALITY IS REAL BUT CONDITIONAL carries forward unchanged.
**Main repo status:** safe restart point is this doc + benchmark state file + Sprint 33 generalization scorecard + Sprint 34 Open Bandit contract

## What The Next Agent Needs To Know

Sprint 34 is complete. It delivered the first executable contract for logged
multi-action policy data via PR #184 (merged), closing issue #182. No code
was written in Sprint 34; the contract is the deliverable. The empirical
work that matters for a fresh agent is:

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
   CONDITIONAL**. That verdict carries forward unchanged; Sprint 34 did
   not rerun any benchmark.
6. Sprint 34 pinned the seven Open Bandit contract decisions: first slice
   (ZOZOTOWN Men / uniform-random logger), new `DomainAdapter` subclass,
   SNIPW-primary OPE stack, verdict at B80, five support gates, OBP as
   an optional extra, and the three-issue Sprint 35 implementation shape.
7. Sprint 35 is the active implementation lane. It opens the adapter,
   OPE stack + gates, and first Men/Random benchmark report as three
   ordered issues (A, B, C) per Section 10 of the contract.

All Sprint 29 synthetic benchmark results carry forward unchanged.

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
2. null control: 11 clean runs across 12 sprint slots (through S29, S26 did not rerun); Hillstrom and Criteo also passed their null controls

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

## What Sprint 35 Should Do

Sprint 34 is complete. Sprint 35 is the active implementation lane, scoped
by Section 10 of the Open Bandit contract ([sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md)):

1. Open three implementation issues: Issue A (adapter), Issue B (OPE stack + gates), Issue C (first Men/Random benchmark report).
2. Run the Issue A smoke test first; it must confirm the OBD `action_prob` schema (conditional vs joint) and document the chosen three-to-five context features before Issues B and C can be finalized.
3. Hold every PR to the contract's Section 7 gates, now specified in relative terms (5% relative null-control band, 10% relative propensity sanity band, ESS floor, zero-support fraction, DR/SNIPW cross-check).
4. Do not subclass `MarketingLogAdapter`. The contract requires a new `DomainAdapter` subclass for the multi-action path.
5. Do not expand the first-run scope beyond the Men/Random slice, SNIPW-primary OPE, and the 6-to-8 variable item-scoring parameterization.
6. Do not reopen Hillstrom or Criteo as the main lane.

## Sprint 34 Contract Decisions (Merged)

The Sprint 34 Open Bandit contract merged as PR #184, closing issue [#182](https://github.com/datablogin/causal-optimizer/issues/182). See the full document at [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md). Summary of the contract decisions:

1. **First slice:** ZOZOTOWN Men campaign, uniform-random logger (~453K rows, 34 actions, 3 positions, binary click reward).
2. **Adapter:** a new `DomainAdapter` subclass (not a subclass of `MarketingLogAdapter`). Parameterizes an item-scoring policy in a 6-to-8 variable search space (softmax temperature, exploration epsilon, a small set of context-feature weights, a position-handling flag).
3. **OPE stack:** SNIPW primary, DM and DR secondary, DRos deferred to Sprint 35+.
4. **Objective:** maximize SNIPW-estimated CTR (`policy_value`). No revenue, no cost column, no multi-objective.
5. **Support gates:** null control (5% relative band above permuted baseline mean), ESS floor (`max(1000, n_rows/100)`), zero-support fraction `<= 10%`, propensity-mean sanity band (10% relative to schema-dependent target), and DR/SNIPW cross-check within 25% relative.
6. **OBP dependency:** accepted as an optional extra. OBP powers the data loader and estimators; OBP types are hidden behind the adapter boundary and the adapter fails fast if OBP is missing.
7. **Sprint 35 shape:** three ordered issues -- adapter, OPE stack + gates, and first Men/Random benchmark report.

## Files To Read First

1. [07-benchmark-state.md](../plans/07-benchmark-state.md)
2. [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md)
3. [sprint-33-generalization-scorecard.md](sprint-33-generalization-scorecard.md)
4. [sprint-33-criteo-benchmark-report.md](sprint-33-criteo-benchmark-report.md)
5. [sprint-31-hillstrom-benchmark-report.md](sprint-31-hillstrom-benchmark-report.md)
6. [sprint-31-hillstrom-lessons-learned.md](sprint-31-hillstrom-lessons-learned.md)
7. [sprint-31-open-bandit-access-and-gap-audit.md](sprint-31-open-bandit-access-and-gap-audit.md)
8. [sprint-30-reality-and-generalization-scorecard.md](sprint-30-reality-and-generalization-scorecard.md)
9. [sprint-30-general-causal-portability-brief.md](sprint-30-general-causal-portability-brief.md)
10. [24-sprint-34-recommendation.md](../plans/24-sprint-34-recommendation.md)

## Immediate Instructions For The Next Agent

Sprint 34 is complete. Sprint 35 is the active lane:

1. Read the [Sprint 34 Open Bandit contract](sprint-34-open-bandit-contract.md) first. It pins the Men/Random slice, the adapter interface, the SNIPW-primary OPE stack, the Section 7 support gates (all specified in relative terms), and the OBP-as-optional-extra dependency decision.
2. Open the three Sprint 35 implementation issues described in Section 10 of that contract (adapter, OPE stack + gates, first benchmark report) as ordered, with partial overlap permitted between Issues A and B.
3. Do not subclass `MarketingLogAdapter` or silently reshape it to multi-action.
4. Do not expand scope to the Women / All campaigns, BTS logger, or slate-level OPE in the first implementation sprint.
5. Do not reopen Hillstrom or Criteo as the main lane.
6. Do not claim `causal` beats `random` on real data; ERCOT has not closed that gap.

## One-Line Situation Summary

Sprint 33 closed with verdict **GENERALITY IS REAL BUT CONDITIONAL** (ERCOT COAST p=0.008 certified, Hillstrom pooled slice certified surrogate-only under RF, Criteo near-parity under Ax/BoTorch after the heterogeneous follow-up). Sprint 34 merged the Open Bandit contract as PR #184, pinning Men/Random as the first slice and OBP as an optional extra for the OPE stack. Sprint 35 opens the three implementation issues (adapter, OPE stack + gates, first benchmark report) against that contract.
