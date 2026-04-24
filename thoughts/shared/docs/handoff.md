# Causal Optimizer Handoff Document

**Date:** 2026-04-22
**Current sprint:** 38 planning (Option B / C / D from the Sprint 37 report) -- Sprint 37 complete
**Current state:** Sprint 37 complete (PR #198 merged; first preregistered Open Bandit prior graph + A1 minimal-focus engine flag; Men/Random rerun produced near-parity at B80 with `p = 0.7337`, all five Section 7 gates PASS, backend `ax_botorch` only). Sprint 36 (PR #195/#196) merged the docs-only preregistration + restart-doc sync. Sprint 35 (PRs #188, #189, #191, #192) is the predecessor empirical row. Sprint 33 verdict GENERALITY IS REAL BUT CONDITIONAL still carries forward unchanged.
**Main repo status:** safe restart point is this doc + benchmark state file + Sprint 33 generalization scorecard + Sprint 34 Open Bandit contract + Sprint 35 Open Bandit benchmark report + Sprint 36 recommendation + **Sprint 37 Open Bandit prior-graph rerun report**

## What The Next Agent Needs To Know

Sprint 37 is complete. The first preregistered Open Bandit prior graph
(seven nodes, six edges, every search variable directly parents
`policy_value`, no bidirected edges) is now wired into `BanditLogAdapter`
on `main`. A new `pomis_minimal_focus` flag on `ExperimentEngine`
(default `False`) opts a workload into the Sprint 37 Option A1
minimal-focus heuristic — the Open Bandit benchmark harness sets it to
`True` only on the `causal` arm; `surrogate_only` and `random` are
mechanically unchanged from Sprint 35. The Sprint 37 Men/Random rerun
under the same Sprint 35 contract (10 seeds, B20/B40/B80, full slice,
Ax/BoTorch primary, permutation seed 20260419) produced **near-parity**
between `causal` and `surrogate_only` at the verdict budget B80
(two-sided MWU `p = 0.7337`, means agree to six decimals: 0.006181 vs
0.006182). Both optimized strategies still beat `random` at certified
significance at every budget (`p = 0.0002`). All five Section 7 gates
PASS. The Sprint 35 bit-identical tie is broken on every seed at every
budget — A1 changes the trajectory exactly as Sprint 36's path-by-path
analysis (path 4: soft-causal reranker) predicted, but the
verdict-budget mean does not move. The Sprint 36 H0 prediction
(`p > 0.15` at B80) is therefore confirmed.

Do **not** open new Sprint 35, 36, or 37 issues. Those lanes are closed.
The empirical work that matters for a fresh agent is:

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
8. Sprint 36 was rescoped to docs-only preregistration after PR review (PR [#195](https://github.com/datablogin/causal-optimizer/pull/195)). It authored the seven-node / six-edge prior graph as a preregistered spec, walked all seven `causal_graph` read sites in `causal_optimizer/optimizer/suggest.py`, and named one Option A engine change and one Option B graph widening that Sprint 37 could pick between. The recommendation leaned toward **Option A1** (one small focus-restriction heuristic gated behind an explicit engine flag, enabled only for the `causal` arm). Sprint 36 also synced the post-Sprint-35 restart docs as PR [#196](https://github.com/datablogin/causal-optimizer/pull/196).
9. Sprint 37 implemented Option A1 in PR [#198](https://github.com/datablogin/causal-optimizer/pull/198) (closes issue [#197](https://github.com/datablogin/causal-optimizer/issues/197)). The PR landed the preregistered prior graph in `BanditLogAdapter.get_prior_graph()`, added `pomis_minimal_focus: bool = False` to `ExperimentEngine`, threaded the flag through both `suggest_next` paths (main + MAP-Elites elite), added the `_apply_minimal_focus_a1` helper in `causal_optimizer/optimizer/suggest.py` (applied in both optimization and exploitation so B80 doesn't revert at the `>= 50` boundary), and wired the harness so only the `causal` arm enables the flag. The Men/Random rerun produced **near-parity at B80** (`p = 0.7337`); see [sprint-37-open-bandit-prior-graph-report.md](sprint-37-open-bandit-prior-graph-report.md).

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

### Sprint 36 Done

1. [PR #195](https://github.com/datablogin/causal-optimizer/pull/195) merged -- Sprint 36 docs-only preregistration of the first Open Bandit prior graph; named Option A1 (engine change) and Option B (graph widening) for Sprint 37
2. [PR #196](https://github.com/datablogin/causal-optimizer/pull/196) merged -- restart-doc sync after Sprint 35 completion

### Sprint 37 Done

1. [PR #198](https://github.com/datablogin/causal-optimizer/pull/198) merged -- Sprint 37 Option A1 implementation: preregistered prior graph in `BanditLogAdapter.get_prior_graph()`, new `pomis_minimal_focus` engine flag, `_apply_minimal_focus_a1` helper applied in optimization + exploitation, harness wires the flag only on the `causal` arm; Men/Random rerun verdict **near-parity at B80 (p = 0.7337)**; both optimized strategies still certified over `random` at every budget; all five Section 7 gates PASS; backend `ax_botorch` only on every cell (closes [#197](https://github.com/datablogin/causal-optimizer/issues/197)); see [sprint-37-open-bandit-prior-graph-report.md](sprint-37-open-bandit-prior-graph-report.md)

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

### Real Multi-Action Benchmark (Sprint 35 baseline + Sprint 37 prior-graph rerun)

Open Bandit Men/Random (Ax/BoTorch, 10 seeds, full 452,949-row slice, 34
actions, 3 positions). Two reports on `main`:
[sprint-35-open-bandit-benchmark-report.md](sprint-35-open-bandit-benchmark-report.md)
(no prior graph) and
[sprint-37-open-bandit-prior-graph-report.md](sprint-37-open-bandit-prior-graph-report.md)
(preregistered prior graph + Option A1 minimal-focus flag).

Sprint 35 (baseline, no graph):

1. `causal` vs `surrogate_only` B20/B40/B80: exact tie on every seed (p=1.000 at every budget). The tie is mechanical: `BanditLogAdapter.get_prior_graph() -> None`, so `causal` reduces to `surrogate_only`.
2. Both optimized strategies certified over `random` at every budget (p=0.0002, 10/10 wins).
3. All five Section 7 gates PASS (null-control max ratio 1.0154, B80 median ESS 49,867, zero-support 0%, propensity sanity rel-dev ~2e-15, DR/SNIPW divergence 0.48%).

Sprint 37 (preregistered prior graph + A1 minimal-focus, `causal` arm only):

1. `causal` vs `surrogate_only` at B80: **near-parity** (`p = 0.7337` two-sided MWU; means agree to six decimals: 0.006181 vs 0.006182). H0 from the Sprint 36 plan confirmed.
2. `causal` vs `surrogate_only` at B40: not significant (`p = 0.2123`); means within 1.7e-5.
3. `causal` vs `surrogate_only` at B20: trending toward `causal < surrogate_only` (`p = 0.0820`, not certified, `causal < surrogate_only` on 8/10 seeds — mean-regret direction only, not a certified regression).
4. Both optimized strategies still certified over `random` at every budget. At B40/B80 both rows are `p = 0.0002` with 10/10 wins. At B20, `causal` vs `random` is `p = 0.0006` with 9/10 wins and `surrogate_only` vs `random` is `p = 0.0002` with 9/10 wins. Absolute B80 lift over `random`: `causal` ≈ 0.000750, `surrogate_only` ≈ 0.000751 (B80 means: causal 0.006181, s.o. 0.006182, random 0.005431).
5. Sprint 35 bit-identical tie broken on every seed at every budget (A1 changes the trajectory exactly as Sprint 36's path 4 — soft-causal reranker — predicted, but the verdict-budget mean does not move). This is an honest near-parity, not a power-limited near-miss: the B80 means agree to six decimals, so more seeds will not change the verdict.
6. All five Section 7 gates PASS with comfortable margins (null-control max ratio 1.0263 vs 1.05 band; B80 aggregate median ESS 51,255 vs floor 4,529; zero-support 0.0% best-of-seed; propensity sanity rel-dev ~1.9e-15 vs 10% band; DR/SNIPW cross-check max divergence 0.481% vs 25% tolerance).
7. Backend `ax_botorch` on every cell, no RF fallback.
8. `causal` arm B80 mean ESS rose ≈14.6% vs Sprint 35 (40,881 → 46,852); `n_effective_actions` rose 24.91 → 27.14. The A1 restriction biases the optimizer away from the most concentrated softmax policies without changing SNIPW.

Important: the Sprint 37 near-parity at B80 is not "the prior graph
made things worse" — it is the H0 outcome the Sprint 36 plan predicted.
The next missing ingredient is either (a) a graph that restricts the
search by itself (Option B widening with non-ancestor structural
nodes), (b) a different focus heuristic (Option C — e.g.
magnitude-thresholded ancestors), or (c) an explicit decision to pivot
off Open Bandit (Option D). The B20 trending row is the wrong
direction (causal lower) and is **not** something Sprint 38 should
chase.

## What Sprint 38 Should Do

Sprint 37 is complete. The full report at
[sprint-37-open-bandit-prior-graph-report.md](sprint-37-open-bandit-prior-graph-report.md)
names three explicit Sprint 38+ options in its "Sprint 38+
Implications" section. Sprint 38 picks exactly one:

1. **Option B (graph widening).** Add one non-ancestor structural node so the graph alone restricts the search (not just via screening). The Sprint 36 plan cites `logged_position_distribution` and `request_item_overlap` as candidates grounded in the adapter code. This is the first option that would make `_get_focus_variables` itself return a proper subset under the preregistered graph.
2. **Option C (different heuristic).** Replace `screened ∩ ancestors` with a magnitude-thresholded variant (e.g. drop ancestors whose screening importance is below some quantile). Keeps the graph unchanged; changes only the engine-surface restriction.
3. **Option D (move on).** Accept that under the Sprint 35 surface, the `causal` and `surrogate_only` paths converge, and reopen the multi-objective or second-dataset workstream instead.

Framing the next agent should start from:

1. The Open Bandit lane is now real on `main`: adapter, OPE stack, Section 7 gates, Sprint 35 baseline report, Sprint 36 preregistration, and the Sprint 37 Option A1 prior-graph rerun report all exist.
2. The causal-vs-surrogate differentiation question on multi-action data remains unanswered under Option A1. The A1 heuristic changes the trajectory but does not move the verdict-budget mean.
3. Do **not** chase the Sprint 37 B20 trending row (direction is `causal < surrogate_only`, not certified, and the Sprint 36 plan explicitly forbade post-hoc convergence chasing).
4. Hillstrom, Criteo, Women, All, BTS, slate OPE, and DRos-primary should not be reopened as the Sprint 38 main lane.
5. The ERCOT 10-seed rerun remains on the backlog but is not the Sprint 38 critical path.
6. The `claude-review.sh` script hangs on the current Claude CLI (`claude chat < FILE` stays interactive); Sprint 37's gauntlet worked around this by posting a manual subagent review. Fixing the script (likely by switching to `claude --print` or equivalent) is a separate cleanup task, not a Sprint 38 blocker.

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
2. [sprint-37-open-bandit-prior-graph-report.md](sprint-37-open-bandit-prior-graph-report.md)
3. [sprint-35-open-bandit-benchmark-report.md](sprint-35-open-bandit-benchmark-report.md)
4. [26-sprint-36-recommendation.md](../plans/26-sprint-36-recommendation.md)
5. [sprint-34-open-bandit-contract.md](sprint-34-open-bandit-contract.md)
6. [sprint-33-generalization-scorecard.md](sprint-33-generalization-scorecard.md)
7. [sprint-33-criteo-benchmark-report.md](sprint-33-criteo-benchmark-report.md)
8. [sprint-31-hillstrom-benchmark-report.md](sprint-31-hillstrom-benchmark-report.md)
9. [sprint-31-hillstrom-lessons-learned.md](sprint-31-hillstrom-lessons-learned.md)
10. [sprint-31-open-bandit-access-and-gap-audit.md](sprint-31-open-bandit-access-and-gap-audit.md)
11. [sprint-30-reality-and-generalization-scorecard.md](sprint-30-reality-and-generalization-scorecard.md)
12. [sprint-30-general-causal-portability-brief.md](sprint-30-general-causal-portability-brief.md)
13. [24-sprint-34-recommendation.md](../plans/24-sprint-34-recommendation.md)

## Immediate Instructions For The Next Agent

Sprint 37 is complete. Do not open new Sprint 35, 36, or 37 issues. The
active work is Sprint 38 planning:

1. Read the [Sprint 37 Open Bandit prior-graph rerun report](sprint-37-open-bandit-prior-graph-report.md) first -- it is the new evidence since the Sprint 35 baseline and pins the A1 outcome (near-parity at B80, all gates PASS, B20 trending toward `causal < surrogate_only` but not certified).
2. Read the [Sprint 35 baseline report](sprint-35-open-bandit-benchmark-report.md) and the [Sprint 36 recommendation](../plans/26-sprint-36-recommendation.md) for the contract Sprint 38 inherits.
3. Read the [Sprint 34 Open Bandit contract](sprint-34-open-bandit-contract.md) for the Men/Random slice, adapter interface, SNIPW-primary OPE stack, Section 7 support gates, and OBP-as-optional-extra dependency decision.
4. Sprint 38 picks exactly one of the three options named in the Sprint 37 report's "Sprint 38+ Implications" section: Option B (graph widening), Option C (different heuristic), or Option D (move on).
5. Do **not** describe the Sprint 35 tie as evidence that causal guidance is inert on multi-action data in general -- the tie was mechanical (null prior graph). Sprint 37's near-parity is the H0 outcome under A1 + the preregistered minimal graph and is similarly not a "causal guidance is inert" claim.
6. Do **not** chase the Sprint 37 B20 trending row -- direction is `causal < surrogate_only`, not certified, and the Sprint 36 plan explicitly forbade post-hoc convergence chasing.
7. Do not reopen Hillstrom or Criteo as the main lane.
8. Do not claim `causal` beats `random` on real data as a general statement; ERCOT has not closed that gap. On Open Bandit Men/Random both optimized strategies beat `random` at certified significance in both Sprint 35 and Sprint 37, but `causal` and `surrogate_only` are at near-parity, so it is not a `causal`-specific claim.
9. Treat one of {Option B graph widening, Option C alternative heuristic, Option D pivot} as the Sprint 38 critical path; pick on the basis of "which one gives `_get_focus_variables` (or its replacement) a chance to return a proper subset under a sensible structural argument."

## One-Line Situation Summary

Sprint 33 closed with verdict **GENERALITY IS REAL BUT CONDITIONAL** (ERCOT COAST p=0.008 certified, Hillstrom pooled slice certified surrogate-only under RF, Criteo near-parity under Ax/BoTorch after the heterogeneous follow-up). Sprint 34 merged the Open Bandit contract as PR #184. Sprint 35 executed that contract in four merged PRs (#188, #189, #191, #192) and produced the first Men/Random benchmark with bit-identical `causal == surrogate_only` tie under a null prior graph. Sprint 36 (PR #195/#196) merged the docs-only preregistration of the first prior graph and the post-Sprint-35 restart-doc sync, naming Option A1 for Sprint 37. Sprint 37 (PR #198) implemented A1 (preregistered seven-node / six-edge graph + `pomis_minimal_focus` engine flag enabled only on the `causal` arm) and reran Men/Random: B80 verdict is **near-parity** (`p = 0.7337`), all five Section 7 gates PASS, both arms still certified over `random`, the Sprint 35 bit-identical tie is broken on every seed but the verdict-budget mean does not move; H0 from the Sprint 36 plan is confirmed. Sprint 38 picks Option B / C / D from the Sprint 37 report.
