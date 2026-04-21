# Benchmark State

Updated: 2026-04-21 (Sprint 35 complete — first Men/Random Open Bandit benchmark merged as PR #192)

## Purpose

This file is the restart point for benchmark, attribution, and sprint-direction work in
`causal-optimizer`.

Use it when:

1. starting a new chat
2. handing context to another reviewer or agent
3. checking what is merged vs still pending
4. recovering after an interruption or crash

## Current Position

The project is now strongest as a **trustworthy research harness with one real-world positive, one clean non-energy boundary, one near-parity large-scale marketing result, and one clean multi-action Open Bandit row where both optimized strategies beat `random` at certified significance but `causal` and `surrogate_only` tie mechanically under a null prior graph**.

What is true today:

1. the benchmark stack is disciplined and reusable
2. positive controls, negative controls, provenance, locked A/B reruns, stability gates, and boundary-mapping artifacts are all in place
3. the project can demonstrate causal advantage on controlled benchmark families where noise burden and categorical barriers matter
4. the Sprint 29 default change produced the first real-world causal vs surrogate-only differentiation on ERCOT (COAST p=0.008, NORTH_C p=0.059) — but causal still does not beat random
5. alignment-only remains the correct production default after Sprint 21 attribution and Sprint 22 confirmation
6. Sprint 25 delivered the first mechanism-matched stability fix that actually met the base-B80 gate
7. Sprint 26 and Sprint 27 expanded benchmark coverage to seven rows and clarified the causal-advantage boundary
8. Sprint 28 separated Ax/BoTorch primary claims from RF-fallback secondary checks
9. Sprint 29 completed diagnosis, intervention, and regression gate — verdict GENERALITY IMPROVED
10. Sprint 30 produced the ERCOT reality report, portability brief, and Hillstrom harness — verdict REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC
11. Sprint 31 ran Hillstrom as the first non-energy real-data benchmark — verdict SURROGATE-ONLY ADVANTAGE (pooled slice) with a near-parity causal B80 tail on the primary slice under RF fallback
12. Sprint 32 merged the Criteo uplift benchmark contract
13. Sprint 33 executed the Criteo contract under Ax/BoTorch on a 1M-row subsample — Run 1 degenerate-surface exact tie and Run 2 heterogeneous-surface near-parity, combined verdict NEAR-PARITY
14. Sprint 33 closure published the generalization scorecard — verdict GENERALITY IS REAL BUT CONDITIONAL
15. Sprint 34 merged the Open Bandit contract and multi-action architecture brief as PR #184 — pinned Men/Random as the first slice, a new `DomainAdapter` subclass, SNIPW-primary OPE, five Section 7 support gates, and OBP as an optional extra
16. Sprint 35 executed the contract end-to-end in four merged PRs (adapter #189, OPE stack #188, bridge #191, benchmark report #192) and produced the first Men/Random benchmark on the full 452,949-row slice under Ax/BoTorch with all five Section 7 gates PASS; `causal` and `surrogate_only` tie bit-identically on every seed because `BanditLogAdapter.get_prior_graph() -> None`, and both beat `random` at certified significance (p=0.0002, 10/10 wins, every budget)

## Current Goal

Sprint 35 is **complete**. The Open Bandit lane is now real on `main`:
adapter (`BanditLogAdapter`), OPE stack (SNIPW primary, DM and DR
secondary), all five Section 7 support gates, bridge between the adapter
and the OPE path, and a published first-run Men/Random benchmark report.
See [sprint-35-open-bandit-benchmark-report.md](../docs/sprint-35-open-bandit-benchmark-report.md)
for the authoritative benchmark evidence and
[sprint-34-open-bandit-contract.md](../docs/sprint-34-open-bandit-contract.md)
for the pinning contract.

The active lane is now **Sprint 36 planning**, not Sprint 35 execution.
The high-level framing is preserved: Sprint 35 resolved the plumbing
question (the Open Bandit path runs cleanly, all five gates pass, both
optimized strategies beat `random`), but it did not and could not resolve
the causal-vs-surrogate differentiation question on multi-action data —
the Sprint 35 exact tie is the expected null result under a null prior
graph (`get_prior_graph() -> None` per Sprint 34 contract Section 4e),
not evidence that causal guidance is inert on multi-action data in
general. Answering that question requires a bandit-log-compatible
multi-action prior graph (authored or principled discovery), which is
the most direct Sprint 36+ move. Sprint 36 detailed planning lives in a
separate recommendation and prompt document owned by a different track
and is not scoped by this file.

## Current Sprint Status

### Sprint 35

Closed issues:

1. [#185](https://github.com/datablogin/causal-optimizer/issues/185) Sprint 35.A: Open Bandit `DomainAdapter` and Men/Random smoke test -- closed by PR #189
2. [#186](https://github.com/datablogin/causal-optimizer/issues/186) Sprint 35.B: SNIPW / DM / DR OPE stack and Section 7 gates -- closed by PR #188
3. [#190](https://github.com/datablogin/causal-optimizer/issues/190) Sprint 35 bridge: wire `BanditLogAdapter` into the Open Bandit OPE path -- closed by PR #191
4. [#187](https://github.com/datablogin/causal-optimizer/issues/187) Sprint 35.C: first Men/Random Open Bandit benchmark report -- closed by PR #192

Merged PRs:

1. `#189` merged
   - PR: [#189](https://github.com/datablogin/causal-optimizer/pull/189)
   - issue: `#185` closed
   - result: Sprint 35.A `BanditLogAdapter` and Men/Random smoke test; confirms the OBD `action_prob` schema is conditional `P(item | position) = 1 / n_items` (on Men/Random, 1/34 ≈ 0.0294) and documents the chosen three-to-five context features in the adapter docstring per Sprint 34 contract Section 4c
2. `#188` merged
   - PR: [#188](https://github.com/datablogin/causal-optimizer/pull/188)
   - issue: `#186` closed
   - result: Sprint 35.B Open Bandit OPE stack (SNIPW primary, DM and DR secondary, DR on an in-module per-action empirical-CTR reward model) and the five Section 7 support gates (null control, ESS, zero-support, propensity sanity, DR/SNIPW cross-check)
3. `#191` merged
   - PR: [#191](https://github.com/datablogin/causal-optimizer/pull/191)
   - issue: `#190` closed
   - result: three bridge seams between Sprint 35.A and Sprint 35.B — `normalize_positions` helper that dense-ranks integer positions to 0-indexed contiguous labels, `BanditLogAdapter.propensity_schema` pinned to conditional, `get_obp_version` helper replacing the Track B `OBP_VERSION_PLACEHOLDER`, and a narrow `BanditLogAdapter.to_bandit_feedback()` helper for the OPE path
4. `#192` merged
   - PR: [#192](https://github.com/datablogin/causal-optimizer/pull/192)
   - issue: `#187` closed
   - result: Sprint 35.C first Men/Random Open Bandit benchmark report — full 452,949-row slice, 34 actions, 3 positions, Ax/BoTorch primary, 10 seeds × B20/B40/B80; **verdict: exact tie between `causal` and `surrogate_only` on every seed (p=1.000), both CERTIFIED over `random` (p=0.0002, 10/10 wins)**; all five Section 7 gates PASS (null control max ratio 1.0154 vs 1.05 band, B80 median ESS 49,867 vs floor 4,530, zero-support 0.0% best-of-seed, propensity sanity deviation ~2e-15 vs 10% band, DR/SNIPW cross-check max divergence 0.48% vs 25% tolerance)

Report document:

1. [sprint-35-open-bandit-benchmark-report.md](../docs/sprint-35-open-bandit-benchmark-report.md)

Current Sprint 35 position:

1. Sprint 35.A adapter and smoke test: **complete** (PR #189 merged)
2. Sprint 35.B OPE stack and Section 7 gates: **complete** (PR #188 merged)
3. Sprint 35 bridge between adapter and OPE stack: **complete** (PR #191 merged)
4. Sprint 35.C first Men/Random benchmark report: **complete** (PR #192 merged)
5. Verdict: clean Men/Random row; **the `causal` vs `surrogate_only` exact tie is mechanical** — `BanditLogAdapter.get_prior_graph() -> None` per Sprint 34 contract Section 4e, so `causal` reduces to `surrogate_only` on this slice and both return bit-identical best-of-seed policy values. This is the "least valuable but acceptable" first-run outcome per Sprint 34 contract Section 12 and carries the same weight as the Criteo near-parity row. The tie is not evidence that causal guidance is inert on multi-action data in general; it is the expected null result under a null prior graph. The Sprint 33 closure verdict (GENERALITY IS REAL BUT CONDITIONAL) carries forward unchanged.

### Sprint 34

Plan:

1. [24-sprint-34-recommendation.md](24-sprint-34-recommendation.md)

Prompts:

1. [sprint-34-open-bandit-contract.md](../prompts/sprint-34-open-bandit-contract.md)

Closed issue:

1. [#182](https://github.com/datablogin/causal-optimizer/issues/182) define the Open Bandit benchmark contract and multi-action architecture brief -- closed by PR #184

Merged PRs:

1. `#184` merged
   - PR: [#184](https://github.com/datablogin/causal-optimizer/pull/184)
   - issue: `#182` closed
   - result: Open Bandit contract / multi-action architecture brief -- pins Men/Random as the first slice, SNIPW as the primary OPE estimator, a new `DomainAdapter` subclass (not a `MarketingLogAdapter` extension), support gates expressed in relative terms (5% relative null-control band, 10% relative propensity sanity band, ESS floor, zero-support fraction, DR/SNIPW cross-check), OBP as an optional extra under a pinned version, and three ordered Sprint 35 implementation issues (adapter, OPE stack + gates, first benchmark report)

Contract document:

1. [sprint-34-open-bandit-contract.md](../docs/sprint-34-open-bandit-contract.md)

Current Sprint 34 position:

1. Open Bandit contract / multi-action architecture brief: **complete** (PR #184 merged)
2. Sprint 35 implementation issues (adapter, OPE stack + gates, first benchmark report): **complete** -- Sprint 35 executed and closed Issues #185, #186, #190, and #187 via PRs #189, #188, #191, and #192
3. Verdict: the contract is executable and has been executed in full on one slice (Men/Random). No synthetic, energy, or binary-marketing benchmark ran in Sprint 34 or Sprint 35, so the synthetic Ax boundary and Sprint 33 closure verdict carry forward unchanged

### Sprint 33

Plan:

1. [23-sprint-33-recommendation.md](23-sprint-33-recommendation.md)
2. [24-sprint-34-recommendation.md](24-sprint-34-recommendation.md)

Prompts:

1. [sprint-33-criteo-benchmark-implementation.md](../prompts/sprint-33-criteo-benchmark-implementation.md)
2. [sprint-33-closure-scorecard.md](../prompts/sprint-33-closure-scorecard.md)

Merged PRs:

1. `#180` merged
   - PR: [#180](https://github.com/datablogin/causal-optimizer/pull/180)
   - result: Criteo uplift benchmark Run 1 (degenerate 2-variable surface) exact-tie near-parity on B20/B40/B80 and mandatory Run 2 (synthesized f0-tertile segments) near-parity on B20/B40/B80; combined verdict NEAR-PARITY; Ax/BoTorch primary, 1M-row subsample of 13.98M-row dataset, 10 seeds, null control PASS, propensity gate PASS, ESS ~849,982 on optimized strategies

Current Sprint 33 position:

1. Criteo benchmark run and report: **complete** (PR #180 merged)
2. Sprint 33 generalization scorecard and restart-doc sync: **complete** ([PR #183](https://github.com/datablogin/causal-optimizer/pull/183) merged)
3. verdict: **GENERALITY IS REAL BUT CONDITIONAL** — ERCOT remains the strongest real-world positive signal, Hillstrom is a real non-energy boundary favoring `surrogate_only` under RF fallback, Criteo is near-parity under Ax/BoTorch even after the mandatory heterogeneous follow-up, project remains a general causal research harness but current causal advantage over `surrogate_only` is conditional on landscape structure, noise burden, and search-space breadth
4. Sprint 34: draft Open Bandit contract and multi-action architecture brief; not another binary marketing rerun

### Sprint 32

Merged PRs:

1. `#178` merged
   - PR: [#178](https://github.com/datablogin/causal-optimizer/pull/178)
   - result: Criteo uplift benchmark contract (primary outcome `visit`, 1M-row fixed-seed subsample, IPS / ESS / null-control gates, mandatory Run 2 on near-parity)

Current Sprint 32 position:

1. Criteo benchmark contract: **complete** (PR #178 merged)
2. verdict: contract pinned; execution deferred to Sprint 33

### Sprint 31

Plan:

1. [22-sprint-31-generalization-research-plan.md](22-sprint-31-generalization-research-plan.md)

Prompts:

1. [sprint-31-hillstrom-benchmark-contract.md](../prompts/sprint-31-hillstrom-benchmark-contract.md)
2. [sprint-31-hillstrom-benchmark-implementation.md](../prompts/sprint-31-hillstrom-benchmark-implementation.md)

Merged PRs:

1. `#174` merged
   - PR: [#174](https://github.com/datablogin/causal-optimizer/pull/174)
   - result: Criteo access and adapter-gap audit
2. `#176` merged
   - PR: [#176](https://github.com/datablogin/causal-optimizer/pull/176)
   - result: Hillstrom benchmark report — pooled slice certified SURROGATE-ONLY ADVANTAGE at B20/B40/B80 (p=0.017 / 0.002 / 0.019); primary slice trending s.o. at B20 (p=0.060), certified s.o. at B40 (p=0.0001), near-parity at B80 (p=0.817) with bimodal causal tail; causal > random at primary B80 (p=0.0004); RF fallback backend; 10 seeds; 3-variable active search space

Current Sprint 31 position:

1. Hillstrom benchmark run and report: **complete** (PR #176 merged)
2. Criteo access / adapter-gap audit: **complete** (PR #174 merged)
3. ERCOT 10-seed rerun: not executed (still on backlog)
4. verdict: **SURROGATE-ONLY ADVANTAGE** on Hillstrom pooled slice under RF fallback; first non-energy boundary is specific and diagnosable, not a general marketing claim

### Sprint 30

Issues:

1. [#162](https://github.com/datablogin/causal-optimizer/issues/162) ERCOT reality gate
2. [#163](https://github.com/datablogin/causal-optimizer/issues/163) general-causal portability brief
3. [#164](https://github.com/datablogin/causal-optimizer/issues/164) reality-and-generalization scorecard
4. [#168](https://github.com/datablogin/causal-optimizer/issues/168) Hillstrom benchmark harness

Merged PRs:

1. `#165` merged
   - PR: [#165](https://github.com/datablogin/causal-optimizer/pull/165)
   - issue: `#163` closed
   - result: portability brief re-anchored project as domain-agnostic
2. `#166` merged
   - PR: [#166](https://github.com/datablogin/causal-optimizer/pull/166)
   - issue: `#162` closed
   - result: ERCOT reality report — COAST causal > s.o. (p=0.008), NORTH_C trending (p=0.059)
3. `#167` merged
   - PR: [#167](https://github.com/datablogin/causal-optimizer/pull/167)
   - result: Hillstrom benchmark contract
4. `#169` merged
   - PR: [#169](https://github.com/datablogin/causal-optimizer/pull/169)
   - issue: `#168` closed
   - result: Hillstrom benchmark harness code merged

Current Sprint 30 position:

1. `#162` ERCOT reality gate: **complete** (PR #166 merged)
2. `#163` portability brief: **complete** (PR #165 merged)
3. `#164` scorecard: **complete** (PR #175 merged)
4. `#168` Hillstrom harness: **complete** (PR #169 merged)
5. verdict: **REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC** — first ERCOT
   causal vs s.o. differentiation (COAST p=0.008, NORTH_C p=0.059),
   but no non-energy empirical results yet
6. Sprint 31: run Hillstrom + extend ERCOT to 10 seeds

### Sprint 29

Plan:

1. [20-sprint-29-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/20-sprint-29-recommendation.md)

Issues:

1. [#152](https://github.com/datablogin/causal-optimizer/issues/152) diagnose interaction and dose-response optimizer trajectories under Ax
2. [#153](https://github.com/datablogin/causal-optimizer/issues/153) implement adaptive causal guidance under Ax-primary gates
3. [#154](https://github.com/datablogin/causal-optimizer/issues/154) run optimizer-core regression gate and publish scorecard

Prompts:

1. [sprint-29-trajectory-diagnosis.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-29-trajectory-diagnosis.md)
2. [sprint-29-adaptive-causal-guidance.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-29-adaptive-causal-guidance.md)
3. [sprint-29-optimizer-core-scorecard.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/prompts/sprint-29-optimizer-core-scorecard.md)

Merged PRs:

1. `#155` merged
   - PR: [#155](https://github.com/datablogin/causal-optimizer/pull/155)
   - result: trajectory diagnosis found that interaction failure is early causal pressure and dose-response looked real but underpowered at 5 seeds
2. `#158` merged
   - PR: [#158](https://github.com/datablogin/causal-optimizer/pull/158)
   - result: dose-response is now a certified Ax-primary causal win at 10 seeds (`p=0.002`, `9/10` wins)
3. `#159` merged
   - PR: [#159](https://github.com/datablogin/causal-optimizer/pull/159)
   - result: interaction ablation showed exploration weighting is the primary cause of the B20 catastrophe; `graph_only` was empirically best on that row
4. `#160` merged
   - PR: [#160](https://github.com/datablogin/causal-optimizer/pull/160)
   - result: production default `causal_exploration_weight` changed from `0.3` to `0.0`; `causal_softness` left unchanged at `0.5`
5. `#161` merged
   - PR: [#161](https://github.com/datablogin/causal-optimizer/pull/161)
   - issue: `#154` closed
   - result: Ax-primary regression gate and optimizer-core scorecard, verdict **GENERALITY IMPROVED**

Current Sprint 29 position:

1. `#152` trajectory diagnosis: **complete** (PR #155 merged)
2. `#153` adaptive causal guidance: **complete** (PR #160 merged)
3. `#154` regression gate and scorecard: **complete** (PR #161 merged)
4. verdict: **GENERALITY IMPROVED** — all rows improved in mean regret,
   interaction flipped to near-parity, but base B80 p loosened from
   0.045 to 0.112 (no longer statistically significant, mean improved)
5. Sprint 29 produced 3 certified wins (medium, high, dose-response),
   1 trending (base), 1 near-parity (interaction), 1 PASS (null control)

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
3. `#151` merged
   - PR: [#151](https://github.com/datablogin/causal-optimizer/pull/151)
   - issue: `#148` closed
   - result: AX PRIMARY, RF SECONDARY verdict

## Immediate Next Step

Sprint 35 is **complete**. All four Sprint 35 PRs (#188, #189, #191, #192)
and their issues (#185, #186, #187, #190) are merged and closed. The first
Men/Random Open Bandit benchmark is published as
[sprint-35-open-bandit-benchmark-report.md](../docs/sprint-35-open-bandit-benchmark-report.md).
The Sprint 33 closure verdict **GENERALITY IS REAL BUT CONDITIONAL**
carries forward unchanged; no synthetic, energy, or binary-marketing
benchmark ran in Sprint 34 or Sprint 35.

The active lane is now **Sprint 36 planning**. Do **not** open new Sprint
35 issues. If resuming:

1. read the [Sprint 35 Open Bandit benchmark report](../docs/sprint-35-open-bandit-benchmark-report.md) first — it pins the exact shape of the Men/Random result (clean diagnostics, all five Section 7 gates PASS, `causal == surrogate_only` exact tie on every seed and every budget, both beat `random` at certified significance)
2. read the [Sprint 34 Open Bandit contract](../docs/sprint-34-open-bandit-contract.md) for the Men/Random slice, adapter interface, OPE stack, Section 7 support gates, and OBP dependency decision
3. read the [Sprint 33 generalization scorecard](../docs/sprint-33-generalization-scorecard.md) for the synthesized verdict across ERCOT, Hillstrom, and Criteo
4. read the [handoff document](../docs/handoff.md) for the post-Sprint-35 restart instructions
5. Sprint 36 detailed planning lives in a separate recommendation and prompt document owned by a different track; do not start new Sprint 36 implementation scope from this file
6. do not describe the Sprint 35 exact tie as evidence that causal guidance is inert on multi-action data in general — the tie is mechanical (null prior graph), not empirical
7. do not reopen Hillstrom or Criteo as the Sprint 36 main lane
8. the ERCOT 10-seed rerun remains on the backlog but is not the Sprint 36 critical path

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
   - [PR #151](https://github.com/datablogin/causal-optimizer/pull/151) Sprint 28 backend baseline scorecard
16. [19-sprint-28-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/19-sprint-28-recommendation.md)
17. [20-sprint-29-recommendation.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/20-sprint-29-recommendation.md)
18. [sprint-29-trajectory-diagnosis-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-trajectory-diagnosis-report.md)
19. [sprint-29-dose-response-10seed-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-dose-response-10seed-report.md)
20. [sprint-29-interaction-ablation-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-interaction-ablation-report.md)
21. [sprint-29-adaptive-causal-guidance-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-29-adaptive-causal-guidance-report.md)
22. [sprint-34-open-bandit-contract.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-34-open-bandit-contract.md)
23. [sprint-35-open-bandit-benchmark-report.md](/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md)

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
3. `#151` backend baseline scorecard (merged)

What we learned:

1. every benchmark artifact now records its optimizer path explicitly
2. the Ax/BoTorch seven-benchmark rerun restores direct comparability to the trusted Ax baselines
3. all demand-response trusted priors reproduced exactly (base matches S25, medium matches S27, high matches S25)
4. null control: 10th clean run (0.2% max delta)
5. two rows have mean-regret direction reverse by backend: base energy (tie under RF, causal win under Ax) and dose-response (s.o. win under RF, causal lower mean under Ax but p=0.142)
6. one row is backend-sensitive in magnitude: interaction (s.o. wins under both, gap narrows under Ax)
7. four rows are backend-invariant: medium-noise, high-noise, confounded, null control
8. verdict: **AX PRIMARY, RF SECONDARY**
9. RF fallback is useful for family-level drift detection but should not be used to certify or refute row-level causal-advantage claims
10. Sprint 29 should return to optimizer-core work

### Sprint 30 (General Causal Autoresearch: Reality Check And Portability)

Merged PRs:

1. `#165` portability brief
2. `#166` ERCOT reality report
3. `#167` Hillstrom benchmark contract
4. `#169` Hillstrom benchmark harness

What we learned:

1. the Sprint 29 default change (`causal_exploration_weight=0.0`) produced the first real-world causal vs surrogate-only differentiation on ERCOT
2. COAST B80: causal certified better than s.o. (p=0.008, 5/5 wins) — this is the strongest real-world signal to date
3. NORTH_C B80: causal trending better than s.o. (p=0.059, 4/5 wins) — marginal, needs 10-seed confirmation
4. causal still does not beat random on either dataset; all strategies converge to ridge
5. the core engine is fully domain-portable; 6 of 7 active regression gate rows are ERCOT-tied
6. the portability brief re-anchored the project identity as domain-agnostic
7. the Hillstrom email campaign harness is the first non-energy benchmark code
8. no non-energy empirical results exist yet — generality claim is structural
9. verdict: **REAL-WORLD IMPROVED BUT DOMAIN-SPECIFIC**
10. Sprint 31: run Hillstrom + extend ERCOT to 10 seeds

### Sprint 31 (First Non-Energy Real-Data Benchmark)

Merged PRs:

1. `#174` Criteo access and adapter-gap audit
2. `#176` Hillstrom benchmark report

What we learned:

1. Hillstrom ran end-to-end on a real marketing dataset under the existing `MarketingLogAdapter`, confirming the generalization program is empirical rather than structural
2. pooled slice returned certified `surrogate_only` advantage at B20/B40/B80 (p=0.017 / 0.002 / 0.019, two-sided MWU)
3. primary slice was trending s.o. at B20 (p=0.060), certified s.o. at B40 (p=0.0001), near-parity at B80 (p=0.817) with a bimodal causal tail on seeds 7/8/9
4. causal beat `random` at primary B80 (p=0.0004, 9/10 wins) — the causal path is not inert on Hillstrom
5. the run was on the RF fallback backend, so it is a real non-energy boundary result under RF, not an Ax-primary refutation
6. the 3-variable active search space is narrower than the rows with clearest synthetic causal wins — a plausible boundary condition on graph leverage
7. the Hillstrom null-control pass has the caveat that permuted-outcome policy values can still exceed the simple baseline for some strategies, so primary-slice gains should not be read as clean treatment-effect evidence on their own
8. the Criteo access audit confirmed that Criteo fits the existing adapter via a wrapper rather than needing a new multi-action architecture
9. verdict: **SURROGATE-ONLY ADVANTAGE** on Hillstrom pooled slice; specific and diagnosable non-energy boundary, not a general marketing-domain claim

### Sprint 32 (Criteo Benchmark Contract)

Merged PRs:

1. `#178` Criteo uplift benchmark contract

What we learned:

1. the contract pinned primary outcome (`visit`), secondary outcome (`conversion`), benchmark scale (1M-row fixed-seed subsample), IPS / ESS gates, null-control rules, propensity-gate behavior, and the mandatory Run 2 requirement on near-parity
2. Criteo adapts to the existing `MarketingLogAdapter` via a wrapper; no adapter evolution required for the first run
3. backend discipline is explicit in the contract: no mixing of Ax and RF within one verdict table
4. verdict: contract merged; execution deferred to Sprint 33

### Sprint 33 (First Ax-Primary Large-Scale Marketing Benchmark)

Merged PRs:

1. `#180` Criteo benchmark report (Run 1 + Run 2)

What we learned:

1. the engine runs cleanly on a 1M-row, 85:15 treatment-imbalanced, binary-outcome marketing log under Ax/BoTorch
2. Run 1 on the degenerate 2-variable surface returned exact-tie near-parity on B20/B40/B80 (p=1.000), as the contract anticipated
3. the mandatory Run 2 with synthesized f0-tertile segments also returned near-parity on B20/B40/B80 (p=0.168 / 1.000 / 0.368); heterogeneity did not unlock a causal advantage
4. the propensity gate passed (max deviation 0.79pp vs 2pp threshold); IPS stack was stable at 85:15 imbalance (ESS ~849,982 on optimized strategies, no zero-support events)
5. null control passed within the 5% band on all cells
6. combined Criteo verdict per the Sprint 32 contract: **near-parity**
7. Hillstrom and Criteo together answer the narrow binary-marketing question; Open Bandit is the right next frontier rather than a third binary uplift rerun
8. closure verdict: **GENERALITY IS REAL BUT CONDITIONAL** — the engine remains a general causal research harness, but current causal advantage over `surrogate_only` is conditional on landscape structure, noise burden, and search-space breadth
9. Sprint 34: Open Bandit contract / multi-action architecture brief

### Sprint 35 (First Multi-Action Open Bandit Benchmark)

Merged PRs:

1. `#189` Sprint 35.A `BanditLogAdapter` and Men/Random smoke test
2. `#188` Sprint 35.B Open Bandit OPE stack (SNIPW / DM / DR) and Section 7 gates
3. `#191` Sprint 35 bridge between adapter and OPE path (position normalization, conditional propensity schema pin, OBP version provenance helper)
4. `#192` Sprint 35.C first Men/Random Open Bandit benchmark report

What we learned:

1. the engine runs cleanly on a 452,949-row, 34-action, 3-position logged bandit feedback dataset under Ax/BoTorch; the existing `ExperimentEngine` + `suggest_parameters` loop handles the six-variable item-scoring surface without any optimizer-core modifications
2. OBD Men/Random's `action_prob` column is the **conditional** propensity `P(item | position) = 1 / n_items` (on Men/Random, 1/34 ≈ 0.0294), not the joint `P(item, position) = 1 / (n_items * n_positions)` — the Sprint 34 contract Section 5c ambiguity is resolved
3. all five Section 7 support gates pass cleanly with comfortable margins (null control max ratio 1.0154 vs 1.05 band; B80 median ESS 49,867 vs floor 4,530; zero-support 0.0% best-of-seed vs 10% band; propensity sanity relative deviation ~2e-15 vs 10% band; DR/SNIPW cross-check max divergence 0.48% vs 25% tolerance) — the verdict-publication bar is trustworthy
4. the verdict at B80 is an **exact bit-identical tie** between `causal` and `surrogate_only` on every seed (p=1.000 at every budget), with both **CERTIFIED** over `random` (p=0.0002, 10/10 wins, every budget; absolute B80 lift 0.000751, ~14% relative over `random` and ~20% relative over logged-policy μ=0.005124)
5. the tie is mechanical, not substantive: `BanditLogAdapter.get_prior_graph() -> None` per Sprint 34 contract Section 4e, so the `causal` strategy operates without causal knowledge and is behaviorally identical to `surrogate_only` on this slice. The first-run adapter was explicitly allowed to return `None`; this is the expected null result under a null prior graph, not evidence about whether causal guidance helps on multi-action data in general
6. both optimized strategies consistently converge to the same softmax policy each seed (low `tau` ≈ 0.285, `eps=0.0`, `w_user_item_affinity=3.0`, negative `w_item_popularity`, `position_handling_flag="position_1_only"`), confirming the tie is driven by the missing causal prior rather than by optimizer-state drift
7. Sprint 34 contract Section 12 classification: "least valuable but acceptable" first-run outcome — a clean Open Bandit row that carries the same weight as the Criteo near-parity row in the generalization scorecard
8. Open Bandit is now a real, merged benchmark lane; causal-vs-surrogate differentiation on multi-action data remains unanswered and the next missing ingredient is causal structure (a defensible multi-action prior graph or a principled way to discover one for bandit logs), not more plumbing
9. Sprint 36: multi-action prior graph authoring or discovery for Open Bandit; detailed planning lives in a separate recommendation and prompt document

### Sprint 29 (Adaptive Causal Guidance Under Clean Backend Gates)

Merged PRs:

1. `#155` trajectory diagnosis
2. `#158` dose-response 10-seed certification
3. `#159` interaction ablation
4. `#160` default change (causal_exploration_weight 0.3 to 0.0)
5. `#161` regression gate and scorecard

What we learned:

1. the interaction B20 catastrophe was caused by causal-weighted exploration, not alignment
2. dose-response is a certified Ax-primary causal win (p=0.003, 9/10 wins)
3. removing causal_exploration_weight improved every row in mean regret
4. interaction flipped from s.o. advantage (p=0.014) to near-parity (p=0.225)
5. base B80 mean improved (1.13 to 1.01) but lost statistical significance (p=0.045 to p=0.112)
6. no row regressed; no row has s.o. statistically significantly better than causal under Ax
7. verdict: **GENERALITY IMPROVED**

## Current Best Evidence

### Real Predictive Benchmarks (Sprint 30)

ERCOT NORTH_C and COAST now show the first causal vs surrogate-only differentiation:

1. COAST B80: causal certified better than s.o. (MAE 104.88 vs 105.72, two-sided MWU `p=0.008`, 5/5 wins)
2. NORTH_C B80: causal trending better than s.o. (MAE 132.48 vs 132.98, two-sided MWU `p=0.059`, 4/5 wins)
3. causal still does not statistically beat random on either dataset
4. all strategies still converge to `ridge`
5. results at 5 seeds only; 10-seed rerun remains on the backlog

Interpretation:

1. the Sprint 29 default change (`causal_exploration_weight=0.0`) broke the "causal == s.o." plateau from Sprint 16
2. the improvement is in ridge hyperparameters, not model class
3. the project has a first real-world causal vs s.o. signal but not a full causal advantage claim

### Positive / Negative Benchmark Boundary (Sprint 29, unchanged)

Trusted Ax-primary references (Sprint 29, after `causal_exploration_weight=0.0`):

1. base B80: causal mean/std `1.01 / 1.10`, catastrophic seeds `0/10`, wins `7/10`, two-sided `p=0.112` (trending, no longer significant)
2. medium-noise B80: causal mean/std `1.19 / 1.52`, wins `10/10`, two-sided `p=0.002` (certified)
3. high-noise B80: causal mean/std `1.08 / 1.72`, wins `10/10`, two-sided `p=0.001` (certified)

Cross-family rows:

1. confounded demand-response: all strategies can be misled; not a clean causal win row (backend-invariant)
2. interaction policy: near-parity (causal `1.90` vs s.o. `2.18`, `p=0.225`), flipped from s.o. advantage after Sprint 29 intervention
3. dose-response clinical: certified Ax-primary causal win (`0.22`, `p=0.003`, `9/10` wins); s.o. wins under RF

Negative control:

1. null control clean across `11` runs over `12` sprint slots (backend-invariant)
2. Sprint 26 intentionally did not rerun null control
3. latest clean reruns stay within the `2%` tolerance

### Non-Energy Benchmarks (Sprint 31 + Sprint 33 + Sprint 35)

1. Hillstrom email campaign (RF fallback, 10 seeds, 3-variable active search space):
   - primary B20: trending s.o. (p=0.060, 8/10 wins s.o.)
   - primary B40: certified s.o. (p=0.0001, 10/10 wins s.o.)
   - primary B80: near-parity with bimodal causal tail (p=0.817); causal > random certified (p=0.0004, 9/10 wins)
   - pooled B20/B40/B80: certified s.o. (p=0.017 / 0.002 / 0.019)
   - null-control pass has the caveat that permuted-outcome policy values can still exceed the simple baseline
2. Criteo uplift (Ax/BoTorch, 10 seeds, 1M-row subsample of 13.98M-row dataset):
   - Run 1 degenerate 2-variable surface B20/B40/B80: exact-tie near-parity (p=1.000 on all three)
   - Run 2 synthesized f0-tertile segments B20/B40/B80: near-parity (p=0.168 / 1.000 / 0.368)
   - propensity gate PASS (max deviation 0.79pp vs 2pp threshold)
   - IPS stack stable at 85:15 treatment imbalance (ESS ~849,982 on optimized strategies, no zero-support events)
   - null control PASS within 5% band on all cells
   - combined Criteo verdict: near-parity
3. Open Bandit Men/Random (Ax/BoTorch, 10 seeds, full 452,949-row slice, 34 actions, 3 positions, 6-variable search space):
   - `causal` vs `surrogate_only` B20/B40/B80: **exact tie** on every seed (p=1.000 at every budget) — mechanical tie, driven by `BanditLogAdapter.get_prior_graph() -> None` per Sprint 34 contract Section 4e
   - `causal` vs `random` B20/B40/B80: **certified** (p=0.0002, 10/10 wins at every budget); absolute B80 lift 0.000751 (~14% relative over `random`, ~20% relative over logged-policy μ=0.005124)
   - `surrogate_only` vs `random` B20/B40/B80: **certified** (p=0.0002, 10/10 wins at every budget), identical to the causal-vs-random row
   - propensity schema confirmed as conditional `P(item | position) = 1 / 34`; Section 7d gate PASS with relative deviation ~2e-15
   - all five Section 7 gates PASS (null control max ratio 1.0154 vs 1.05 band; B80 median ESS 49,867 vs floor 4,530; zero-support 0.0% best-of-seed; DR/SNIPW cross-check max divergence 0.48% vs 25% tolerance)
   - verdict per Sprint 34 contract Section 12: "least valuable but acceptable" first-run outcome — clean diagnostics, no causal vs surrogate-only separation under a null prior graph; carries the same weight as the Criteo near-parity row

## Current Conclusion

The project is a **trustworthy automated causal research harness with one real-world positive (ERCOT), one clean non-energy boundary (Hillstrom under RF), one near-parity large-scale marketing result (Criteo under Ax/BoTorch), and one clean multi-action Open Bandit row where both optimized strategies beat `random` at certified significance under a null prior graph**.

What is now established:

1. causal guidance wins on 3 of 7 synthetic benchmarks under Ax (medium, high, dose-response); base is trending; interaction is near-parity
2. the Sprint 29 default change produced the first real-world causal vs surrogate-only differentiation on ERCOT (COAST p=0.008, NORTH_C p=0.059)
3. causal still does not beat random on real ERCOT data
4. the engine is architecturally domain-portable; the benchmark portfolio is empirically energy-dominated but Hillstrom (RF), Criteo (Ax), and Open Bandit Men/Random (Ax) now exist as real non-energy lanes
5. Hillstrom returned a clean pooled-slice surrogate-only advantage on a narrow 3-variable search space under RF fallback; causal beats random at primary B80
6. Criteo returned near-parity under Ax/BoTorch on both a degenerate 2-variable surface and the mandatory heterogeneous f0-tertile follow-up
7. Open Bandit Men/Random returned an exact bit-identical tie between `causal` and `surrogate_only` under Ax/BoTorch, with both certified over `random`; the tie is mechanical (`BanditLogAdapter.get_prior_graph() -> None`) and is not evidence about causal guidance on multi-action data in general
8. causal advantage over surrogate-only is currently conditional on landscape structure, noise burden, and search-space breadth; it is not a universal property of the engine
9. RF fallback is a secondary family-level regression signal, not a drop-in substitute for Ax-primary baselines; Hillstrom, Criteo, and Open Bandit must not be folded into the same verdict row
10. the Open Bandit lane is now real, but the causal-vs-surrogate differentiation question on multi-action data is still unanswered; the next missing ingredient is causal structure (a defensible multi-action prior graph or a principled discovery path for bandit logs), not more plumbing

## Sprint 33 Exit Condition

Sprint 33 is **complete** (PR #183 merged).  The scorecard verdict is GENERALITY IS REAL BUT CONDITIONAL:

1. ERCOT COAST: causal certified > s.o. (p=0.008, 5 seeds) — strongest real-world positive
2. ERCOT NORTH_C: causal trending > s.o. (p=0.059, 5 seeds)
3. causal still does not beat random on either ERCOT dataset
4. Hillstrom pooled B20/B40/B80: certified s.o. advantage under RF fallback
5. Criteo Run 1 + Run 2: combined near-parity under Ax/BoTorch
6. synthetic Ax boundary unchanged since Sprint 29 (medium, high, dose-response certified; base trending; interaction near-parity)
7. Sprint 34: Open Bandit contract / multi-action architecture brief, not another binary marketing rerun

## Sprint 34 Contract Summary

The Sprint 34 Open Bandit contract ([sprint-34-open-bandit-contract.md](../docs/sprint-34-open-bandit-contract.md)) merged as PR #184 and pins seven decisions for the first implementation sprint. Sprint 35 executed all seven on Men/Random:

1. First slice: ZOZOTOWN Men campaign, uniform-random logger (~453K rows, 34 actions, 3 positions, binary click reward) — Sprint 35.C confirmed 452,949 rows on the full slice
2. Adapter: a new `DomainAdapter` subclass parameterizing an item-scoring policy in a 6-to-8 variable search space (softmax temperature, exploration epsilon, context-feature weights, a position-handling flag) — Sprint 35.A landed this as `BanditLogAdapter` with six variables
3. OPE stack: SNIPW primary, DM and DR secondary, DRos deferred — Sprint 35.B landed this
4. Objective: maximize SNIPW-estimated CTR; no revenue, no cost column, no multi-objective
5. Support gates (all in relative terms): null control within 5% relative above the permuted baseline mean, ESS floor `max(1000, n_rows/100)`, zero-support fraction `<= 10%`, propensity-mean sanity band within 10% relative of the schema-dependent target (Sprint 35.A smoke test confirmed the **conditional** `1/n_items` schema), DR/SNIPW cross-check within 25% relative — all five PASS in Sprint 35.C
6. OBP: optional extra dependency with a pinned version; OBP types are hidden behind the adapter boundary — Sprint 35 uses `obp==0.4.1`
7. Sprint 35 shape: three ordered issues -- adapter, OPE stack + gates, first Men/Random benchmark report — delivered as four merged PRs (adapter #189, OPE stack #188, bridge #191, report #192)

## Sprint 35 Exit Condition

Sprint 35 is **complete**. Summary:

1. Open Bandit adapter (`BanditLogAdapter`), OPE stack (SNIPW / DM / DR), Section 7 gates, bridge between adapter and OPE path, and first Men/Random benchmark report are all merged to `main`
2. Men/Random slice: full 452,949 rows, 34 actions, 3 positions, Ax/BoTorch primary, 10 seeds × B20/B40/B80
3. Propensity schema confirmed as conditional `P(item | position) = 1 / 34`
4. All five Section 7 support gates PASS with comfortable margins
5. `causal` vs `surrogate_only`: exact bit-identical tie on every seed (p=1.000 at every budget) — mechanical tie driven by `BanditLogAdapter.get_prior_graph() -> None`
6. `causal` vs `random` and `surrogate_only` vs `random`: both certified (p=0.0002, 10/10 wins, every budget)
7. Sprint 33 closure verdict GENERALITY IS REAL BUT CONDITIONAL carries forward unchanged; the Sprint 35 Open Bandit row is a clean "least valuable but acceptable" first-run outcome per Sprint 34 contract Section 12
8. Sprint 36: multi-action prior graph authoring or discovery for Open Bandit; detailed planning lives in a separate recommendation and prompt document owned by a different track

## Practical Next Step

If resuming from here:

1. read the [Sprint 35 Open Bandit benchmark report](../docs/sprint-35-open-bandit-benchmark-report.md) first — it is the latest evidence and pins the exact shape of the Open Bandit result
2. read the [Sprint 34 Open Bandit contract](../docs/sprint-34-open-bandit-contract.md) for the pinning contract (slice, adapter interface, OPE stack, Section 7 gates, OBP dependency decision)
3. read the [Sprint 33 generalization scorecard](../docs/sprint-33-generalization-scorecard.md) for the synthesized verdict across ERCOT, Hillstrom, and Criteo
4. read the [handoff document](../docs/handoff.md) for the post-Sprint-35 restart instructions
5. do **not** open new Sprint 35 issues — the lane is complete; Sprint 36 detailed planning lives in a separate recommendation and prompt document owned by a different track
6. do not describe the Sprint 35 exact tie as evidence that causal guidance is inert on multi-action data in general — the tie is mechanical (null prior graph), not empirical
7. keep the ERCOT 10-seed rerun on the backlog but do not make it the Sprint 36 critical path
8. do not reopen Hillstrom or Criteo as the Sprint 36 main lane
