## Open Bandit Contract And Multi-Action Architecture Brief

**Date:** 2026-04-18
**Sprint:** 34 (Open Bandit contract / multi-action architecture)
**Issue:** [#182](https://github.com/datablogin/causal-optimizer/issues/182)
**Status:** Contract sprint. No adapter or evaluator code is written in this issue.
**Predecessors:**
- Sprint 31 [Open Bandit access and adapter-gap audit](sprint-31-open-bandit-access-and-gap-audit.md)
- Sprint 31 [Hillstrom lessons learned](sprint-31-hillstrom-lessons-learned.md)
- Sprint 33 [Criteo benchmark report](sprint-33-criteo-benchmark-report.md)
- Sprint 33 [Generalization scorecard](sprint-33-generalization-scorecard.md)
- Sprint 30 [General-causal portability brief](sprint-30-general-causal-portability-brief.md)
- Plan [16 agentic science architecture](../plans/16-agentic-science-architecture.md)
- Plan [07 benchmark state](../plans/07-benchmark-state.md)
- Plan [24 Sprint 34 recommendation](../plans/24-sprint-34-recommendation.md)

## 1. Purpose And Scope

This document is the first executable contract for the next problem class the
causal-optimizer will validate against: **logged multi-action policy data**.
It converts the Sprint 31 Open Bandit audit from a feasibility note into a
contract a future implementation sprint can execute without reopening the
design questions.

The contract settles seven decisions, in order:

1. the first Open Bandit data slice and campaign scope
2. the minimum multi-action adapter interface
3. the minimum OPE stack for the first trustworthy run
4. the benchmark objective and primary reward metric
5. null-control and support diagnostics for the multi-action setting
6. what remains out of scope for the first implementation sprint
7. whether the first implementation depends directly on OBP or stays internal

### What this contract is not

1. not a claim that Open Bandit is a drop-in extension of
   `MarketingLogAdapter`. It is not. The Sprint 31 audit already ruled that out
   and this contract preserves that boundary.
2. not a relitigation of Hillstrom or Criteo. Sprint 33 closed those lanes.
   This contract builds on the result, not around it.
3. not an implementation spec for any component. It is a contract for the
   shapes, the gates, and the first believable run.
4. not a promise to pass a benchmark. It is a promise to produce a first
   honest, diagnosable run on a new problem class.

## 2. Framing And Boundary Conditions

### 2a. Why Open Bandit is the next frontier, not a third binary marketing rerun

After Hillstrom (RF fallback, pooled slice certified surrogate-only advantage)
and Criteo (Ax/BoTorch, near-parity on both the degenerate and heterogeneous
runs), the narrow binary-marketing question is now empirically answered at the
scope the project can currently test: under the current adapter contract, on
two real marketing datasets with 2-to-3 active variables, causal guidance does
not automatically beat `surrogate_only`. A third binary uplift rerun would
mostly re-test the same question. Open Bandit is the first intervention
structure in the benchmark queue that the current engine cannot already encode.

### 2b. What changes structurally versus what does not

The domain-portable core engine (`ExperimentEngine`, `suggest_parameters`,
`EffectEstimator`, `OffPolicyPredictor`, `ScreeningDesigner`, `MAPElites`,
`GraphLearner`, `BenchmarkRunner`, provenance) does not need to change for this
contract. What needs to exist is a new intervention structure at the adapter
and evaluator boundary: a categorical action drawn from a logged multi-action
policy, an inverse-propensity estimator that handles many actions with small
propensities, and a search space that parameterizes an item-scoring policy
rather than a binary threshold rule.

### 2c. Evidence standards that carry over unchanged

1. 10 seeds, per-seed provenance, MWU two-sided tests, the "certified /
   trending / not significant" labels from the Sprint 33 scorecard.
2. Backend provenance preserved: Ax/BoTorch and RF fallback are not mixed in a
   single verdict row.
3. Null control is a first-class pass/fail gate, not a nice-to-have.
4. The final report separates observed policy value from causal attribution,
   and the optimizer-path provenance is recorded on every run.

### 2d. What a Sprint 34 implementation sprint is allowed to claim

A first honest Open Bandit run does not need to produce a causal win. It needs
to produce:

1. a reproducible multi-action benchmark artifact under a trustworthy OPE
   estimator
2. a clean null control
3. a support-diagnostic pass that shows the estimator is not silently variance
   pathological
4. a backend-matched verdict row on one campaign-policy slice

That is sufficient to add Open Bandit as a real non-energy lane in the
benchmark portfolio, independent of whether causal beats `surrogate_only` on
the first slice.

## 3. First Data Slice And Campaign Scope

### 3a. First slice: ZOZOTOWN Men campaign, logged uniform-random policy

The first Open Bandit run should use **one** slice. Specifically:

| Decision | Choice |
|----------|--------|
| Campaign | Men |
| Logging policy | Uniform Random |
| Rows (nominal) | ~452,949 |
| Action space | 34 distinct items |
| Positions | 3 (left, center, right) |
| Reward | binary click |
| Propensity source | true logged propensities in the `action_prob` column |

Why this slice:

1. **Smallest honest footprint.** Men/Random is the smallest campaign-policy
   slice that still has real logged propensities. Women/Random is larger at
   ~865K rows; All/Random is larger still at ~1.37M rows. The first run must
   be cheap enough that the benchmark can iterate, and Men/Random is the
   cheapest honest option.
2. **Uniform Random propensities are the cleanest starting point.** Under
   Random, per-action propensity is approximately `1/n_items`. That is small
   (`~1/34 ≈ 0.029`) but uniform, so it avoids the additional variance burden
   of Bernoulli Thompson Sampling, which concentrates probability mass on
   high-performing items and pushes low-probability actions toward the
   structural positivity limit.
3. **34 actions is the first honest multi-action count.** It is large enough
   that the problem is structurally different from binary treatment, small
   enough that a reasonable item-scoring policy can be parameterized without
   embedding tricks in the first run.
4. **One campaign removes cross-campaign aggregation risk.** The three
   campaigns (All / Men / Women) are independent experiments on different item
   pools (Saito et al. 2021, Table 1). They are not nested subsets. Mixing
   them in one verdict would be a category error. The first run picks one.

### 3b. Optional evaluator-side addition

If the first implementation sprint completes Men/Random within its time box, a
second slice — Men/BTS on the same item pool — can be added as a secondary
evaluator check, held to exactly the same gates. BTS is included here only as
a "distribution shift" sanity check, not as the primary result.

The first benchmark verdict must be reported on Men/Random alone. Men/BTS is
optional and additive.

### 3c. Subsampling and caching

For development, a deterministic row subsample (seed-locked, same Criteo-style
convention: `DataFrame.sample(n=..., random_state=<fixed_seed>)`) is
acceptable. The first benchmark **report** must run on the full Men/Random
slice. Subsampling is acceptable only for smoke tests and CI.

No fixture is committed to the repo. The full data comes from the OBP loader
(or equivalent direct download path); the repo ships only a provenance record
(source URL, content hash, row count, column count) that future runs must
reproduce.

### 3d. What is explicitly deferred

1. the Women and All campaigns
2. any cross-campaign aggregation
3. Bernoulli Thompson Sampling as a primary logging policy
4. any policy-over-policy counterfactual (e.g., evaluating BTS under a Random
   logger) that is not required for the first verdict row

## 4. Minimum Multi-Action Adapter Interface

### 4a. Position within the existing adapter hierarchy

The first implementation should **not** subclass `MarketingLogAdapter`. The
binary treatment contract there is load-bearing (the 0/1 check, the
`1/p(T=1)` and `1/(1-p)` IPS formulas, the threshold-plus-budget policy rule,
the cost column). Reshaping it to multi-action would hide the structural
break.

Two acceptable placements:

1. a new sibling adapter class (e.g. `BanditLogAdapter`) at
   `causal_optimizer/domain_adapters/bandit_log.py`, implementing
   `DomainAdapter`
2. or, a shared abstract base (`MultiActionLogAdapter`) alongside
   `DomainAdapter`, if the project decides to add a second multi-action
   dataset later in the same sprint group

The first implementation sprint should pick one. The **contract** only
requires that the class implements `DomainAdapter` directly and does not
subclass `MarketingLogAdapter`.

### 4b. Required interface methods

At minimum, the adapter must implement:

| Method | Contract |
|--------|----------|
| `get_search_space` | returns a `SearchSpace` over parameters of an **item-scoring** policy (see Section 4c) |
| `run_experiment(parameters)` | computes and returns `{"policy_value": ..., ...diagnostics}` using an OPE estimator over the logged data |
| `get_prior_graph` | may return `None` in the first run. A minimal prior graph is optional, not required. |
| `get_objective_name` | returns `"policy_value"` (see Section 5) |
| `get_minimize` | returns `False` (maximize expected CTR) |
| `get_strategy` | returns `"bayesian"` (unchanged from default) |

### 4c. Search space: first-run parameterization

The search space should parameterize a **contextual item-scoring policy**
whose output is a distribution over actions. The first run should use a
narrow, honest parameterization:

1. **Softmax temperature** `tau` (continuous, bounded, e.g. `[0.1, 10.0]`):
   controls how sharply the evaluation policy peaks on its highest-scoring
   action.
2. **Exploration epsilon** `eps` (continuous, `[0.0, 0.5]`): probability of
   falling back to uniform over all actions, used both to prevent structural
   zero propensity under the evaluation policy and to regularize the softmax.
3. **Context-feature weights** (a small, fixed number — e.g. 3 to 5
   continuous variables, not the full context vector): linear weights on a
   chosen subset of user or user-item affinity features that the scoring
   policy uses to rank items.
4. **Position handling flag** (categorical: `"marginalize"` or
   `"position_1_only"`): determines whether the policy is evaluated across
   all three positions (marginal reward) or restricted to a single position.
   The first run should default to `"position_1_only"` to avoid position bias
   as a hidden confounder in the first verdict; `"marginalize"` is available
   as a secondary run.

What the contract does **not** require for the first run:

1. action embeddings
2. per-item scoring parameters (34 items × weights would blow up the search
   dimensionality; use a fixed low-dimensional feature-weighted scoring
   function instead)
3. learned positional weighting
4. neural or tree-based scoring models

The first run must stay parameterizable inside a six-to-nine variable search
space. That is the same order of magnitude as the existing marketing search
spaces and is believable under `suggest_parameters()` without new optimizer
work.

### 4d. What `run_experiment` must return

For each parameterization, `run_experiment` must return a dict containing:

| Key | Meaning |
|-----|---------|
| `policy_value` | Primary objective: OPE-estimated CTR of the evaluation policy. See Section 5. |
| `ess` | Effective sample size of the IPS weights over the logged data. |
| `weight_cv` | Coefficient of variation of the IPS weights. |
| `max_weight` | Maximum IPS weight observed in this policy evaluation. |
| `zero_support_fraction` | Fraction of logged rows where the evaluation policy assigns strictly positive probability to the logged action. Must be reported, not silently filtered. |
| `n_effective_actions` | Number of distinct actions the evaluation policy actually places non-negligible mass on (e.g., top-k items whose combined policy mass exceeds 0.95). |

These diagnostics mirror the Criteo contract's IPS-stability fields. They are
not optional. A run that omits any of them does not satisfy this contract.

### 4e. Causal graph: optional, minimal, or deferred

The Sprint 31 audit lists position as a confounder candidate and item features
(price, brand, category) as potential covariates. A prior causal graph is
**optional** for the first run. The first implementation sprint may return
`None` from `get_prior_graph` and let the engine run without a causal graph.

If a prior graph is authored, it must follow the existing semantics: nodes are
either search-space variables or outcomes, and edges must be justified from
the dataset documentation, not from surface intuition. A bad graph is worse
than no graph; the default for the first run is "no graph authored yet."

This contract does not require graph authoring in the first sprint.

## 5. Minimum OPE Stack For The First Trustworthy Run

### 5a. Estimator choice: SNIPW primary, DR secondary, DRos on the shortlist

| Estimator | Role in the first run |
|-----------|-----------------------|
| Vanilla IPW | Reported as a diagnostic only. Variance-pathological at small propensities. Not the headline. |
| Self-Normalized IPW (SNIPW) | **Primary estimator for the first verdict row.** Standard for OBD at scale, variance-bounded, honest under uniform-random logging. |
| Direct Method (DM) | Reported as a secondary sanity check. Expected to be biased; used to detect extreme reward-model failure. |
| Doubly Robust (DR) | Reported as a secondary estimator. First run may defer implementing its own DR and use OBP's DR wrapper if the OBP dependency is accepted (see Section 7). |
| DRos (Doubly Robust with Optimistic Shrinkage) | On the shortlist for Sprint 35. Not required for the first verdict row. The OBP paper (Saito et al. 2021) shows DRos outperforms SNIPW by 30-60% on OBD, but it is not required to make the first run trustworthy. |

The first benchmark **verdict** is quoted against SNIPW. DM and DR are
reported for cross-estimator stability, not as the headline.

### 5b. Why SNIPW is the first-run primary

1. It is the cheapest estimator that is robust enough to be trustworthy on
   OBD (per the OBP paper and prior audit).
2. It is a direct generalization of the self-normalized IPW path already used
   in `MarketingLogAdapter`, so it is easiest to review.
3. It makes no assumption about the reward model being correct, so it cannot
   silently manufacture a causal story via a miscalibrated DM component.
4. It is well-understood by the benchmark reviewers; a first-run SNIPW number
   is interpretable even if downstream estimators are added in later sprints.

### 5c. Propensity clipping policy

SNIPW still needs a floor on the logged-policy propensity when the logger
never sampled certain (action, context) pairs. The first run must:

1. clip logged propensities from below at a fixed floor `min_propensity_clip`
   (default `1 / (n_items * 3)` for Men/Random, i.e. roughly
   `1 / (34 * 3) ≈ 0.0098`, matching the expected Random propensity per
   position).
2. freeze `min_propensity_clip` in the first run (not tuned by the
   optimizer). This mirrors the Criteo contract Section 5 decision.
3. report the **number of clipped rows** as a diagnostic.

The first benchmark may revisit the clip threshold only if Run 1 post-hoc
shows that >5% of rows are clipped; that would be a support-failure signal
and would trigger a Run 2 with a narrower evaluation policy, not a clip
tuning pass.

### 5d. Bootstrap and confidence intervals

Per-seed bootstrap is optional in Run 1. The primary variance control comes
from the 10-seed distribution of final policy values, the same way every
prior benchmark reports. An estimator-internal bootstrap (OBP offers this)
is allowed if it does not break the 10-seed-per-cell budget, but it is not
required.

### 5e. What is deferred to Sprint 35 or later

1. DRos, Switch-DR, Continuous-Action IPW, MIPS, and embedding-based OPE
2. full cross-estimator regression gates (OBP provides several; only SNIPW
   gates the first verdict)
3. counterfactual ranking / slate-level OPE
4. any BTS-as-logger evaluation
5. per-action variance-reduction tricks beyond self-normalization and
   clipping

## 6. Benchmark Objective And Primary Reward Metric

### 6a. Single primary objective

The Sprint 34 contract pins one primary objective:

**Objective:** maximize SNIPW-estimated expected click-through rate
(`policy_value = CTR_hat_SNIPW`).

Direction: maximize. Units: probability of click per exposure (bounded in
`[0, 1]`). Typical values on OBD Men/Random are in the `0.005` range.

### 6b. Secondary reported metrics

The benchmark report must additionally report:

1. Vanilla IPW CTR (diagnostic only; not used for verdicts)
2. DM CTR (diagnostic only; expected to be biased)
3. DR CTR if OBP is used (secondary sanity check)
4. Random-policy CTR under the logged data as a null reference
5. BTS-logger-policy CTR as a reference point (already published in Saito et
   al. 2021, Table 1). Used as a comparability check, not as a gate.

### 6c. No revenue or conversion metric

OBD does not carry a revenue or conversion signal. The contract must not
invent one. Any future multi-objective work on this dataset is out of scope.

### 6d. No cost column

OBD does not have a per-observation cost column. The contract must not bolt
on a fake one. `total_cost` from `MarketingLogAdapter` has no analogue here
and should be absent from the adapter return value.

### 6e. Verdict rule

The first-run verdict is quoted on Men/Random under SNIPW at budget B80, as a
two-sided MWU test of `causal` vs `surrogate_only` policy values across 10
seeds. The Sprint 33 classification labels apply unchanged:

1. "certified" at `p <= 0.05`
2. "trending" at `0.05 < p <= 0.15`
3. "not significant" at `p > 0.15`
4. "near-parity" when the two strategies produce within-noise identical
   distributions

Secondary comparisons (`causal` vs `random`, `surrogate_only` vs `random`) are
reported but are not the Sprint 34 verdict gate.

## 7. Null-Control And Support Diagnostics For Multi-Action Data

Multi-action OPE needs more gates than binary IPS. The first run must pass
all four before a verdict is claimed.

### 7a. Null-control gate

Permute the logged reward column within each row (not across rows, to
preserve context-action joint structure), then rerun the full strategy sweep
on the permuted dataset. The null-control pass requires that no strategy
produces a policy value more than **5 percentage points** above the permuted
baseline mean for more than one of the three strategies at any budget.

The 5 percentage-point band mirrors the Criteo contract convention
(translated into SNIPW-CTR units rather than visit-rate units).

If null control fails, the benchmark must be rejected and the report must
document which strategy inflated on permuted outcomes.

### 7b. Support / effective sample size gate

Across the 10 seeds, the median ESS of the IPS weights under the optimized
policies must be at least `max(1000, n_rows / 100)`. For Men/Random
(~453K rows), that is ~4,530 as the ESS floor. A run whose optimized
policies' median ESS falls below this threshold is a support-failure row, and
its verdict must be quoted with an explicit support-weakness caveat.

This threshold is deliberately conservative for the first run. Tightening it
is a Sprint 35 conversation, not a Sprint 34 one.

### 7c. Zero-support fraction gate

For every evaluated policy, the adapter must compute the fraction of logged
rows for which the evaluation policy places **structurally zero** probability
on the logged action (i.e., the policy would never have taken that action
under any exploration). The gate fails if this fraction exceeds **10%** for
the best-of-seed policy at any budget. Rows with `eps > 0` should have zero
zero-support in expectation; a failure here points at a search-space or
parameterization bug, not a data bug.

### 7d. Propensity and policy-mass sanity gate

On Men/Random, the empirical mean propensity in the logged data should be
within 2 percentage points of `1/34 ≈ 0.0294` per (position, item) slot. A
deviation outside this band suggests the loader, slice, or subsampling has
contaminated the logged propensity and the slice should be re-loaded before
the benchmark is rerun.

### 7e. Per-estimator cross-check

If the DR / OBP path is available, DR and SNIPW must not diverge by more than
**25% relative** on the optimized policy values for any seed. A larger
divergence means one of the two estimators is unstable on this slice; the
report must flag the seed and defer the verdict until the cause is
identified.

### 7f. Backend recording

Every verdict cell must record its optimizer path (Ax/BoTorch vs RF fallback)
in the provenance file. Per Sprint 28 and Sprint 33, results from different
backends are not mixed in one verdict row.

## 8. OBP Dependency Decision

### 8a. Decision: first implementation depends on OBP for the OPE stack

The first implementation sprint **should** depend directly on Open Bandit
Pipeline (OBP, `pip install obp`, Apache 2.0) for:

1. the data loader (`OpenBanditDataset`)
2. the SNIPW estimator and the DM / DR estimators used for secondary
   diagnostics
3. the existing bandit-feedback dict schema

The adapter code must still live in the causal-optimizer repository and
conform to `DomainAdapter`. OBP is used as a library underneath; its
estimator outputs feed `policy_value` and the diagnostic fields.

### 8b. Why this is the right call for the first run

1. OBP's estimators are the published reference implementations on OBD,
   validated in Saito et al. 2021 and the follow-up OBP papers. The project
   cannot outperform them in the first implementation sprint without burning
   the sprint on OPE reimplementation.
2. The first run is about validating the **engine on a new problem class**,
   not about validating a new OPE estimator. Reusing OBP isolates the
   variable the contract cares about.
3. OBP is Apache 2.0, does not bundle the dataset, and can be added as an
   optional extra (`uv sync --extra bandit` or similar) so the core package
   does not inherit a new hard dependency.
4. OBP's output format (the bandit-feedback dict) is stable and well-audited,
   which makes the adapter's shape easy to review.

### 8c. What the contract requires about the OBP integration

1. OBP must be an optional extra, not a core dependency. The core
   causal-optimizer install must still work without OBP.
2. The adapter must fail fast with a clear error if OBP is missing.
3. The adapter must not expose OBP types at its public interface. The
   adapter accepts policy parameters and returns the dict from Section 4d;
   OBP internals are hidden.
4. The OBP version used for the verdict must be pinned in the provenance
   record.
5. The run must remain reproducible under a frozen OBP version: upgrading
   OBP is a Sprint 35 decision, not a silent one.

### 8d. When to re-open this decision

1. if OBP becomes unmaintained or a licensing question changes
2. if the project ever wants to claim a **novel** OPE estimator on OBD (then
   reimplementing it in-house becomes the point of the sprint)
3. if the first run needs an estimator OBP does not implement (unlikely for
   Sprint 35, since DRos, Switch-DR, and MIPS are all in OBP)

## 9. Out Of Scope For The First Implementation Sprint

In order of "most likely to get tempting during implementation":

1. Women and All campaigns; cross-campaign aggregation
2. Bernoulli Thompson Sampling as the primary logger
3. Online learning, bandit training from scratch, or any non-offline
   evaluation path
4. Slate-level / ranking-aware OPE
5. Action embeddings, MIPS, and Switch-DR
6. Continuous-action OPE
7. Multi-objective optimization (click + revenue, click + diversity, etc.)
8. A deep or tree-based item-scoring model inside `suggest_parameters()`
9. Position-bias causal modeling beyond "marginalize" vs "position_1_only"
10. Auto-discovered causal graphs on OBD (Sprint 35+ conversation)
11. A second dataset (e.g., MovieLens, Outbrain, Yahoo! R6)
12. Any online-decisioning or A/B-test claim

Each of these is a legitimate research direction. None are required for the
first honest Sprint 35 run, and bundling any of them into Sprint 35 would
blur the contract.

## 10. Suggested Implementation Sprint Shape

Sprint 34 delivers this contract. The recommended Sprint 35 issue shape is:

1. **Sprint 35 Issue A:** implement `BanditLogAdapter` (or chosen name) with
   the Section 4 interface, wired to OBP under the optional extra.
2. **Sprint 35 Issue B:** implement the multi-action OPE stack wrapper
   (SNIPW primary, DM and DR secondary) and the Section 7 diagnostic gates.
3. **Sprint 35 Issue C:** run the Men/Random benchmark at 10 seeds x
   B20/B40/B80, run the null control, produce the Sprint 35 Open Bandit
   benchmark report under the same format as the Criteo report.

Each issue is one PR, reviewed independently. Issue A can land before Issues
B and C; Issue B can be partially tested against a synthetic bandit feedback
dict before Issue C consumes it.

No implementation issue should open until this contract is merged.

## 11. Success Criteria For Sprint 34

Sprint 34 is successful if:

1. this contract document is merged
2. every Section 3-8 decision is explicit and a future agent can act on it
   without asking follow-up questions
3. the handoff and benchmark-state docs reflect Open Bandit as the next
   active research lane
4. the first Sprint 35 implementation issue can be opened after this contract
   merges without unresolved architectural ambiguity
5. Sprint 34 does not re-litigate Hillstrom or Criteo

Sprint 34 is **not** successful if:

1. the contract mixes architecture design with implementation guesses not
   validated against the OBP API
2. the contract silently depends on `MarketingLogAdapter`
3. the contract leaves the OBP-dependency question open
4. the contract expands scope beyond one slice, one estimator, and one
   verdict row

## 12. What A Good Sprint 35 Outcome Looks Like

Best case: the Men/Random run produces a clean SNIPW verdict with all four
support gates green, and the verdict row is a new non-energy,
non-binary-treatment benchmark entry. Direction of the verdict is secondary;
presence of a clean, trustworthy row is primary.

Still valuable: the first run fails one gate (e.g. ESS below the Section 7b
floor, or zero-support fraction above the Section 7c bound). That is a
well-specified blocker that tells Sprint 36 exactly what to fix, and it is
strictly better than an ambiguous tie.

Least valuable but acceptable: the Men/Random run is clean on diagnostics but
shows no separation between `causal` and `surrogate_only` under SNIPW. That
is a useful data point in the generalization scorecard and carries the same
weight as the Criteo near-parity row.

## 13. Status After This Contract

Sprint 33 result: GENERALITY IS REAL BUT CONDITIONAL. That is unchanged.

Sprint 34 result (after this contract merges): the project has a first
executable contract for logged multi-action policy data and a scoped plan for
the first Open Bandit run. Open Bandit is the active frontier; Hillstrom and
Criteo are not reopened.

Sprint 35 trigger: merging this contract. Sprint 35 should open Issues A, B,
and C in Section 10.

## 14. Statistical And Reporting Conventions

Inherited unchanged from the Sprint 33 scorecard:

1. all p-values are two-sided Mann-Whitney U unless otherwise noted
2. population std (ddof=0) in tables; sample-pooled std (ddof=1) is used
   when Cohen's d is quoted in source reports
3. "certified" = `p <= 0.05`; "trending" = `0.05 < p <= 0.15`; "not
   significant" = `p > 0.15`
4. "winner" is reserved for statistically significant rows; directional
   improvements without significance are "trending" or "mean improved"
5. backend provenance is preserved; Ax/BoTorch and RF fallback verdicts are
   not mixed in a single row

## 15. Attribution

The Open Bandit Dataset and Pipeline are owned and published by ZOZO, Inc.
The reference paper is Saito, Aihara, Matsutani, and Narita, "Open Bandit
Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy
Evaluation," NeurIPS Datasets and Benchmarks 2021
(arXiv:2008.07146). Any merged Sprint 35 benchmark artifact must cite this
paper and acknowledge ZOZO as the dataset publisher.

License terms from the Sprint 31 audit carry over: the dataset is CC BY 4.0,
the OBP library is Apache 2.0, and the project redistributes neither.
