# Sprint 36 Recommendation

Updated: 2026-04-21

## Sprint Theme

**Preregister the first Open Bandit prior graph and the engine-surface
change that would make it matter. Defer the rerun.**

Sprint 35 shipped the first Men/Random benchmark. The verdict was clean:
all five Section 7 gates passed, both optimized strategies certified over
`random` at every budget, and `causal` vs `surrogate_only` produced a
bit-identical tie on every seed at every budget (p = 1.000 at B20, B40,
B80). The tie is mechanical — `BanditLogAdapter.get_prior_graph()`
returns `None`, so the `causal` path has no graph to exploit and reduces
to `surrogate_only` on this slice.

An earlier draft of Sprint 36 proposed authoring a minimal, semantically
correct graph and rerunning Men/Random under it. Human review (PR #195,
2026-04-21) pointed out the self-contradiction: the same document
argued the proposed 7-node / 6-edge graph makes every search variable
an ancestor of `policy_value`, so `_get_focus_variables()` returns the
full search space and behavior is predicted to match the no-graph case,
*and* forbade any engine or optimizer change that could make the graph
matter. Framed that way, the sprint was an expensive rerun of a result
the plan already predicted would be a no-op (or near-no-op) under the
present engine surface.

Sprint 36 is therefore rescoped to a **docs-only preregistration** that
(a) authors the minimal graph as a preregistered spec, (b) documents
exactly how the current engine would consume it, (c) identifies where
the plan's earlier "pure no-op" framing was imprecise, and (d) names
one small engine-surface change or one graph widening that would make
a Sprint 37 rerun able to move the `causal` vs `surrogate_only`
comparison. No Python change to `BanditLogAdapter`, no rerun, no new
verdict row.

## Goal

Sprint 36 should end with:

1. a merged preregistration document (this plan) that names the
   minimal graph edge by edge from the adapter's scoring code, not
   from post-hoc convergence noise
2. an engine-surface analysis that enumerates every place
   `causal_optimizer/optimizer/suggest.py` reads `causal_graph` and
   describes how each path behaves when every search variable is an
   ancestor of the objective (including the soft-causal reranker path
   at `suggest.py:751` that an earlier draft under-weighted)
3. a falsifiable exit criterion that names the one small change
   Sprint 37 should make — either (a) a specific engine-surface
   modification that gives `causal` a real mechanical advantage over
   `surrogate_only` when every search variable is an ancestor, or
   (b) a specific, code-grounded reason to widen the graph beyond
   ancestors-of-policy-value (e.g., a non-ancestor structural node)
   — and states the preregistered H0/H1/H2 that Sprint 37's rerun
   would test

The sprint must not fabricate edges, invent a richer graph to chase
differentiation, ship a Python change to `BanditLogAdapter`, or run
the benchmark. A bad graph or a premature rerun is worse than no graph.

## Why This Sprint Now

The Sprint 35 tie was exact across all 30 seed × budget cells. That is
information. It constrains the space of graphs and engine changes that
could plausibly have produced a different result, and the review
comment on PR #195 is correct that walking that constraint forward
*before* spending a benchmark run is the higher-value move.

The engine currently reads `self.causal_graph` in the following places
inside `causal_optimizer/optimizer/suggest.py` (line numbers against
`HEAD` of `sprint-36/open-bandit-prior-graph-plan`):

1. `_get_focus_variables` (lines 1122–1134). Returns
   `[v for v in search_space.variable_names if v in causal_graph.ancestors(objective_name)]`,
   falling back to the full search space if that list is empty. When
   every search variable is an ancestor of `objective_name`, this list
   equals the no-graph case.
2. `_suggest_optimization` screening intersection (lines 425–437).
   Combines `graph_focus` with the screening result. When
   `graph_focus == search_space.variable_names`, the intersection
   equals `screened_variables`, matching the no-graph case.
3. `_suggest_exploitation` parent-weighted perturbation (lines 569–586).
   Uses `causal_graph.parents(objective_name)` to bias the random choice
   of which variable to perturb. The weighting activates only when
   `parent_focus` is a *proper* subset of `eligible_vars` (line 575:
   `if parent_focus and len(parent_focus) < len(eligible_vars)`). When
   every search variable is a parent, `weights = None` and the behavior
   matches the no-graph case.
4. Soft-causal reranker in `_suggest_bayesian` (lines 751–812). Here
   `use_soft = causal_graph is not None and causal_softness <
   _HARD_FOCUS_THRESHOLD`. The engine default
   (`engine/loop.py:124`) sets `causal_softness = 0.5` and
   `_HARD_FOCUS_THRESHOLD = 1e5` (`suggest.py:77`), so passing a graph
   silently turns soft mode on. Soft mode generates 5 Ax candidates,
   injects categorical diversity, and reranks by
   `_causal_alignment_score` against the best-known params, averaged
   over all ancestor variables. The score is
   `mean(|c_norm - b_norm|)` across ancestors
   (`suggest.py:982–1021` for the full function, including the
   `_normalize_value` call that makes the two-candidate example
   below rescale `tau` onto `[0, 1]` before differencing; the
   per-ancestor inner loop at lines 1010–1019 appends
   `abs(c_norm - b_norm)` and line 1021 returns
   `float(np.mean(diffs))`), and `_rerank_alignment_only`
   picks the candidate with the **highest** score
   (`suggest.py:840–842`: `if adjusted > best_score`) — i.e. the one
   whose normalized ancestor-parameter vector is *farthest* from the
   best-known params, not closest. Despite the name "alignment", the
   reranker is an exploration-biased distance-from-best selector.
   The derivation is load-bearing for the Sprint 37 hypothesis
   direction, so for a two-candidate sanity check: if `best_params`
   has `tau = 0.2` and two Ax candidates propose `tau = 0.21` and
   `tau = 0.28` (with every other ancestor equal), the per-variable
   normalized displacements are `|0.01 / range|` and
   `|0.08 / range|`, the means on the ancestor set are strictly
   ordered the same way, and `adjusted > best_score` at
   `suggest.py:840` picks the second (`tau = 0.28`) candidate — the
   one *farther* from `best_params`. A Sprint 37 unit test that
   replays this two-candidate case against `_rerank_alignment_only`
   would fail immediately if the sign convention is ever flipped.
   With every search variable as an ancestor, this reranker ranks
   across the full space — **not** a pure no-op.
5. Soft-causal path in `_suggest_surrogate` (lines 883–979). The RF
   fallback uses the same alignment-based reranking. Under the Section 7
   no-RF-fallback rule, this path should not execute on the Sprint 37
   rerun, but it is reachable if Ax fails to import and must therefore
   be covered by the engine-surface analysis.
6. `causal_exploration_weight` LHS bias (lines 246–280). Has been
   pinned to `0.0` since Sprint 29 (PR #160), so the ancestor-bias
   candidate-replacement path is inert under the production default.
   The Sprint 37 prediction assumes this pin stays at `0.0`; a future
   PR that changes that default would require revisiting the
   prediction.
7. `_suggest_causal_gp` routing (lines 450–458; guard at line 451,
   call at line 458). Activated only when `strategy == "causal_gp"`
   is explicitly requested. The benchmark `causal` strategy goes
   through `_suggest_bayesian`, not `_suggest_causal_gp`, so this
   path is out of scope for the Sprint 37 rerun.

The earlier Sprint 36 draft leaned on path 1 and predicted a pure
no-op ("this honest minimal graph should not separate `causal` from
`surrogate_only`"). That prediction was subtly wrong: path 4 is
active under the default engine config, and its alignment reranker
*does* use the graph — it just happened to produce a bit-identical
tie in Sprint 35 because, empirically, the branches on each side of
`use_soft` generate a single winning candidate under the Sprint 35
seed sequence. In the no-graph case (`use_soft = False`) Ax returns
exactly one candidate and the reranker is skipped entirely
(`suggest.py:771–773`); in the with-graph case (`use_soft = True`)
Ax generates five candidates and `_rerank_alignment_only` picks one,
but with an empty history and matching seeds the five-candidate pool
and the one-candidate call in the control arm select the same
parameter dict. That empirical coincidence is not guaranteed under a
non-trivial graph, and it is not the kind of prediction Sprint 36
should stake a benchmark rerun on.

Running Men/Random again with the minimal graph wired in therefore
does not answer a well-posed question. Either the tie persists (weakly
interesting — confirms the empirical coincidence but does not tell us
which of paths 1–5 was responsible) or it breaks (the rerun then
needs a diagnostic harness Sprint 36 has not specified to attribute
the break to a specific code path). Preregistering the graph, the
engine-surface analysis, and a single candidate Sprint 37 change is
the scope that pays for itself inside one PR.

## Minimal Preregistered Graph

The spec below is authored here, not in `bandit_log.py`. Sprint 37
decides whether to land it as code (and under what engine change).

### Nodes (7)

Search-space variables (exactly the six variables declared in
`BanditLogAdapter.get_search_space`,
`causal_optimizer/domain_adapters/bandit_log.py` lines 316–346):

1. `tau`
2. `eps`
3. `w_item_feature_0`
4. `w_user_item_affinity`
5. `w_item_popularity`
6. `position_handling_flag`

Outcome node (the primary objective; Sprint 34 contract Section 5a,
and the name returned by
`BanditLogAdapter.get_objective_name()` in
`causal_optimizer/domain_adapters/bandit_log.py` line 526):

7. `policy_value`

### Directed edges (6)

Every search variable → `policy_value`. Each edge has a one-line code
justification (line numbers are against `HEAD` of this branch):

| Edge | Code site | Justification |
|------|-----------|---------------|
| `tau → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` line 393 (`scaled = scores / safe_tau`) | `tau` is the softmax temperature; it scales every score directly into the policy that SNIPW evaluates. |
| `eps → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` line 398 (`policy = (1.0 - eps) * softmax + eps * uniform`) | `eps` linearly mixes the softmax and the uniform fallback into the evaluation policy. |
| `w_item_feature_0 → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` line 386 (`item_term = w_item * self._item_feature_0[None, :]`) | Per-item continuous feature weight; a summand in `scores` which feeds the softmax. |
| `w_user_item_affinity → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` line 388 (`affinity_term = w_affinity * self._affinity[mask]`) | Per-row, per-candidate affinity weight; a summand in `scores`. |
| `w_item_popularity → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` line 387 (`pop_term = w_popularity * self._item_popularity[None, :]`) | Popularity prior weight; a summand in `scores`. |
| `position_handling_flag → policy_value` | `causal_optimizer/domain_adapters/bandit_log.py` lines 366–369 (row mask on `self._position == 0` vs all rows) | Controls which logged rows enter the SNIPW sum; the flag changes both the policy value and the number of rows `n_active`. |

### Bidirected edges (none)

The six search-space variables are independently sampled by the
optimizer — they have no shared latent cause, they are knobs. The
coupling among the three IPS-weight diagnostics (`ess`, `weight_cv`,
`max_weight`) is a shared computation, not an unobserved confounder;
modeling it as a bidirected edge would misrepresent the code.
Bidirected edges from auto-discovery are explicitly "heuristic proxies,
not formally identified confounders" (CLAUDE.md discovery notes,
Sprint 34 contract Section 4e). Without bidirected edges POMIS
collapses to the trivial case. The derivation, step by step against
`causal_optimizer/optimizer/pomis.py`:

1. With no bidirected edges, every node's c-component is a singleton
   (`types.py:214`).
2. `_muct` (`pomis.py:59–124`) starts from `{outcome}`, takes the
   c-component of each frontier node (a singleton here, so no
   expansion), and adds descendants-of-c-component within the
   ancestral subgraph. Descendants of `policy_value` inside
   `An(policy_value)` ∪ `{policy_value}` is the empty set, so the
   frontier never grows. Therefore `MUCT = {policy_value}`.
3. `_interventional_border` (`pomis.py:127–132`) returns
   `parents(MUCT) - MUCT = parents(policy_value) - {policy_value}`,
   which under the preregistered graph is exactly the six search
   variables (every search variable has a direct edge to
   `policy_value`). So the interventional border equals the full
   search space.
4. `compute_pomis` (`pomis.py:17–56`) adds `frozenset(ib)` to the
   result at line 54 — i.e. the full-search-space frozenset.
   The POMIS list is therefore a single element equal to the full
   search space.

This is the "trivial case" in that the sole POMIS is the full search
space, but the route is `IB = parents(Y)`, **not** `MUCT = An(Y)`.
The same conclusion (POMIS does not restrict the search) holds, and
it is the reason Option A in the Exit Criterion section must pick
between A1 (a non-POMIS minimality heuristic) and A2 (a graph
widening that introduces bidirected edges or non-ancestor
structural nodes so POMIS itself begins to restrict — e.g. by
shrinking IB below the full variable set).

### Why the diagnostic outcomes are not nodes

`ess`, `weight_cv`, `max_weight`, `zero_support_fraction`, and
`n_effective_actions` are returned by `run_experiment`
(`causal_optimizer/domain_adapters/bandit_log.py` lines 470–477) but
they are not the objective and the engine's `_get_focus_variables`
only dispatches on ancestors of `objective_name`. Adding these as
nodes in the first graph would add edges that the current engine
never reads. Keeping the first graph to seven nodes mirrors the
discipline Sprint 32 applied when it shrank `MarketingLogAdapter`'s
prior graph to the Criteo projected 5-edge graph
(`causal_optimizer/benchmarks/criteo.py` lines 220–237).

### What the engine would do with this graph, path by path

Under the Sprint 37 rerun assumptions (`causal_softness = 0.5`,
`causal_exploration_weight = 0.0`, `strategy != "causal_gp"`, Ax
available):

- Path 1 (`_get_focus_variables`): returns the full search space →
  identical to no-graph.
- Path 2 (screening intersection): identical to no-graph.
- Path 3 (parent-weighted exploitation perturbation): `weights = None`
  → identical to no-graph.
- Path 4 (soft-causal Bayesian reranker): **active and not a no-op.**
  Generates 5 Ax candidates, injects categorical diversity, reranks
  by `_causal_alignment_score` averaged across all six ancestors
  (`suggest.py:806–812`). Picks the candidate whose normalized
  parameter vector is farthest from the best-known params. With the
  graph absent (`causal_graph is None`), `use_soft = False` at
  `suggest.py:751`, Ax returns one candidate, reranking is skipped
  entirely, and categorical diversity is not injected.
- Path 5 (RF fallback soft-causal): not reached when Ax is available
  and the no-RF-fallback rule holds.
- Path 6 (`causal_exploration_weight`): inert (pinned to `0.0`).
- Path 7 (`_suggest_causal_gp`): inert (not requested by the
  benchmark).

This means the minimal graph is **not** a pure no-op under the current
engine surface — path 4 produces a different candidate reranking.
Empirically Sprint 35 saw a bit-identical tie, but that is the
coincidence described above (the one-candidate control arm and the
five-candidate soft-mode arm pick the same parameter dict under the
Sprint 35 seed sequence with an empty history), not a guarantee.
Sprint 37's rerun must explicitly report both the winning candidate
and, on the `causal` arm, the five-candidate pool and their
`_causal_alignment_score` values, so a recurrence of the Sprint 35
tie is diagnosable rather than mysterious.

## Exit Criterion (what Sprint 37 must do)

Sprint 37 picks exactly one of the following two options. Sprint 36's
only remaining job (post-merge) is to name which option it was — a
single-sentence follow-up comment on the Sprint 37 issue, not a
plan rewrite.

**Option A — small engine change that makes the graph matter.** The
minimal candidate is to extend `_get_focus_variables` (or a new
sibling helper) to return a *proper subset* of the search space when
every search variable is an ancestor but a minimality criterion can
identify one. Scope guardrails:

1. Single function edit inside `causal_optimizer/optimizer/suggest.py`,
   plus unit tests.
2. Default off behind an explicit engine flag
   (`pomis_minimal_focus: bool = False` on `ExperimentEngine`).
3. Sprint 37 benchmark rerun sets the flag to `True` only for the
   `causal` strategy.
4. Preregistered H0/H1/H2 copied from the "Falsifiable Outcome" block
   below, with `causal` re-interpreted to mean "graph + flag on" and
   `surrogate_only` unchanged.

The natural minimality criterion is POMIS, since
`causal_optimizer/optimizer/pomis.py` already computes minimal
intervention sets for a given graph. However, under the preregistered
bidirected-edge-free graph above, POMIS returns the full set, so
plain-POMIS Option A reduces to a no-op. For Option A to bind,
Sprint 37 must additionally pick **one** of:

- **A1.** A non-POMIS minimality heuristic (for example, return the
  subset of ancestors whose estimated effect-on-objective magnitude
  exceeds a threshold, using the screening result that
  `_suggest_optimization` already computes at `suggest.py:425–437`).
  This keeps the graph unchanged and the change is isolated inside
  `_get_focus_variables` + its screening input.
- **A2.** A companion graph widening that adds one or more
  bidirected edges or non-ancestor structural nodes so that POMIS
  itself returns a proper subset (see Option B for the graph-only
  version of this path).

Sprint 36 does not pick between A1 and A2; Sprint 37's issue body
does, and the close-out comment records the choice with a one-line
code-grounded justification. The recommendation leans toward **A1**
because the screening result needed for a magnitude-thresholded
minimality heuristic is already computed by `_suggest_optimization`
at `suggest.py:425–437`, so A1 is a genuinely small one-function
change, whereas A2 reopens the bidirected-edge scope that Sprint 34
contract Section 4e and this plan's "Bidirected edges (none)"
section both explicitly defer.

**Option B — graph widening (add one non-ancestor structural node).**
Add one or more non-ancestor structural nodes to the graph (for
example, a `logged_position_distribution` node that parents
`position_handling_flag` but is not itself a knob, or a
`request_item_overlap` node that models the row-mask / action-support
coupling) so that `ancestors(policy_value) ⊊ search_space.variable_names`
and path 1 becomes a real restriction. Scope guardrails:

1. Graph change is authored by hand from the adapter code, not from
   convergence noise. Any added node must cite a specific
   `bandit_log.py` line that makes it a real structural variable, not
   a summary statistic of a knob.
2. Sprint 37 also ships the preregistered H0/H1/H2 below.
3. No bidirected edges still.

Neither option is committed here; the recommendation is that Sprint 36
end with **Option A preferred** because it isolates the cause
(a specific engine-surface change) from the effect, and because a
Sprint 37 with Option B would still need Sprint 38 to answer "did the
tie break because of the graph, or because of an engine path we
under-weighted in Sprint 36's analysis?".

### Preregistered H0 / H1 / H2 for the Sprint 37 rerun

> **H0 (predicted under Option A with flag off, or under the
> current engine with the minimal graph only).** On the Men/Random
> slice, B80 two-sided Mann-Whitney U p-value on SNIPW policy values
> between `causal` and `surrogate_only`, 10 seeds per arm, satisfies
> `p > 0.15` (Sprint 34 contract Section 6e "not significant" band).
>
> The prediction assumes `causal_softness = 0.5`,
> `causal_exploration_weight = 0.0`, `strategy` routes through
> `_suggest_bayesian` and not `_suggest_causal_gp`, Ax/BoTorch is
> available (no RF fallback), and no new engine path reads
> `causal_graph` between now and the Sprint 37 rerun. If any of those
> change, the prediction must be revisited before the verdict is
> quoted.

> **H1 (alternative).** `p <= 0.05` at B80. The mean-SNIPW direction
> (`causal` higher or lower than `surrogate_only`) is reported as
> observed; this is a "certified" Sprint 33 label in either direction.
> A certified `causal > surrogate_only` row would be the first
> graph-induced multi-action causal advantage in the scorecard; a
> certified `causal < surrogate_only` row would be a real regression
> that must be diagnosed.

> **H2 (trending).** `0.05 < p <= 0.15` at B80. Same two-sided
> convention as H1: the direction is reported as observed, matching
> the Sprint 31 and ERCOT NORTH_C "trending" label usage.

B20 and B40 are reported for trajectory analysis but do not gate the
Sprint 37 verdict, matching the Criteo and Sprint 35 conventions.

Success gates that must also hold for any Sprint 37 verdict row to
publish:

1. all five Section 7 gates PASS on the rerun (unchanged thresholds)
2. `null_control` permutation seed is `20260419` (the same as
   Sprint 35, taken from the committed Sprint 35 report — see the
   Sprint 36 implementation prompt for the exact source-of-truth rule)
3. backend provenance records `ax_botorch` on every verdict cell; no
   RF fallback mixing

### Power and the H0-vs-H1 boundary

Sprint 37 keeps the Sprint 33 / Sprint 35 convention of 10 seeds per
arm. At n=10 per arm, a two-sided Mann-Whitney U test has limited
power to detect small SNIPW differences: as a rough guide, it can
reliably certify a separation only when the mean shift is comparable
to or larger than the seed-level standard deviation (Cohen's d on the
order of 1). Sprint 35's optimized-strategy B80 std is
≈ 8e-6 (report "Per-Budget Outcome Tables" section), so a real but
small (~1%) causal advantage could still land in H2 rather than H1
under 10 seeds. A Sprint 38 power-extension rerun is the explicit
answer to that risk, not a Sprint 37 widening.

## Recommended Issue Shape

**One issue for Sprint 36.** Sprint 36 is narrow enough to fit in a
single docs-only PR: this plan plus an updated implementation prompt.
No Python change. No rerun.

Sprint 37 is a separate issue, created at the end of Sprint 36 with a
title roughly **"Sprint 37: Open Bandit prior graph rerun (Option A
or Option B)"** depending on which of the two options Sprint 36's
close-out comment picks.

## Empirical Setup (deferred to Sprint 37)

The Sprint 35 setup is the starting point for Sprint 37 and is
preregistered here so Sprint 37 cannot silently drift:

1. **Slice:** full Men/Random, 452,949 rows, SHA-256 match on
   `men.csv` per the Sprint 35 report "Data Provenance" section
2. **Budgets:** 20, 40, 80 (B80 gates the verdict)
3. **Seeds:** 0..9 (10 seeds per cell)
4. **Strategies:** `random`, `surrogate_only`, `causal`
5. **Estimator (primary):** SNIPW, `min_propensity_clip =
   1 / (2 · 34 · 3) ≈ 4.9019608e-03` (Sprint 34 contract Section 5c)
6. **Estimators (secondary):** DM, DR (Dudík et al. 2011 in-module
   form; OBP DR wrapper remains optional)
7. **Backend:** Ax/BoTorch; no RF-fallback mixing in the verdict row
8. **Section 7 gates:** all five, unchanged thresholds
9. **Null-control permutation seed:** `20260419` (same as Sprint 35).
   The authoritative source is the committed Sprint 35 report
   (`thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md`,
   "Permutation seed" line and the Section 7a header). The local-only
   artifact JSON that Sprint 35 also produced is not required and is
   optional corroboration only.
10. **Propensity schema:** conditional `P(item | position) = 1/34`
    (confirmed Sprint 35.A smoke test;
    `causal_optimizer/domain_adapters/bandit_log.py`
    `propensity_schema` class constant line 183)
11. **Verdict rule:** two-sided Mann-Whitney U on B80 SNIPW values,
    10 seeds per strategy; Sprint 33 scorecard labels unchanged
    (`p <= 0.05` certified, `0.05 < p <= 0.15` trending)

## What Sprint 36 Must Not Do

Hard exclusions, inherited from the Sprint 34 contract Section 9 and
from the Sprint 35 report's explicit scope-boundary list, plus the
Sprint 36 scope rewrite:

1. **No Python change in this sprint.** `BanditLogAdapter.get_prior_graph`
   stays as `return None`. The minimal graph exists only as a
   preregistered spec in this document. Authoring it as code is a
   Sprint 37 deliverable, conditional on Option A or Option B.
2. **No benchmark rerun in this sprint.** The Sprint 35 rerun
   (`scripts/open_bandit_benchmark.py` with Sprint 35 flags) is
   deferred to Sprint 37. This sprint does not re-execute the
   benchmark, does not regenerate the artifact JSON, and does not
   update the Sprint 35 report.
3. **No new slice.** Women, All, BTS-primary, and cross-campaign
   aggregation are all out of scope here and in Sprint 37.
4. **No DRos-primary.** DRos remains on the Sprint 35+ shortlist per
   contract Section 5a; the first prior-graph result must quote SNIPW.
5. **No slate-level or ranking-aware OPE.**
6. **No second dataset.** MovieLens, Outbrain, Yahoo! R6 stay on the
   Sprint 37+ backlog.
7. **No bidirected edges in the first graph.** Deferred per the
   Sprint 34 contract Section 4e discipline.
8. **No edges into `ess`, `weight_cv`, `max_weight`,
   `zero_support_fraction`, or `n_effective_actions`.** These outcome
   nodes are not consumed by `_get_focus_variables` in the current
   engine; adding them would inflate the graph without changing
   behavior.
9. **No relaxation of any Section 7 gate.** Sprint 37's rerun must
   hold them as-is.
10. **No auto-discovered graph overlay.** Sprint 34 contract Section
    4e defers auto-discovery on OBD to Sprint 37+. Hybrid mode
    remains unused in Sprint 36 and in the preregistered Sprint 37
    rerun.
11. **No multi-objective extension.** The objective remains
    `policy_value` maximize.
12. **No engine-code edits in this sprint.** Option A in the Exit
    Criterion describes a Sprint 37 engine change; Sprint 36 only
    names and scopes it.
13. **No force-push, no hook skipping, no auto-merge.** Standard
    project rules.

## Strategy

Sprint 36 runs in two stages inside one docs-only PR:

1. **Preregister the graph and the engine-surface analysis** — this
   document.
2. **Update the implementation prompt** to match the rescoped sprint
   (`thoughts/shared/prompts/sprint-36-open-bandit-prior-graph.md`).
   The prompt must (a) not ask for the adapter change or the rerun,
   (b) cite the committed Sprint 35 report as the authoritative
   source for `permutation_seed = 20260419`, and (c) phrase the
   sprint so a fresh agent on a clean checkout with no local
   artifact can execute it end-to-end.

## Success Criteria

Sprint 36 is successful if:

1. this recommendation and the updated prompt merge
2. the preregistered graph is authored edge by edge from the adapter
   code, not from post-hoc convergence noise
3. the engine-surface analysis names every `causal_graph` read site in
   `causal_optimizer/optimizer/suggest.py` and correctly characterizes
   what each one does when every search variable is an ancestor of
   `policy_value`
4. the Exit Criterion names exactly one Option A engine change and
   one Option B graph widening, both small enough to fit in a single
   Sprint 37 PR
5. the sprint does not touch anything under `causal_optimizer/`, does
   not rerun the benchmark, and does not update `handoff.md` or
   `07-benchmark-state.md`

Sprint 36 is **not** successful if:

1. any Python file under `causal_optimizer/` changes in this PR
2. the benchmark artifact JSON is regenerated in this PR
3. the Sprint 35 report is edited to reflect a Sprint 36 rerun
4. the preregistered H0/H1/H2 text is tuned to match an unpublished
   observation
5. the Sprint 37 exit condition is widened to more than one Option A
   change plus one Option B widening

## What A Good Outcome Looks Like

Best case: Sprint 36 merges with a clean preregistration, a Sprint 37
issue opened naming Option A, and a one-line update in the sprint
tracker that Sprint 37 will ship a single focus-restriction engine
change plus a rerun.

Acceptable case: same, but Sprint 37 picks Option B (graph widening)
because Option A runs into an engine refactor that does not fit one
PR. The Sprint 36 close-out comment names Option B and cites the
specific node or edge being added.

Unacceptable case: Sprint 36 expands its own scope mid-PR, ships the
adapter change anyway, or reruns the benchmark. Any of those violate
the scope lock the PR #195 review asked for.

## References

- Sprint 34 Open Bandit contract: [sprint-34-open-bandit-contract.md](../docs/sprint-34-open-bandit-contract.md)
- Sprint 35 Open Bandit benchmark report: [sprint-35-open-bandit-benchmark-report.md](../docs/sprint-35-open-bandit-benchmark-report.md)
- `BanditLogAdapter`: `causal_optimizer/domain_adapters/bandit_log.py`
- Open Bandit OPE stack and Section 7 gates: `causal_optimizer/benchmarks/open_bandit.py`
- Open Bandit benchmark runner module: `causal_optimizer/benchmarks/open_bandit_benchmark.py`
  (exposes `OpenBanditScenario` and `load_men_random_slice`)
- Open Bandit CLI entry point: `scripts/open_bandit_benchmark.py`
  (translates `--data-path` / `--budgets` / `--seeds` flags into an
  invocation of the runner module and writes the artifact JSON)
- Engine graph usage sites:
  - `causal_optimizer/optimizer/suggest.py` lines 1122–1134
    (`_get_focus_variables`)
  - `causal_optimizer/optimizer/suggest.py` lines 425–437
    (screening intersection inside `_suggest_optimization`)
  - `causal_optimizer/optimizer/suggest.py` lines 569–586
    (parent-weighted perturbation inside `_suggest_exploitation`)
  - `causal_optimizer/optimizer/suggest.py` lines 751–812
    (soft-causal reranker inside `_suggest_bayesian`)
  - `causal_optimizer/optimizer/suggest.py` lines 883–979
    (soft-causal path inside `_suggest_surrogate`)
  - `causal_optimizer/optimizer/suggest.py` lines 246–280
    (`causal_exploration_weight` LHS bias, pinned to `0.0`)
  - `causal_optimizer/optimizer/suggest.py` lines 982–1021
    (`_causal_alignment_score`, the distance metric the
    `_suggest_bayesian` reranker and the `_suggest_surrogate`
    scorer both consume)
- POMIS implementation: `causal_optimizer/optimizer/pomis.py`
  (`compute_pomis` at lines 17–56, `_muct` at lines 59–124,
  `_interventional_border` at lines 127–132; the Option A / POMIS
  trivial-case derivation in the "Bidirected edges (none)" section
  walks through all three of these against the preregistered graph)
- `CausalGraph` type: `causal_optimizer/types.py` lines 167–263
- `DomainAdapter.get_prior_graph` contract:
  `causal_optimizer/domain_adapters/base.py` line 39
- Existing prior-graph examples:
  `causal_optimizer/domain_adapters/marketing_logs.py` line 362
  (`MarketingLogAdapter.get_prior_graph`),
  `causal_optimizer/benchmarks/criteo.py` line 220
  (`criteo_projected_prior_graph`)
- PR #195 human review (2026-04-21) that prompted this rescope.
