# Sprint 38 Recommendation

Updated: 2026-04-22

## Sprint Theme

**Option B — graph widening. Add one non-search-space structural node
(`logged_position_distribution`) to the Open Bandit prior graph. The
widening is a preregistered structural test of the Sprint 37
engine-surface analysis — confirmed below as a predicted no-op along
every engine path that already existed under the Sprint 37
`pomis_minimal_focus=True` wiring. Under the widened graph,
`_get_focus_variables` still returns the full search space (the new
node is not a search-space variable), `_apply_minimal_focus_a1` still
binds and still returns the same `screened ∩ ancestors` intersection
as Sprint 37, and the soft-causal reranker's alignment score is
unchanged. The no-op prediction *is* the falsifiable claim — a B80
certified or trending row in either direction would prove the
Sprint 37 engine-path analysis incomplete.**

**Terminology note.** The Sprint 38 orchestration prompt and the
Sprint 36 plan both used the phrase "non-ancestor structural node"
as shorthand for "a structural node that is not itself a
search-space knob." That shorthand is graph-theoretically imprecise
once the edge
`logged_position_distribution -> position_handling_flag` is added,
because `logged_position_distribution` then *is* an ancestor of
`policy_value` via the two-hop path through `position_handling_flag`.
To keep the engine-path reasoning unambiguous, this recommendation
uses **"non-search-space structural node"** (or simply "structural
node") instead of "non-ancestor" from this point forward. The A1
binding condition holds under the widened graph precisely because
`logged_position_distribution` is not a search-space variable
(`_ancestors_in_space` filters it out of its search-space
intersection), not because it is a non-ancestor of the objective.

Sprint 37 landed Option A1 cleanly (PR #198). The preregistered
seven-node / six-edge graph is now live in
[`BanditLogAdapter.get_prior_graph()`](../../causal_optimizer/domain_adapters/bandit_log.py),
and the new `pomis_minimal_focus` flag on `ExperimentEngine`
(default `False`) is enabled only on the Open Bandit `causal` arm. The
rerun broke the Sprint 35 bit-identical `causal == surrogate_only` tie
on every seed at every budget — the graph now matters mechanically —
but the verdict-budget impact was null: B80 two-sided MWU
`p = 0.7337`, means `0.006181` vs `0.006182`. The B20 row trended
`causal < surrogate_only` (`p = 0.0820`, 8/10 seeds), which the
Sprint 37 report and the Sprint 38 orchestration prompt both
explicitly say not to chase. See
[sprint-37-open-bandit-prior-graph-report.md](../docs/sprint-37-open-bandit-prior-graph-report.md).

Sprint 38 picks exactly one of the three follow-ups the Sprint 37
report named. The pick is **Option B** — graph widening with one
non-search-space structural node, because it is the next
structurally distinct test of whether the Men/Random slice can
separate `causal` from `surrogate_only` under the verdict rule.
Options C and D are rejected below, on the evidence, not queue
pressure.

## Goal

Sprint 38 should end with:

1. one new non-search-space structural node
   (`logged_position_distribution`) added to the preregistered graph
   returned by `BanditLogAdapter.get_prior_graph()`, with a single
   directed edge `logged_position_distribution -> position_handling_flag`
   and no bidirected edges
2. a Sprint 37-shape rerun of Men/Random under that widened graph:
   same budgets (20, 40, 80), same 10 seeds, same strategies
   (`random`, `surrogate_only`, `causal`), same SNIPW-primary / DM /
   DR estimators, same Section 7 gates, same null-control permutation
   seed (`20260419`), Ax/BoTorch-only (no RF fallback mixing)
3. a preregistered falsifiable outcome: H0 (predicted) = `p > 0.15`
   at B80; H1 (alternative) = `p <= 0.05` at B80; H2 (trending) =
   `0.05 < p <= 0.15` at B80 — same two-sided MWU convention as
   Sprint 35 and Sprint 37
4. a Sprint 38 report that answers one question end-to-end: does
   pure graph widening (one new non-search-space ancestor of
   `policy_value`, same search space, same engine surface) move the
   B80 verdict, or does it recapitulate the Sprint 37 near-parity as
   the engine-path analysis predicts?

No second node in Sprint 38. No bidirected edges. No new slice. No
DRos-primary. No second dataset. No new heuristic. No power extension
on A1. The scope is one node, one edge, one rerun, one verdict row.

## Why This Sprint Now

### Why Option B (this sprint, Men/Random, one non-search-space structural node)

Under the preregistered graph Sprint 37 landed, every search variable
is already an ancestor of `policy_value`. That means
`_get_focus_variables` at
[`causal_optimizer/optimizer/suggest.py`](../../causal_optimizer/optimizer/suggest.py)
lines 1170–1180 returns the full search space — the
graph-ancestors path by itself is a no-op restriction. The A1 flag
added a second step: when every search variable is an ancestor AND
the screening intersection is a non-empty proper subset, return the
intersection (`_apply_minimal_focus_a1`, `suggest.py` lines
1183–1236). Sprint 37's B80 row says that indirect path does
produce a different trajectory (the Sprint 35 exact tie is broken)
but does not move the verdict-budget mean.

The natural next test is to add structure to the graph itself —
specifically, one non-search-space structural node that makes the
ancestor DAG non-trivial (some search-space variables gain
non-search-space parents; the objective gains an ancestor that is
not itself a knob). This is Option B in the Sprint 36 plan and in
the Sprint 37 "Sprint 38+ Implications" section. The Sprint 36
plan's "Exit Criterion" section names two candidate structural
nodes: `logged_position_distribution` and `request_item_overlap`.
Sprint 38 commits to exactly one: `logged_position_distribution`
with a directed edge
`logged_position_distribution -> position_handling_flag`. (The
Sprint 36 plan and the Sprint 38 orchestration prompt called these
"non-ancestor" nodes; after adding the two-hop edge,
`logged_position_distribution` is in fact an ancestor of
`policy_value`. The accurate term is "non-search-space" — see the
Sprint Theme terminology note above.)

**Honest prediction about what this widening changes.** Under the
Sprint 37 engine surface, the widened graph is predicted to be a
no-op along every path that already existed. The per-path breakdown
(path 1 `_get_focus_variables`, path 4 soft-causal reranker, etc.)
is in
[What the engine would do with the widened graph, path by path](#what-the-engine-would-do-with-the-widened-graph-path-by-path)
below. The mechanism is straightforward: the new node sits outside
the search space, so `_get_focus_variables` (which filters ancestors
through `search_space.variable_names`) still returns the six
search-space variables; the A1 helper
`_apply_minimal_focus_a1` checks
`len(ancestors_in_space) == len(all_var_names)`, and the widened
graph leaves that equal, so A1 still binds and still returns the
same `screened ∩ ancestors` intersection it did in Sprint 37.

**That is exactly the point.** Sprint 38's value is that the no-op
prediction is preregistered and falsifiable. Sprint 37 broke the
Sprint 35 bit-identical tie by turning `pomis_minimal_focus` on; if
Sprint 38 produces a B80 certified or trending row in either
direction, the Sprint 37 engine-surface analysis is provably
incomplete and must be redone before any further Open Bandit sprint.
If Sprint 38 reproduces the Sprint 37 B80 near-parity, that is
strong evidence the Open Bandit A1 lane is exhausted along the pure
graph-widening axis under this engine surface, and Sprint 39 has
clean grounds to pick between a bidirected-edge A2 variant, Option
C, or Option D (see
[What Happens After Sprint 38](#what-happens-after-sprint-38)).

The structural claim that justifies this node is narrow and
code-grounded: the adapter's `position_handling_flag` knob chooses
between two concrete row-mask behaviors (`marginalize` over all
positions or `position_1_only`), and which of those dominates the
SNIPW objective depends on the *logged* position distribution — a
property of the data, not of any search-space knob. That property
is structurally upstream of the search-space knob but is not itself
a knob. See
[Minimal Preregistered Graph (Sprint 38 widening)](#minimal-preregistered-graph-sprint-38-widening)
for the line-level justification. No bidirected edges are added;
the Sprint 34 contract Section 4e discipline holds.

### Why not Option C (different heuristic on the existing graph)

Option C swaps the A1 `screened ∩ ancestors` rule for another
screening-derived heuristic — the Sprint 36 plan lists a
magnitude-thresholded variant (drop ancestors whose screening
importance is below some quantile). The Sprint 37 report's Sprint 38+
implications section leaves this on the table but does not endorse
it, and the orchestration prompt is explicit that Option C "carries a
higher burden of proof than Option B; another heuristic tweak must
show why it is not post-hoc tuning."

Concretely, the evidence against spending Sprint 38 on Option C:

1. **Option C tunes against the only non-certified trend Sprint 37
   produced.** The B20 `p = 0.0820` row is the only budget where A1
   moved a p-value near the certified band, and the direction is
   *worse* (`causal < surrogate_only` on 8/10 seeds). A new heuristic
   that "fixes" that row without a preregistered hypothesis would be
   a direct post-hoc chase — exactly what the Sprint 36 plan and the
   Sprint 37 report forbid.
2. **Option B is structurally prior to Option C.** If graph widening
   produces no verdict movement, a heuristic change on top of a
   trivially-ancestored graph is even less likely to, because the
   heuristic consumes the same screening output and the same
   ancestor set.
3. **The Sprint 37 near-parity is honest, not power-limited.** At
   B80 the means agree to six decimals. Sprint 37's report calls
   this out explicitly: "this is not a power-limited needs-more-seeds
   result, it is an honest near-parity." A different heuristic on the
   same graph would be fighting an empirical convergence, not a
   power gap.

Option C stays on the backlog. It becomes interesting *after* a
Sprint 38 Option B rerun, if the widened graph produces a B80
trending or certified row and Option C can then be framed as "pick
between two heuristics that both see a non-trivial ancestor
structure."

### Why not Option D (move on, reopen another lane)

Option D would declare the Open Bandit A1 lane exhausted and reopen
Women, All, BTS-primary, slate OPE, DRos-primary, or a second dataset.
The orchestration prompt is explicit: Option D "is acceptable only if
argued directly from the evidence rather than from queue pressure or
impatience."

The evidence does not yet support Option D:

1. **A1 is one structural test.** Sprint 37 ran exactly one prior
   graph shape (the preregistered minimal seven-node / six-edge
   graph) with exactly one focus-restriction heuristic
   (`screened ∩ ancestors`) on exactly one slice (Men/Random) with
   exactly one logger (uniform-random). The lane has not been
   stress-tested along the graph-structure axis. Declaring it
   exhausted after one shape is premature.
2. **Option B is the canonical next test on this lane.** The Sprint
   36 plan names it, the Sprint 37 report names it, and the Sprint
   36 plan's "Why This Sprint Now" section walks through why graph
   widening is the highest-signal next move after a no-op
   ancestor-exclusive graph. Skipping it to reopen another slice
   would leave a specific, preregistered question unanswered.
3. **Moving on without Option B would leave an interpretive hole.**
   If Sprint 38 opens Women/All/BTS/a second dataset without first
   running Option B, a future null result on a new slice cannot
   distinguish "causal advantage does not generalize" from "causal
   advantage needs non-trivial graph structure to appear at all."
   Option B disambiguates those before Sprint 39 picks where to go.
4. **Option B fits in one implementation PR.** It is a single-edge
   graph widening plus a 6,100-second rerun (Sprint 37's runtime
   budget) — strictly smaller than any Option D lane reopen.

Option D becomes the right move if Sprint 38's Option B result is
itself near-parity at B80 *and* the widened-graph trajectory at
B20/B40 is also near-parity. In that case the Open Bandit A1 lane is
genuinely exhausted along the graph-structure axis under
`pomis_minimal_focus=True`, and Sprint 39 reopens another slice or
dataset. See [What Happens After Sprint 38](#what-happens-after-sprint-38).

## Minimal Preregistered Graph (Sprint 38 widening)

The Sprint 37 graph (seven nodes, six edges) is preserved verbatim.
Sprint 38 adds exactly one new node and exactly one new directed edge.

### Added node (1)

8. `logged_position_distribution`

**Code-grounded justification.** The adapter's `run_experiment`
branches on the `position_handling_flag` categorical to pick the
logged-row subset that enters the SNIPW sum
([`causal_optimizer/domain_adapters/bandit_log.py`](../../causal_optimizer/domain_adapters/bandit_log.py)
lines 352–369):

```python
if position_flag == "position_1_only":
    mask = self._position == 0
else:
    mask = np.ones(self._n_rounds, dtype=bool)
```

`self._position` is the logged position array (line 195) — a
property of the dataset, not of any knob. Which of the two
`position_handling_flag` choices is better depends on the *logged*
position distribution: if most logged clicks happen at position 1
(index 0 after `rankdata`), `position_1_only` is the right mask; if
clicks are roughly uniform across positions, `marginalize` is. That
distribution is a structural upstream variable — it is not itself a
knob, but it causally determines which `position_handling_flag`
choice dominates the objective.

Naming it `logged_position_distribution` matches the Sprint 36 plan's
Option B candidate list (plan "Exit Criterion" section, Option B
bullet) and the Sprint 37 report's "Sprint 38+ Implications" section.

### Added edge (1)

`logged_position_distribution -> position_handling_flag`

**Code-grounded justification.** The flag's behavior is a direct
function of `self._position`
(`bandit_log.py` lines 352–369, cited above). Per the Sprint 36
plan's edge-justification discipline, each added edge must cite a
specific line that makes the upstream variable structural. The line
range above — the row-mask branch inside `run_experiment` — is the
site that makes `logged_position_distribution` structurally upstream
of `position_handling_flag`.

No edge from `logged_position_distribution` directly to `policy_value`
is added. The flow is:
`logged_position_distribution -> position_handling_flag -> policy_value`
(the second edge already exists from Sprint 37).

### Bidirected edges (none, still)

The Sprint 34 contract Section 4e discipline holds. No bidirected
edges are added in Sprint 38. The "Bidirected edges (none)"
derivation in the Sprint 36 plan carries forward unchanged —
Sprint 38's single new node and single new edge do not introduce any
confounder structure.

### Graph after widening (Sprint 38 state)

```
nodes (8):
  tau, eps, w_item_feature_0, w_user_item_affinity, w_item_popularity,
  position_handling_flag, policy_value,
  logged_position_distribution  # NEW in Sprint 38

directed edges (7):
  tau                            -> policy_value
  eps                            -> policy_value
  w_item_feature_0               -> policy_value
  w_user_item_affinity           -> policy_value
  w_item_popularity              -> policy_value
  position_handling_flag         -> policy_value
  logged_position_distribution   -> position_handling_flag  # NEW in Sprint 38

bidirected edges: none
```

### What the engine would do with the widened graph, path by path

Under the Sprint 37 engine surface (unchanged — Sprint 38 ships no
engine or optimizer change):

1. `_get_focus_variables` (`suggest.py` lines 1170–1180): the ancestor
   set `ancestors(policy_value)` still equals the six search-space
   variables plus the new `logged_position_distribution`. The
   search-space intersection
   `{v in search_space.variable_names | v in ancestors(policy_value)}`
   still equals the full search space, because
   `logged_position_distribution` is **not** itself a search-space
   variable. So path 1 on its own remains a no-op — the full
   search-space fallback triggers.
2. A1 minimal-focus (`_apply_minimal_focus_a1`, `suggest.py` lines
   1183–1236): the binding condition — every search-space variable
   is an ancestor of `policy_value` — still holds, so the A1 helper
   still applies `screened ∩ ancestors` whenever the screening
   result is a non-empty proper subset. Under the widened graph the
   ancestor set inside the search space is unchanged, so the A1
   result is the same set the Sprint 37 rerun computed at every
   (seed, budget, phase) triple.
3. Parent-weighted exploitation (`suggest.py` lines 569–586): uses
   `causal_graph.parents(objective_name)`. Sprint 37 graph gave
   `parents(policy_value) = 6` search-space vars → `weights = None`
   (no restriction). Sprint 38 widened graph still gives
   `parents(policy_value) = 6` search-space vars (the new edge
   targets `position_handling_flag`, not `policy_value`) → still
   `weights = None`, still no-op.
4. Soft-causal reranker (`_rerank_alignment_only` +
   `_causal_alignment_score`, `suggest.py` lines 751–812 and
   982–1021): averages over ancestors of the objective. The
   ancestor set grows by one (`logged_position_distribution`), but
   that variable is not in the search space, so
   `_causal_alignment_score` skips it when it iterates over
   ancestors in the search space (the per-variable normalized
   displacement loop at lines 1010–1019 only fires when the
   ancestor name is a key in `candidate` and `best_params`). The
   reranker's per-seed output is therefore identical to Sprint 37.
5. `_suggest_causal_gp`: inert (not requested by the benchmark).
6. `causal_exploration_weight`: pinned to `0.0`, inert.

So — honestly — the Sprint 38 widened graph is **predicted to be a
no-op** along every engine path that already existed. That is
exactly the point. Sprint 37 broke the Sprint 35 bit-identical tie
by turning `pomis_minimal_focus` on; if Sprint 38's widened graph
reproduces the Sprint 37 B80 near-parity, that is strong evidence
the Open Bandit A1 lane is genuinely exhausted along the pure
graph-widening axis under this engine surface, and Sprint 39 has
clean grounds to pick between a bidirected-edge Option A2 variant
or Option D.

**This is a preregistered no-op prediction, not a hidden one.** The
value of running it is that the *prediction itself* is the
falsifiable claim: if Sprint 38 produces a B80 certified or trending
row in either direction, the Sprint 37 engine-surface analysis is
incomplete and must be redone before any further Open Bandit sprint.

## Exit Criterion

Sprint 38 is successful if:

1. `BanditLogAdapter.get_prior_graph()` returns the widened eight-node
   / seven-edge graph above, with no bidirected edges and no other
   structural changes
2. A Sprint 37-shape rerun of Men/Random under that graph produces a
   complete verdict table at B20 / B40 / B80 for `random` /
   `surrogate_only` / `causal`, 10 seeds per cell, all five Section 7
   gates PASS, Ax/BoTorch-only backend provenance on every verdict
   cell
3. The Sprint 38 report records the B80 verdict row under the Sprint
   33 / Sprint 35 / Sprint 37 labels: certified (`p <= 0.05`),
   trending (`0.05 < p <= 0.15`), or not significant (`p > 0.15`) —
   with the two-sided MWU convention, same as Sprint 37
4. The report reconciles the observed outcome against the
   preregistered H0 / H1 / H2 below, and the Sprint 37 report's
   per-path engine analysis
5. The report calls the Option B Open Bandit A1 lane status for
   Sprint 39: exhausted (go to Option D), or still productive (run
   one more focused iteration — either A2 bidirected-edge variant or
   Option C heuristic, with a preregistered hypothesis)

### Preregistered H0 / H1 / H2 for the Sprint 38 rerun

> **H0 (predicted).** On the Men/Random slice, B80 two-sided
> Mann-Whitney U p-value on SNIPW policy values between `causal` and
> `surrogate_only`, 10 seeds per arm, satisfies `p > 0.15` (Sprint 34
> contract Section 6e "not significant" band). This is the same H0
> Sprint 37 confirmed. The Sprint 38 engine-path analysis above
> predicts the widened graph is a no-op along every path that
> already existed under Sprint 37's minimal-focus wiring, so the B80
> row should recapitulate the Sprint 37 near-parity.
>
> The prediction assumes `causal_softness = 0.5`,
> `causal_exploration_weight = 0.0`, `strategy` routes through
> `_suggest_bayesian` (not `_suggest_causal_gp`), Ax/BoTorch is
> available (no RF fallback), `pomis_minimal_focus = True` on the
> `causal` arm only, and no new engine path reads `causal_graph`
> between Sprint 37 and the Sprint 38 rerun. If any of those change,
> the prediction must be revisited before the verdict is quoted.

> **H1 (alternative — certified).** `p <= 0.05` at B80. The
> mean-SNIPW direction (`causal` higher or lower than
> `surrogate_only`) is reported as observed; this is a "certified"
> Sprint 33 label in either direction. A certified
> `causal > surrogate_only` row would be the first graph-induced
> multi-action causal advantage in the scorecard and would
> immediately invalidate the Sprint 38 engine-path no-op prediction,
> requiring a diagnostic rerun before Sprint 39 plans. A certified
> `causal < surrogate_only` row would be a real regression and must
> be diagnosed before any further Open Bandit iteration.

> **H2 (trending).** `0.05 < p <= 0.15` at B80. Same two-sided
> convention as H1: the direction is reported as observed. A
> trending row in either direction is a soft signal the Sprint 37
> engine-path analysis is incomplete and warrants a diagnostic
> rerun, but does not, on its own, invalidate the Sprint 38 verdict
> rule.

B20 and B40 are reported for trajectory analysis but do not gate the
Sprint 38 verdict, matching the Criteo / Sprint 35 / Sprint 37
conventions. The B20 row is explicitly **not** the focus — Sprint 38
is not a B20-chase sprint, and the Sprint 37 B20 trend (`p = 0.0820`,
`causal < surrogate_only` on 8/10 seeds) is a trajectory artifact,
not a Sprint 38 target.

Success gates that must also hold for the Sprint 38 verdict row to
publish (inherited from Sprint 34 contract Section 7 and Sprint 37):

1. all five Section 7 gates PASS on the rerun (unchanged thresholds)
2. `null_control` permutation seed is `20260419` (the same as
   Sprint 35 and Sprint 37, taken from the committed Sprint 35 and
   Sprint 37 reports)
3. backend provenance records `ax_botorch` on every verdict cell; no
   RF fallback mixing

### Power and the H0-vs-H1 boundary

Sprint 38 keeps the Sprint 33 / Sprint 35 / Sprint 37 convention of
10 seeds per arm. At n=10 per arm, a two-sided Mann-Whitney U test
has limited power to detect small SNIPW differences: Sprint 37's
optimized-strategy B80 std is ≈ 8e-6, so a real but small (~1%)
causal advantage could still land in H2 rather than H1 under 10
seeds. A Sprint 39 power-extension rerun is the explicit answer to
that risk if Sprint 38 produces a trending row (H2); **Sprint 38
itself does not extend power on A1 or on Option B**.

## Recommended Issue Shape

**One issue for Sprint 38.** The scope is a single graph widening
(one new node, one new edge) plus a Sprint 37-shape rerun and report.
This fits cleanly in one implementation PR — same footprint as
Sprint 37's PR #198 (adapter change + engine flag + rerun + report).
Sprint 37 did two things (graph + engine flag); Sprint 38 does one
thing (graph-only widening), so the scope is strictly smaller.

The issue title should read roughly **"Sprint 38: Open Bandit prior
graph widening (Option B — `logged_position_distribution`)"**. The
issue body should pin the preregistered H0 / H1 / H2, the Option B
guardrails (one node, one edge, no bidirected edges, no engine
change, no new heuristic), and the exhaustion criterion for Sprint
39.

## Empirical Setup

The Sprint 37 setup is the starting point for Sprint 38 and is
preregistered here so Sprint 38 cannot silently drift:

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
9. **Null-control permutation seed:** `20260419` (same as Sprint 35
   and Sprint 37). The authoritative source is the committed
   Sprint 35 report
   (`thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md`,
   "Permutation seed" line and the Section 7a header) and the
   committed Sprint 37 report
   (`thoughts/shared/docs/sprint-37-open-bandit-prior-graph-report.md`,
   "Configuration" section)
10. **Propensity schema:** conditional `P(item | position) = 1/34`
    (confirmed Sprint 35.A smoke test;
    `causal_optimizer/domain_adapters/bandit_log.py`
    `propensity_schema` class constant line 181)
11. **A1 flag:** `pomis_minimal_focus = True` only on the `causal`
    arm; `False` for `surrogate_only` and `random` (unchanged from
    Sprint 37)
12. **Verdict rule:** two-sided Mann-Whitney U on B80 SNIPW values,
    10 seeds per strategy; Sprint 33 scorecard labels unchanged
    (`p <= 0.05` certified, `0.05 < p <= 0.15` trending,
    `p > 0.15` not significant)

## What Sprint 38 Must Not Do

Hard exclusions, inherited from the Sprint 34 contract Section 9,
the Sprint 35 report, the Sprint 36 plan, and the Sprint 37 report:

1. **No second new node.** The widened graph adds exactly one node
   (`logged_position_distribution`) and exactly one edge. Any
   additional node — including `request_item_overlap`, which the
   Sprint 36 plan also listed as a candidate — is deferred to
   Sprint 39+. Mid-sprint scope expansion into a two-node widening
   is the exact anti-pattern the Sprint 36 review caught.
2. **No bidirected edges.** The Sprint 34 contract Section 4e
   discipline holds. Bidirected edges remain deferred.
3. **No engine or optimizer change.** `pomis_minimal_focus` stays
   default-off at the engine level and is enabled only for the
   `causal` arm by the Open Bandit benchmark harness (unchanged from
   Sprint 37). The soft-causal reranker, `_get_focus_variables`,
   `_apply_minimal_focus_a1`, the parent-weighted exploitation
   perturbation, and `_suggest_causal_gp` are not edited in
   Sprint 38.
4. **No new heuristic.** Option C (magnitude-thresholded minimality
   heuristic or any other replacement for `screened ∩ ancestors`)
   is deferred. Sprint 38 tests one axis at a time — graph
   structure, not heuristic.
5. **No new slice.** Women, All, BTS-primary, and cross-campaign
   aggregation remain out of scope. Men/Random only.
6. **No DRos-primary.** DRos remains on the Sprint 35+ shortlist;
   the first Option B result must quote SNIPW.
7. **No slate-level or ranking-aware OPE.**
8. **No second dataset.** MovieLens, Outbrain, Yahoo! R6 stay on the
   Sprint 39+ backlog.
9. **No power extension on A1.** Sprint 37's B80 row is unambiguous;
   a Sprint 38 power extension would re-open a settled question.
10. **No B20 chase.** Sprint 37's B20 trend (`p = 0.0820`,
    `causal < surrogate_only` on 8/10 seeds) is explicitly not the
    Sprint 38 target. Sprint 38 reports B20 for trajectory analysis
    only; the verdict row is B80.
11. **No relaxation of any Section 7 gate.** Sprint 38's rerun must
    hold all five gates as-is at Sprint 34 contract thresholds.
12. **No auto-discovered graph overlay.** Hybrid mode remains unused.
    The widened graph is preregistered, not learned.
13. **No multi-objective extension.** The objective remains
    `policy_value` maximize.
14. **No force-push, no hook skipping, no auto-merge.** Standard
    project rules.
15. **No updates to `handoff.md` or `07-benchmark-state.md` from
    Sprint 38's implementation PR** beyond the post-merge canonical
    sync. Track A owns those; Sprint 38's implementation PR should
    only touch adapter code, unit tests, the Sprint 38 report, and
    the report-linked run command.

## Strategy

Sprint 38 runs in two stages inside one implementation PR:

1. **Graph widening.** Edit
   `BanditLogAdapter.get_prior_graph()` to return the eight-node /
   seven-edge widened graph above. Add unit tests that pin the
   added node, the added edge, the preserved existing edges, and
   the absence of bidirected edges. Update the docstring to cite
   the Sprint 38 plan (this document).
2. **Rerun and report.** Execute the Sprint 37 reproducibility
   command verbatim against the widened graph. Write the Sprint 38
   report to
   `thoughts/shared/docs/sprint-38-open-bandit-graph-widening-report.md`
   in the Sprint 37 report's shape (summary, verdict rows,
   per-budget tables, Section 7 gates, per-seed detail, ESS
   diagnostics, hypothesis reconciliation, interpretation, scope
   boundaries, attribution).

The rerun command is the Sprint 37 command with no CLI-flag changes
— graph widening lives inside `BanditLogAdapter.get_prior_graph()`,
not at the benchmark surface:

```bash
uv run python scripts/open_bandit_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/open_bandit/open_bandit_dataset \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --strategies random,surrogate_only,causal \
  --null-control \
  --permutation-seed 20260419 \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/sprint-38-open-bandit-graph-widening/men_random_results.json
```

## Success Criteria

Sprint 38 is successful if:

1. this recommendation and its execution prompt merge (planning PR)
2. a separate implementation PR lands the widened graph plus a
   complete Sprint 38 rerun report
3. the report's B80 row is labeled under the Sprint 33 / Sprint 35 /
   Sprint 37 scorecard (certified / trending / not significant) with
   the two-sided MWU convention
4. the report reconciles the observed B80 outcome against the
   preregistered H0 / H1 / H2 without re-tuning them to fit the
   observation
5. all five Section 7 gates PASS at Sprint 34 contract thresholds
6. the report calls Sprint 39's next move explicitly: exhausted
   (Option D), A2 bidirected-edge variant, Option C heuristic, or a
   second Option B node (`request_item_overlap`)

Sprint 38 is **not** successful if:

1. the implementation PR adds more than one node or more than one
   edge to the graph
2. the implementation PR introduces a bidirected edge
3. the implementation PR touches `causal_optimizer/optimizer/suggest.py`,
   `causal_optimizer/engine/loop.py`, or any other engine surface
4. the preregistered H0 / H1 / H2 text is tuned to match an
   unpublished observation
5. the Sprint 38 report reframes the B20 trend as a Sprint 38
   target
6. the rerun uses a permutation seed other than `20260419`
7. any Section 7 gate is relaxed
8. the rerun mixes RF-fallback and Ax/BoTorch verdict cells

## What Happens After Sprint 38

The Sprint 38 report must close by naming Sprint 39's next move, in
one of four shapes. Sprint 38's job is to run the rerun and record
the outcome; Sprint 39's issue body records the choice, the same
way Sprint 36 recorded Sprint 37's Option A1 vs A2 pick.

1. **Sprint 38 B80 is near-parity (H0 confirmed, same as Sprint 37).**
   The Open Bandit A1 lane is exhausted along the pure graph-widening
   axis. Sprint 39 goes to **Option D**: reopen Women, BTS-primary,
   or a second dataset (MovieLens, Outbrain, Yahoo! R6) with a fresh
   preregistered hypothesis.
2. **Sprint 38 B80 is trending (H2, either direction).** The Sprint
   37 engine-path no-op prediction was incomplete. Sprint 39 runs a
   diagnostic rerun (5 extra seeds, same graph) before picking any
   next structural move, to rule out a 10-seed power-limited H2
   miss.
3. **Sprint 38 B80 is certified `causal > surrogate_only` (H1,
   positive).** First graph-induced multi-action causal advantage in
   the scorecard. Sprint 39 replicates on a held-out slice (Women or
   BTS-primary) before any Option C or Option A2 work, to confirm
   the advantage is not Men/Random-specific.
4. **Sprint 38 B80 is certified `causal < surrogate_only` (H1,
   negative).** Real regression. Sprint 39 reverts Option B, reruns
   Option A1 with the Sprint 37 graph as a sanity check, and opens a
   diagnostic issue to attribute the regression before any further
   Open Bandit iteration.

In all four cases, the Sprint 39 issue body records which path
Sprint 38's outcome selected and cites the specific rows / p-values
that forced the choice.

## References

- Sprint 34 Open Bandit contract: [sprint-34-open-bandit-contract.md](../docs/sprint-34-open-bandit-contract.md)
- Sprint 35 Open Bandit benchmark report: [sprint-35-open-bandit-benchmark-report.md](../docs/sprint-35-open-bandit-benchmark-report.md)
- Sprint 36 preregistration plan: [26-sprint-36-recommendation.md](26-sprint-36-recommendation.md)
- Sprint 37 Open Bandit prior-graph report: [sprint-37-open-bandit-prior-graph-report.md](../docs/sprint-37-open-bandit-prior-graph-report.md)
- `BanditLogAdapter`: `causal_optimizer/domain_adapters/bandit_log.py`
  (see `get_prior_graph`, lines 513–541, and `run_experiment` row-mask
  branch, lines 352–369, for the Option B node justification)
- Open Bandit OPE stack and Section 7 gates: `causal_optimizer/benchmarks/open_bandit.py`
- Open Bandit benchmark runner: `causal_optimizer/benchmarks/open_bandit_benchmark.py`
  (enables `pomis_minimal_focus` only on the `causal` arm, line 541)
- Open Bandit CLI entry point: `scripts/open_bandit_benchmark.py`
- Engine graph-usage sites:
  - `causal_optimizer/optimizer/suggest.py` lines 1170–1180
    (`_get_focus_variables`)
  - `causal_optimizer/optimizer/suggest.py` lines 1183–1236
    (`_apply_minimal_focus_a1`, Sprint 37 A1 helper)
  - `causal_optimizer/optimizer/suggest.py` lines 751–812
    (soft-causal reranker inside `_suggest_bayesian`)
  - `causal_optimizer/optimizer/suggest.py` lines 982–1021
    (`_causal_alignment_score`)
- `CausalGraph` type: `causal_optimizer/types.py`
- `DomainAdapter.get_prior_graph` contract:
  `causal_optimizer/domain_adapters/base.py`
- Existing prior-graph examples:
  `causal_optimizer/domain_adapters/marketing_logs.py`
  (`MarketingLogAdapter.get_prior_graph`),
  `causal_optimizer/benchmarks/criteo.py`
  (`criteo_projected_prior_graph`)
- Sprint 38 tracked issue: `#201`
