# Sprint 36 Recommendation

Updated: 2026-04-21

## Sprint Theme

**First Open Bandit prior-graph experiment: does a minimal, semantically
correct graph move the Men/Random tie?**

Sprint 35 shipped the first Men/Random benchmark. The verdict was clean:
all five Section 7 gates passed, both optimized strategies certified over
`random` at every budget, and `causal` vs `surrogate_only` produced a
bit-identical tie on every seed at every budget (p = 1.000 at B20, B40,
B80). The tie is mechanical — `BanditLogAdapter.get_prior_graph()`
returns `None`, so the `causal` path has no graph to exploit and reduces
to `surrogate_only` on this slice.

Sprint 36 asks the narrowest next question: **can a small,
code-grounded prior graph create real `causal` vs `surrogate_only`
separation on the same slice, or does the tie persist under a correctly
authored graph?** Either answer is valuable evidence for the
generalization scorecard. A persisted tie is a specific, falsifiable
statement about the engine surface (ancestor-based focus does nothing
when every knob is already an ancestor of the objective); a genuine
separation would be the first graph-induced causal advantage on a
multi-action logged-policy benchmark.

## Goal

Sprint 36 should end with:

1. a merged prior graph on `BanditLogAdapter` that is justified edge by
   edge from the adapter's scoring code and from the Sprint 35
   benchmark report, not from post-hoc convergence noise
2. a Men/Random rerun under the same Section 7 gates and the same
   SNIPW-primary verdict rule
3. a clean verdict row that is either:
   - a `causal` vs `surrogate_only` separation at any budget (new
     benchmark entry), or
   - a substantive tie under a correctly authored graph (first
     engine-surface data point for the "ancestor focus is trivial when
     every search variable is an ancestor of the objective" hypothesis)

The sprint must not fabricate edges, invent a richer graph, or relax
any Section 7 gate. A bad graph is worse than no graph.

## Why This Sprint Now

Sprint 35 produced an exact tie because of the engine's current graph
usage path, not because multi-action OPE is insensitive to causal
structure:

1. `optimizer/suggest.py::_get_focus_variables` is the only place the
   graph directly alters optimization during exploitation
   (`suggest.py` line 185:
   `focus_variables = _get_focus_variables(search_space, causal_graph, objective_name)`)
   and the optimization-phase screening intersection
   (`suggest.py` lines 425–437). It returns
   `[v for v in search_space.variable_names if v in causal_graph.ancestors(objective_name)]`
   (`suggest.py` line 1132). When that list equals the full search
   space, focus behaves exactly like the no-graph case.
2. `causal_exploration_weight` has been pinned to `0.0` since
   Sprint 29 (PR #160), so the LHS-candidate ancestor-bias code path
   in `suggest.py` lines 246–280 is inactive under the production
   default. The H0 prediction below assumes this pin stays at `0.0`
   for the Sprint 36 rerun; if a future PR changes that default
   before Sprint 36 runs, revisit the prediction first.
3. POMIS requires bidirected edges to produce non-trivial
   intervention sets (`optimizer/pomis.py` lines 30–56). A graph with
   no bidirected edges collapses POMIS to the trivial case.

Running a Men/Random rerun with `get_prior_graph() -> None` again
answers no new question. Running it with an *honest* graph isolates a
single cause: whether the engine's current graph usage is load-bearing
on a smooth multi-action scoring policy surface.

## Why not a richer graph

The Sprint 35 seed-0 B80 best policy converged to
`tau ≈ 0.285`, `eps = 0.0`, `w_user_item_affinity = 3.0` (upper
boundary), `w_item_popularity = -2.461`, `w_item_feature_0 ≈ 0.231`,
`position_handling_flag = "position_1_only"` (Sprint 35 report,
"Per-Budget Outcome Tables (SNIPW — primary)" section). The smaller absolute weight on `w_item_feature_0` is tempting
evidence to prune that variable as a non-ancestor of `policy_value`.
That temptation is precisely what this contract rejects. `item_term =
w_item * self._item_feature_0[None, :]` is a summand in `scores` inside
`bandit_log.py` line 386 — the code path makes `w_item_feature_0` a
direct structural ancestor of `policy_value` regardless of how small
the optimizer's best weight turned out to be. Dropping it from the
graph to manufacture a differentiation lever would be noise chasing,
not causal structure.

The same argument applies to every other search variable: `tau`, `eps`,
`w_user_item_affinity`, `w_item_popularity`, and
`position_handling_flag` each appear directly in the softmax scoring
pipeline or the row-mask selector (`bandit_log.py` lines 366–398). The
minimal, semantically correct graph therefore contains a directed edge
from every search variable to `policy_value`. This is the small graph
Sprint 36 should author.

## First Graph (proposed)

### Nodes (7)

Search-space variables (exactly the six variables declared in
`BanditLogAdapter.get_search_space`, `bandit_log.py` lines 316–346):

1. `tau`
2. `eps`
3. `w_item_feature_0`
4. `w_user_item_affinity`
5. `w_item_popularity`
6. `position_handling_flag`

Outcome node (the primary objective; Sprint 34 contract Section 5a,
and the name returned by
`BanditLogAdapter.get_objective_name()` in `bandit_log.py` line 526):

7. `policy_value`

### Directed edges (6)

Every search variable → `policy_value`. Each edge has a one-line code
justification:

| Edge | Code site | Justification |
|------|-----------|---------------|
| `tau → policy_value` | `bandit_log.py` line 393 (`scaled = scores / safe_tau`) | `tau` is the softmax temperature; it scales every score directly into the policy that SNIPW evaluates. |
| `eps → policy_value` | `bandit_log.py` line 398 (`policy = (1.0 - eps) * softmax + eps * uniform`) | `eps` linearly mixes the softmax and the uniform fallback into the evaluation policy. |
| `w_item_feature_0 → policy_value` | `bandit_log.py` line 386 (`item_term = w_item * self._item_feature_0[None, :]`) | Per-item continuous feature weight; a summand in `scores` which feeds the softmax. |
| `w_user_item_affinity → policy_value` | `bandit_log.py` line 388 (`affinity_term = w_affinity * self._affinity[mask]`) | Per-row, per-candidate affinity weight; a summand in `scores`. |
| `w_item_popularity → policy_value` | `bandit_log.py` line 387 (`pop_term = w_popularity * self._item_popularity[None, :]`) | Popularity prior weight; a summand in `scores`. |
| `position_handling_flag → policy_value` | `bandit_log.py` lines 366–369 (row mask on `self._position == 0` vs all rows) | Controls which logged rows enter the SNIPW sum; the flag changes both the policy value and the number of rows `n_active`. |

### Bidirected edges (none)

Sprint 36 must not add bidirected edges. The six search-space variables
are independently sampled by the optimizer — they have no shared
latent cause, they are knobs. The coupling among the three IPS-weight
diagnostics (`ess`, `weight_cv`, `max_weight`) is a shared
computation, not an unobserved confounder; modeling it as a bidirected
edge would misrepresent the code. Bidirected edges from auto-discovery
are explicitly "heuristic proxies, not formally identified
confounders" (CLAUDE.md discovery notes, Sprint 34 contract Section
4e). Without bidirected edges POMIS collapses to the trivial case,
which is the correct behavior for this slice.

### Why the diagnostic outcomes are not nodes

`ess`, `weight_cv`, `max_weight`, `zero_support_fraction`, and
`n_effective_actions` are returned by `run_experiment`
(`bandit_log.py` lines 470–477) but they are not the objective and the
engine's `_get_focus_variables` only dispatches on ancestors of
`objective_name`. Adding these as nodes in the first graph would add
edges that the current engine never reads. Keeping the first graph to
seven nodes mirrors the discipline Sprint 32 applied when it shrank
`MarketingLogAdapter`'s prior graph to the Criteo projected 5-edge
graph (`benchmarks/criteo.py` lines 220–237).

### What `focus_variables` will equal under this graph

`ancestors("policy_value") = {tau, eps, w_item_feature_0,
w_user_item_affinity, w_item_popularity, position_handling_flag}`.
That set equals the full search space. Under
`optimizer/suggest.py::_get_focus_variables` line 1132, that means
`focus_variables` equals the full search space — the same list the
engine produces when `causal_graph is None`.

This is the contract's central prediction: **under the current engine
surface, this honest minimal graph should not separate `causal` from
`surrogate_only`.** The Sprint 35 tie should reproduce. See the
falsifiable-outcome section below.

The prediction mechanically depends on the private helper
`optimizer/suggest.py::_get_focus_variables`. The Sprint 36
implementation PR's unit test for H0 (see the "Strategy" section)
therefore imports that private function by name — a deliberate
coupling so a future refactor that moves the focus logic out of
`_get_focus_variables` will immediately break the Sprint 36 test and
force a plan review rather than silently invalidating the prediction.

## Falsifiable Outcome

The primary Sprint 36 hypothesis, stated as a concrete inequality the
rerun either confirms or refutes:

> **H0 (predicted).** With the minimal graph above wired into
> `BanditLogAdapter.get_prior_graph()`, the B80 two-sided
> Mann-Whitney U p-value on SNIPW policy values between `causal` and
> `surrogate_only`, computed over 10 seeds on the full Men/Random
> slice, satisfies `p > 0.15` (the Sprint 34 contract Section 6e "not
> significant" band).

H0 being confirmed is a non-trivial result — it is the first direct
evidence that the engine's ancestor-based focus path is inert when
every search variable is an ancestor of the objective, which is the
default situation for smooth scoring-policy surfaces.

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
Sprint 36 verdict, matching the Criteo and Sprint 35 conventions.

Success gates that must also hold for the verdict row to publish:

1. all five Section 7 gates PASS on the rerun (unchanged thresholds)
2. `null_control` permutation seed is the same as Sprint 35
   (`20260419`) so the permuted-baseline comparison is apples-to-apples
3. backend provenance records `ax_botorch` on every verdict cell; no
   RF fallback mixing

### Power and the H0-vs-H1 boundary

Sprint 36 keeps the Sprint 33 / Sprint 35 convention of 10 seeds per
arm. At n=10 per arm, a two-sided Mann-Whitney U test has limited
power to detect small SNIPW differences: as a rough guide, it can
reliably certify a separation only when the mean shift is comparable
to or larger than the seed-level standard deviation (Cohen's d on the
order of 1). Sprint 35's optimized-strategy B80 std is
≈`8e-6` (report "Per-Budget Outcome Tables" section), so a real but
small (~1%) causal advantage could still land in H2 rather than H1
under 10 seeds. A Sprint 37 power-extension rerun (see "Exit
Criterion" below) is the explicit answer to that risk, not a
Sprint 36 widening.

## Recommended Issue Shape

**One issue.** Sprint 36 is narrow enough to fit in a single PR:
the graph change is a few lines in
`causal_optimizer/domain_adapters/bandit_log.py`, the rerun is the
same CLI entry point Sprint 35 already exercised
(`scripts/open_bandit_benchmark.py`), and the report is a structured
rewrite of the Sprint 35 report under a new title.

Rationale for one issue (not two): the graph authoring and the rerun
are information-coupled — the rerun's only purpose is to falsify the
authored graph, and the report must quote both the exact graph and
the exact rerun numbers. Splitting the issue in two would force the
rerun PR to either pin the graph before it is reviewable or accept a
flaky cross-PR dependency. Neither tradeoff pays for itself. If the
graph review surfaces a genuine redesign (not predicted), *that* is a
natural point to split — but the default is one issue.

Suggested issue title: **Sprint 36: first Open Bandit prior graph and
Men/Random rerun (#TBD)**.

## Empirical Setup (locked)

Sprint 36 must reuse the Sprint 35 setup verbatim unless an empirical
observation forces a change:

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
9. **Null-control permutation seed:** 20260419 (same as Sprint 35)
10. **Propensity schema:** conditional `P(item | position) = 1/34`
    (confirmed Sprint 35.A smoke test; `bandit_log.py`
    `propensity_schema` class constant line 183)
11. **Verdict rule:** two-sided Mann-Whitney U on B80 SNIPW values,
    10 seeds per strategy; Sprint 33 scorecard labels unchanged
    (`p <= 0.05` certified, `0.05 < p <= 0.15` trending)

## What Sprint 36 Must Not Do

Hard exclusions, inherited from the Sprint 34 contract Section 9 and
from the Sprint 35 report's explicit scope-boundary list:

1. **No new slice.** Women, All, BTS-primary, and cross-campaign
   aggregation are all out of scope.
2. **No DRos-primary.** DRos remains on the Sprint 35+ shortlist per
   contract Section 5a; the first prior-graph result must quote SNIPW.
3. **No slate-level or ranking-aware OPE.**
4. **No second dataset.** MovieLens, Outbrain, Yahoo! R6 stay on the
   Sprint 37+ backlog.
5. **No unchanged rerun with `get_prior_graph() -> None`.** That is
   the Sprint 35 result; Sprint 36 exists to vary exactly that
   function.
6. **No bidirected edges in the first graph.** Deferred per the
   Sprint 34 contract Section 4e discipline.
7. **No edges into `ess`, `weight_cv`, `max_weight`,
   `zero_support_fraction`, or `n_effective_actions`.** These outcome
   nodes are not consumed by `_get_focus_variables` in the current
   engine; adding them would inflate the graph without changing
   behavior.
8. **No relaxation of any Section 7 gate.** A support-gate failure
   on the rerun is a blocker, not an excuse to widen the band.
9. **No auto-discovered graph overlay.** Sprint 34 contract Section
   4e defers auto-discovery on OBD to Sprint 37+. Hybrid mode
   remains unused in Sprint 36.
10. **No multi-objective extension.** The objective remains
    `policy_value` maximize.
11. **No new engine features and no engine-code edits.** The graph
    must be testable against the existing engine surface
    (`types.py::CausalGraph`,
    `domain_adapters/base.py::DomainAdapter.get_prior_graph`). The
    implementation PR must not modify `causal_optimizer/optimizer/`,
    `causal_optimizer/engine/`, or the OPE stack under
    `causal_optimizer/benchmarks/open_bandit*.py`.
12. **No force-push, no hook skipping, no auto-merge.** Standard
    project rules.

## Strategy

Sprint 36 should run in three deliberate stages inside one PR:

1. **Graph authoring** — rewrite `BanditLogAdapter.get_prior_graph`
   to return the seven-node, six-edge graph above. Add a unit test
   that asserts every search-space variable is an ancestor of
   `policy_value` and that there are no bidirected edges.
2. **Rerun** — re-execute `scripts/open_bandit_benchmark.py` with
   the same flags Sprint 35 used. Regenerate the artifact JSON.
3. **Report** — write
   `thoughts/shared/docs/sprint-36-open-bandit-prior-graph-report.md`
   under the same structure as the Sprint 35 report, but with the
   graph diff, the H0/H1/H2 hypothesis box, and the resulting verdict.

## Success Criteria

Sprint 36 is successful if:

1. the first prior graph on `BanditLogAdapter` merges
2. the Men/Random rerun completes cleanly on the same setup as
   Sprint 35 and all five Section 7 gates PASS
3. the report publishes a verdict row that classifies one of H0, H1,
   or H2 against the preregistered thresholds, without redefining the
   thresholds mid-sprint
4. the scorecard gains a new data point — the first `causal`-path
   result on a multi-action logged-policy benchmark with a
   non-trivial causal graph
5. the sprint does not touch anything out of scope

Sprint 36 is **not** successful if:

1. the graph is widened after the rerun to chase a significant p-value
2. the report calls a trending result "certified", or a certified
   result "trending", or recategorizes the row after the fact
3. the rerun mixes backends or reshapes the Section 7 gate thresholds
4. the graph includes edges the adapter code does not support

## What A Good Outcome Looks Like

Best case (least likely per the engine-surface argument): H1 — a
certified `causal > surrogate_only` separation at B80 with all gates
green. That would be the project's first graph-induced multi-action
causal advantage and a genuinely new scorecard row.

Most likely (and still valuable): H0 — the tie persists, gates pass,
and the report cleanly attributes the tie to the engine's
ancestor-based focus path being inert when every search variable is
already an ancestor of the objective. This is a specific, reusable
statement that informs the Sprint 37+ direction.

Least valuable but acceptable: H2 — trending in either direction.
That ends the sprint with an honest "needs more seeds or a
structurally different graph" conclusion, which is diagnosable.

## Exit Criterion

At the end of Sprint 36 we should know:

1. whether the current engine's graph usage creates `causal`
   differentiation on a smooth multi-action scoring policy surface
   under a correctly authored minimal graph
2. what Sprint 37 must address next, with the trigger depending on
   which hypothesis lands:
   - **H0 landed (tie persists):** Sprint 37 chooses between "author
     a larger graph with non-ancestor nodes so `focus_variables` is
     actually a proper subset" and "add an engine feature that uses
     the graph beyond ancestor-based focus" (e.g., parent-conditional
     priors on the Ax surrogate)
   - **H1 landed (certified separation):** Sprint 37 reruns on a
     second slice (Men/BTS as the contract-Section 3b sanity check)
     to confirm the effect is not slice-specific, and updates the
     generalization scorecard with the first multi-action causal
     advantage row
   - **H2 landed (trending):** Sprint 37 extends to 20 seeds on the
     same graph / same slice to close the power gap before deciding
     between the H0 and H1 follow-ups above

Each of those three follow-ups is a concrete Sprint 37 trigger, which
is the working definition of a good Sprint 36 exit.

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
- Engine graph usage: `causal_optimizer/optimizer/suggest.py` lines 1122–1134 (`_get_focus_variables`)
- `CausalGraph` type: `causal_optimizer/types.py` lines 167–263
- `DomainAdapter.get_prior_graph` contract:
  `causal_optimizer/domain_adapters/base.py` line 39
- Existing prior-graph examples:
  `causal_optimizer/domain_adapters/marketing_logs.py` line 362
  (`MarketingLogAdapter.get_prior_graph`),
  `causal_optimizer/benchmarks/criteo.py` line 220
  (`criteo_projected_prior_graph`)
