Work on Sprint 38: land Option B — Open Bandit prior-graph widening
with one non-search-space structural node
(`logged_position_distribution`) — and run the Sprint 37-shape rerun
against the widened graph.

**Terminology.** The orchestration prompt and the Sprint 36 plan
used the phrase "non-ancestor structural node" for this
(`logged_position_distribution`); the Sprint 38 recommendation
retires that phrase because after adding the edge
`logged_position_distribution -> position_handling_flag`, the node
*is* an ancestor of `policy_value` via the two-hop path. The A1
binding condition still holds because the node is not a
search-space variable, not because it is a non-ancestor. See the
Sprint 38 recommendation's "Sprint Theme" terminology note.

Canonical source: the Sprint 38 recommendation is authoritative for
every contract decision (graph shape, hard constraints,
anti-patterns, Exit Criterion, what is in vs out of scope). This
prompt duplicates those items so an implementation agent has a
self-contained brief, but any conflict between the two documents is
resolved in favor of the recommendation. If you amend this prompt,
update the recommendation first.

All paths below are **repo-relative** — resolve them from the root of
your `causal-optimizer` checkout (or worktree).

Primary recommendation:
- `thoughts/shared/plans/27-sprint-38-recommendation.md`

Read first:
1. `CLAUDE.md`
2. `thoughts/shared/plans/27-sprint-38-recommendation.md`
3. `thoughts/shared/docs/sprint-37-open-bandit-prior-graph-report.md`
4. `thoughts/shared/plans/26-sprint-36-recommendation.md`
5. `thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md`
6. `thoughts/shared/docs/sprint-34-open-bandit-contract.md`
7. `thoughts/shared/docs/handoff.md`
8. `thoughts/shared/plans/07-benchmark-state.md`
9. `causal_optimizer/domain_adapters/bandit_log.py`
   Specifically re-read `run_experiment` lines 346–475 (the row-mask
   branch at 352–369 is the code site that grounds the
   `logged_position_distribution -> position_handling_flag` edge) and
   `get_prior_graph` lines 513–541 (the function you are editing).
10. `causal_optimizer/benchmarks/open_bandit.py`
11. `causal_optimizer/benchmarks/open_bandit_benchmark.py`
    Specifically the `pomis_minimal_focus=(strategy == "causal")`
    wiring at line 541 — Sprint 38 does not change this. The flag
    stays on for the `causal` arm only.
12. `causal_optimizer/types.py` (`CausalGraph`)
13. `causal_optimizer/optimizer/suggest.py`
    All the `causal_graph` read sites the Sprint 36 plan enumerated,
    in particular `_get_focus_variables` (lines 1170–1180),
    `_apply_minimal_focus_a1` (lines 1183–1236), and
    `_causal_alignment_score` (lines 982–1021). Sprint 38 does **not**
    edit any of these — the sprint is graph-only.
14. `tests/unit/test_bandit_log_prior_graph.py`
15. `tests/unit/test_a1_minimal_focus.py`

Goal:
- Ship an implementation PR that (a) widens the Open Bandit
  preregistered prior graph by exactly one node
  (`logged_position_distribution`) and exactly one directed edge
  (`logged_position_distribution -> position_handling_flag`), (b)
  updates the unit tests that pin the graph shape, and (c) reruns
  the Sprint 37-shape Men/Random benchmark against the widened
  graph and writes a Sprint 38 report.
- No engine change, no optimizer change, no new heuristic, no new
  slice, no second dataset, no new search-space variable, no
  bidirected edge, no relaxation of any Section 7 gate.

Produce in one PR:

1. Updated adapter: `causal_optimizer/domain_adapters/bandit_log.py`
   - `get_prior_graph()` returns the eight-node / seven-edge widened
     graph from the Sprint 38 recommendation's "Minimal Preregistered
     Graph (Sprint 38 widening)" section, verbatim.
   - **Implementation note.** The Sprint 37 `get_prior_graph` body
     programmatically enumerates the search space:
     `edges = [(name, "policy_value") for name in self.get_search_space().variable_names]`
     (`bandit_log.py` line 540). That pattern cannot express the new
     Sprint 38 edge, because
     `logged_position_distribution -> position_handling_flag` has a
     source that is not a search-space variable. The implementation
     must switch to an explicit edge list — build
     `search_edges = [(name, "policy_value") for name in self.get_search_space().variable_names]`
     and then extend with the Sprint 38 edge, or write out all seven
     edges as literal tuples. Do not keep the list-comprehension-only
     pattern; it silently drops the new edge.
   - Unit tests must pin edges by name tuple (see item 2 below), not
     by counting `len(search_space.variables)`. The Sprint 37 shape of
     `len(edges) == len(search_space.variables)` no longer holds
     under the widened graph (seven edges, six search-space
     variables).
   - The function's docstring cites Sprint 38 (this document and the
     plan) and keeps the Sprint 37 lane reference for context.
   - No other edits to `bandit_log.py`. The search space stays at six
     variables; `run_experiment` is not edited; `to_bandit_feedback`
     is not edited.

2. Updated unit tests: `tests/unit/test_bandit_log_prior_graph.py`
   - Pin the eight-node graph: exactly
     `{"tau", "eps", "w_item_feature_0", "w_user_item_affinity",
     "w_item_popularity", "position_handling_flag", "policy_value",
     "logged_position_distribution"}`.
   - Pin the seven directed edges: the six Sprint 37 edges plus the
     new `("logged_position_distribution", "position_handling_flag")`
     edge.
   - Pin `bidirected_edges` is empty.
   - Pin `ancestors(policy_value)` to include
     `logged_position_distribution` (via the two-hop path through
     `position_handling_flag`).
   - Pin that `logged_position_distribution` is **not** a search-space
     variable.
   - Keep every Sprint 37 assertion that is still true; update only
     the cells that changed.

3. Sprint 38 benchmark rerun:
   - Execute the Sprint 37 reproducibility command verbatim
     (reproduced in the Sprint 38 recommendation's "Strategy"
     section). No CLI-flag changes; the widening lives inside
     `BanditLogAdapter.get_prior_graph()`.
   - Write the artifact JSON to the Sprint 38 path under
     `artifacts/sprint-38-open-bandit-graph-widening/men_random_results.json`
     (the `--output` flag in the reproducibility command already
     points here).
   - The artifact JSON is **local only**, not committed, matching the
     Sprint 35 / Sprint 37 convention.

4. Sprint 38 benchmark report:
   `thoughts/shared/docs/sprint-38-open-bandit-graph-widening-report.md`
   - Same shape as
     `thoughts/shared/docs/sprint-37-open-bandit-prior-graph-report.md`.
   - Summary names the Sprint 38 verdict row under the Sprint 33 /
     Sprint 35 / Sprint 37 scorecard labels (certified / trending /
     not significant), two-sided MWU convention.
   - Verdict rows for B20 / B40 / B80, `causal` vs
     `surrogate_only` vs `random`, 10 seeds each cell.
   - Per-budget outcome tables (population std, ddof=0 — same as
     Sprint 37).
   - Per-seed detail tables at B80 and B20 (for trajectory
     diagnosis, not for a B20 verdict).
   - Secondary estimator rows (DM and DR), with the DR / SNIPW
     cross-check number for Section 7e.
   - Section 7 support gates, all five, with the same thresholds
     and observed-value columns as Sprint 37.
   - ESS diagnostics table at B80 (the five columns Sprint 37's
     "ESS Diagnostics" table used).
   - Hypothesis reconciliation against the preregistered H0 / H1 /
     H2 copied verbatim from the Sprint 38 recommendation. **Do not
     re-tune the hypothesis text.**
   - Interpretation section: explicitly answer "did the Sprint 37
     engine-path no-op prediction hold under the widened graph, yes
     or no, and with what p-value?"
   - Scope boundaries section: same list as Sprint 37's "Scope
     Boundaries" section.
   - Next-move section: name Sprint 39's next move under the four
     branches the Sprint 38 recommendation's "What Happens After
     Sprint 38" section enumerates (exhausted-D /
     trending-diagnostic / certified-positive-H1 /
     certified-negative-H1).

5. Docstring + reference updates in the adapter:
   - `BanditLogAdapter.get_prior_graph` docstring cites the Sprint 38
     recommendation by path.
   - Inline comment near the `get_prior_graph` return explains the
     two-hop ancestor path through `logged_position_distribution`.

6. Nothing else. In particular, do **not**:
   - edit `causal_optimizer/optimizer/suggest.py`
   - edit `causal_optimizer/engine/loop.py`
   - edit `causal_optimizer/benchmarks/open_bandit.py` or
     `causal_optimizer/benchmarks/open_bandit_benchmark.py`
   - add any new engine flag
   - add a second new node (including `request_item_overlap`)
   - add any bidirected edge
   - edit `tests/unit/test_a1_minimal_focus.py`
     (the A1 helper behavior is unchanged under the widened graph
     — the binding condition "every search-space variable is an
     ancestor of `policy_value`" still holds)
   - edit `thoughts/shared/docs/handoff.md` or
     `thoughts/shared/plans/07-benchmark-state.md` (Track A will
     sync those after Sprint 38 lands)
   - edit `README.md`
   - edit the Sprint 35 or Sprint 37 reports
   - relax any Section 7 gate threshold
   - change the null-control permutation seed away from `20260419`
   - mix RF-fallback and Ax/BoTorch verdict cells in the report

Hard constraints (inherited from the Sprint 38 recommendation's
"What Sprint 38 Must Not Do" section):

1. **Exactly one new node**, `logged_position_distribution`. No
   `request_item_overlap`. No third node. Mid-sprint scope expansion
   is the specific anti-pattern the Sprint 36 review caught.
2. **Exactly one new edge**,
   `logged_position_distribution -> position_handling_flag`. No edge
   from the new node directly to `policy_value`. No reverse
   direction.
3. **No bidirected edges.** Sprint 34 contract Section 4e.
4. **No engine or optimizer change.** `pomis_minimal_focus` stays
   default-off at the engine level; the benchmark harness keeps it
   `True` only on the `causal` arm — same wiring as Sprint 37
   (`open_bandit_benchmark.py` line 541).
5. **No new heuristic.** Option C is deferred to Sprint 39+ if
   Sprint 38's outcome warrants it.
6. **No new slice.** Men/Random only.
7. **No DRos-primary.** SNIPW is the primary estimator.
8. **No slate-level OPE.**
9. **No second dataset.**
10. **No power extension on A1.** Sprint 37's B80 row is unambiguous.
11. **No B20 chase.** Sprint 38's verdict row is B80.
12. **No Section 7 relaxation.**
13. **No auto-discovery overlay.**
14. **No multi-objective extension.**
15. **No force-push, no hook skipping, no auto-merge.**

Workflow (mandatory, in this order):

```text
/tdd -> implement -> /polish -> gh pr create -> /gauntlet -> report PR URL
```

- Start from a fresh worktree based on `origin/main` after the
  Sprint 38 planning PR merges.
- `/tdd`: write failing tests first that pin the widened graph shape
  (node set, edge set, absence of bidirected edges, two-hop ancestor
  path, `logged_position_distribution` not in search space). Then
  implement the adapter change to make them pass.
- `/polish`: run before creating the PR. Do not skip. Fix every
  simplify / lint / format / mypy / coverage issue it surfaces.
- `gh pr create`: PR title under 70 characters, e.g.
  "feat: sprint 38 open bandit graph widening (option B)". Body
  uses a HEREDOC, references the Sprint 38 issue, and includes a
  Summary + Test plan.
- `/gauntlet`: run after the PR is open. Let it iterate until
  greploop + claudeloop + check-pr all pass.
- **Do not merge.** Human review and explicit approval are required
  per the PR merge policy in `CLAUDE.md`.

Falsifiable outcome (copied verbatim from the Sprint 38 recommendation):

> **H0 (predicted).** On the Men/Random slice, B80 two-sided
> Mann-Whitney U p-value on SNIPW policy values between `causal` and
> `surrogate_only`, 10 seeds per arm, satisfies `p > 0.15`. This is
> the same H0 Sprint 37 confirmed. The Sprint 38 engine-path
> analysis predicts the widened graph is a no-op along every path
> that already existed under Sprint 37's minimal-focus wiring, so
> the B80 row should recapitulate the Sprint 37 near-parity.
>
> The prediction assumes `causal_softness = 0.5`,
> `causal_exploration_weight = 0.0`, `strategy` routes through
> `_suggest_bayesian` (not `_suggest_causal_gp`), Ax/BoTorch is
> available (no RF fallback), `pomis_minimal_focus = True` on the
> `causal` arm only, and no new engine path reads `causal_graph`
> between Sprint 37 and the Sprint 38 rerun.

> **H1 (alternative — certified).** `p <= 0.05` at B80. A certified
> row in either direction invalidates the Sprint 38 no-op
> prediction and forces a diagnostic rerun before Sprint 39 plans.

> **H2 (trending).** `0.05 < p <= 0.15` at B80. Two-sided; direction
> reported as observed.

Success gates that must hold for the Sprint 38 verdict row to
publish:

1. all five Section 7 gates PASS at Sprint 34 contract thresholds
2. null-control permutation seed is `20260419`
3. backend provenance is `ax_botorch` on every verdict cell; no
   RF-fallback mixing

Reproducibility command (Sprint 38):

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

The CLI requires no new flags — the widening lives inside
`BanditLogAdapter.get_prior_graph()` and the benchmark harness
already enables `pomis_minimal_focus` only on the `causal` arm.

Deliverables (report back in the PR body):

1. PR URL
2. the observed B80 verdict row and p-value
3. whether H0 held, and with what Sprint 33 label
4. all five Section 7 gate statuses
5. the Sprint 39 next-move branch Sprint 38's outcome selected
   (exhausted-D / trending-diagnostic / certified-positive-H1 /
   certified-negative-H1)
6. gauntlet status

Do not merge. Leave the PR open for human review.
