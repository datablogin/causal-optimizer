Work on Sprint 36: preregister the first Open Bandit prior graph and
the engine-surface change that would make it matter. Defer the rerun
to Sprint 37.

Canonical source: the Sprint 36 recommendation below is authoritative
for every contract decision (graph shape, hard constraints,
anti-patterns, the Option A / Option B Exit Criterion, what is in vs
out of scope). This prompt duplicates those items so an
implementation agent has a self-contained brief, but any conflict
between the two documents is resolved in favor of the recommendation.
If you amend this prompt, update the recommendation first.

All paths below are **repo-relative** — resolve them from the root of
your `causal-optimizer` checkout (or worktree).

Primary recommendation:
- `thoughts/shared/plans/26-sprint-36-recommendation.md`

Read first:
1. `CLAUDE.md`
2. `thoughts/shared/plans/26-sprint-36-recommendation.md`
3. `thoughts/shared/docs/sprint-34-open-bandit-contract.md`
4. `thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md`
5. `thoughts/shared/docs/handoff.md`
6. `thoughts/shared/plans/07-benchmark-state.md`
7. `causal_optimizer/domain_adapters/bandit_log.py`
8. `causal_optimizer/benchmarks/open_bandit.py`
9. `causal_optimizer/benchmarks/open_bandit_benchmark.py`
10. `causal_optimizer/types.py` (`CausalGraph`)
11. `causal_optimizer/optimizer/suggest.py`
    All seven `causal_graph` read sites enumerated in the Sprint 36
    recommendation's "Why This Sprint Now" section. In particular,
    confirm both the ancestor-based focus behavior (lines 1122–1134)
    **and** the soft-causal reranker behavior (lines 751–812) before
    writing the preregistered graph spec. Missing the soft-causal
    path is the specific misreading the PR #195 review flagged in an
    earlier draft.
12. `causal_optimizer/domain_adapters/marketing_logs.py`
    (existing prior-graph example: `MarketingLogAdapter.get_prior_graph`)
13. `causal_optimizer/benchmarks/criteo.py`
    (existing prior-graph example: `criteo_projected_prior_graph`)

Goal:
- Ship a docs-only preregistration PR that (a) authors the first
  semantically correct prior graph for the Men/Random Open Bandit
  slice as a preregistered spec, (b) documents how the current engine
  surface would consume that exact graph path by path, and (c) names
  exactly one Option A engine change and one Option B graph widening
  that Sprint 37 could pick between.
- The implementation must not change any Python file. No adapter
  edits, no engine edits, no test changes, no benchmark rerun. If
  you find yourself opening `causal_optimizer/domain_adapters/bandit_log.py`
  in an editor for anything other than reading line numbers, stop
  and re-read the Sprint 36 recommendation's "What Sprint 36 Must
  Not Do" section.

Produce in one PR:

1. Updated Sprint 36 recommendation:
   `thoughts/shared/plans/26-sprint-36-recommendation.md`
   - Matches the structure of the canonical recommendation merged on
     this branch. If you are re-running this prompt after a further
     rescope, confirm the committed file still reflects the
     preregistration-only framing before editing.

2. Updated Sprint 36 implementation prompt (this file):
   `thoughts/shared/prompts/sprint-36-open-bandit-prior-graph.md`
   - If you amend the scope of Sprint 36 again, the prompt must not
     drift back into asking for the adapter change or the rerun.

3. Nothing else. In particular, do **not**:
   - edit `causal_optimizer/domain_adapters/bandit_log.py`
   - edit `tests/unit/test_bandit_log_adapter.py`
   - regenerate any benchmark artifact JSON
   - edit `thoughts/shared/docs/handoff.md` or
     `thoughts/shared/plans/07-benchmark-state.md` (Track A owns those)
   - edit `README.md` (Track A owns that)
   - edit the Sprint 35 report

Hard constraints (inherited from the Sprint 36 recommendation):

1. No Python change in this sprint. `BanditLogAdapter.get_prior_graph`
   stays as `return None` until Sprint 37.
2. No benchmark rerun in this sprint.
3. Do not widen the preregistered graph to chase differentiation.
4. Do not add bidirected edges to the first graph.
5. Do not add edges into `ess`, `weight_cv`, `max_weight`,
   `zero_support_fraction`, or `n_effective_actions`.
6. Do not change any Section 7 gate threshold.
7. Do not touch `causal_optimizer/optimizer/`,
   `causal_optimizer/engine/`, or the OPE stack.
8. Do not preregister a run on Women, All, BTS, a second dataset,
   DRos-primary, or slate OPE.
9. Do not relabel a trending row as certified, or vice versa, in the
   Sprint 37 preregistration.
10. Do not preregister a run that mixes Ax/BoTorch and RF-fallback
    within a verdict row.
11. Do not auto-merge the PR. Stop after `/gauntlet` and wait for
    human approval.

Null-control permutation seed — source of truth:

- The authoritative source for the Sprint 37 rerun's null-control
  permutation seed is the committed Sprint 35 report,
  `thoughts/shared/docs/sprint-35-open-bandit-benchmark-report.md`.
  That report records `permutation seed = 20260419` in both its
  "Configuration" block (line 83 at the time of writing) and the
  Section 7a header. This is the single source Sprint 36 must cite
  and Sprint 37 must use.
- The Sprint 35 run also produced a local artifact JSON at
  `artifacts/sprint-35-open-bandit-benchmark/men_random_results.json`
  (local-only, not committed — see the Sprint 35 report, "Data
  Provenance" section). That artifact's `provenance.permutation_seed`
  field is optional *corroboration* only; a fresh agent on a clean
  checkout is not required to have it. If the artifact is present,
  an agent may cross-check the report's value against it, but a
  mismatch is diagnosed by re-reading the report, not by trusting
  the local artifact.
- Sprint 36 itself does not run the benchmark, so neither source is
  strictly required for this sprint. The seed is called out here so
  that the Sprint 37 follow-up issue inherits the correct pointer.

Environment prerequisites:

- None beyond a normal repo checkout. Sprint 36 does not run the
  benchmark; Ax/BoTorch, the `bandit` extra, and the full Men/Random
  slice at `--data-path` are **not** required for this sprint and
  must not be listed as blockers on the Sprint 36 PR.
- Sprint 37 will inherit the Sprint 35 environment prerequisites
  (ax-platform, botorch, and OBP installed; full Men/Random slice
  local) unchanged. Capturing those as Sprint 37 prerequisites is
  part of the Sprint 37 issue the Sprint 36 close-out creates, not
  part of this PR.

Required workflow:

1. Review the merged Sprint 36 recommendation end to end. Confirm
   that it names all seven `causal_graph` read sites in
   `causal_optimizer/optimizer/suggest.py`, that the per-edge code
   citations in the "Minimal Preregistered Graph" section match the
   current `bandit_log.py` line numbers, and that the Option A /
   Option B Exit Criterion fits in a single Sprint 37 PR each.
2. If the engine surface in `suggest.py` has changed since the
   recommendation was written (check `git log -p
   causal_optimizer/optimizer/suggest.py` against the branch point),
   update the line-number citations in the recommendation *before*
   anything else. The recommendation's engine-surface analysis is
   load-bearing — stale line numbers invalidate the preregistration.
3. Run the `polish` skill on the two changed files
   (`thoughts/shared/plans/26-sprint-36-recommendation.md` and
   `thoughts/shared/prompts/sprint-36-open-bandit-prior-graph.md`).
   Address every finding.
4. Open a PR with base `main`. Title roughly
   "Sprint 36: preregister first Open Bandit prior graph (docs-only)".
   Link the Sprint 36 issue with `Closes #<n>`.
5. Run the `gauntlet` skill on the PR. Iterate until clean.
6. Deliver back the PR URL and gauntlet status. Do not merge.
7. After human approval merges the PR, open the Sprint 37 issue
   titled roughly "Sprint 37: Open Bandit prior graph rerun (Option
   A or Option B)" and record in the Sprint 36 close-out comment
   which option was picked and why.

Reporting discipline (preserved for when Sprint 37's report is
written, not required in Sprint 36 itself):

- All p-values are two-sided Mann-Whitney U unless otherwise noted.
- Reserve "winner" for `p <= 0.05`; use "trending" for
  `0.05 < p <= 0.15`; use "near-parity" for within-noise identical
  distributions.
- Population std (ddof=0) in report tables; sample-pooled std
  (ddof=1) only when Cohen's d is quoted.
- Record backend provenance explicitly on every verdict cell.
- Null-control seed must be `20260419`.

Deliverables to include in the PR body:

1. one-paragraph summary citing the rescoped Sprint 36
   recommendation and the PR #195 review that prompted the rescope
2. the preregistered seven-node / six-edge graph printed explicitly
   (a fenced block is fine)
3. a one-paragraph engine-surface summary covering all seven
   `causal_graph` read sites, explicitly calling out the soft-causal
   reranker (`suggest.py:751`) as active under the default engine
   config
4. the Option A engine change and the Option B graph widening named
   in the Exit Criterion, one sentence each
5. a Test plan checkbox list covering: docs-only diff (no Python
   files touched), engine-surface line numbers still match `HEAD`,
   preregistration matches the `BanditLogAdapter` scoring code,
   Sprint 37 issue stub drafted

Anti-patterns to flag and refuse:

- "Author the graph as code too, it's only a few lines" — the point
  of Sprint 36 is that authoring the graph without a matching engine
  surface change is what the Sprint 35 tie already told us about. The
  adapter edit belongs in Sprint 37 gated by Option A or Option B.
- "Run the benchmark anyway as a sanity check" — Sprint 35 is the
  sanity check. Rerunning before naming the engine change that would
  move the verdict is the scope expansion PR #195 asked the plan to
  close.
- "Drop `w_item_feature_0` because the optimizer picked a small
  weight" — post-hoc noise chasing. The adapter's scoring code makes
  it a direct ancestor; do not drop it from the preregistered graph.
- "Add a bidirected edge between `ess` and `weight_cv` because they
  are correlated" — correlated by shared computation, not by an
  unobserved confounder.
- "Require the local Sprint 35 artifact JSON as a Sprint 36 (or
  Sprint 37) prerequisite" — it is not committed; the committed
  report text is the source of truth for the permutation seed and
  every other Sprint 35 configuration value.
