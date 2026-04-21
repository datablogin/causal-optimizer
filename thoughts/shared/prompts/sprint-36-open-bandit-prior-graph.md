Work on Sprint 36: first Open Bandit prior graph and Men/Random rerun.

Canonical source: the Sprint 36 recommendation below is authoritative
for every contract decision (graph shape, hard constraints,
anti-patterns, Section 7 gates). This prompt duplicates those items
so an implementation agent has a self-contained brief, but any
conflict between the two documents is resolved in favor of the
recommendation. If you amend this prompt, update the recommendation
first.

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
    (lines 1122–1134 `_get_focus_variables`; confirm ancestor-based
    focus behavior before touching the graph)
12. `causal_optimizer/domain_adapters/marketing_logs.py`
    (existing prior-graph example: `MarketingLogAdapter.get_prior_graph`)
13. `causal_optimizer/benchmarks/criteo.py`
    (existing prior-graph example: `criteo_projected_prior_graph`)

Goal:
- Author the first semantically correct prior graph for the
  Men/Random Open Bandit slice, rerun the Sprint 35 benchmark
  with that graph active, and publish a preregistered-hypothesis
  verdict row in a new sprint-36 report.
- The implementation must change exactly one adapter function
  (`BanditLogAdapter.get_prior_graph`) plus the tests that cover
  it; engine code is out of scope.

Produce in one PR:

1. Updated adapter:
   `causal_optimizer/domain_adapters/bandit_log.py`
   - Replace `get_prior_graph(self) -> CausalGraph | None: return None`
     with a returned `CausalGraph` matching the Sprint 36 recommendation
     exactly. Required shape:
     - 7 nodes: `tau`, `eps`, `w_item_feature_0`,
       `w_user_item_affinity`, `w_item_popularity`,
       `position_handling_flag`, `policy_value`
     - 6 directed edges, each from a search variable to `policy_value`
     - 0 bidirected edges
   - Keep the existing docstring content about Sprint 34 contract
     Section 4e — rewrite it to cite Sprint 36 now that the adapter
     ships a non-null graph.

2. Adapter tests (new or extended):
   `tests/unit/test_bandit_log_adapter.py`
   (the file already exists from Sprint 35.A; extend it in place)

   First, delete or rewrite the existing
   `test_prior_graph_is_none_by_default` at
   `tests/unit/test_bandit_log_adapter.py:112`. That assertion
   (`adapter.get_prior_graph() is None`) encodes the Sprint 35
   behavior and will fail the moment Sprint 36's graph lands; replace
   it with the six tests below rather than leaving it as a broken
   "known failing" test.
   - `test_get_prior_graph_returns_expected_nodes` — exactly the 7
     nodes above.
   - `test_get_prior_graph_has_six_directed_edges` — exactly 6 edges.
   - `test_every_search_variable_is_ancestor_of_policy_value` — call
     `graph.ancestors("policy_value")` and compare.
   - `test_no_bidirected_edges` — `graph.bidirected_edges == []`.
   - `test_search_space_variables_match_graph_nodes_minus_outcome` —
     every name returned by `get_search_space().variable_names` is a
     non-`policy_value` node of the returned graph.
   - `test_focus_variables_equals_full_search_space_under_prior_graph`
     — call `causal_optimizer.optimizer.suggest._get_focus_variables(
     adapter.get_search_space(), adapter.get_prior_graph(),
     adapter.get_objective_name())` and assert it returns a list whose
     set equals `set(adapter.get_search_space().variable_names)`. This
     mechanically verifies the Sprint 36 recommendation's H0
     prediction before the expensive rerun runs. Annotate the test
     with a comment flagging that `_get_focus_variables` is a private
     helper (leading underscore) so a future refactor of that module
     knows to update this test alongside the implementation.

3. Benchmark rerun and report:
   - Regenerate the Men/Random artifact JSON by running
     `scripts/open_bandit_benchmark.py` with the same flags the Sprint
     35 report documents (same `--data-path`, `--budgets 20,40,80`,
     `--seeds 0,1,2,3,4,5,6,7,8,9`, `--strategies
     random,surrogate_only,causal`, `--null-control`,
     `--permutation-seed 20260419`). The rerun must execute under the
     same Ax/BoTorch backend.
   - Write:
     `thoughts/shared/docs/sprint-36-open-bandit-prior-graph-report.md`
     Same sectioning as the Sprint 35 report. In addition it must:
     - embed the full seven-node / six-edge graph in the Configuration
       section, including the per-edge code-line justifications copied
       from the Sprint 36 recommendation
     - quote preregistered H0 / H1 / H2 from the recommendation
       verbatim before the verdict table
     - classify the verdict under the unchanged Sprint 33 scorecard
       labels (`p <= 0.05` certified, `0.05 < p <= 0.15` trending,
       `p > 0.15` not significant, `within-noise identical` near-parity)

4. Restart-doc updates (only if the rerun's outcome materially
   changes the current scorecard line):
   - `thoughts/shared/docs/handoff.md`
   - `thoughts/shared/plans/07-benchmark-state.md`
   If H0 is confirmed and the tie persists, a one-sentence update to
   each file is sufficient. Do not rewrite prior-sprint sections.

Hard constraints (inherited from Sprint 36 recommendation):

1. Do not widen the graph after the rerun.
2. Do not add bidirected edges to the first graph.
3. Do not add edges into `ess`, `weight_cv`, `max_weight`,
   `zero_support_fraction`, or `n_effective_actions`.
4. Do not change any Section 7 gate threshold.
5. Do not touch `causal_optimizer/optimizer/`,
   `causal_optimizer/engine/`, or the OPE stack.
6. Do not run on Women, All, BTS, a second dataset, DRos-primary, or
   slate OPE.
7. Do not relabel a trending row as certified, or vice versa.
8. Do not mix Ax/BoTorch and RF-fallback within a verdict row.
9. Do not auto-merge the PR. Stop after `/gauntlet` and wait for human
   approval.

Environment prerequisites:

- Ax/BoTorch must be installed and importable before the rerun starts
  (`uv sync --extra bayesian` or `uv sync --extra all`). The
  recommendation's "no RF-fallback mixing" rule means a BoTorch import
  failure blocks the sprint; verify `ax-platform` and `botorch` both
  load in a Python REPL before running the benchmark. If either
  import still fails after `uv sync --extra all`, stop and report the
  failure to the human reviewer — do not fall back to RF and do not
  publish a verdict row.
- The `bandit` extra (OBP) must be installed to load the Men/Random
  slice: `uv sync --extra bandit`.
- The full Men/Random slice must be available locally at the Sprint 35
  `--data-path` location; do not substitute the 10,000-row OBP-bundled
  sample.
- Confirm the null-control permutation seed matches Sprint 35 by
  reading the Sprint 35 artifact JSON's provenance dict (key
  `permutation_seed`), not just the report text. The two must agree at
  `20260419` for the null-control comparison to be apples-to-apples.
- Before launching the rerun, confirm the CLI flags the Sprint 35
  report quotes still exist on the current `scripts/open_bandit_benchmark.py`
  (run `uv run python scripts/open_bandit_benchmark.py --help` and
  check for `--data-path`, `--budgets`, `--seeds`, `--strategies`,
  `--null-control`, `--permutation-seed`, `--output`). If any flag
  has been renamed or removed since Sprint 35, stop and report —
  silently adapting flag names would invalidate the
  apples-to-apples comparison.

Required workflow:

1. Use the `tdd` skill first:
   - write the six adapter tests from Produce item 2 above; confirm
     they fail against the current `return None` implementation
2. Implement the graph change in `bandit_log.py`; run the adapter tests
   locally until green; run the full fast test suite (`uv run pytest
   -m "not slow"`) to confirm no regressions.
3. Re-run `scripts/open_bandit_benchmark.py` as described above and
   capture the artifact JSON.
4. Write the Sprint 36 report; copy the per-edge justifications and
   preregistered H0/H1/H2 text from the Sprint 36 recommendation.
5. Run the `polish` skill on the changed files; address every finding.
6. Open a PR with base `main`. Title roughly
   "Sprint 36: first Open Bandit prior graph and Men/Random rerun".
   Link the Sprint 36 issue with `Closes #<n>`.
7. Run the `gauntlet` skill on the PR. Iterate until clean.
8. Deliver back the PR URL and gauntlet status. Do not merge.

Reporting discipline (carried over from Sprint 35):

- All p-values are two-sided Mann-Whitney U unless otherwise noted.
- Reserve "winner" for `p <= 0.05`; use "trending" for `0.05 < p <=
  0.15`; use "near-parity" for within-noise identical distributions.
- Population std (ddof=0) in report tables; sample-pooled std
  (ddof=1) only when Cohen's d is quoted.
- Record backend provenance explicitly on every verdict cell.
- Null-control seed must be `20260419`.

Deliverables to include in the PR body:

1. one-paragraph summary citing the Sprint 36 recommendation
2. the seven-node / six-edge graph printed explicitly (a fenced block
   is fine)
3. the H0 / H1 / H2 preregistered hypotheses, copied verbatim
4. the B80 verdict row with the chosen Sprint 33 label
5. a Test plan checkbox list covering: unit tests green, fast suite
   green, rerun Section 7 gates green, verdict preregistered

Anti-patterns to flag and refuse:

- "Drop `w_item_feature_0` because the optimizer picked a small
  weight" — post-hoc noise chasing. The adapter's scoring code makes
  it a direct ancestor; do not drop it.
- "Add a bidirected edge between `ess` and `weight_cv` because they
  are correlated" — correlated by shared computation, not by an
  unobserved confounder.
- "Tighten the 7a null-control band because the rerun inflated past
  it" — a gate failure is a blocker; fix the run, not the band.
