# Sprint 13b — Coverage Diagnostics Fix + Energy Adapter Hardening

## Context

Post-merge audit surfaced two issues (GitHub #48, #49). One is a diagnostics bug where discarded experiments are invisible to coverage analysis, producing misleading "never intervened" reports. The other hardens the energy adapter's time-axis validation to fail fast on duplicate timestamps and surface cadence gaps.

**Base branch:** `main`

Read `CLAUDE.md` for project conventions. Read the GitHub issues for full context:
- `gh issue view 48` — Coverage diagnostics should count discarded experiments
- `gh issue view 49` — EnergyLoadAdapter should fail fast on duplicates and surface cadence gaps

---

## Prompt

Stand up one agent per feature. Each agent must follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Features this sprint:

- **sprint-13b/coverage-discarded** (GitHub #48) — Fix `analyze_coverage()` in `diagnostics/coverage.py` so discarded experiments count toward intervention coverage. (1) Change the status filter on line 36 from `status == KEEP` to `status != CRASH` — this includes both KEEP and DISCARD experiments when computing `varied_vars`. A variable was "tested" if it was varied in any non-crash experiment, regardless of whether the result was kept. (2) Add a second set `kept_varied_vars` that filters to KEEP-only, for downstream consumers that care about the retained frontier. Add `kept_varied_vars` as a new field on `CoverageAnalysis` in `diagnostics/models.py` — default to `None`, populate when available. (3) The existing `ancestors_never_intervened` and POMIS unexplored lists should use the broader `varied_vars` (all non-crash). (4) Add unit tests in `tests/unit/test_coverage.py`: reproduce the exact scenario from the issue — one KEEP experiment and one DISCARD experiment for the same variable — and assert `ancestors_never_intervened` does NOT include that variable. Also test that a variable only present in CRASH experiments IS reported as never intervened. Also test that the new `kept_varied_vars` field correctly reflects only KEEP experiments. Reference: `causal_optimizer/diagnostics/coverage.py:34-42`, `causal_optimizer/diagnostics/models.py`.

- **sprint-13b/energy-time-validation** (GitHub #49) — Harden `EnergyLoadAdapter` time-axis handling to fail fast on duplicates and surface cadence gaps. (1) Change duplicate timestamp handling in `energy_load.py:86-92`: instead of silently dropping duplicates with a warning, **raise `ValueError`** with message `"Found {n} duplicate timestamps. This adapter requires single-series data with unique timestamps. If you have multi-area data, filter to one area before passing."` This is a breaking behavior change — the issue explicitly says the adapter should fail fast, not silently drop. (2) After sorting, infer the dominant cadence using `pd.Series.diff().mode()` on the timestamp column. Add a `cadence_gaps` metric to `run_experiment()` output: count of rows where the gap to the previous row is >1.5x the dominant cadence. This surfaces irregular data without failing. (3) Add a `cadence_regularity` metric: fraction of consecutive rows that match the dominant cadence (within 10% tolerance). Value of 1.0 means perfectly regular; 0.5 means half the rows have unexpected gaps. (4) Store the inferred cadence as `self._cadence` (a `pd.Timedelta`) in `__init__` after sorting, for use in `run_experiment()`. (5) Update unit tests in `tests/unit/test_energy_load_adapter.py`: the existing `test_duplicate_timestamps_handled` test now expects a `ValueError` instead of success — update the assertion. Add new tests: `test_duplicate_timestamps_raises` (explicit), `test_cadence_metrics_present` (assert both new metrics exist), `test_regular_cadence_near_one` (fixture data is hourly, expect `cadence_regularity > 0.95`), `test_irregular_cadence_detected` (drop every 3rd row from fixture, expect `cadence_regularity < 0.8` and `cadence_gaps > 0`). Reference: `causal_optimizer/domain_adapters/energy_load.py:79-92`.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Each agent works in an isolated worktree
- Do NOT merge — leave PRs open for human review
- Merge order after human approval: coverage-discarded → energy-time-validation (independent, but coverage is smaller/safer to merge first)
- After each merge, next PR rebases on main and re-runs `uv run pytest -m "not slow"`
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
