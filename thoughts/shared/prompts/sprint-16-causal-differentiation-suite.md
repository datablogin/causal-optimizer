# Sprint 16 — Causal Differentiation + Multi-Benchmark Suite

## Context

The first real benchmark (Sprint 15, PR #67) proved the harness works but showed
that `causal` and `surrogate_only` are **identical** in every run. The benchmark
is now a system-learning milestone, not a product win.

This sprint delivers the three changes recommended in
`thoughts/shared/plans/08-optimizer-improvement-briefs.md`:

1. Make the causal path genuinely different under the RF-surrogate fallback
2. Add a second real benchmark (ERCOT COAST + Houston weather)
3. Build a multi-benchmark evaluation suite and re-run both benchmarks

## Root Cause Analysis

Why `causal == surrogate_only` on every seed, budget, and metric:

1. The energy adapter's prior graph is a **star**: all 7 search variables are
   direct parents of `mae`.
2. `_get_focus_variables()` calls `causal_graph.ancestors("mae")` which returns
   all 7 variables — the same set returned when `causal_graph is None`.
3. Without Ax/BoTorch, both strategies fall back to `_suggest_surrogate()` with
   identical `focus_variables`.
4. No bidirected edges → POMIS returns one set containing all variables → no
   pruning.
5. Result: the causal graph is present but has zero effect on optimization.

The fix must inject causal information into the RF-surrogate path in a way that
changes behavior even when all variables are ancestors.

## Design Principle

From Plan 08: improve the optimizer in a domain-general way. Do not optimize
for ERCOT NORTH_C specifically. Every change must be judged across both
benchmarks.

## Steps

Steps 1 and 2 are **independent** and should be run in parallel (separate
worktrees). Step 3 depends on both being merged first.

```
Phase A (parallel):
  Step 1: Causal Fallback Differentiation  →  PR  →  human review  →  merge
  Step 2: Second Real Benchmark Prep       →  PR  →  human review  →  merge

Phase B (after Phase A merges):
  Step 3: Multi-Benchmark Suite + Reports  →  PR  →  human review  →  merge
```

---

## Step 1: Causal Fallback Differentiation

### GitHub Issue

Create issue titled:
**"Make causal strategy differ from surrogate_only under RF-surrogate fallback"**

Body:

```
## Problem

The first real benchmark (PR #67) showed that `causal` and `surrogate_only`
produce identical results in every configuration. Root cause: the RF-surrogate
fallback path in `_suggest_surrogate()` uses `focus_variables` as its only
causal input, but when the causal graph is a star (all vars → objective),
focus_variables is the full variable set — same as no graph.

## What to do

Inject causal graph structure into `_suggest_surrogate()` so that the causal
strategy generates meaningfully different candidates.

## Acceptance

1. `causal` and `surrogate_only` produce observably different suggestions on
   a deterministic test fixture
2. the change is domain-general, not ERCOT-specific
3. existing tests remain green
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md` (Agent 1 brief)
- `causal_optimizer/optimizer/suggest.py` (the file you will change)
- `causal_optimizer/types.py` (CausalGraph API: .ancestors(), .parents(), .edges)
- `causal_optimizer/engine/loop.py` (how suggest_parameters is called)
- `causal_optimizer/domain_adapters/energy_load.py` (the star graph that exposes the bug)
- `tests/unit/test_suggest.py` (existing suggest tests)
- `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md` (benchmark evidence)

Branch name: `sprint-16/causal-fallback-differentiation`

Feature:

Make `_suggest_surrogate()` in `causal_optimizer/optimizer/suggest.py` behave
differently when a causal graph is available, even when focus_variables is the
full variable set.

### Root Cause

`_suggest_surrogate()` receives `focus_variables: list[str]` but no causal
graph. When all variables are ancestors (star graph), focus_variables is the
same list as "all variables" — the function cannot tell causal from
surrogate_only.

### Required Change: Causal Targeted Candidate Generation

1. Add an optional `causal_graph: CausalGraph | None = None` parameter to
   `_suggest_surrogate()`.

2. Pass `causal_graph` from `_suggest_optimization()` at the fallback call
   site (line ~228).

3. When `causal_graph is not None`, change candidate generation:

   **Current behavior (keep for surrogate_only):**
   - Generate 100 LHS candidates
   - Pick best by RF prediction

   **New behavior (when causal_graph is provided):**
   - Generate 50 LHS candidates (broad exploration)
   - Generate 50 "targeted intervention" candidates:
     - Start from the best-known parameter set
     - For each targeted candidate, randomly select 1 or 2 direct parents of
       the objective (using `causal_graph.parents(objective_name)` — add a
       `parents()` method to `CausalGraph` if it doesn't exist)
     - Perturb only those selected parents while holding all other variables
       at best-known values
     - Use the same perturbation logic as `_suggest_exploitation()`:
       continuous ±10-30% of range, integer ±1-2, categorical random choice
   - Score all 100 candidates with the RF model
   - Pick the best-scoring candidate

   This creates genuine behavioral differentiation because:
   - surrogate_only: 100 random LHS → best RF prediction (explores broadly)
   - causal: 50 LHS + 50 targeted → best RF prediction (explores + targets
     direct causes)

4. If `CausalGraph` does not already have a `parents()` method, add one:
   ```python
   def parents(self, target: str) -> set[str]:
       """Direct parents of target (nodes with an edge directly to target)."""
       return {u for u, v in self.edges if v == target}
   ```

5. Also pass `causal_graph` through `_suggest_exploitation()` to bias
   perturbation toward direct parents when the graph is available:
   - When causal_graph is provided and has parents of the objective that
     overlap with focus_variables, prefer perturbing those parents
   - When causal_graph is None (surrogate_only), keep current uniform
     random variable selection

### Tests to Add

Add these in `tests/unit/test_suggest.py` (or a new file
`tests/unit/test_causal_differentiation.py` if cleaner):

1. **test_causal_and_surrogate_produce_different_suggestions**:
   - Create a simple star-graph CausalGraph (3 vars → objective)
   - Create a SearchSpace with 3 continuous variables
   - Create an ExperimentLog with 15 results (past exploration phase)
   - Call `suggest_parameters()` with `causal_graph=graph` 10 times with
     fixed seeds
   - Call `suggest_parameters()` with `causal_graph=None` 10 times with
     the same seeds
   - Assert that at least 3 of the 10 suggestion pairs differ
   - This is the acceptance gate: causal and surrogate_only must be
     observably different

2. **test_targeted_candidates_perturb_few_variables**:
   - With a causal graph, generate suggestions and verify that some
     candidates differ from the best-known in only 1-2 variables
   - This tests the targeted intervention mechanism

3. **test_causal_graph_parents_method**:
   - Test the `parents()` method on CausalGraph with a known graph

4. **test_surrogate_only_unchanged**:
   - Verify that `_suggest_surrogate()` with `causal_graph=None` produces
     the same results as before (regression guard)

### Design Notes

- Keep the 100-candidate total budget the same (50 + 50) so runtime impact
  is minimal
- The targeted candidates must respect variable bounds from SearchSpace
- Use `np.random.default_rng(seed)` for reproducible targeted generation
- Do not change the Ax/BoTorch path — only the RF-surrogate fallback
- Do not change the exploration phase (LHS) — only optimization phase
- The change must be generic: it works on any CausalGraph, not just the
  energy adapter's star graph

Conventions: from __future__ import annotations, type hints on all public
methods, ruff line length 100, mypy strict, np.random.default_rng(seed).

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
```

---

## Step 2: Second Real Benchmark Data Prep

### GitHub Issue

Create issue titled:
**"Add second real benchmark: ERCOT COAST + Houston weather"**

Body:

```
## Goal

Add a second real predictive-energy benchmark using a different ERCOT zone
so optimizer changes can be judged across tasks, not just on NORTH_C.

## Dataset

- Load series: ERCOT COAST weather zone
- Weather: Houston-area NOAA station (prefer major airport with full 2022-2024
  hourly coverage, e.g., IAH — George Bush Intercontinental, USW00012960)
- Date range: 2022-01-01 through 2024-12-31
- Saved timezone: UTC

## Acceptance

1. Prepared Parquet passes the same QA gates as the NORTH_C dataset
2. Smoke run completes with the existing benchmark script (no code changes)
3. Full artifact and summary CSV are generated
4. Benchmark report committed using the shared template
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  Data prep → QA → smoke → full run → report → gh pr create → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md` (Agent 2 brief)
- `thoughts/shared/prompts/sprint-15-real-energy-benchmark-run.md` (the first
  benchmark prompt — follow the same pattern for data prep, QA, and reporting)
- `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`
  (template to follow for the new report)
- `thoughts/shared/templates/benchmark-iteration-report.md`
- `causal_optimizer/benchmarks/predictive_energy.py` (harness — do not modify)
- `scripts/energy_predictive_benchmark.py` (runner — do not modify)

Branch name: `sprint-16/ercot-coast-benchmark`

### Phase A: Data Preparation

Own:
1. Source download from official ERCOT and NOAA sites
2. Timestamp normalization (ERCOT local market time → UTC)
3. Load/weather join on exact UTC hour
4. Saved Parquet dataset
5. Provenance notes for the report

Dataset choice:
- **Load series:** ERCOT `COAST` weather zone hourly load
- **Weather station:** Houston-area NOAA hourly station — prefer George Bush
  Intercontinental Airport (IAH), station ID `USW00012960`. If that station
  has coverage gaps, use Hobby Airport `USW00012918` instead. Document which
  station was used and why.
- **Date range:** `2022-01-01 00:00:00` through `2024-12-31 23:00:00`
- **Saved timezone:** `UTC`

Official source pages only:
1. ERCOT load archives:
   https://www.ercot.com/gridinfo/load/load_hist
2. NOAA NCEI ISD / Global Hourly:
   https://www.ncei.noaa.gov/index.php/products/land-based-station/integrated-surface-database

Dataset preparation contract (same as NORTH_C):
1. Columns: timestamp, target_load, temperature (required); humidity,
   hour_of_day, day_of_week, is_holiday (optional)
2. timestamp: hourly, unique, parseable, saved in UTC
3. target_load: ERCOT COAST series
4. temperature: from the selected NOAA Houston-area station
5. If multiple weather obs per hour, use hourly mean
6. Join on exact UTC hour after both normalized
7. Drop duplicate timestamps — stop and report if they persist
8. Drop rows with missing target_load or temperature
9. Save to Parquet

Time handling rules:
1. Treat ERCOT timestamps as US/Central market time, convert to UTC
2. Calendar features (hour_of_day, day_of_week, is_holiday) derived from
   ERCOT local market time BEFORE converting to UTC
3. No DST duplicate timestamps (UTC conversion handles this)

Minimum QA gates (must all pass before benchmark run):
1. Row count >= 24,000
2. Required columns present
3. Timestamps unique
4. Timestamps monotonic after sort
5. No missing target_load
6. No missing temperature

Deliver:
1. Prepared Parquet at:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet`
2. Row count, start/end timestamps, QA results
3. Source URLs and station ID
4. Prep note describing join and timezone handling

### Phase B: Benchmark Execution

Smoke command:
```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_smoke.json
```

Full command:
```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_results.json
```

Rules:
1. Smoke must pass before full run
2. Do not change benchmark code — if smoke fails, stop and report the blocker
3. If the smoke run fails, the report becomes a blocker report

Local-only outputs (do NOT commit to git):
1. Parquet: `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet`
2. Smoke: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_smoke.json`
3. Full: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_results.json`
4. Summary CSV: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_summary.csv`

Summary CSV schema (same as NORTH_C):
dataset_id, strategy, budget, n_runs, test_mae_mean, test_mae_std,
best_validation_mae_mean, best_validation_mae_std, validation_test_gap_mean,
validation_test_gap_std, runtime_seconds_mean, runtime_seconds_std,
best_seed_by_test_mae

### Phase C: Reporting

Commit the report here:
`thoughts/shared/docs/ercot-coast-houston-2022-2024-benchmark-report.md`

Use the shared template. The report must include:
1. Exact dataset path and source URLs
2. NOAA station ID used and why
3. Saved timezone UTC
4. Actual split row counts and timestamp boundaries
5. Exact smoke and full commands
6. Artifact paths
7. Test MAE mean and std by strategy and budget
8. Validation-test gap by strategy and budget
9. Whether causal == surrogate_only (expected: yes, until Step 1 lands)
10. Clear conclusion: positive, negative, or inconclusive

Stop conditions:
1. Duplicate timestamps after UTC conversion
2. Smoke run fails on shipped benchmark
3. Obviously invalid metrics
4. Benchmark requires a contract change

Rules:
- Do not skip /polish before creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, QA summary, benchmark results summary
```

---

## Step 3: Multi-Benchmark Suite + Re-Run

### Prerequisites

Both Step 1 and Step 2 must be merged to main before starting Step 3.

### GitHub Issue

Create issue titled:
**"Multi-benchmark evaluation suite with cross-benchmark acceptance rules"**

Body:

```
## Goal

Build a suite runner that evaluates optimizer changes across both real
benchmarks (NORTH_C and COAST) and produces a combined report with
acceptance/rejection decision.

## Acceptance

1. One command runs both real benchmarks
2. One report summarizes per-benchmark and aggregate outcomes
3. Acceptance rules reject overfit changes:
   - improve at least one benchmark
   - no material regression on the others
   - stable across seeds
4. Re-run shows whether the causal differentiation change (Step 1)
   actually improves results
```

### Agent Prompt

```
Stand up one agent in an isolated worktree. Rebase on main first (must include
Steps 1 and 2). Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/08-optimizer-improvement-briefs.md` (Agent 3 brief)
- `scripts/energy_predictive_benchmark.py` (single-benchmark runner)
- `causal_optimizer/benchmarks/predictive_energy.py` (harness)
- `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`
- `thoughts/shared/docs/ercot-coast-houston-2022-2024-benchmark-report.md`
- `thoughts/shared/templates/benchmark-iteration-report.md`

Branch name: `sprint-16/multi-benchmark-suite`

Feature:

Build `scripts/energy_benchmark_suite.py` — a suite runner that executes the
predictive energy benchmark across multiple datasets and produces a combined
evaluation.

### Suite Runner

Create `scripts/energy_benchmark_suite.py` with:

1. **argparse CLI:**
   - `--datasets` (required) — comma-separated paths to local Parquet files
   - `--dataset-ids` (required) — comma-separated short IDs matching datasets
     (e.g., `ercot_north_c,ercot_coast`)
   - `--budgets` (default "20,40,80")
   - `--seeds` (default "0,1,2,3,4")
   - `--strategies` (default "random,surrogate_only,causal")
   - `--output-dir` (required) — directory to write per-benchmark and suite
     artifacts

2. **Suite execution:**
   - For each dataset: call the existing single-benchmark logic (import from
     `scripts/energy_predictive_benchmark.py` or invoke it as a subprocess)
   - Write per-benchmark JSON artifacts to `output-dir/<dataset_id>_results.json`
   - Write per-benchmark summary CSVs

3. **Suite summary:**
   - Aggregate results across datasets
   - Compute per-strategy rankings (which strategy is best on each benchmark)
   - Apply acceptance rules (see below)
   - Write `output-dir/suite_summary.json` with:
     - per_benchmark: list of per-benchmark summaries
     - aggregate: cross-benchmark comparison
     - acceptance: pass/fail with reasons

4. **Acceptance rules** (from Plan 08):
   - `improved`: strategy beats baseline on at least one benchmark
   - `no_regression`: no material regression (>2% relative) on other benchmarks
   - `stable`: std across seeds is < 5% of mean for all benchmarks
   - `differentiated`: causal and surrogate_only produce different results
   - Overall: PASS only if all four rules pass; CONDITIONAL if differentiated
     fails but others pass; FAIL otherwise

5. **Suite report:**
   - Print a combined comparison table to stdout
   - Generate `output-dir/suite_report.md` using the summary data
   - The report must answer:
     - Did the causal differentiation change actually change results?
     - Did causal improve on either benchmark?
     - Did any strategy regress?
     - Are results stable across seeds?
     - Overall recommendation: PROMOTE / INVESTIGATE / REJECT

### Tests

Add `tests/unit/test_benchmark_suite.py`:

1. **test_acceptance_rules_pass**: Mock per-benchmark results where causal
   improves on one, no regression on other → PASS
2. **test_acceptance_rules_fail_regression**: Mock results where causal
   improves one but regresses other → FAIL
3. **test_acceptance_rules_conditional**: Mock results where strategies are
   identical → CONDITIONAL
4. **test_suite_summary_schema**: Verify the JSON summary contains required
   fields

### Re-Run Both Benchmarks

After the suite is implemented, run the full suite:

```bash
uv run python scripts/energy_benchmark_suite.py \
  --datasets /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet,/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --dataset-ids ercot_north_c,ercot_coast \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output-dir /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/suite_sprint16
```

Include the suite report in the PR as:
`thoughts/shared/docs/sprint-16-suite-report.md`

This report is the main deliverable. It tells us whether Sprint 16's causal
differentiation change actually moved the needle.

Conventions: from __future__ import annotations, type hints, ruff line length
100, mypy strict.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, suite acceptance result, key findings
```

---

## Execution Sequence

```
Phase A — run in parallel:
  Worktree 1: Step 1 prompt → agent delivers PR → human reviews → merge
  Worktree 2: Step 2 prompt → agent delivers PR → human reviews → merge

Phase B — after both Phase A PRs merge:
  Worktree 3: Step 3 prompt → agent delivers PR → human reviews → merge
```

Post-merge verification after all three:

```bash
# Unit tests still green
uv run pytest -m "not slow" -v

# Suite smoke (budget=3, seed=0)
uv run python scripts/energy_benchmark_suite.py \
  --datasets /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet,/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --dataset-ids ercot_north_c,ercot_coast \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output-dir /tmp/suite_smoke
```

## Acceptance Checklist

Sprint 16 is complete when:

1. `causal` and `surrogate_only` produce observably different suggestions
   (proven by deterministic unit test)
2. Second real benchmark (ERCOT COAST) is prepared, run, and reported
3. Multi-benchmark suite runner exists and produces combined reports
4. Suite report shows whether causal differentiation improved results
5. All fast tests pass (`uv run pytest -m "not slow"`)
6. Benchmark evidence is strong enough to guide the next optimizer iteration
