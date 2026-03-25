# Benchmark State

Updated: 2026-03-25

## Purpose

This file is the current restart point for the real predictive-model benchmark work in `causal-optimizer`.

Use it when:

1. starting a new chat
2. handing context to another reviewer or agent
3. checking what is merge-ready vs still pending

## Current Goal

Build the first real predictive-model benchmark that answers:

1. Can `causal-optimizer` improve a real predictive model on unseen data under a fixed experiment budget?

The benchmark target is:

1. day-ahead energy load forecasting
2. local CSV or Parquet data
3. locked chronological `train` / `validation` / `test` split
4. strategy comparison across `random`, `surrogate_only`, and `causal`

## Canonical Planning Docs

Primary references:

1. `thoughts/shared/plans/05-real-predictive-model-benchmark.md`
2. `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`
3. `thoughts/shared/prompts/sprint-14-energy-benchmark.md`

PR 63 review handoff reference:

1. `thoughts/shared/prompts/sprint-14-pr63-verification.md`

## Benchmark Contract

These are the important design decisions that should not drift:

1. Dataset split is locked and chronological: `60 / 20 / 20`
2. Optimization sees only `train` and `validation`
3. Test is evaluated once after selecting the best validation configuration
4. Lag features must use only past data
5. After feature generation and any row dropping, the effective training window must remain strictly earlier than the effective validation/test window
6. If preprocessing makes that impossible, the run must fail rather than leak held-out rows into training
7. The first pass is single-series and local-data only

## Issue / PR State

### Issue #59 / PR #62

Scope:

1. split harness
2. predictive energy benchmark data loading
3. locked split helpers
4. validation runner
5. held-out test evaluation

Key implementation details:

1. `causal_optimizer/benchmarks/predictive_energy.py`
2. `EnergyLoadAdapter` gained `split_timestamp`
3. degenerate post-preprocessing splits now raise instead of leaking held-out rows into training

Review state:

1. last review outcome: no blocking findings after the degenerate split fix
2. recommended next step: merge if not already merged

Important contract note:

1. the split boundary must survive lag-induced row dropping
2. forcing a held-out row back into training is not acceptable

### Issue #60 / PR #63

Scope:

1. benchmark runner CLI
2. strategy execution across budgets and seeds
3. JSON artifact writing
4. compact summary output

Local review context:

1. PR URL: `https://github.com/datablogin/causal-optimizer/pull/63`
2. branch: `sprint-14/benchmark-runner`
3. reviewed worktree: `/Users/robertwelborn/Projects/causal-optimizer/.claude/worktrees/agent-a16e6d3e`
4. reviewed commit: `d075662`

Key fixes that were required and are now present:

1. `runtime_seconds` now includes held-out test evaluation
2. all-crash combinations are skipped instead of serialized as sentinel rows

Verification rerun on `d075662`:

1. `uv run pytest tests/unit/test_energy_predictive_benchmark_script.py tests/integration/test_predictive_energy_benchmark.py -q`
   Result: `49 passed, 1 skipped`
2. `uv run python scripts/energy_predictive_benchmark.py --data-path tests/fixtures/energy_load_fixture.csv --budgets 3 --seeds 0 --strategies random,surrogate_only,causal --output /tmp/predictive_energy_results_pr63_all.json`
   Result: completed successfully and wrote 3 real result rows

Review state:

1. last review outcome: no blocking findings
2. recommended next step: merge if branch state still matches `d075662` or later equivalent

Non-blocking follow-up idea:

1. normalize random-strategy crash handling so it is as tolerant as engine-based strategies

### Issue #61

Scope:

1. benchmark smoke tests
2. reproducibility regression test
3. benchmark documentation

Recommended start condition:

1. begin only after `#59` and `#60` are merged to `main`

## Recommended Next Steps

1. merge PR `#62` if it is not already merged
2. merge PR `#63` if it is not already merged
3. start `#61` on top of merged `main`
4. keep the benchmark docs and prompts aligned to the shipped implementation, not the older scaffold

## Review Workflow That Worked

For future PR review handoffs, provide:

1. PR URL
2. branch name
3. exact commit hash
4. exact local worktree path
5. a verification note with:
   - `Changes since last review`
   - exact commands run
   - focused repro or smoke commands for the requested fixes

The most effective review loop was:

1. inspect the local worktree first
2. rerun only the focused tests and smoke commands tied to the review findings
3. post GitHub comments only for real contract, bug, or integration issues

## Local Workspace Note

Several planning and prompt files under `thoughts/shared/` are currently present locally but untracked in git.

If this state file is meant to be durable team documentation, it should be committed along with the related planning docs.
