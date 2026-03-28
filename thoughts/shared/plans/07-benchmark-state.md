# Benchmark State

Updated: 2026-03-28 (Sprint 17 complete)

## Purpose

This file is the current restart point for the real predictive-model benchmark work in `causal-optimizer`.

Use it when:

1. starting a new chat
2. handing context to another reviewer or agent
3. checking what is merge-ready vs still pending

## Current Goal

Use the first real predictive benchmark to answer:

1. Can `causal-optimizer` improve a real predictive model on unseen data under a fixed experiment budget?

The current benchmark target is:

1. day-ahead energy load forecasting
2. local CSV or Parquet data
3. locked chronological `train` / `validation` / `test` split
4. strategy comparison across `random`, `surrogate_only`, and `causal`
5. one real benchmark report that is strong enough to guide the next optimizer iteration

## Canonical Planning Docs

Primary references:

1. `thoughts/shared/plans/05-real-predictive-model-benchmark.md`
2. `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`
3. `thoughts/shared/prompts/sprint-14-energy-benchmark.md`
4. `thoughts/shared/prompts/sprint-15-real-energy-benchmark-run.md`
5. `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`
6. `thoughts/shared/plans/08-optimizer-improvement-briefs.md`
7. `thoughts/shared/prompts/sprint-16-causal-differentiation-suite.md`

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

## Benchmark Outcome

### First Real Benchmark Run

Dataset:

1. `ERCOT NCENT/NORTH_C` hourly load
2. NOAA station `USW00003927`
3. saved as UTC single-series Parquet at:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet`

Key verified dataset properties:

1. `26,291` rows
2. no duplicate timestamps
3. no missing `target_load`
4. no missing `temperature`
5. calendar features now derived from ERCOT local market time before UTC storage

Locked split boundaries:

1. train: `2022-01-01 06:00:00 UTC` to `2023-10-20 11:00:00 UTC`
2. validation: `2023-10-20 12:00:00 UTC` to `2024-05-26 13:00:00 UTC`
3. test: `2024-05-26 14:00:00 UTC` to `2024-12-31 23:00:00 UTC`

Benchmark result:

1. outcome: **inconclusive / negative for causal advantage**
2. `random` was marginally better on test MAE at all budgets
3. `causal` and `surrogate_only` were effectively identical
4. all strategies converged to `ridge`
5. benchmark behavior was highly stable across seeds

Interpretation:

1. The benchmark harness is functioning correctly.
2. The current causal path is not meaningfully differentiated under the RF-surrogate fallback.
3. The benchmark currently provides stronger evidence about optimizer limitations than about causal benefit.

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

1. merged
2. no outstanding benchmark-contract blockers from review

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

1. merged
2. no outstanding runner blockers from review

Non-blocking follow-up idea:

1. normalize random-strategy crash handling so it is as tolerant as engine-based strategies

### Issue #61

Scope:

1. benchmark smoke tests
2. reproducibility regression test
3. benchmark documentation

Review state:

1. shipped via PR `#64`
2. smoke tests, reproducibility test, and benchmark docs are in place

### Issue / PR #65

Scope:

1. first real energy benchmark run
2. dataset preparation
3. benchmark artifact generation
4. benchmark report

Branch / review context:

1. branch: `sprint-15/energy-benchmark-report`
2. latest reviewed commit: `8ddf271`
3. report path: `thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`

Important review outcome:

1. local-time calendar feature bug was found and fixed
2. split boundary reporting was corrected to actual timestamps
3. the benchmark report is now review-ready
4. the substantive result remains: `causal` is not outperforming `random` on this benchmark

Local artifact paths:

1. dataset: `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet`
2. smoke: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_smoke.json`
3. full results: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json`
4. summary CSV: `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_summary.csv`

## Recommended Next Steps

1. ~~merge PR `#65` if branch state still matches the reviewed benchmark-report state~~ — done (PR #67 merged)
2. treat the first real benchmark result as a system-learning milestone, not a product win
3. ~~prioritize optimizer improvements that differentiate `causal` from `surrogate_only` in a domain-general way~~ — Sprint 16 Step 1
4. ~~re-run the benchmark only after one general optimizer improvement lands~~ — Sprint 16 Step 3
5. ~~add at least one more real benchmark before making stronger predictive-model claims~~ — Sprint 16 Step 2

## Sprint 16 Scope

Sprint prompt: `thoughts/shared/prompts/sprint-16-causal-differentiation-suite.md`

Three steps:

1. **Causal Fallback Differentiation** — inject causal graph structure into
   `_suggest_surrogate()` via targeted intervention candidates (50 LHS + 50
   causal-targeted). Root cause: star graph makes `ancestors()` == all vars.
   Fix: pass `causal_graph` to surrogate, generate candidates that perturb
   only 1-2 direct parents of the objective.

2. **Second Real Benchmark** — ERCOT COAST + Houston-area NOAA weather.
   Same harness, same contract, different load/weather regime. Parquet at
   `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet`

3. **Multi-Benchmark Suite** — suite runner across both datasets, combined
   report with acceptance rules (improve >= 1, no regression, stable, differentiated).
   Re-runs both benchmarks after the differentiation change lands.

Steps 1 and 2 are parallel. Step 3 depends on both.

## Sprint 17 Scope

Sprint prompt: `thoughts/shared/prompts/sprint-17-skip-calibration-counterfactual-benchmark.md`

Three steps (Steps 1 and 2 parallel, Step 3 depends on both):

1. **Skip Calibration Diagnostics** (PR #77) — `SkipDiagnostics`, `AuditResult`,
   `AnytimeMetrics` dataclasses. Engine instrumented with `_skip_log`,
   `skip_diagnostics` property, `anytime_metrics()` method, `audit_skip_rate` param.
   Benchmark suite report includes skip calibration section.

2. **Semi-Synthetic Counterfactual Benchmark** (PR #76) — `DemandResponseScenario`
   generates semi-synthetic data from real ERCOT covariates with known treatment
   effects. Non-trivial causal graph with genuine non-parents (humidity, day_of_week).
   `CounterfactualBenchmarkResult` with policy value, regret, decision error rate.

3. **Suite Re-Run + Combined Report** (PR #79) — Full suite with skip diagnostics
   + counterfactual benchmark. Combined report at
   `thoughts/shared/docs/sprint-17-combined-report.md`.

Sprint 17 results:

- **Promotion verdict: INVESTIGATE**
- Skip calibration: 33% false-skip rate (too high for production)
- Counterfactual benchmark: degenerate (treatment cost 50.0 > benefit, oracle = never treat)
- Causal vs surrogate_only: <0.1% difference at budget=80, below 0.1% threshold
- Causal is 100x slower than surrogate_only on counterfactual benchmark (screening formula enumeration)
- Plan 08 fully complete (all 5 agents delivered)

## Priority Improvements

These are the highest-value next changes suggested by Sprint 17:

1. ~~Make causal guidance active under the RF-surrogate fallback instead of effectively no-op.~~ — done (Sprint 16)
2. ~~Add a second real benchmark so optimizer changes are judged across tasks, not just on ERCOT.~~ — done (Sprint 16)
3. ~~Add cross-benchmark acceptance rules.~~ — done (Sprint 16)
4. ~~Add skip calibration diagnostics.~~ — done (Sprint 17)
5. ~~Add semi-synthetic counterfactual benchmark.~~ — done (Sprint 17)
6. Fix counterfactual benchmark: lower treatment cost or strengthen effects so oracle is non-trivial.
7. Improve skip calibration: higher confidence thresholds, calibrated probabilities, or burn-in.
8. Address causal differentiation at low budgets: causal == surrogate_only at budget 20/40.
9. Performance: causal 100x slower than surrogate_only due to screening formula enumeration.

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

The real benchmark artifacts and prep script live outside the repo under:

1. `/Users/robertwelborn/Projects/_local/causal-optimizer/`

That keeps large files out of git, but long-term reproducibility depends on
preserving those local files or moving the prep logic into a durable tracked
location.
