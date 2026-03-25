# Energy Predictive Benchmark Handoff

## Purpose

This file turns the real predictive-model benchmark idea into an implementation-ready handoff for agent teams.

The immediate goal is to answer one real question with evidence:

1. Can `causal-optimizer` improve a real predictive model on unseen data under a fixed experiment budget?

This benchmark is intentionally narrow:

1. one domain
2. one real modeling task
3. one locked split strategy
4. three strategies to compare

If we cannot make this benchmark convincing, we should not make broad claims about predictive-model research.

## Scope

Build a **real day-ahead energy load forecasting benchmark** using a local hourly dataset and compare:

1. `random`
2. `surrogate_only`
3. `causal`

The benchmark must report:

1. best validation MAE
2. test MAE for the best validation-selected configuration
3. validation-test gap
4. runtime and summary stats across seeds

## Shared Rules

All agents should follow this workflow:

1. `/tdd`
2. implement
3. `/polish`
4. `gh pr create`
5. `/gauntlet`
6. report PR URL

All work should assume:

1. no network access during benchmark execution
2. real dataset path provided explicitly
3. fixture data remains for unit tests only
4. `test` split is never touched during optimization
5. the first pass targets a single-series dataset

## Agent Feedback Loop Policy

Different assumptions are allowed early if they improve the design, but they must converge quickly into one shared public contract.

Use these rules:

1. Divergence is welcome in implementation ideas, edge cases, and internal design.
2. Divergence is not welcome for long on public surfaces such as split semantics, CLI arguments, artifact fields, or test expectations.
3. `#59` owns the benchmark data contract and split semantics.
4. `#60` owns the public runner entrypoint and result artifact schema.
5. `#61` must test and document the public surfaces that actually merged, not a parallel interpretation of the scaffold.
6. If an agent discovers a better assumption than the handoff doc, update the handoff doc in the same PR so the next agent inherits the corrected contract.
7. Downstream agents should consume upstream helpers and interfaces instead of reimplementing them locally.
8. If a downstream agent must deviate, the PR should call out the mismatch explicitly and propose the contract update.

Recommended merge order:

1. merge `#59` first
2. rebase `#60` on merged `#59`
3. rebase `#61` on merged `#59` and `#60`

Review standard:

1. We prefer productive disagreement over silent incompatibility.
2. A better design is welcome if it is reconciled back into the shared contract.
3. The merged public API, tests, and docs become the source of truth for the next handoff.

## Target File Layout

Shipped implementation layout:

```text
causal_optimizer/benchmarks/
  predictive_energy.py          # PR #59 — data harness, split, runner, result dataclass

scripts/
  energy_predictive_benchmark.py  # PR #60 — CLI runner, strategy dispatch, JSON output

tests/integration/
  test_predictive_energy_benchmark.py   # PR #59 — harness unit/integration tests
  test_predictive_energy_smoke.py       # PR #64 (issue #61) — end-to-end smoke tests

tests/regression/
  test_predictive_energy_reproducibility.py  # PR #64 (issue #61) — seed determinism checks

thoughts/shared/docs/
  predictive-energy-benchmark.md  # PR #64 (issue #61) — benchmark documentation
```

## Shipped API — What Actually Merged

The scaffold below was the starting point.  This section documents what
actually shipped in PRs #59 and #60, with key divergences noted.

### `PredictiveBenchmarkResult` (in `causal_optimizer/benchmarks/predictive_energy.py`)

```python
@dataclass
class PredictiveBenchmarkResult:
    strategy: str
    budget: int
    seed: int
    best_validation_mae: float
    test_mae: float
    selected_parameters: dict[str, Any]
    runtime_seconds: float                    # NEW — not in scaffold
    validation_test_gap: float = field(init=False)  # CHANGED — auto-computed via __post_init__
```

**Divergences from scaffold:**
- `runtime_seconds` added as a required field (covers full run including test eval).
- `validation_test_gap` is `field(init=False)`, computed as `test_mae - best_validation_mae` in `__post_init__`.  The scaffold had it as a regular init field.

### `split_time_frame`

```python
def split_time_frame(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
```

**Enhancements over scaffold:**
- Validates `train_frac + val_frac < 1.0`, both positive, both non-zero.
- Raises on duplicate timestamps (single-series constraint).
- Enforces minimum 10 rows per partition (`_MIN_PARTITION_ROWS`).
- Uses `mergesort` for stable sort.

### `ValidationEnergyRunner`

```python
class ValidationEnergyRunner:
    def __init__(self, train_df, val_df, seed=None) -> None: ...
    def run(self, parameters: dict[str, Any]) -> dict[str, float]: ...
```

**Key divergence from scaffold:** The shipped version creates the `EnergyLoadAdapter` once at `__init__` time using `split_timestamp` (the first validation timestamp) rather than `train_ratio`.  This prevents lag-induced NaN dropping from shifting the split boundary.  The scaffold re-created the adapter on each `run()` call using ratio-based splitting.

### `evaluate_on_test`

```python
def evaluate_on_test(train_df, val_df, test_df, parameters, seed=None) -> dict[str, float]:
```

**Key divergence from scaffold:** Uses the first test timestamp as `split_timestamp` instead of ratio-based splitting.  Same leakage-prevention rationale as `ValidationEnergyRunner`.

### `run_strategy` (in `scripts/energy_predictive_benchmark.py`)

```python
def run_strategy(strategy, budget, seed, train_df, val_df, test_df) -> PredictiveBenchmarkResult | None:
```

**Divergences from scaffold:**
- `random` strategy uses direct `sample_random_params` + `ValidationEnergyRunner.run()` loop instead of `ExperimentEngine`.  This is more efficient and avoids engine overhead for pure random search.
- Returns `None` (not raises) when no valid result is produced (all experiments crashed).
- `runtime_seconds` covers the full run including `evaluate_on_test`.

### CLI (`main()` in `scripts/energy_predictive_benchmark.py`)

**Enhancements over scaffold:**
- Added `--strategies` flag (scaffold hardcoded all three).
- Added `_sanitize_for_json` to replace `inf`/`nan` with `None` for RFC 8259 compliance.
- Added `_print_summary` table (mean +/- std per strategy/budget).
- Added `--area-id` flag for multi-area datasets.
- Fail-fast: validates strategies and creates output directory before computation.

## Benchmark Contract

### Dataset

Required columns:

1. `timestamp`
2. `target_load`
3. `temperature`

Optional:

1. `humidity`
2. `hour_of_day`
3. `day_of_week`
4. `is_holiday`
5. `area_id`

### Split

Use a locked chronological split:

1. `train`: first 60%
2. `validation`: next 20%
3. `test`: final 20%

Rules:

1. Optimization sees only `train` and `validation`
2. Test is evaluated once per completed run
3. Lag features must use only past data
4. After feature generation and any row dropping, the effective training window must remain strictly earlier than the effective validation/test window. If preprocessing makes that impossible, the run must fail rather than leak held-out rows into training.

### Search Space

First-pass search variables:

1. `model_type`
2. `lookback_window`
3. `use_temperature`
4. `use_humidity`
5. `use_calendar`
6. `regularization`
7. `n_estimators`

### Strategies

Compare:

1. `random`
2. `surrogate_only`
3. `causal`

### Budget Grid

Run:

1. `20`
2. `40`
3. `80`

Across:

1. `5` seeds

## Agent Briefs

### Agent 1: Split Harness (#59 — merged)

Goal:

Build the benchmark data and evaluation harness so experiments optimize on validation while preserving a locked final test evaluation.

Deliverables:

1. split helper for `train` / `validation` / `test`
2. wrapper runner or adapter that trains on `train` and scores on `validation`
3. helper to evaluate the selected best config on `test`
4. explicit checks for single-series data and time ordering

Status: **Shipped and merged.**

### Agent 2: Runner And Results (#60 — merged as #63)

Goal:

Build the benchmark entrypoint that runs all strategies, budgets, and seeds and emits results.

Deliverables:

1. benchmark script
2. config loader or command-line arguments
3. per-run artifact output in JSON
4. aggregate summary table with mean/std across seeds

Status: **Shipped and merged.**

### Agent 3: Tests And Docs (#61 / PR #64)

Goal:

Add a small but meaningful test surface and document how to run the benchmark.

Deliverables:

1. one tiny-budget integration smoke test
2. one reproducibility regression test
3. benchmark doc describing dataset contract, split strategy, outputs, and limitations

Status: **PR #64 open, under review.**

## Notes On The Scaffold

The original scaffold was intentionally first-pass.  All items below were
addressed in the shipped implementation:

1. `random` strategy now uses direct sampling instead of engine — **addressed**.
2. Result aggregation via `_print_summary` — **addressed**.
3. CLI upgraded with `--strategies`, `--area-id`, validation — **addressed**.
4. `PredictiveBenchmarkResult` lives in `causal_optimizer/benchmarks/predictive_energy.py` — **addressed**.
5. `EnergyLoadAdapter` `split_timestamp` support used by `ValidationEnergyRunner` and `evaluate_on_test` — **addressed**.

## GitHub Issue Split

The implementation was split into three issues:

1. `#59` — build the data/split harness and locked test evaluation (merged)
2. `#60`/`#63` — build the benchmark runner and results artifacts (merged)
3. `#61` — add smoke tests, reproducibility checks, and benchmark docs (in progress)

## Acceptance Checklist

Before calling the benchmark ready, confirm all of these:

1. one command runs all strategies for at least one budget and one seed
2. a results artifact is produced
3. test MAE is reported separately from validation MAE
4. same seed gives the same result on a fixed small run
5. the benchmark can be run on a local real dataset without code edits

## What I Will Review

When these agent branches come back, I will review for:

1. leakage risk
2. split correctness
3. benchmark reproducibility
4. result schema clarity
5. whether the benchmark actually answers the predictive-model question rather than only producing another demo
