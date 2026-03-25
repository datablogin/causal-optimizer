# Sprint 14 — Real Energy Predictive Benchmark

## Context

Build the first real predictive-model benchmark to answer: can `causal-optimizer` improve a real predictive model on unseen data under a fixed experiment budget?

Three sequential issues, each reviewed and merged before the next begins:

1. **#59** — Split harness (data contract, locked split, validation runner, test evaluation)
2. **#60** — Benchmark runner (CLI entrypoint, strategy dispatch, result artifacts)
3. **#61** — Tests and docs (smoke test, reproducibility test, benchmark documentation)

Plans:
- `thoughts/shared/plans/05-real-predictive-model-benchmark.md`
- `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`

---

## Step 1: Split Harness (#59)

Run this first. Review and merge before Step 2.

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md` (full scaffold and contract)
- `causal_optimizer/domain_adapters/energy_load.py` (existing adapter to reuse)
- `causal_optimizer/benchmarks/runner.py` (existing benchmark patterns)
- `tests/fixtures/energy_load_fixture.csv` (200-row fixture for tests)

Feature:

- **sprint-14/split-harness** (GitHub #59) — Build the benchmark data and
  evaluation harness in `causal_optimizer/benchmarks/predictive_energy.py`.

  Public API to implement:

  1. `load_energy_frame(data_path: str, area_id: str | None = None) -> pd.DataFrame`
     — Load CSV or Parquet (detect by Path.suffix). If area_id is provided, filter
     to that value. Raise ValueError on: empty result, missing required columns
     (timestamp, target_load, temperature), area_id filter requested but column
     missing. If area_id column exists but area_id param is None and
     df["area_id"].nunique() > 1, raise ValueError telling the user to specify
     --area-id.

  2. `split_time_frame(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
     — Parse timestamps, sort chronologically, return (train, val, test) DataFrames.
     Raise ValueError on: empty partitions, duplicate timestamps, fractions that
     don't leave room for test (train_frac + val_frac >= 1.0). Each partition must
     have at least 10 rows.

  Benchmark contract:
  - After feature generation and any row dropping, the effective training window
    must remain strictly earlier than the effective validation/test window.
  - If preprocessing makes that impossible, the run must fail rather than leak
    held-out rows into training.

  3. `class ValidationEnergyRunner`
     — Implements ExperimentRunner protocol (has `run(parameters) -> dict[str, float]`).
     Constructor takes train_df, val_df, seed. Internally concatenates train+val,
     preserves the original train/validation time boundary after feature
     generation, creates an EnergyLoadAdapter using that explicit split boundary,
     and delegates to adapter.run_experiment(). Deterministic for fixed seed.

  4. `evaluate_on_test(train_df, val_df, test_df, parameters, seed) -> dict[str, float]`
     — Combines all three frames, preserves the original train+validation/test
     boundary after feature generation, creates EnergyLoadAdapter, runs
     experiment. This is the one-shot post-run test evaluation.

  5. `@dataclass class PredictiveBenchmarkResult`
     — Fields: strategy (str), budget (int), seed (int), best_validation_mae (float),
     test_mae (float), validation_test_gap (float), selected_parameters (dict),
     runtime_seconds (float).

  Tests to add in `tests/integration/test_predictive_energy_benchmark.py`:
  - test split produces correct approximate proportions on fixture data (use
    relaxed fracs like 0.5/0.25/0.25 since fixture is only 200 rows)
  - test no leakage: all test timestamps are strictly after all validation
    timestamps, all validation timestamps are strictly after all train timestamps
  - regression test: with `lookback_window > 1`, the effective post-feature
    training window remains strictly earlier than the effective validation/test
    window, or the run fails cleanly if no valid pre-boundary training rows remain
  - test duplicate timestamps raise ValueError
  - test multi-series guard raises ValueError when area_id column exists with
    multiple values but area_id param is None
  - test ValidationEnergyRunner produces deterministic metrics for fixed seed
  - test evaluate_on_test returns metrics with mae key
  - test empty DataFrame raises ValueError
  - test fractions summing to >= 1.0 raises ValueError

  Conventions: from __future__ import annotations, type hints on all public
  methods, ruff line length 100, mypy strict, np.random.default_rng(seed).

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
```

**After human review and merge of #59**, proceed to Step 2.

---

## Step 2: Benchmark Runner (#60)

Run after #59 is merged. Rebase on main first.

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md` (scaffold)
- `causal_optimizer/benchmarks/predictive_energy.py` (the harness from #59 — this is your upstream dependency, import from it)
- `causal_optimizer/benchmarks/runner.py` (existing benchmark runner patterns)
- `causal_optimizer/engine/loop.py` (ExperimentEngine constructor signature)

Feature:

- **sprint-14/benchmark-runner** (GitHub #60) — Build the benchmark entrypoint
  that runs all strategies, budgets, and seeds against the harness from #59.

  Create `scripts/energy_predictive_benchmark.py` with:

  1. argparse CLI:
     - `--data-path` (required) — path to local CSV or Parquet
     - `--area-id` (optional) — filter to one balancing area
     - `--budgets` (default "20,40,80") — comma-separated experiment budgets
     - `--seeds` (default "0,1,2,3,4") — comma-separated RNG seeds
     - `--strategies` (default "random,surrogate_only,causal") — comma-separated
     - `--output` (default "predictive_energy_results.json") — artifact path

  2. `run_strategy(strategy, budget, seed, train_df, val_df, test_df) -> PredictiveBenchmarkResult`
     — Import load_energy_frame, split_time_frame, ValidationEnergyRunner,
     evaluate_on_test, PredictiveBenchmarkResult from
     causal_optimizer.benchmarks.predictive_energy. Create ExperimentEngine with
     the adapter's search space, graph (only for "causal" strategy, None for
     others), descriptor_names, objective_name="mae", minimize=True, and the
     given seed. Run engine.run_loop(budget). Extract best result via
     log.best_result("mae", minimize=True). Call evaluate_on_test for the best
     config. Record runtime_seconds using time.perf_counter around the full
     strategy run. Return PredictiveBenchmarkResult with all fields.

  3. `main()` function:
     — Parse args. Load frame via load_energy_frame. Split via split_time_frame.
     Loop over budget × seed × strategy, call run_strategy for each. Write
     results to JSON via json.dumps([dataclasses.asdict(r) for r in results]).
     After writing, print a compact summary table to stdout: for each
     strategy × budget, show mean±std of best_validation_mae, test_mae, and
     validation_test_gap across seeds. Use simple f-string formatting, no
     external table library needed.

  4. Handle edge cases:
     — If best result is None (all experiments crashed), log a warning and skip
     that combination rather than crashing the whole benchmark.
     — Validate strategy names against {"random", "surrogate_only", "causal"}.

  Do NOT add tests in this PR — tests belong to #61. But do verify manually:
  `uv run python scripts/energy_predictive_benchmark.py --data-path tests/fixtures/energy_load_fixture.csv --budgets 3 --seeds 0 --strategies random`
  should complete without error and produce a JSON file.

  Conventions: from __future__ import annotations, type hints, ruff, mypy strict.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
```

**After human review and merge of #60**, proceed to Step 3.

---

## Step 3: Tests and Docs (#61)

Run after #59 and #60 are both merged. Rebase on main first.

```
Stand up one agent in an isolated worktree. Follow this exact workflow:

  /tdd → implement → /polish → gh pr create → /gauntlet → report PR URL

Read these before starting:
- `CLAUDE.md`
- `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`
- `causal_optimizer/benchmarks/predictive_energy.py` (merged from #59)
- `scripts/energy_predictive_benchmark.py` (merged from #60)
- Test and document what actually shipped, not the scaffold.

Feature:

- **sprint-14/benchmark-tests-docs** (GitHub #61) — Add tests and documentation
  for the real energy predictive benchmark.

  1. Create `tests/integration/test_predictive_energy_smoke.py`:
     - Smoke test: import run_strategy from the benchmark script (or call the
       harness directly). Run strategy="random", budget=3, seed=0 on the fixture
       data with relaxed split fracs (0.5/0.25/0.25). Assert the result is a
       PredictiveBenchmarkResult with all fields populated, test_mae > 0,
       best_validation_mae > 0, runtime_seconds > 0.
     - Test each strategy ("random", "surrogate_only", "causal") completes
       without error for budget=3 on fixture data.

  2. Create `tests/regression/test_predictive_energy_reproducibility.py`:
     - Run the benchmark harness twice with identical seed=42, budget=3,
       strategy="random" on fixture data. Assert best_validation_mae and
       test_mae are exactly equal across runs.
     - Mark all tests in this file `@pytest.mark.slow`.

  3. Create `thoughts/shared/docs/predictive-energy-benchmark.md`:
     - Dataset contract: required columns (timestamp, target_load, temperature),
       optional columns, single-series requirement, local-data only.
     - Locked split: 60/20/20 chronological, test touched once after optimization.
     - Strategies: random, surrogate_only, causal.
     - Default budgets (20, 40, 80) and seeds (0-4).
     - Output artifact: JSON array of records with fields (strategy, budget,
       seed, best_validation_mae, test_mae, validation_test_gap,
       selected_parameters, runtime_seconds).
     - Example command:
       `uv run python scripts/energy_predictive_benchmark.py --data-path path/to/data.csv --budgets 20,40 --seeds 0,1,2`
     - Key rule: predictive claims depend on untouched test performance.
       Validation-only wins are not sufficient evidence.
     - Limitations: single-series, narrow 7-variable search space, no
       feature engineering, Ridge/RF/GBM only.

  4. If the merged API from #59/#60 differs from the handoff doc scaffold,
     update `thoughts/shared/plans/06-energy-predictive-benchmark-handoff.md`
     to reflect what actually shipped.

  Conventions: from __future__ import annotations, type hints, ruff, mypy strict.

Rules:
- Do not skip /polish before creating the PR
- Do not skip /gauntlet after creating the PR
- Do NOT merge — leave PR open for human review
- Report: PR URL, polish summary, gauntlet summary (iterations + issues fixed)
```

---

## Execution Sequence

```
Step 1: Run #59 prompt → agent delivers PR → human reviews → merge
Step 2: Run #60 prompt → agent delivers PR → human reviews → merge
Step 3: Run #61 prompt → agent delivers PR → human reviews → merge
```

Post-merge verification after all three:
```bash
uv run pytest -m "not slow" -v
uv run python scripts/energy_predictive_benchmark.py \
  --data-path tests/fixtures/energy_load_fixture.csv \
  --budgets 3 --seeds 0 --strategies random
```
