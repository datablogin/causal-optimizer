# Sprint 15: Real Energy Benchmark Run

## Goal

Run the first real predictive-model benchmark for `causal-optimizer` on a
single, well-defined energy dataset and produce a review-ready report.

This sprint is about **execution and evidence**, not expanding the benchmark
surface. Do not change the benchmark contract, search space, or runner unless a
real-data blocker makes the run impossible.

## Benchmark Question

Using the shipped benchmark on one real local dataset:

1. does `causal` beat `random` on held-out `test_mae`?
2. does `causal` beat `surrogate_only` on held-out `test_mae`?
3. do any validation gains survive on the untouched test window?

## Dataset Choice

Use this dataset for the first run:

1. **Load series:** ERCOT `NORTH_C` weather-zone hourly load
2. **Weather series:** NOAA DFW airport hourly temperature
3. **Date range:** `2022-01-01 00:00:00` through `2024-12-31 23:00:00`
4. **Saved timezone:** `UTC`

### Why This Dataset

1. `NORTH_C` should have a tighter weather relationship than ERCOT total.
2. The benchmark already expects a single time series with `timestamp`,
   `target_load`, and `temperature`.
3. This is operationally meaningful without creating live-deployment risk.

## Official Sources

Use official source pages only:

1. ERCOT load archives:
   [https://www.ercot.com/gridinfo/load/load_hist](https://www.ercot.com/gridinfo/load/load_hist)
2. ERCOT current weather-zone load display reference:
   [https://www.ercot.com/gridinfo/load](https://www.ercot.com/gridinfo/load)
3. NOAA NCEI ISD / Global Hourly:
   [https://www.ncei.noaa.gov/index.php/products/land-based-station/integrated-surface-database](https://www.ncei.noaa.gov/index.php/products/land-based-station/integrated-surface-database)
4. NOAA station to use:
   `USW00003927` (`DAL FTW WSCMO AIRPORT, TX US`)

## Required Outputs

### Local-Only Outputs

Do **not** commit raw data or large JSON artifacts to git unless the repo owner
explicitly asks.

Write these to a local path outside the repo:

1. Prepared dataset:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet`
2. Smoke artifact:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_smoke.json`
3. Full artifact:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json`
4. Summary CSV:
   `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_summary.csv`

### Repo Output

Commit the final report here:

1. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`

Use this template:

1. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/templates/benchmark-iteration-report.md`

### Summary CSV Schema

Write one aggregate row per `(strategy, budget)` combination with these columns:

1. `dataset_id`
2. `strategy`
3. `budget`
4. `n_runs`
5. `test_mae_mean`
6. `test_mae_std`
7. `best_validation_mae_mean`
8. `best_validation_mae_std`
9. `validation_test_gap_mean`
10. `validation_test_gap_std`
11. `runtime_seconds_mean`
12. `runtime_seconds_std`
13. `best_seed_by_test_mae`

## Dataset Preparation Contract

The prepared dataset must satisfy all of these rules:

1. Columns:
   - required: `timestamp`, `target_load`, `temperature`
   - optional: `humidity`, `hour_of_day`, `day_of_week`, `is_holiday`
2. `timestamp` must be hourly, unique, parseable, and saved in `UTC`.
3. `target_load` must be the ERCOT `NORTH_C` series.
4. `temperature` must come from NOAA station `USW00003927`.
5. If multiple weather observations fall in the same UTC hour, aggregate them
   deterministically to one hourly value. Use the hourly mean unless the source
   already provides exactly one hourly observation.
6. Join load and weather on exact UTC hour after both are normalized to the
   same timezone and hourly cadence.
7. Drop duplicate timestamps. If duplicates remain after preparation, stop and
   report the issue instead of guessing.
8. Drop rows with missing `target_load` or `temperature` in the final saved
   dataset.
9. Save to Parquet.

### Time Handling Rules

1. Treat ERCOT timestamps as local market time, then convert to `UTC` in the
   saved dataset.
2. Preserve the series as one row per hour after timezone normalization.
3. The saved dataset must not contain DST duplicate timestamps. Converting to
   `UTC` is the intended way to avoid that problem.

### Minimum QA Gates

Do not run the full benchmark until all of these are true:

1. dataset row count is at least `24,000`
2. required columns are present
3. timestamps are unique
4. timestamps are monotonic after sort
5. no missing `target_load`
6. no missing `temperature`

## Agent Roles

### Agent A: Data Prep And Provenance

Own:

1. source download
2. timestamp normalization
3. load/weather join
4. saved Parquet dataset
5. provenance notes for the report

Deliver:

1. prepared Parquet file at the exact path above
2. row count, start/end timestamps, and QA check results
3. source URLs and station ID
4. a short prep note describing the join and timezone handling

### Agent B: Benchmark Execution

Own:

1. smoke run
2. full benchmark run
3. local JSON artifacts
4. summary CSV

Run from:

1. `/Users/robertwelborn/Projects/causal-optimizer`

Smoke command:

```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_smoke.json
```

Full command:

```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json
```

Rules:

1. smoke must pass before the full run starts
2. do not change benchmark code in the same pass just to make the run succeed
3. if the smoke run fails, stop and capture the traceback in the report draft

### Agent C: Reporting And Review Handoff

Own:

1. summary CSV
2. benchmark report markdown
3. exact reproduction commands
4. interpretation against the success criteria in Plan 05

The report must answer:

1. which strategy had the best mean `test_mae` at each budget?
2. did `causal` beat `random`?
3. did `causal` beat `surrogate_only`?
4. how large was the validation-test gap?
5. were there crashes, skips, or suspiciously unstable seeds?

## Report Requirements

Use the template and fill every section that can be known from this run.

Important details that must be present:

1. exact dataset path
2. exact source URLs
3. NOAA station ID `USW00003927`
4. saved timezone `UTC`
5. actual split row counts and timestamp boundaries
6. exact smoke and full commands
7. artifact paths
8. test MAE mean and std across seeds by strategy and budget
9. validation-test gap mean and std across seeds by strategy and budget
10. clear conclusion on whether this run is positive, negative, or inconclusive

### Model Distribution Section

Use the actual model names from the adapter:

1. `ridge`
2. `rf`
3. `gbm`

## Stop Conditions

Stop and hand back evidence instead of improvising if any of these happen:

1. the prepared dataset still has duplicate timestamps after conversion to UTC
2. smoke run fails on the shipped benchmark
3. the full run produces obviously invalid metrics
4. the benchmark requires a contract change to handle the real dataset

If blocked, the report should become a blocker report rather than pretending the
benchmark was completed.

## Acceptance Checklist

This sprint is complete when:

1. the Parquet dataset exists at the specified local path
2. smoke artifact exists and contains 3 completed rows
3. full artifact exists for the full budget grid
4. summary CSV exists
5. final markdown report is committed in the repo
6. the report contains enough detail for Codex to review the evidence without
   rerunning data prep

## Workflow

Follow the normal agent workflow where it makes sense:

1. `/tdd`
2. implement
3. `/polish`
4. `gh pr create`
5. `/gauntlet`
6. report PR URL

For this sprint, "implement" mostly means data preparation, execution, and
reporting. Avoid benchmark code changes unless a real bug blocks execution.
