# Benchmark Iteration Report Template

## Metadata

| Field | Value |
|-------|-------|
| Report ID | `{YYYYMMDD-HHMM}-{benchmark_name}` |
| Benchmark | `{benchmark_name}` (e.g., `predictive_energy`) |
| Date | `{YYYY-MM-DD}` |
| Commit | `{short_hash}` on branch `{branch_name}` |
| Dataset ID | `{dataset_id}` (e.g., `ercot_north_c_dfw_2022_2024`) |
| Dataset | `{path_or_description}` |
| Dataset Rows | `{N}` |
| Split | `{train_frac}/{val_frac}/{test_frac}` (e.g., `60/20/20`) |
| Reporter | `{agent_id or human_name}` |
| Timezone | `{timezone_of_saved_dataset}` (recommended: `UTC`) |
| Target Series | `{area_or_series_name}` |
| Weather Source | `{station_name_or_description}` |

## Dataset Provenance

### Source Inputs

| Input | Value |
|-------|-------|
| Load Source | `{official_source_name}` |
| Load URL | `{official_url}` |
| Load Series | `{series_name}` |
| Weather Source | `{official_source_name}` |
| Weather URL | `{official_url}` |
| Weather Station | `{station_id} - {station_name}` |
| Raw Coverage | `{start_timestamp}` to `{end_timestamp}` |
| Prepared Dataset Path | `{local_path_to_prepared_csv_or_parquet}` |
| Prepared File Size | `{size}` |
| Prepared File SHA256 | `{sha256_or_n/a}` |

### Preparation Notes

1. `{How timestamps were normalized and converted to the saved timezone}`
2. `{How load and weather were joined}`
3. `{How duplicate timestamps were handled}`
4. `{How missing temperature/humidity values were handled}`

## Dataset QA

### Structural Checks

| Check | Result | Detail |
|-------|--------|--------|
| Required columns present | `{PASS/FAIL}` | |
| Timestamps parse cleanly | `{PASS/FAIL}` | |
| Timestamps are unique | `{PASS/FAIL}` | |
| Timestamps monotonic after sort | `{PASS/FAIL}` | |
| Single-series dataset | `{PASS/FAIL}` | |
| No missing `target_load` | `{PASS/FAIL}` | |
| No missing `temperature` | `{PASS/FAIL}` | |
| Saved timezone is explicit | `{PASS/FAIL}` | |

### Split Boundaries

| Partition | Rows | Start | End |
|-----------|------|-------|-----|
| Train | `{N}` | `{timestamp}` | `{timestamp}` |
| Validation | `{N}` | `{timestamp}` | `{timestamp}` |
| Test | `{N}` | `{timestamp}` | `{timestamp}` |

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Strategies | `{comma-separated list}` |
| Budgets | `{comma-separated list}` |
| Seeds | `{comma-separated list}` |
| Search Space | `{N}` variables |
| Artifact Path | `{path_to_json_output}` |

## Results Summary

### Primary Metric: Test MAE (mean ± std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| surrogate_only | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| causal | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |

### Validation MAE (mean ± std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| surrogate_only | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| causal | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |

### Validation-Test Gap (mean ± std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| surrogate_only | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |
| causal | `{mean} ± {std}` | `{mean} ± {std}` | `{mean} ± {std}` |

### Runtime (mean seconds per run)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `{mean}` | `{mean}` | `{mean}` |
| surrogate_only | `{mean}` | `{mean}` | `{mean}` |
| causal | `{mean}` | `{mean}` | `{mean}` |

## Diagnostic Signals

### Win/Loss Matrix (test MAE, per-seed head-to-head)

| Comparison | Budget 20 | Budget 40 | Budget 80 |
|------------|-----------|-----------|-----------|
| causal < random | `{W}/{L}/{T}` | `{W}/{L}/{T}` | `{W}/{L}/{T}` |
| causal < surrogate_only | `{W}/{L}/{T}` | `{W}/{L}/{T}` | `{W}/{L}/{T}` |
| surrogate_only < random | `{W}/{L}/{T}` | `{W}/{L}/{T}` | `{W}/{L}/{T}` |

W = wins, L = losses, T = ties (within 1% relative)

### Crash / Skip Rate

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `{N_crash}/{N_total}` | | |
| surrogate_only | `{N_crash}/{N_total}` | | |
| causal | `{N_crash}/{N_total}` | | |

### Selected Model Distribution

| Strategy | ridge | rf | gbm |
|----------|-------|---------------|----------|
| random | `{count}` | `{count}` | `{count}` |
| surrogate_only | `{count}` | `{count}` | `{count}` |
| causal | `{count}` | `{count}` | `{count}` |

Counts are across all seeds and budgets for the best-selected configuration per run.

### Artifact Inventory

| Artifact | Path | Notes |
|----------|------|-------|
| Smoke JSON | `{path}` | `{optional}` |
| Full Results JSON | `{path}` | |
| Summary CSV | `{path}` | |
| Report Markdown | `{path}` | |

## Interpretation

### Success Criteria Check (from Plan 05)

| Criterion | Result | Evidence |
|-----------|--------|----------|
| causal or surrogate_only beats random on test MAE | `{PASS/FAIL/INCONCLUSIVE}` | `{one-line reference}` |
| Results hold across multiple seeds | `{PASS/FAIL/INCONCLUSIVE}` | `{one-line reference}` |
| Gains not erased by validation-test gap | `{PASS/FAIL/INCONCLUSIVE}` | `{one-line reference}` |
| Chosen models not pathological on runtime | `{PASS/FAIL/INCONCLUSIVE}` | `{one-line reference}` |

### Failure Signal Check (from Plan 05)

| Signal | Observed? | Detail |
|--------|-----------|--------|
| Validation gains do not transfer to test | `{YES/NO}` | |
| causal indistinguishable from random | `{YES/NO}` | |
| causal consistently worse than surrogate_only | `{YES/NO}` | |
| Results vary wildly across seeds | `{YES/NO}` | |
| Benchmark only winnable by exploiting one val slice | `{YES/NO}` | |

### Observations

_Free-form notes on what this iteration revealed. Include:_

1. _Any surprising parameter selections_
2. _Whether more budget helps or plateaus_
3. _Whether the causal graph is actually influencing suggestions_
4. _Anything that should change before the next iteration_

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | `{description}` | `{agent/human}` | `{P0/P1/P2}` |

## Reproducibility

To reproduce this report:

```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path {data_path} \
  --budgets {budgets} \
  --seeds {seeds} \
  --strategies {strategies} \
  --output {artifact_path}
```

Commit: `{full_hash}`

If dataset preparation was part of this iteration, include the exact prep command
or script path here as well.
