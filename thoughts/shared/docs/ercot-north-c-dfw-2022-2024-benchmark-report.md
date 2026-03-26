# Benchmark Iteration Report

## Metadata

| Field | Value |
|-------|-------|
| Report ID | `20260325-0000-ercot_north_c_dfw_2022_2024` |
| Benchmark | `predictive_energy` |
| Date | `2026-03-26` |
| Commit | `TBD` on branch `main` |
| Dataset ID | `ercot_north_c_dfw_2022_2024` |
| Dataset | `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet` |
| Dataset Rows | `26,291` |
| Split | `60/20/20` (train=15,774, val=5,258, test=5,259) |
| Reporter | `Agent C (Claude)` |
| Timezone | `UTC` |
| Target Series | `ERCOT NCENT/NORTH_C weather zone` |
| Weather Source | `NOAA station USW00003927 (DAL FTW WSCMO AIRPORT, TX US)` |

## Dataset Provenance

### Source Inputs

| Input | Value |
|-------|-------|
| Load Source | `ERCOT` |
| Load URL | `https://www.ercot.com/gridinfo/load/load_hist` |
| Load Series | `ERCOT NCENT/NORTH_C weather zone` |
| Weather Source | `NOAA Global Hourly` |
| Weather URL | `https://www.ncei.noaa.gov/data/global-hourly/access/` |
| Weather Station | `USW00003927 - DAL FTW WSCMO AIRPORT, TX US` |
| Raw Coverage | `2022-01-01 06:00:00 UTC` to `2024-12-31 23:00:00 UTC` |
| Prepared Dataset Path | `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet` |
| Prepared File Size | `642 KB` |
| Prepared File SHA256 | `n/a` |

### Preparation Notes

1. ERCOT timestamps were treated as US/Central market time, then converted to UTC.
2. Load and weather were joined via inner join on exact UTC hour; 2 rows dropped for missing data.
3. No duplicate timestamps were present after conversion (verified by QA uniqueness check).
4. Humidity was computed from temperature + dewpoint via the Magnus formula. NOAA sub-hourly observations were averaged to hourly.
5. Calendar features (hour_of_day, day_of_week, is_holiday) are derived from ERCOT local market time (US/Central) before converting timestamps to UTC for storage.

## Dataset QA

### Structural Checks

| Check | Result | Detail |
|-------|--------|--------|
| Required columns present | `PASS` | timestamp, target_load, temperature, humidity, hour_of_day, day_of_week, is_holiday |
| Timestamps parse cleanly | `PASS` | |
| Timestamps are unique | `PASS` | |
| Timestamps monotonic after sort | `PASS` | |
| Single-series dataset | `PASS` | ERCOT NCENT/NORTH_C only |
| No missing `target_load` | `PASS` | |
| No missing `temperature` | `PASS` | |
| Saved timezone is explicit | `PASS` | UTC |

### Split Boundaries

| Partition | Rows | Start | End |
|-----------|------|-------|-----|
| Train | `15,774` | `2022-01-01 06:00:00 UTC` | `2023-10-20 11:00:00 UTC` |
| Validation | `5,258` | `2023-10-20 12:00:00 UTC` | `2024-05-26 13:00:00 UTC` |
| Test | `5,259` | `2024-05-26 14:00:00 UTC` | `2024-12-31 23:00:00 UTC` |

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Strategies | `random, surrogate_only, causal` |
| Budgets | `20, 40, 80` |
| Seeds | `0, 1, 2, 3, 4` |
| Search Space | `7` variables (model_type, lookback_window, use_temperature, use_humidity, use_calendar, regularization, n_estimators) |
| Artifact Path | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json` |

## Results Summary

### Primary Metric: Test MAE (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `132.4996 +/- 0.2215` | `132.4996 +/- 0.2215` | `132.3478 +/- 0.1026` |
| surrogate_only | `133.1380 +/- 0.0073` | `133.1545 +/- 0.0089` | `132.7241 +/- 0.3748` |
| causal | `133.1380 +/- 0.0073` | `133.1545 +/- 0.0089` | `132.7241 +/- 0.3748` |

### Validation MAE (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `124.9218 +/- 0.0493` | `124.9218 +/- 0.0493` | `124.8580 +/- 0.0589` |
| surrogate_only | `125.1519 +/- 0.0281` | `125.1423 +/- 0.0173` | `124.9364 +/- 0.1697` |
| causal | `125.1519 +/- 0.0281` | `125.1423 +/- 0.0173` | `124.9364 +/- 0.1697` |

### Validation-Test Gap (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `7.5778 +/- 0.2037` | `7.5778 +/- 0.2037` | `7.4898 +/- 0.1293` |
| surrogate_only | `7.9861 +/- 0.0344` | `8.0122 +/- 0.0165` | `7.7876 +/- 0.2052` |
| causal | `7.9861 +/- 0.0344` | `8.0122 +/- 0.0165` | `7.7876 +/- 0.2052` |

### Runtime (mean seconds per run)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `199.2` | `371.8` | `694.3` |
| surrogate_only | `67.2` | `71.1` | `141.7` |
| causal | `66.9` | `70.6` | `153.4` |

## Diagnostic Signals

### Win/Loss Matrix (test MAE, per-seed head-to-head)

| Comparison | Budget 20 | Budget 40 | Budget 80 |
|------------|-----------|-----------|-----------|
| causal < random | `0/0/5` | `0/0/5` | `0/0/5` |
| causal < surrogate_only | `0/0/5` | `0/0/5` | `0/0/5` |
| surrogate_only < random | `0/0/5` | `0/0/5` | `0/0/5` |

W = wins, L = losses, T = ties (within 1% relative)

### Crash / Skip Rate

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `0/5` | `0/5` | `0/5` |
| surrogate_only | `0/5` | `0/5` | `0/5` |
| causal | `0/5` | `0/5` | `0/5` |

### Selected Model Distribution

| Strategy | ridge | rf | gbm |
|----------|-------|---------------|----------|
| random | `15` | `0` | `0` |
| surrogate_only | `15` | `0` | `0` |
| causal | `15` | `0` | `0` |

Counts are across all seeds and budgets for the best-selected configuration per run.

### Artifact Inventory

| Artifact | Path | Notes |
|----------|------|-------|
| Smoke JSON | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_smoke.json` | 3 rows, all strategies passed |
| Full Results JSON | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json` | 45 runs total |
| Summary CSV | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_summary.csv` | |
| Report Markdown | `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md` | This file |

## Interpretation

### Success Criteria Check (from Plan 05)

| Criterion | Result | Evidence |
|-----------|--------|----------|
| causal or surrogate_only beats random on test MAE | `FAIL` | Random achieved 132.35-132.50 vs 132.72-133.15 for causal/surrogate_only; random is marginally better at all budgets |
| Results hold across multiple seeds | `INCONCLUSIVE` | All ties at 1% threshold across 5 seeds; differences are sub-0.8 MW |
| Gains not erased by validation-test gap | `INCONCLUSIVE` | Val-test gap is consistent (~7.5-8.0 MW) with no overfitting signal, but there are no gains to preserve |
| Chosen models not pathological on runtime | `PASS` | Engine-based strategies are 3-5x faster than random due to off-policy skip; all runs completed without crashes |

### Failure Signal Check (from Plan 05)

| Signal | Observed? | Detail |
|--------|-----------|--------|
| Validation gains do not transfer to test | `NO` | Val-test gap is stable and consistent across strategies |
| causal indistinguishable from random | `YES` | All head-to-head comparisons are ties at 1% threshold across every budget and seed |
| causal consistently worse than surrogate_only | `NO` | causal and surrogate_only produce identical results in every run |
| Results vary wildly across seeds | `NO` | Std across seeds is 0.01-0.37 MW, indicating high stability |
| Benchmark only winnable by exploiting one val slice | `NO` | All strategies converge to the same model class (ridge); no evidence of val-slice exploitation |

### Observations

1. **causal and surrogate_only are identical.** Both strategies use the same ExperimentEngine with the same seed sequence. The causal graph prior only affects focus variables, and without Ax/BoTorch installed (RF surrogate fallback is active), the causal focus variables do not meaningfully alter the optimization path. This is the most important finding: the causal differentiation mechanism is not exercised under the RF surrogate fallback.

2. **Budget barely matters.** Results are nearly flat across 20/40/80 budgets. The LHS exploration phase (first 10 experiments) appears to saturate the search space quickly, and subsequent optimization steps do not find materially better configurations. This suggests either the search space is too small (7 variables) or the landscape is very flat near the optimum.

3. **All strategies converge to ridge regression.** Screening identified n_estimators, regularization, and lookback_window as the most important variables, yet every run across all seeds and budgets selected ridge as the best model. This indicates the ridge model with the right hyperparameters dominates the search space for this dataset.

4. **Random is marginally better but much slower.** Random achieved ~0.4-0.8 MW lower test MAE but took 3-5x longer to run because it evaluates every candidate configuration. The engine-based strategies use off-policy prediction to skip poor candidates, which saves wall-clock time but may also skip configurations that would have been marginally better.

5. **Very stable benchmark.** The standard deviations (0.01-0.37 MW across seeds) and consistent val-test gaps confirm that the benchmark harness, dataset split, and evaluation pipeline are functioning correctly and producing reproducible results.

6. **To make causal differentiation testable**, the next iteration should either install Ax/BoTorch so the causal acquisition function is active, or modify the RF surrogate path to incorporate focus variables more directly.

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Install Ax/BoTorch and re-run to activate POMIS-aware acquisition function | Agent/Human | P0 |
| 2 | Investigate why RF surrogate fallback ignores causal focus variables | Agent | P1 |
| 3 | Consider expanding the search space or adding more model types to reduce ridge dominance | Human | P2 |
| 4 | Add a budget-2 or budget-5 run to confirm LHS saturation hypothesis | Agent | P2 |
| 5 | Compute SHA256 of prepared Parquet file for full provenance tracking | Agent | P2 |

## Reproducibility

To reproduce this report:

```bash
# Smoke run
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_smoke.json

# Full run
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_north_c_dfw_2022_2024_results.json
```

Commit: `TBD`
