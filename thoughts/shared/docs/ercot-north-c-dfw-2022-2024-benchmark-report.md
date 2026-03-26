# Benchmark Iteration Report

## Metadata

| Field | Value |
|-------|-------|
| Report ID | `20260325-0000-ercot_north_c_dfw_2022_2024` |
| Benchmark | `predictive_energy` |
| Date | `2026-03-25` |
| Commit | `49e42a9` on branch `main` |
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
| Train | `15,774` | row 0 | row 15,773 |
| Validation | `5,258` | row 15,774 | row 21,031 |
| Test | `5,259` | row 21,032 | row 26,290 |

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
| random | `132.6230 +/- 0.0923` | `132.6230 +/- 0.0923` | `132.5803 +/- 0.0820` |
| surrogate_only | `133.0555 +/- 0.1829` | `132.9002 +/- 0.2027` | `132.7890 +/- 0.1782` |
| causal | `133.0555 +/- 0.1829` | `132.9002 +/- 0.2027` | `132.7890 +/- 0.1782` |

### Validation MAE (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `124.9251 +/- 0.0420` | `124.9251 +/- 0.0420` | `124.8988 +/- 0.0195` |
| surrogate_only | `125.0886 +/- 0.0981` | `124.9841 +/- 0.1352` | `124.8968 +/- 0.1141` |
| causal | `125.0886 +/- 0.0981` | `124.9841 +/- 0.1352` | `124.8968 +/- 0.1141` |

### Validation-Test Gap (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `7.6979 +/- 0.0716` | `7.6979 +/- 0.0716` | `7.6814 +/- 0.1003` |
| surrogate_only | `7.9669 +/- 0.0880` | `7.9162 +/- 0.0696` | `7.8922 +/- 0.0641` |
| causal | `7.9669 +/- 0.0880` | `7.9162 +/- 0.0696` | `7.8922 +/- 0.0641` |

### Runtime (mean seconds per run)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `190.7` | `342.0` | `625.7` |
| surrogate_only | `64.8` | `66.3` | `128.2` |
| causal | `62.6` | `70.0` | `141.8` |

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
| causal or surrogate_only beats random on test MAE | `FAIL` | Random achieved 132.58-132.62 vs 132.79-133.06 for causal/surrogate_only; random is marginally better at all budgets |
| Results hold across multiple seeds | `INCONCLUSIVE` | All ties at 1% threshold across 5 seeds; differences are sub-0.5 MW |
| Gains not erased by validation-test gap | `INCONCLUSIVE` | Val-test gap is consistent (~7.7-8.0 MW) with no overfitting signal, but there are no gains to preserve |
| Chosen models not pathological on runtime | `PASS` | Engine-based strategies are 3-5x faster than random due to off-policy skip; all runs completed without crashes |

### Failure Signal Check (from Plan 05)

| Signal | Observed? | Detail |
|--------|-----------|--------|
| Validation gains do not transfer to test | `NO` | Val-test gap is stable and consistent across strategies |
| causal indistinguishable from random | `YES` | All head-to-head comparisons are ties at 1% threshold across every budget and seed |
| causal consistently worse than surrogate_only | `NO` | causal and surrogate_only produce identical results in every run |
| Results vary wildly across seeds | `NO` | Std across seeds is 0.08-0.20 MW, indicating high stability |
| Benchmark only winnable by exploiting one val slice | `NO` | All strategies converge to the same model class (ridge); no evidence of val-slice exploitation |

### Observations

1. **causal and surrogate_only are identical.** Both strategies use the same ExperimentEngine with the same seed sequence. The causal graph prior only affects focus variables, and without Ax/BoTorch installed (RF surrogate fallback is active), the causal focus variables do not meaningfully alter the optimization path. This is the most important finding: the causal differentiation mechanism is not exercised under the RF surrogate fallback.

2. **Budget barely matters.** Results are nearly flat across 20/40/80 budgets. The LHS exploration phase (first 10 experiments) appears to saturate the search space quickly, and subsequent optimization steps do not find materially better configurations. This suggests either the search space is too small (7 variables) or the landscape is very flat near the optimum.

3. **All strategies converge to ridge regression.** Screening identified n_estimators, regularization, and lookback_window as the most important variables, yet every run across all seeds and budgets selected ridge as the best model. This indicates the ridge model with the right hyperparameters dominates the search space for this dataset.

4. **Random is marginally better but much slower.** Random achieved ~0.2-0.4 MW lower test MAE but took 3-5x longer to run because it evaluates every candidate configuration. The engine-based strategies use off-policy prediction to skip poor candidates, which saves wall-clock time but may also skip configurations that would have been marginally better.

5. **Very stable benchmark.** The tiny standard deviations (0.08-0.20 MW across seeds) and consistent val-test gaps confirm that the benchmark harness, dataset split, and evaluation pipeline are functioning correctly and producing reproducible results.

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

Commit: `49e42a9053c4934cfb1f63e18d08c5a897c4d016`
