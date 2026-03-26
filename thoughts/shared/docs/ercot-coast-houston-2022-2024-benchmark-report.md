# Benchmark Iteration Report

## Metadata

| Field | Value |
|-------|-------|
| Report ID | `20260326-ercot_coast_houston_2022_2024` |
| Benchmark | `predictive_energy` |
| Date | `2026-03-26` |
| Commit | `e6112d30c6f5a62944a8e35593cc7da24ea49643` on branch `main` |
| Dataset ID | `ercot_coast_houston_2022_2024` |
| Dataset | `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet` |
| Dataset Rows | `26,297` |
| Split | `60/20/20` (train=15,778, val=5,259, test=5,260) |
| Reporter | `Agent 2 (Claude)` |
| Timezone | `UTC` |
| Target Series | `ERCOT COAST weather zone` |
| Weather Source | `NOAA station USW00012960 (HOUSTON INTERCONTINENTAL AIRPORT, TX US)` |

## Dataset Provenance

### Source Inputs

| Input | Value |
|-------|-------|
| Load Source | `ERCOT` |
| Load URL | `https://www.ercot.com/gridinfo/load/load_hist` |
| Load Series | `ERCOT COAST weather zone` |
| Weather Source | `NOAA Global Hourly (ISD)` |
| Weather URL | `https://www.ncei.noaa.gov/data/global-hourly/access/` |
| Weather Station | `USW00012960 - HOUSTON INTERCONTINENTAL AIRPORT, TX US` |
| Raw Coverage | `2022-01-01 07:00:00 UTC` to `2024-12-31 23:00:00 UTC` |
| Prepared Dataset Path | `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet` |
| Prepared File Size | `626 KB` |
| Prepared File SHA256 | `bc8d56c6f7e3c3e7a21963d02147ad790994c4f8afcaa309935898d9e6e4a570` |

### Weather Station Selection

George Bush Intercontinental Airport (IAH), station ID `USW00012960`, was selected as the Houston-area NOAA station. This station was preferred over Hobby Airport (`USW00012918`) because:

1. IAH has complete hourly coverage across the full 2022-2024 date range with no gaps.
2. IAH is a major international airport station with high-quality, continuous automated observations.
3. The station is centrally located in the Houston metropolitan area within the ERCOT COAST weather zone.

### Preparation Notes

1. ERCOT timestamps were treated as US/Central market time (CST/CDT), then converted to UTC. The "Hour Ending" convention was preserved: each timestamp represents the end of the hourly interval.
2. DST fall-back handling: ERCOT marks the repeated hour with a "DST" suffix. The DST-tagged rows were converted using the CDT offset (UTC-5) directly, while the 01:00 AM ambiguous hours on fall-back dates were localized as CDT. Spring-forward nonexistent 02:00 timestamps were shifted forward. All 6 DST transitions across the 3-year dataset were handled correctly with zero duplicate UTC timestamps.
3. The "24:00" ERCOT convention (midnight expressed as hour 24 of the previous day) was converted to 00:00 of the next day (1,096 rows across 3 years).
4. Load and weather were joined via inner join on exact UTC hour; 7 hours dropped where NOAA coverage started later than ERCOT (first 7 hours of 2022-01-01). No rows were dropped for missing data after the join.
5. NOAA sub-hourly observations were averaged to hourly means. Temperature was parsed from the ISD TMP field (tenths of Celsius). Humidity was computed from temperature and dewpoint via the Magnus formula.
6. Calendar features (hour_of_day, day_of_week, is_holiday) were derived from ERCOT local market time (US/Central) before converting timestamps to UTC for storage.

## Dataset QA

### Structural Checks

| Check | Result | Detail |
|-------|--------|--------|
| Required columns present | `PASS` | timestamp, target_load, temperature, humidity, hour_of_day, day_of_week, is_holiday |
| Timestamps parse cleanly | `PASS` | |
| Timestamps are unique | `PASS` | |
| Timestamps monotonic after sort | `PASS` | |
| Single-series dataset | `PASS` | ERCOT COAST only |
| No missing `target_load` | `PASS` | |
| No missing `temperature` | `PASS` | |
| Saved timezone is explicit | `PASS` | UTC |
| Row count >= 24,000 | `PASS` | 26,297 rows |

### Split Boundaries

| Partition | Rows | Start | End |
|-----------|------|-------|-----|
| Train | `15,778` | `2022-01-01 07:00:00 UTC` | `2023-10-20 16:00:00 UTC` |
| Validation | `5,259` | `2023-10-20 17:00:00 UTC` | `2024-05-26 19:00:00 UTC` |
| Test | `5,260` | `2024-05-26 20:00:00 UTC` | `2024-12-31 23:00:00 UTC` |

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Strategies | `random, surrogate_only, causal` |
| Budgets | `20, 40, 80` |
| Seeds | `0, 1, 2, 3, 4` |
| Search Space | `7` variables (model_type, lookback_window, use_temperature, use_humidity, use_calendar, regularization, n_estimators) |
| Artifact Path | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_results.json` |

## Results Summary

### Primary Metric: Test MAE (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `105.4586 +/- 0.3414` | `105.3919 +/- 0.2845` | `105.3848 +/- 0.2442` |
| surrogate_only | `105.6230 +/- 0.2154` | `105.6759 +/- 0.1890` | `105.5189 +/- 0.2984` |
| causal | `105.6230 +/- 0.2154` | `105.6759 +/- 0.1890` | `105.5189 +/- 0.2984` |

### Validation MAE (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `90.4509 +/- 0.0518` | `90.4208 +/- 0.0517` | `90.3680 +/- 0.0491` |
| surrogate_only | `90.4388 +/- 0.0581` | `90.4293 +/- 0.0714` | `90.3377 +/- 0.0668` |
| causal | `90.4388 +/- 0.0581` | `90.4293 +/- 0.0714` | `90.3377 +/- 0.0668` |

### Validation-Test Gap (mean +/- std across seeds)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `15.0077 +/- 0.3588` | `14.9711 +/- 0.3281` | `15.0168 +/- 0.2450` |
| surrogate_only | `15.1842 +/- 0.2010` | `15.2466 +/- 0.1979` | `15.1812 +/- 0.2479` |
| causal | `15.1842 +/- 0.2010` | `15.2466 +/- 0.1979` | `15.1812 +/- 0.2479` |

### Runtime (mean seconds per run)

| Strategy | Budget 20 | Budget 40 | Budget 80 |
|----------|-----------|-----------|-----------|
| random | `190.9` | `347.1` | `665.7` |
| surrogate_only | `65.0` | `67.1` | `131.6` |
| causal | `62.8` | `68.8` | `163.3` |

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
| Smoke JSON | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_smoke.json` | 3 rows, all strategies passed |
| Full Results JSON | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_results.json` | 45 runs total |
| Summary CSV | `/Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_summary.csv` | |
| Report Markdown | `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-coast-houston-2022-2024-benchmark-report.md` | This file |

## Interpretation

### Success Criteria Check (from Plan 05)

| Criterion | Result | Evidence |
|-----------|--------|----------|
| causal or surrogate_only beats random on test MAE | `FAIL` | Random achieved 105.38-105.46 vs 105.52-105.68 for causal/surrogate_only; random is marginally better at all budgets |
| Results hold across multiple seeds | `INCONCLUSIVE` | All comparisons are ties at 1% threshold across 5 seeds; differences are sub-0.3 MW |
| Gains not erased by validation-test gap | `INCONCLUSIVE` | Val-test gap is consistent (~15.0-15.2 MW) with no overfitting signal, but there are no gains to preserve |
| Chosen models not pathological on runtime | `PASS` | Engine-based strategies are 2.5-5x faster than random due to off-policy skip; all runs completed without crashes |

### Failure Signal Check (from Plan 05)

| Signal | Observed? | Detail |
|--------|-----------|--------|
| Validation gains do not transfer to test | `NO` | Val-test gap is stable and consistent across strategies |
| causal indistinguishable from random | `YES` | All head-to-head comparisons are ties at 1% threshold across every budget and seed |
| causal consistently worse than surrogate_only | `NO` | causal and surrogate_only produce identical results in every run |
| Results vary wildly across seeds | `NO` | Std across seeds is 0.05-0.34 MW, indicating high stability |
| Benchmark only winnable by exploiting one val slice | `NO` | All strategies converge to the same model class (ridge); no evidence of val-slice exploitation |

### Observations

1. **causal and surrogate_only are identical.** This replicates the NORTH_C finding exactly. Both strategies produce identical results at every budget and seed combination. The causal graph prior does not alter the optimization path under the RF surrogate fallback. This is the same root cause identified in the NORTH_C benchmark: the causal focus variables do not meaningfully influence the fallback optimization path without Ax/BoTorch.

2. **Consistent with NORTH_C results.** The COAST benchmark shows the same pattern as NORTH_C: random is marginally better than engine-based strategies, causal == surrogate_only, all strategies converge to ridge, and budget barely matters. This confirms these are systemic properties of the optimizer, not dataset-specific artifacts.

3. **Budget barely matters.** Results are nearly flat across 20/40/80 budgets (test MAE varies by less than 0.3 MW across budgets for any strategy). The LHS exploration phase appears to saturate the search space quickly, consistent with the NORTH_C observation.

4. **All strategies converge to ridge regression.** Every single run across all 45 strategy-budget-seed combinations selected ridge as the best model. This confirms that ridge dominates the 7-variable search space for energy load forecasting on both COAST and NORTH_C zones.

5. **Larger validation-test gap than NORTH_C.** The val-test gap is approximately 15.0-15.2 MW for COAST vs 7.5-8.0 MW for NORTH_C. This is expected: COAST is a much larger load zone (7k-24k MW range vs 8.5k-17k MW range for NORTH_C), and the test window (May-Dec 2024) includes the high-demand summer peak which differs materially from validation (Oct 2023-May 2024).

6. **Random is marginally better but much slower.** Random achieved ~0.1-0.3 MW lower test MAE but took 2.5-5x longer. The engine-based strategies use off-policy prediction to skip poor candidates, saving wall-clock time but potentially missing marginally better configurations.

7. **Very stable benchmark.** Standard deviations across seeds (0.05-0.34 MW) confirm that the benchmark harness and dataset are producing reproducible results, consistent with NORTH_C stability.

### Cross-Benchmark Comparison (COAST vs NORTH_C)

| Metric | COAST | NORTH_C |
|--------|-------|---------|
| Dataset rows | 26,297 | 26,291 |
| Load range (MW) | 7,128 - 23,963 | 8,573 - 16,965 |
| Best test MAE (random, B80) | 105.38 | 132.35 |
| Val-test gap | ~15.0 MW | ~7.5 MW |
| causal == surrogate_only | Yes | Yes |
| random marginally better | Yes | Yes |
| All ridge | Yes | Yes |
| Budget matters | No | No |

## Action Items

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Land causal fallback differentiation (Agent 1) so causal != surrogate_only | Agent | P0 |
| 2 | Build multi-benchmark suite to evaluate optimizer changes across both datasets | Agent | P1 |
| 3 | Investigate why RF surrogate fallback ignores causal focus variables | Agent | P1 |
| 4 | Consider expanding search space or model types to reduce ridge dominance | Human | P2 |

## Reproducibility

To reproduce this report:

```bash
# Smoke run
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --budgets 3 \
  --seeds 0 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_smoke.json

# Full run
uv run python scripts/energy_predictive_benchmark.py \
  --data-path /Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_coast_houston_2022_2024.parquet \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output /Users/robertwelborn/Projects/_local/causal-optimizer/artifacts/ercot_coast_houston_2022_2024_results.json
```

Commit: `e6112d30c6f5a62944a8e35593cc7da24ea49643`
