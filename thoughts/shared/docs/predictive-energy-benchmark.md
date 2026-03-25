# Predictive Energy Benchmark

A locked-split benchmark for comparing causal-optimizer strategies on real
day-ahead electricity load forecasting data.

## Purpose

Answer one question with evidence: **can `causal-optimizer` improve a real
predictive model on unseen data under a fixed experiment budget?**

The benchmark is intentionally narrow: one domain, one modeling task, one
split strategy, three strategies to compare.  If the results are not
convincing on this benchmark, broad predictive-model claims are not
warranted.

## Dataset Contract

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime-parseable | Unique per row, hourly or sub-hourly |
| `target_load` | numeric | Electricity load (MW or similar) |
| `temperature` | numeric | Ambient temperature |

### Optional columns

| Column | Type | Description |
|--------|------|-------------|
| `humidity` | numeric | Relative humidity (%) |
| `hour_of_day` | integer | Hour (0-23) |
| `day_of_week` | integer | Day (0=Monday, 6=Sunday) |
| `is_holiday` | integer/boolean | Holiday indicator |
| `area_id` | string | Balancing area identifier |

### Constraints

- **Single-series**: the dataset must contain exactly one time series.  If
  an `area_id` column exists with multiple values, pass `--area-id` to
  select one.  The harness raises `ValueError` on multi-series data
  without an explicit filter.
- **Local data only**: the benchmark loads a local CSV or Parquet file.
  No network access occurs during execution.
- **No duplicate timestamps**: `split_time_frame` raises on duplicates.

## Locked Split

The dataset is split chronologically (no shuffling) into three partitions:

| Partition | Default fraction | Role |
|-----------|-----------------|------|
| Train | 60% | Model fitting during optimization |
| Validation | 20% | Metric used by the optimizer to select configurations |
| Test | 20% | Held-out evaluation, touched once per completed run |

### Rules

1. **Optimization sees only train + validation.**  The `ValidationEnergyRunner`
   concatenates train and val and uses the first validation timestamp as an
   explicit split boundary (`split_timestamp`).  This survives lag-induced
   NaN dropping.
2. **Test is evaluated once** after the optimizer selects its best
   configuration.  `evaluate_on_test` uses the first test timestamp as the
   split boundary.
3. **Lag features use only past data.**  The `EnergyLoadAdapter` generates
   `load_lag_1` through `load_lag_N` using `.shift()`.  Rows with NaN from
   lag creation are dropped before training.
4. **If preprocessing makes the split impossible, the run fails rather than
   leaking held-out rows into training.**

### Minimum partition size

Each partition must have at least 10 rows after splitting.  Smaller
datasets or extreme fractions raise `ValueError`.

## Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random sampling from the 7-variable search space |
| `surrogate_only` | `ExperimentEngine` without a causal graph (RF surrogate only) |
| `causal` | `ExperimentEngine` with the EnergyLoadAdapter's prior causal graph |

All three strategies use the same search space, runner, and test evaluation.

## Search Space

7 variables controlling model type, features, and hyperparameters:

| Variable | Type | Range |
|----------|------|-------|
| `model_type` | categorical | `ridge`, `rf`, `gbm` |
| `lookback_window` | integer | 1 to 48 |
| `use_temperature` | boolean | true/false |
| `use_humidity` | boolean | true/false |
| `use_calendar` | boolean | true/false |
| `regularization` | continuous | 0.001 to 10.0 |
| `n_estimators` | integer | 10 to 200 |

Models: Ridge regression, RandomForest, GradientBoosting (all scikit-learn).

## Default Budgets and Seeds

- **Budgets**: 20, 40, 80 experiments per strategy
- **Seeds**: 0, 1, 2, 3, 4

Each (strategy, budget, seed) triple produces one `PredictiveBenchmarkResult`.

## Output Artifact

A JSON array of records.  Each record has these fields:

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | string | `"random"`, `"surrogate_only"`, or `"causal"` |
| `budget` | integer | Number of experiments in the run |
| `seed` | integer | RNG seed |
| `best_validation_mae` | float | Best MAE found during optimization (on val set) |
| `test_mae` | float | MAE on the held-out test set using the selected parameters |
| `validation_test_gap` | float | `test_mae - best_validation_mae` |
| `selected_parameters` | object | Parameters that achieved the best validation MAE |
| `runtime_seconds` | float | Wall-clock time for the full run (including test evaluation) |

Non-finite values (`inf`, `nan`) are replaced with `null` for RFC 8259
compliance.  Runs where all experiments crash are omitted from the output.

## Example Command

```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path path/to/data.csv \
  --budgets 20,40 \
  --seeds 0,1,2
```

Full options:

```bash
uv run python scripts/energy_predictive_benchmark.py \
  --data-path path/to/data.csv \
  --area-id AREA_A \
  --budgets 20,40,80 \
  --seeds 0,1,2,3,4 \
  --strategies random,surrogate_only,causal \
  --output predictive_energy_results.json
```

The script prints a summary table to stdout after writing the JSON artifact.

## Key Rule

**Predictive claims depend on untouched test performance.**  Validation-only
wins are not sufficient evidence.  A strategy must show competitive or
superior test MAE (not just validation MAE) to support claims about
real-world prediction quality.  The `validation_test_gap` field exists
specifically to flag overfitting to the validation set.

## Limitations

- **Single-series only**: the benchmark does not handle panel data or
  multi-area aggregation.  Each run operates on one contiguous time series.
- **Narrow search space**: 7 variables covering model type, lag window,
  feature toggles, and regularization.  No feature engineering, no deep
  learning, no external regressors beyond those in the dataset.
- **Ridge / RF / GBM only**: the model zoo is limited to three scikit-learn
  regressors.  Results may not generalize to more expressive model families.
- **No feature engineering**: the benchmark uses raw columns plus simple lag
  features.  No rolling statistics, Fourier terms, or calendar embeddings
  beyond what the dataset provides.
- **Fixture data is synthetic**: the 200-row fixture in
  `tests/fixtures/energy_load_fixture.csv` is for testing only.  Real
  benchmark runs require a real energy dataset.
