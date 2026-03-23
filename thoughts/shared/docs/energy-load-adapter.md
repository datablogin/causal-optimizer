# EnergyLoadAdapter

Day-ahead electricity load forecasting on a fixed historical dataset. Optimizes model configuration (type, features, hyperparameters) to minimize MAE on a held-out validation window.

## Dataset Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `timestamp` | datetime/string | Yes | Hourly timestamps |
| `target_load` | float | Yes | Electricity load in MW |
| `temperature` | float | No | Degrees Celsius |
| `humidity` | float | No | Relative humidity (0-100%) |
| `hour_of_day` | int | No | 0-23 |
| `day_of_week` | int | No | 0-6 (Monday=0) |
| `is_holiday` | int | No | 0 or 1 |

At least one covariate column beyond `timestamp` and `target_load` is required. The adapter accepts a `pd.DataFrame` or a path to a local CSV file.

## Search Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `model_type` | CATEGORICAL | `["ridge", "rf", "gbm"]` | Sklearn model family |
| `lookback_window` | INTEGER | [1, 48] | Number of lagged load features (hours) |
| `use_temperature` | BOOLEAN | | Include temperature as a feature |
| `use_humidity` | BOOLEAN | | Include humidity as a feature |
| `use_calendar` | BOOLEAN | | Include hour_of_day, day_of_week, is_holiday |
| `regularization` | CONTINUOUS | [0.001, 10.0] | Ridge alpha; tree models use `max(1, int(reg))` as `min_samples_leaf` |
| `n_estimators` | INTEGER | [10, 200] | Number of trees (RF/GBM only; ignored by Ridge) |

## Metrics Returned

| Metric | Type | Description |
|--------|------|-------------|
| `mae` | float | Mean absolute error on validation set (primary objective) |
| `rmse` | float | Root mean squared error on validation set |
| `mape` | float | Mean absolute percentage error (%) |
| `runtime_seconds` | float | Wall-clock training + prediction time |
| `feature_count` | float | Number of features used |

## Objective

- **Name:** `mae`
- **Direction:** minimize (`get_minimize() = True`)

## Split Strategy

Blocked time split with no shuffling:

1. Data is sorted by timestamp (assumed already ordered).
2. First `train_ratio` fraction (default 0.7) of rows = training set.
3. Remaining rows = validation set.
4. Lagged features are built using `pd.Series.shift()`, which uses only past data.
5. Rows with NaN (from lagging) are forward-filled, then any remaining NaN rows are dropped.
6. Split index is recomputed after dropping to maintain the temporal boundary.

No data from the validation window is used during training. Feature generation (lags) uses only past information relative to each row.

## Prior Causal Graph

10 directed edges encoding domain knowledge about how search variables affect metrics:

```
lookback_window  --> mae
lookback_window  --> runtime_seconds
use_temperature  --> mae
use_humidity     --> mae
use_calendar     --> mae
regularization   --> mae
model_type       --> mae
model_type       --> runtime_seconds
n_estimators     --> mae
n_estimators     --> runtime_seconds
```

## Descriptors (MAP-Elites)

`["runtime_seconds", "feature_count"]` — enables diversity tracking across fast/slow and feature-sparse/feature-rich configurations.

## Known Assumptions and Limitations

1. **Single time series only.** The adapter does not handle multiple locations or balancing areas. If the dataset contains multiple series, filter to one before passing.
2. **No learning rate search variable.** GBM uses a fixed `learning_rate=0.1`. This could be added as a search variable in a future iteration.
3. **Regularization mapping for tree models is coarse.** The continuous `regularization` parameter is cast to `int` for `min_samples_leaf`, so values 0.001-0.999 all map to `min_samples_leaf=1`.
4. **Validation set size varies with lookback.** Large `lookback_window` values drop more rows (NaN from shifts), shrinking the validation set. The adapter raises `ValueError` if the validation set becomes empty.
5. **No test set.** The adapter uses a train/validation split only. A three-way split (train/val/test) is left to the user.
6. **Fixture data is synthetic.** The 200-row fixture dataset has realistic patterns (daily/weekly seasonality, temperature dependence) but is generated, not real utility data.

## Fixture Dataset

`tests/fixtures/energy_load_fixture.csv` — 200 rows, ~8 days of hourly data, generated deterministically with `numpy` seed 42. Contains daily load peaks, weekend dips, and temperature-correlated demand.
