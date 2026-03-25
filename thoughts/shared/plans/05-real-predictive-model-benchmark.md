# Real Predictive-Model Benchmark

## Purpose

We need one benchmark that answers a sharper question than the current synthetic suites:

1. Can `causal-optimizer` improve a real predictive model under a fixed experiment budget?
2. Do validation gains transfer to an untouched test window?
3. Does causal guidance beat weaker baselines on the same search space?

This benchmark should measure **real model-building performance**, not just simulator behavior.

## First Benchmark To Build

Start with **day-ahead electricity load forecasting** on a real hourly dataset.

Why this first:

1. We already have `EnergyLoadAdapter`, so the shortest path is to upgrade an existing adapter into a real benchmark harness.
2. The objective is clear and stable: forecast error on future demand.
3. The domain has strong causal structure (weather, calendar, lagged demand) without the operational risk of live finance or ad spend.
4. It is a genuine predictive-model task, not an offline policy-evaluation surrogate.

## Benchmark Question

Given the same data, search space, and experiment budget:

1. Does `causal` beat `random`?
2. Does `causal` beat `surrogate_only`?
3. Does the selected configuration improve **test** MAE over a fixed baseline model?

This is the minimum benchmark needed to justify claims about predictive-model optimization.

## Dataset Contract

Use one real hourly load dataset stored locally outside the repo, plus the existing fixture only for unit tests.

Required columns:

1. `timestamp`
2. `target_load`
3. `temperature`

Optional but recommended:

1. `humidity`
2. `hour_of_day`
3. `day_of_week`
4. `is_holiday`
5. `area_id` if the source contains multiple balancing areas

Dataset expectations:

1. At least 1 year of hourly history
2. No network access during benchmark execution
3. Data path passed explicitly by the user or benchmark config

## Locked Split Strategy

The benchmark must use a **three-way time split**:

1. `train`: first 60%
2. `validation`: next 20%
3. `test`: final 20%

Rules:

1. The optimizer may only see `train` and `validation`.
2. `test` is touched once per completed run, after the best validation configuration is chosen.
3. Lag features must use only past data.
4. If the dataset is multi-series, filter to one `area_id` before running the benchmark.

## Runner Definition

The benchmark runner should train a real forecasting model per experiment, not a simulator.

Recommended model families:

1. `ridge`
2. `random_forest`
3. `hist_gbm`

Experiment output during optimization:

1. `mae`
2. `rmse`
3. `runtime_seconds`
4. `feature_count`

Final run summary after budget is exhausted:

1. best validation MAE
2. test MAE of the selected configuration
3. validation-test gap
4. selected parameter set

## Search Space

Keep the first benchmark narrow and real:

1. model family
2. lookback window
3. temperature on/off
4. humidity on/off
5. calendar features on/off
6. regularization or tree leaf control
7. estimator count for tree models

Do not add open-ended feature engineering yet.

## Strategies To Compare

Every benchmark run must compare the same budgets across:

1. `random`
2. `surrogate_only`
3. `causal`

Optional fourth baseline:

1. `optuna_tpe` or other standard AutoML baseline, only if we are willing to maintain that dependency

## Budget Grid

Run at least:

1. budget `20`
2. budget `40`
3. budget `80`

Seeds:

1. `5` seeds per strategy

This gives enough signal to distinguish genuine improvement from lucky early hits.

## Primary Score

The benchmark score is:

1. **test MAE of the best validation-selected configuration**

This prevents a false win where the optimizer only learns to exploit the validation window.

Secondary scores:

1. best validation MAE
2. validation-test gap
3. runtime per run
4. experiments to reach within X% of the best observed test-backed configuration

## Acceptance Criteria

The benchmark is ready when all of the following are true:

1. One command runs all strategies on the same dataset and budget grid.
2. Results are reproducible for fixed seeds.
3. A results table is emitted with mean and std across seeds.
4. Test performance is reported separately from validation performance.
5. The benchmark can show when a strategy overfits validation.

## What Would Count As Success

I would count the optimizer as credibly useful for predictive-model work if this benchmark shows:

1. `causal` or `surrogate_only` consistently beats `random` on **test** MAE
2. results hold across multiple seeds
3. gains are not erased by validation-test gap
4. the chosen models are not pathological from a runtime perspective

## What Would Count As Failure

Any of these would be important negative evidence:

1. validation gains do not transfer to test
2. `causal` is indistinguishable from `random`
3. `causal` is consistently worse than `surrogate_only`
4. results vary wildly across seeds
5. the benchmark is only winnable by exploiting one validation slice

## Deliverables

To implement this benchmark cleanly, add:

1. a benchmark runner script under `examples/` or `scripts/`
2. a small benchmark config file describing data path, split ratios, and budgets
3. a results artifact format (`csv` or `json`) with per-seed outcomes
4. one regression test for seed reproducibility and one smoke test for a tiny budget

## Recommended Next Step After This Benchmark

If the energy benchmark is successful, the second real benchmark should be:

1. tabular classification/regression on a real public dataset with a true train/validation/test split

That would test whether the optimizer can generalize beyond time series into standard predictive-model workflows.
