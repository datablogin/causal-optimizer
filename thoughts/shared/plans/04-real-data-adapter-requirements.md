# Real-Data Adapter Requirements

## Purpose

This document is the handoff spec for developer agents implementing the next adapters needed to validate `causal-optimizer` on real data instead of only synthetic simulators.

The immediate goal is not "add more domains." The goal is to create adapters that let us answer:

1. Does causal guidance improve results under a fixed experiment budget?
2. Does the observation vs. intervention logic save work without hiding bad decisions?
3. How does the system behave on noisy, imperfect, partially confounded real data?

## Adapters Required

Implement these two adapters first:

1. `EnergyLoadAdapter`
2. `MarketingLogAdapter`

Do not build a finance adapter yet unless it includes a simulator or logged actions. Raw market prices alone are not a good fit for this engine because the engine optimizes interventions, not passive forecasts.

## Deliverables

Each adapter must include all of the following:

1. A new adapter module under `causal_optimizer/domain_adapters/`.
2. A runnable example under `examples/`.
3. Unit tests under `tests/unit/`.
4. Integration tests under `tests/integration/`.
5. A small deterministic fixture dataset under `tests/fixtures/` or `tests/data/`.
6. Adapter documentation describing dataset schema, assumptions, and metrics.

## Shared Adapter Contract

Each adapter must inherit from `DomainAdapter` and implement:

1. `get_search_space()`
2. `run_experiment(parameters)`

Each adapter must also define, when applicable:

1. `get_prior_graph()`
2. `get_descriptor_names()`
3. `get_objective_name()`
4. `get_minimize()`
5. `get_constraints()`
6. `get_discovery_method()`

The adapter must be directly passable as `runner=adapter` into `ExperimentEngine`.

## Shared Non-Negotiable Requirements

1. No network access during tests or examples.
2. No hidden downloads inside adapter constructors.
3. Same seed + same data + same parameters must produce the same metrics.
4. Evaluation must use a fixed, immutable split. No leakage across train, validation, and test windows.
5. The adapter must validate required dataset columns and fail with explicit error messages.
6. `run_experiment()` must return only numeric metrics.
7. At least one metric must represent experiment cost or efficiency.
8. The adapter must run fast enough for repeated engine loops on fixture data.

## EnergyLoadAdapter Requirements

### Goal

Support day-ahead electricity load forecasting on a fixed historical dataset.

### Suggested file

`causal_optimizer/domain_adapters/energy_load.py`

### Dataset shape

The adapter must accept a local CSV or Parquet file containing, at minimum:

1. `timestamp`
2. `target_load`
3. One location or balancing-area identifier if multiple series are present
4. Weather or calendar covariates known at prediction time

Allowed exogenous features include:

1. Temperature
2. Humidity
3. Hour of day
4. Day of week
5. Holiday indicator
6. Lagged load features built only from past data

### Experiment interpretation

One experiment should mean:

1. Train a forecasting configuration on a fixed train window.
2. Evaluate it on a fixed validation window.
3. Return forecast metrics and cost metrics.

### Search space

Use optimizer-controlled choices that are realistic and bounded. Examples:

1. Model family or model variant
2. Lookback window
3. Feature toggles
4. Regularization strength
5. Learning rate
6. Tree depth or estimator count
7. Horizon-specific weighting

Do not expose raw timestamps, row indices, or target leakage knobs as variables.

### Metrics

Required metrics:

1. `mae`
2. One scale-aware metric such as `rmse` or `mape`
3. One cost metric such as `runtime_seconds`

Recommended descriptors:

1. `runtime_seconds`
2. `feature_count`

### Objective

Primary objective should be `mae` with `minimize=True`.

### Prior graph

Provide a conservative prior graph over tunable choices and derived metrics only. Do not pretend to discover the physical grid graph from adapter code.

Acceptable examples:

1. `lookback_window -> mae`
2. `weather_features_enabled -> mae`
3. `model_complexity -> runtime_seconds`
4. `regularization -> mae`

### Validation rules

The adapter must enforce:

1. Rolling or blocked time split
2. No fitting on validation or test windows
3. Feature generation using past information only
4. Explicit handling of missing timestamps or missing weather rows

## MarketingLogAdapter Requirements

### Goal

Support logged marketing or uplift-style data where actions, outcomes, and context are all present.

### Suggested file

`causal_optimizer/domain_adapters/marketing_logs.py`

### Dataset shape

The adapter must accept a local CSV or Parquet file containing, at minimum:

1. Context features
2. A treatment or action column
3. An outcome column
4. A cost or spend column

Strongly preferred:

1. Propensity column from the logging policy
2. Timestamp column
3. Campaign or channel identifier

### Experiment interpretation

One experiment should mean evaluating a fixed policy or uplift configuration on a locked logged dataset.

Acceptable experiment types:

1. Threshold policy over uplift scores
2. Treatment assignment rule over segments
3. Budget allocation policy over channels

Not acceptable:

1. Adapters that only fit a predictor and report in-sample AUC
2. Adapters with no action/treatment representation

### Search space

Use policy-relevant, interpretable choices. Examples:

1. Segment threshold
2. Channel allocation share
3. Treatment eligibility rule
4. Regularization strength
5. Minimum propensity clip

### Metrics

Required metrics:

1. `policy_value` or `incremental_outcome`
2. `total_cost`
3. One support metric such as `effective_sample_size`, `coverage`, or `treated_fraction`

Recommended descriptors:

1. `total_cost`
2. `treated_fraction`

### Objective

Primary objective should be a value metric with `minimize=False`.

### Prior graph

Provide a cautious graph that reflects policy and outcome structure, not unverifiable business folklore.

Acceptable examples:

1. `eligibility_threshold -> treated_fraction`
2. `treated_fraction -> total_cost`
3. `channel_share -> total_cost`
4. `creative_policy -> policy_value`

### Validation rules

The adapter must enforce:

1. Clear separation between logging policy information and learned policy evaluation
2. Positivity checks or explicit support warnings
3. Defensive handling when propensities are missing
4. No using future outcomes to define current actions

## Fixture Data Requirements

Each adapter must include a small fixture dataset that is:

1. Checked into the repo
2. Small enough for fast tests
3. Deterministic
4. Documented with column descriptions

The full real dataset may live outside the repo, but tests must not depend on it.

## Required Tests

Each adapter must ship with unit tests covering:

1. Search space shape and variable types
2. Objective name and minimize direction
3. Prior graph existence and basic structure
4. Determinism under fixed seed
5. Dataset schema validation
6. Failure behavior on missing columns or bad parameters
7. Metric presence and numeric types

Each adapter must ship with integration tests covering:

1. `ExperimentEngine.run_loop()` for at least 15 experiments without crashes
2. Phase transition behavior
3. `diagnose()` producing a report
4. Reproducibility with a fixed seed
5. A basic optimization sanity check against a naive baseline

## Example Requirements

Each adapter example must:

1. Load local fixture data or a clearly provided local path
2. Construct `ExperimentEngine` with the adapter
3. Run a short loop
4. Print the best result and a small diagnostic summary

Examples must not require external services.

## Documentation Requirements

Each adapter must include a short markdown document covering:

1. Dataset schema
2. Search variables
3. Metrics returned
4. Objective definition
5. Split strategy
6. Known assumptions and limitations

## What I Will Test After Handoff

When the adapters are ready, I will:

1. Run the adapter unit tests.
2. Run the adapter integration tests.
3. Run short end-to-end engine loops on both adapters.
4. Compare behavior against existing synthetic adapters.
5. Audit failures, flaky behavior, data leakage risks, and metric inconsistencies.
6. Identify bugs, new requirements, and feature opportunities.

The minimum expected local command set is:

```bash
uv run pytest \
  tests/unit/test_energy_load_adapter.py \
  tests/unit/test_marketing_log_adapter.py \
  tests/integration/test_energy_load_adapter.py \
  tests/integration/test_marketing_log_adapter.py
```

And, if examples are added:

```bash
uv run python examples/energy_load.py
uv run python examples/marketing_logs.py
```

## Acceptance Checklist

An adapter is ready for my review only when all of the following are true:

1. Code compiles and imports cleanly.
2. New tests are present and passing locally.
3. Fixture data is committed.
4. Example script runs without network access.
5. Required documentation is present.
6. The adapter uses a fixed evaluation split and documents it.
7. The adapter returns stable numeric metrics.

## Nice-to-Have Features

These are not required for first pass, but they are useful:

1. Constraint support
2. Multi-objective support
3. Optional discovered-graph mode
4. Policy support diagnostics for logged data
5. Runtime telemetry
6. Explicit warning metrics for leakage or support violations

## Out of Scope For First Pass

Do not spend time on these before the basic adapters are complete:

1. Finance price-only adapters
2. Remote dataset ingestion
3. Dashboard work
4. Async execution
5. LLM-generated causal graphs
