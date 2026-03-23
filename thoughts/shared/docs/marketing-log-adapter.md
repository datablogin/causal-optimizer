# MarketingLogAdapter

Offline marketing policy evaluation from logged observational data. Evaluates counterfactual treatment policies using inverse propensity score (IPS/IPW) weighting on a fixed logged dataset.

## Dataset Schema

### Required columns

| Column | Type | Default name | Description |
|--------|------|-------------|-------------|
| Treatment | int (0/1) | `treatment` | Binary treatment indicator |
| Outcome | float | `outcome` | Revenue or conversion value |
| Cost | float | `cost` | Per-observation treatment cost |

### Optional columns (used if present)

| Column | Type | Default name | Description |
|--------|------|-------------|-------------|
| Propensity | float (0-1) | `propensity` | Logging policy probability of treatment. If absent, uniform propensity (marginal treatment rate) is assumed. |
| Channel | string | `channel` | Marketing channel (`"email"`, `"social"`, `"search"`). Used for channel-weighted uplift scoring. |
| Segment | string | `segment` | Customer segment (`"high_value"`, `"medium"`, `"low"`). Used for segment-based uplift scoring. |
| Timestamp | datetime | `timestamp` | Date of observation (not used in evaluation, available for analysis). |

Column names for treatment, outcome, cost, and propensity are configurable via constructor parameters (`treatment_col`, `outcome_col`, `cost_col`, `propensity_col`).

### Validation

- Raises `ValueError` on missing required columns, empty DataFrames, or NaN values in required/propensity columns.
- Exactly one of `data` (DataFrame) or `data_path` (CSV path) must be provided.

## Search Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `eligibility_threshold` | CONTINUOUS | [0.0, 1.0] | Minimum normalized uplift score to be eligible for treatment |
| `email_share` | CONTINUOUS | [0.0, 1.0] | Budget fraction allocated to email channel |
| `social_share_of_remainder` | CONTINUOUS | [0.0, 1.0] | Fraction of remaining budget (after email) allocated to social. Simplex parameterization: `social_share = social_share_of_remainder * (1 - email_share)`, `search_share = 1 - email_share - social_share`. All three shares always sum to 1.0. |
| `min_propensity_clip` | CONTINUOUS | [0.01, 0.5] | Floor (and ceiling = 1 - clip) for propensity scores to stabilize IPS weights |
| `regularization` | CONTINUOUS | [0.001, 10.0] | Smooths uplift scores toward the mean. Higher values = more uniform treatment. |
| `treatment_budget_pct` | CONTINUOUS | [0.1, 1.0] | Fraction of eligible users to actually treat (budget constraint) |

## Metrics Returned

| Metric | Type | Description |
|--------|------|-------------|
| `policy_value` | float | Self-normalized IPS-weighted average outcome under the proposed policy (primary objective) |
| `total_cost` | float | IPS-weighted total cost for treated observations, normalized by population size |
| `treated_fraction` | float | Fraction of observations assigned treatment by the policy |
| `effective_sample_size` | float | Kish's effective sample size from IPS weights. 0.0 when no observations match the policy. |

## Objective

- **Name:** `policy_value`
- **Direction:** maximize (`get_minimize() = False`)

## Split Strategy

No train/validation split. The adapter evaluates policies on the full logged dataset using IPS weighting.

Each "experiment" evaluates a different policy configuration (threshold, channel allocation, budget) on the same fixed data. The policy does not train a model — it scores observations using a deterministic function of the policy parameters and dataset features, then computes IPS-weighted outcomes.

This is consistent with the offline policy evaluation paradigm where the dataset is a fixed log from a known logging policy.

## Prior Causal Graph

14 directed edges encoding how policy variables affect metrics:

```
eligibility_threshold        --> treated_fraction
treatment_budget_pct         --> treated_fraction
treated_fraction             --> total_cost
treated_fraction             --> policy_value
treated_fraction             --> effective_sample_size
email_share                  --> total_cost
email_share                  --> policy_value
social_share_of_remainder    --> total_cost
social_share_of_remainder    --> policy_value
regularization               --> treated_fraction
regularization               --> policy_value
min_propensity_clip          --> total_cost
min_propensity_clip          --> effective_sample_size
min_propensity_clip          --> policy_value
```

## Descriptors (MAP-Elites)

`["total_cost", "treated_fraction"]` — enables diversity tracking across cheap/expensive and selective/broad policies.

## Known Assumptions and Limitations

1. **Uplift scores are heuristic, not a trained CATE model.** Scores are computed from channel weights, segment membership, and regularization — not from a fitted uplift model. This is intentional: the adapter parameterizes a *policy*, not a prediction model, and the optimizer searches over policy configurations.
2. **Hardcoded optional column names.** The `channel` and `segment` columns are detected by exact name (`"channel"`, `"segment"`). Unlike the required columns, these are not configurable via constructor parameters.
3. **Segment scoring defaults unknown segments to low-value.** Any segment value other than `"high_value"` or `"medium"` receives a score of 0.2 (the `"low"` weight), with no warning.
4. **`seed` parameter is unused.** Accepted for API consistency with other adapters, but evaluation is fully deterministic given fixed data and parameters.
5. **No positivity violation warnings.** Extreme propensities are clipped silently. The adapter returns `effective_sample_size = 0.0` when no observations match the policy, signaling a degenerate estimate.
6. **Fixture data is synthetic.** The 300-row fixture dataset has realistic confounding (segment affects both propensity and outcome) but is generated, not real marketing data.

## Fixture Dataset

`tests/fixtures/marketing_log_fixture.csv` — 300 rows, generated deterministically with `numpy` seed 42. Contains user_id, segment, channel, treatment, propensity, outcome, cost, timestamp, and age_group columns with realistic correlations.
