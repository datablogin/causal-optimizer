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

The adapter validates data at construction time. The following checks are applied:

- **Empty DataFrame** — raises `ValueError` if the DataFrame has zero rows.
- **Missing required columns** — raises `ValueError` if any of the treatment, outcome, or cost columns are absent.
- **NaN values** — raises `ValueError` if the treatment, outcome, cost, or propensity columns contain any NaN values.
- **Binary treatment enforcement** — raises `ValueError` if the treatment column contains values other than `{0, 1}`.
- **Propensity range** — raises `ValueError` if propensity values fall outside `[0, 1]`.
- **Boundary propensity warning** — emits a `logger.warning` when propensity values include 0.0 or 1.0, since IPS weights will be clipped during evaluation.
- **Single-arm data warning** — emits a `logger.warning` when all observations have the same treatment value (all 0 or all 1), because IPS weighting requires both treated and control observations for reliable estimates.

Exactly one of `data` (DataFrame) or `data_path` (file path) must be provided. `data_path` accepts both CSV and `.parquet` files, detected by file extension.

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
| `propensity_clip_fraction` | float | Fraction of observations whose propensity was clipped by `min_propensity_clip` |
| `max_ips_weight` | float | Largest IPS weight in the evaluation. High values signal variance risk. |
| `weight_cv` | float | Coefficient of variation of positive IPS weights. Higher values indicate less stable estimates. |
| `zero_support` | float | 1.0 when no logged observations match the proposed policy (zero IPS support), 0.0 otherwise |

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
5. **Zero-support fallback is intentionally pessimistic.** When no logged observations match the proposed policy (i.e., the IPS weight sum is zero), the adapter cannot compute a meaningful IPS estimate. Instead of returning an arbitrary value, it sets `zero_support = 1.0` and uses a pessimistic fallback for `policy_value`: the minimum observed outcome (when maximizing) or the maximum observed outcome (when minimizing). This prevents the optimizer from rewarding policies that have no empirical support in the logged data. The adapter also emits a `logger.warning` in this case. At construction time, propensity bounds are validated (values must be in `[0, 1]`), and boundary propensities (exactly 0.0 or 1.0) trigger a warning. During evaluation, propensities are clipped to `[min_propensity_clip, 1 - min_propensity_clip]` to stabilize IPS weights.
6. **Fixture data is synthetic.** The 300-row fixture dataset has realistic confounding (segment affects both propensity and outcome) but is generated, not real marketing data.

## Runtime Notes

The off-policy predictor (`OffPolicyPredictor`) gates observational (DoWhy) causal estimation behind `obs_min_history`, which defaults to 20. When `epsilon_mode=True` is enabled on the engine, the predictor will not attempt observational estimates until the experiment log contains at least 20 results. This avoids expensive DoWhy calls on small experiment logs where there is insufficient data for reliable causal estimation. Users who enable `epsilon_mode` and wonder why observational estimates only appear after 20 experiments can adjust this threshold via the `obs_min_history` parameter on `OffPolicyPredictor`.

## Fixture Dataset

`tests/fixtures/marketing_log_fixture.csv` — 300 rows, generated deterministically with `numpy` seed 42. Contains user_id, segment, channel, treatment, propensity, outcome, cost, timestamp, and age_group columns with realistic correlations.
