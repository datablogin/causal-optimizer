# Time-Series Calendar Profiler

Updated: 2026-03-26

## Purpose

Define a first-pass `TimeSeriesCalendarProfiler` that inspects a time-series
dataset and recommends time-handling and calendar-feature strategies before the
optimizer runs.

This is meant to reduce churn between the researcher and the optimizer by
catching common time-index and calendar mistakes early, with explicit
recommendations instead of silent failure.

This document combines:

1. feature spec
2. agent brief
3. recommendation / output schema

## Why This Should Exist

The first real ERCOT benchmark exposed a class of problems that are not really
modeling failures:

1. UTC vs local-market-time confusion
2. holiday flags computed in the wrong timezone
3. hour-ending vs interval-start labeling
4. DST duplicates or missing hours
5. sub-hourly weather aggregated incorrectly

These are usually better handled by:

1. diagnostics
2. explicit warnings
3. normalization recommendations

not by teaching the predictive optimizer to guess around bad semantics.

## Philosophy

Start with a **rule-based profiler**, not a full ML model.

Why:

1. these issues are structural and explainable
2. we want deterministic, reviewable recommendations
3. we do not yet have enough labeled dataset cases to train a reliable meta-model
4. a profiler can later generate training data for a learned recommender

Longer term:

1. log profiler outputs
2. log human accept/reject decisions
3. train a meta-model over datasets later if it becomes useful

## Scope

The profiler should answer questions like:

1. what is the dominant cadence?
2. are there duplicate or missing timestamps?
3. is there evidence of DST artifacts?
4. do calendar effects appear to align better in a local timezone than in UTC?
5. does the timestamp convention look like hour-ending instead of interval-start?
6. should holiday features be derived in a local timezone?
7. how should sub-hourly signals be rolled up to hourly?

This is **not** a forecasting model.

## Inputs

Required:

1. `timestamp` column

Optional but highly useful:

1. `target` or `target_load`
2. `temperature`
3. `humidity`
4. `area_id`
5. declared market or locale metadata
6. declared source system metadata

Runtime inputs:

1. `data`: DataFrame or path to local CSV/Parquet
2. `timestamp_col`: default `timestamp`
3. `target_col`: optional
4. `candidate_timezones`: optional list
5. `candidate_holiday_calendars`: optional list
6. `expected_cadence`: optional
7. `market_hint`: optional, e.g. `ercot`, `pjm`, `caiso`

## Core Checks

### 1. Timestamp Parse And Monotonicity

Detect:

1. unparsable timestamps
2. mixed timezone awareness
3. non-monotonic order
4. duplicate timestamps

Outputs:

1. parse success/failure
2. duplicate count
3. monotonicity flag
4. recommendation to sort, deduplicate, or stop

### 2. Cadence Inference

Infer:

1. dominant interval
2. cadence regularity
3. gap count
4. sub-hourly vs hourly vs daily pattern

Outputs:

1. inferred cadence
2. regularity score
3. list of unusual interval frequencies
4. recommendation to resample or leave as-is

### 3. DST / Timezone Artifact Detection

Look for:

1. 23-hour / 25-hour day patterns
2. duplicate local-like hours after conversion candidates
3. suspicious hour-of-day shifts
4. exact one-hour discontinuities around likely DST dates

Outputs:

1. DST suspicion flag
2. recommended storage timezone
3. recommendation for deriving calendar features in local time vs UTC

### 4. Interval Convention Detection

Try to detect:

1. hour-start labeling
2. hour-ending labeling
3. whether shifting timestamps backward by one interval improves alignment

Heuristic examples:

1. compare seasonal strength or lag alignment under current vs shifted timestamps
2. compare daily/hourly pattern sharpness before and after interval shift

Outputs:

1. inferred interval convention
2. confidence
3. recommendation:
   - keep as interval-start
   - shift to interval-start
   - needs human review

### 5. Calendar-Basis Comparison

If a target column exists, compare candidate calendar encodings under:

1. UTC
2. candidate local timezones

Example signals:

1. variance explained by hour-of-day buckets
2. variance explained by day-of-week buckets
3. holiday effect magnitude and consistency
4. lag-1 and lag-24 correlation structure

Outputs:

1. preferred calendar basis
2. ranked candidate timezones
3. confidence score
4. recommendation for where to derive:
   - `hour_of_day`
   - `day_of_week`
   - `is_holiday`

### 6. Holiday Calendar Recommendation

If local timezone is known or inferred, compare holiday alignment under:

1. no holiday calendar
2. US federal calendar
3. market-specific holiday calendar if applicable

Outputs:

1. recommended holiday calendar
2. recommendation confidence
3. warning if holiday choice is low-confidence

### 7. Sub-Hourly Aggregation Recommendation

If input cadence is sub-hourly but benchmark expects hourly:

1. recommend rollup rule
2. recommend whether mean, last, sum, or weighted average is appropriate

This should be heuristic-first and source-aware when hints exist.

Outputs:

1. recommended aggregation method
2. reason
3. warning if multiple methods look plausible

## Recommendations The Profiler Should Be Able To Emit

Examples:

1. derive calendar features in `US/Central` before converting to `UTC` for storage
2. convert hour-ending labels to interval-start
3. store timestamps in `UTC` but generate holidays in local market time
4. aggregate weather observations to hourly mean
5. stop: duplicate timestamps remain after normalization
6. stop: cadence ambiguity too high for automatic conversion

## Output Schema

The profiler should emit one structured report object plus optional human-readable markdown.

### Structured Schema

```json
{
  "dataset_id": "string",
  "timestamp_column": "string",
  "target_column": "string | null",
  "summary": {
    "row_count": 0,
    "timestamp_start": "YYYY-MM-DD HH:MM:SS",
    "timestamp_end": "YYYY-MM-DD HH:MM:SS",
    "parse_ok": true,
    "monotonic": true,
    "duplicate_timestamps": 0,
    "inferred_cadence": "hourly",
    "cadence_regularity": 0.0
  },
  "timezone_analysis": {
    "candidate_timezones": [
      {
        "timezone": "UTC",
        "score": 0.0,
        "notes": ["string"]
      }
    ],
    "recommended_storage_timezone": "UTC",
    "recommended_calendar_timezone": "US/Central",
    "dst_suspected": true,
    "confidence": 0.0
  },
  "interval_analysis": {
    "recommended_convention": "interval_start | shift_back_one_hour | needs_review",
    "confidence": 0.0,
    "notes": ["string"]
  },
  "calendar_analysis": {
    "derive_hour_of_day_in": "US/Central",
    "derive_day_of_week_in": "US/Central",
    "derive_is_holiday_in": "US/Central",
    "holiday_calendar": "US_FEDERAL | NONE | CUSTOM",
    "confidence": 0.0,
    "notes": ["string"]
  },
  "aggregation_analysis": {
    "recommended_rollup": "mean | last | sum | none",
    "target_cadence": "hourly",
    "confidence": 0.0,
    "notes": ["string"]
  },
  "warnings": [
    {
      "code": "string",
      "severity": "info | warning | error",
      "message": "string"
    }
  ],
  "recommendations": [
    {
      "id": "string",
      "priority": "P0 | P1 | P2",
      "action": "string",
      "reason": "string",
      "confidence": 0.0
    }
  ],
  "stop": false
}
```

### Human-Readable Report Sections

1. dataset summary
2. cadence / gap diagnostics
3. timezone / DST diagnostics
4. interval convention recommendation
5. calendar feature recommendation
6. holiday recommendation
7. aggregation recommendation
8. warnings
9. recommended next step

## Minimum Recommendation Set

For the first version, the profiler is successful if it can reliably issue:

1. `derive calendar features in local market time before UTC storage`
2. `store timestamps in UTC`
3. `shift hour-ending to interval-start`
4. `aggregate sub-hourly weather to hourly mean`
5. `stop due to duplicate timestamps or unresolved cadence ambiguity`

## API Shape

Suggested first-pass Python API:

```python
@dataclass
class CalendarProfileRecommendation:
    id: str
    priority: str
    action: str
    reason: str
    confidence: float


@dataclass
class TimeSeriesCalendarProfile:
    dataset_id: str
    summary: dict[str, object]
    timezone_analysis: dict[str, object]
    interval_analysis: dict[str, object]
    calendar_analysis: dict[str, object]
    aggregation_analysis: dict[str, object]
    warnings: list[dict[str, object]]
    recommendations: list[CalendarProfileRecommendation]
    stop: bool


class TimeSeriesCalendarProfiler:
    def profile(self, data: pd.DataFrame, ...) -> TimeSeriesCalendarProfile: ...
```

## Agent Brief

### Goal

Build the first rule-based `TimeSeriesCalendarProfiler` so researchers get
early, explicit guidance on timestamp normalization and calendar-feature
construction before benchmark or optimizer runs.

### Why

1. common time-index mistakes create bad evidence and wasted benchmark cycles
2. these mistakes are often diagnosable with rules and scoring
3. we want a durable preprocessing assistant, not another hidden heuristic inside the optimizer

### Deliverables

1. `TimeSeriesCalendarProfiler` implementation
2. structured output object matching the schema above
3. one small CLI or helper script that runs the profiler on a local dataset
4. one markdown example report for ERCOT-like data
5. tests covering:
   - duplicate timestamps
   - cadence inference
   - local-vs-UTC calendar recommendation
   - DST suspicion
   - hour-ending shift recommendation

### Acceptance Criteria

1. the profiler can reproduce the ERCOT local-calendar recommendation
2. the profiler can explain why it made that recommendation
3. the profiler emits `stop=true` on obviously unsafe cases
4. the profiler is rule-based and deterministic in v1
5. no benchmark contract changes are required to use it

### Non-Goals For V1

1. full ML-based meta-learning
2. automatic data rewriting without review
3. perfect timezone inference for arbitrary global datasets
4. replacing source-specific prep logic entirely

### Suggested File Placement

If implemented in-repo, something like:

1. `causal_optimizer/diagnostics/time_calendar_profiler.py`
2. `tests/unit/test_time_calendar_profiler.py`
3. `scripts/profile_time_series_calendar.py`
4. `thoughts/shared/docs/time-series-calendar-profiler.md`

### Workflow

1. `/tdd`
2. implement deterministic checks
3. add structured output schema
4. add CLI or helper script
5. `/polish`
6. `gh pr create`
7. `/gauntlet`
8. report PR URL

## Suggested Future Evolution

After the rule-based version exists:

1. log recommendations and human overrides
2. collect a small corpus of datasets and accepted preprocessing choices
3. train a meta-model to rank or refine recommendations
4. keep the rule-based system as the interpretable fallback

## Immediate Next Step

If we want to pursue this soon, the first sprint should:

1. implement the deterministic profiler
2. validate it on the ERCOT dataset
3. add one synthetic DST / hour-ending test fixture
4. use the profiler output in benchmark-prep review, not yet in automatic execution
