# Sprint 17: Time-Series Calendar Profiler

## Goal

Build the first rule-based `TimeSeriesCalendarProfiler` for `causal-optimizer`.

This profiler should inspect a time-series dataset and recommend:

1. timestamp normalization strategy
2. calendar-feature basis
3. interval convention handling
4. aggregation strategy for sub-hourly inputs
5. stop conditions when automatic handling would be unsafe

This sprint is about **data semantics diagnostics**, not about changing the
predictive optimizer itself.

## Why This Matters

The first real ERCOT benchmark exposed a class of problems that can produce bad
evidence without ever showing up as code failures:

1. UTC vs local-market-time confusion
2. holiday flags derived in the wrong timezone
3. hour-ending vs interval-start labeling mistakes
4. DST duplicate or missing hours
5. sub-hourly weather aggregated incorrectly

We do not want the optimizer silently adapting to those mistakes. We want an
explicit diagnostics layer that helps the researcher normalize the dataset
correctly before benchmark runs.

## What To Build

Implement a first-pass `TimeSeriesCalendarProfiler` that is:

1. deterministic
2. rule-based
3. explainable
4. safe by default

Do **not** build a learned meta-model in this sprint.

## Scope

The profiler should answer:

1. can timestamps be parsed cleanly?
2. are timestamps monotonic and unique?
3. what is the dominant cadence?
4. are there DST-like artifacts?
5. does the data look more like UTC or local market time for calendar features?
6. does the timestamp convention look like interval-start or hour-ending?
7. how should sub-hourly inputs be aggregated if hourly output is needed?

## Inputs

Required:

1. `timestamp` column

Optional:

1. `target` or `target_load`
2. `temperature`
3. `humidity`
4. `area_id`
5. `market_hint`
6. candidate timezones
7. candidate holiday calendars

## Core Checks

### 1. Timestamp Parse And Order

Detect:

1. parse failures
2. mixed timezone awareness
3. non-monotonic rows
4. duplicate timestamps

### 2. Cadence Inference

Infer:

1. dominant interval
2. cadence regularity
3. gap count
4. whether the data is sub-hourly, hourly, or daily

### 3. DST / Timezone Artifact Detection

Detect:

1. 23-hour / 25-hour day patterns
2. suspicious one-hour discontinuities
3. local-calendar alignment patterns

### 4. Interval Convention Detection

Try to distinguish:

1. interval-start
2. hour-ending
3. unresolved / needs review

### 5. Calendar-Basis Recommendation

If a target column exists, compare candidate calendar encodings under:

1. UTC
2. candidate local timezones

The profiler should recommend where to derive:

1. `hour_of_day`
2. `day_of_week`
3. `is_holiday`

### 6. Holiday Recommendation

Recommend:

1. no holiday calendar
2. US federal holiday calendar
3. market-specific holiday calendar if later supported

### 7. Aggregation Recommendation

If source cadence is sub-hourly and target cadence is hourly, recommend:

1. `mean`
2. `last`
3. `sum`
4. `none`

## Minimum Recommendation Set

The first version is successful if it can reliably emit:

1. `derive calendar features in local market time before UTC storage`
2. `store timestamps in UTC`
3. `shift hour-ending labels to interval-start`
4. `aggregate sub-hourly weather to hourly mean`
5. `stop` when duplicate timestamps or unresolved cadence ambiguity make automation unsafe

## Output Contract

Return one structured profile object and support a readable markdown report.

### Structured Fields

The profile should include at least:

1. dataset summary
2. timezone analysis
3. interval analysis
4. calendar analysis
5. aggregation analysis
6. warnings
7. recommendations
8. `stop` boolean

Use the schema in:

1. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/09-time-series-calendar-profiler.md`

## Suggested API

First-pass implementation shape:

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

## Suggested File Placement

1. `causal_optimizer/diagnostics/time_calendar_profiler.py`
2. `tests/unit/test_time_calendar_profiler.py`
3. `scripts/profile_time_series_calendar.py`
4. `thoughts/shared/docs/time-series-calendar-profiler.md`

## Required Tests

Add focused tests for:

1. duplicate timestamps -> warning or stop
2. cadence inference on regular hourly data
3. DST suspicion on a fixture with DST-like behavior
4. local-vs-UTC recommendation on an ERCOT-like fixture
5. hour-ending shift recommendation on a controlled fixture

## Required Example / CLI

Add one small CLI or helper script that can run on a local CSV/Parquet file and
emit:

1. structured JSON
2. readable summary to stdout or markdown

Suggested command shape:

```bash
uv run python scripts/profile_time_series_calendar.py \
  --data-path path/to/data.parquet \
  --timestamp-col timestamp \
  --target-col target_load \
  --market-hint ercot
```

## Acceptance Criteria

This sprint is complete when:

1. the profiler can reproduce the ERCOT recommendation to derive calendar
   features in local market time before UTC storage
2. the profiler can explain why it made that recommendation
3. the profiler emits `stop=true` on obviously unsafe timestamp cases
4. the implementation is deterministic and rule-based
5. no predictive benchmark contract changes are required to use it

## Non-Goals

Do not do these in sprint 17:

1. train an ML model to recommend preprocessing
2. automatically rewrite datasets without review
3. solve arbitrary global timezone inference perfectly
4. fold this logic invisibly into the optimizer core

## Good Design Constraints

1. recommendations should be auditable
2. warnings should be concrete, not generic
3. confidence scores should be conservative
4. unsafe ambiguity should produce a stop condition, not a guess

## ERCOT Validation Target

Use the first real benchmark dataset as one validation case:

1. `/Users/robertwelborn/Projects/_local/causal-optimizer/data/ercot_north_c_dfw_2022_2024.parquet`

The profiler should be able to surface a recommendation equivalent to:

1. derive `hour_of_day`, `day_of_week`, and `is_holiday` in `US/Central`
2. store timestamps in `UTC`

## Workflow

Follow the normal agent workflow:

1. `/tdd`
2. implement deterministic checks
3. add tests
4. add CLI or helper script
5. `/polish`
6. `gh pr create`
7. `/gauntlet`
8. report PR URL

## Handoff References

Use these as source-of-truth context:

1. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/09-time-series-calendar-profiler.md`
2. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/plans/08-optimizer-improvement-briefs.md`
3. `/Users/robertwelborn/Projects/causal-optimizer/thoughts/shared/docs/ercot-north-c-dfw-2022-2024-benchmark-report.md`
