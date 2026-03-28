"""Rule-based TimeSeriesCalendarProfiler for timestamp and calendar diagnostics.

Inspects a time-series dataset and emits explicit recommendations for
timestamp normalization and calendar-feature construction.  Deterministic
and rule-based — no ML models involved.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Market-hint lookup tables
# ---------------------------------------------------------------------------

_MARKET_TIMEZONES: dict[str, list[str]] = {
    "ercot": ["US/Central"],
    "pjm": ["US/Eastern"],
    "caiso": ["US/Pacific"],
    "miso": ["US/Central", "US/Eastern"],
    "nyiso": ["US/Eastern"],
    "isone": ["US/Eastern"],
    "spp": ["US/Central"],
}

_MARKET_HOLIDAYS: dict[str, str] = {
    "ercot": "US_FEDERAL",
    "pjm": "US_FEDERAL",
    "caiso": "US_FEDERAL",
    "miso": "US_FEDERAL",
    "nyiso": "US_FEDERAL",
    "isone": "US_FEDERAL",
    "spp": "US_FEDERAL",
}

# US federal holidays (fixed-date subset, approximate for detection)
_US_FEDERAL_HOLIDAYS_FIXED: list[tuple[int, int]] = [
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (12, 25),  # Christmas
]


def _us_federal_holidays(year: int) -> set[date]:
    """Return approximate US federal holiday dates for *year*.

    Includes fixed-date holidays and common Monday-observed holidays:
    MLK, Presidents, Memorial, Labor, Columbus, Thanksgiving, Veterans.
    """
    holidays: set[date] = set()
    # Fixed-date
    for m, d in _US_FEDERAL_HOLIDAYS_FIXED:
        holidays.add(date(year, m, d))

    # MLK: 3rd Monday in January
    holidays.add(_nth_weekday(year, 1, 0, 3))
    # Presidents: 3rd Monday in February
    holidays.add(_nth_weekday(year, 2, 0, 3))
    # Memorial: last Monday in May
    holidays.add(_last_weekday(year, 5, 0))
    # Labor: 1st Monday in September
    holidays.add(_nth_weekday(year, 9, 0, 1))
    # Columbus: 2nd Monday in October
    holidays.add(_nth_weekday(year, 10, 0, 2))
    # Veterans: Nov 11
    holidays.add(date(year, 11, 11))
    # Thanksgiving: 4th Thursday in November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* (0=Mon) in *month*."""
    d = date(year, month, 1)
    # Advance to first occurrence of weekday
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(weeks=n - 1)
    return d


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* (0=Mon) in *month*."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------


@dataclass
class CalendarProfileRecommendation:
    """A single actionable recommendation from the profiler."""

    id: str
    priority: Literal["P0", "P1", "P2"]
    action: str
    reason: str
    confidence: float


@dataclass
class TimeSeriesCalendarProfile:
    """Full profiler output."""

    dataset_id: str
    summary: dict[str, object]
    timezone_analysis: dict[str, object]
    interval_analysis: dict[str, object]
    calendar_analysis: dict[str, object]
    aggregation_analysis: dict[str, object]
    warnings: list[dict[str, object]] = field(default_factory=list)
    recommendations: list[CalendarProfileRecommendation] = field(default_factory=list)
    stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def human_summary(self) -> str:
        """Return a human-readable text summary."""
        lines: list[str] = []
        lines.append(f"=== Time-Series Calendar Profile: {self.dataset_id} ===")
        lines.append("")

        s = self.summary
        lines.append(f"Rows: {s.get('row_count', '?')}")
        lines.append(f"Range: {s.get('timestamp_start', '?')} -> {s.get('timestamp_end', '?')}")
        lines.append(
            f"Cadence: {s.get('inferred_cadence', '?')} "
            f"(regularity: {s.get('cadence_regularity', '?')})"
        )
        lines.append(
            f"Parse OK: {s.get('parse_ok')}, Monotonic: {s.get('monotonic')}, "
            f"Duplicates: {s.get('duplicate_timestamps')}"
        )
        lines.append("")

        tz = self.timezone_analysis
        lines.append(f"Storage TZ: {tz.get('recommended_storage_timezone', '?')}")
        lines.append(f"Calendar TZ: {tz.get('recommended_calendar_timezone', '?')}")
        lines.append(f"DST suspected: {tz.get('dst_suspected', False)}")
        lines.append("")

        iv = self.interval_analysis
        lines.append(
            f"Interval convention: {iv.get('recommended_convention', '?')} "
            f"(confidence: {iv.get('confidence', '?')})"
        )
        lines.append("")

        cal = self.calendar_analysis
        lines.append(f"Derive hour_of_day in: {cal.get('derive_hour_of_day_in', '?')}")
        lines.append(f"Derive day_of_week in: {cal.get('derive_day_of_week_in', '?')}")
        lines.append(f"Holiday calendar: {cal.get('holiday_calendar', '?')}")
        lines.append("")

        agg = self.aggregation_analysis
        lines.append(
            f"Aggregation: {agg.get('recommended_rollup', 'none')} "
            f"-> {agg.get('target_cadence', '?')}"
        )
        lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  [{w.get('severity', '?')}] {w.get('code')}: {w.get('message')}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  [{r.priority}] {r.id}: {r.action}")
                lines.append(f"         Reason: {r.reason}")
            lines.append("")

        if self.stop:
            lines.append("*** STOP: profiler recommends halting — see warnings above ***")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class TimeSeriesCalendarProfiler:
    """Rule-based profiler for time-series timestamp and calendar diagnostics."""

    def profile(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        target_col: str | None = None,
        candidate_timezones: list[str] | None = None,
        market_hint: str | None = None,
        expected_cadence: str | None = None,
        dataset_id: str = "unknown",
    ) -> TimeSeriesCalendarProfile:
        """Run all profiler checks and return a structured profile."""
        profile_warnings: list[dict[str, object]] = []
        recommendations: list[CalendarProfileRecommendation] = []
        stop = False

        # --- Resolve candidate timezones from market hint ----------------
        if candidate_timezones is None:
            candidate_timezones = ["UTC"]
        if market_hint and market_hint.lower() in _MARKET_TIMEZONES:
            for tz in _MARKET_TIMEZONES[market_hint.lower()]:
                if tz not in candidate_timezones:
                    candidate_timezones.append(tz)

        holiday_calendar = "NONE"
        if market_hint and market_hint.lower() in _MARKET_HOLIDAYS:
            holiday_calendar = _MARKET_HOLIDAYS[market_hint.lower()]

        # --- 1. Timestamp parse + monotonicity ---------------------------
        ts_result = self._check_timestamps(data, timestamp_col)
        ts_series: pd.Series[Any] = ts_result["series"]
        summary: dict[str, object] = ts_result["summary"]

        if ts_result["stop"]:
            stop = True
        profile_warnings.extend(ts_result["warnings"])

        if summary["duplicate_timestamps"]:
            dup_count = summary["duplicate_timestamps"]
            recommendations.append(
                CalendarProfileRecommendation(
                    id="dedup-timestamps",
                    priority="P0",
                    action=f"Resolve {dup_count} duplicate timestamp(s) before proceeding",
                    reason="Duplicate timestamps cause ambiguous joins and incorrect aggregations",
                    confidence=1.0,
                )
            )

        # --- 2. Cadence inference ----------------------------------------
        cadence_result = self._infer_cadence(ts_series)
        summary["inferred_cadence"] = cadence_result["inferred_cadence"]
        summary["cadence_regularity"] = cadence_result["cadence_regularity"]

        if cadence_result["stop"]:
            stop = True
        profile_warnings.extend(cadence_result["warnings"])

        # --- 3. DST / timezone artifact detection ------------------------
        dst_result = self._detect_dst(ts_series)
        dst_suspected = dst_result["dst_suspected"]

        # --- 4. Interval convention detection ----------------------------
        interval_result = self._detect_interval_convention(ts_series)

        if interval_result["recommended_convention"] == "shift_back_one_hour":
            recommendations.append(
                CalendarProfileRecommendation(
                    id="shift-hour-ending",
                    priority="P1",
                    action="Shift timestamps back by one hour to convert from hour-ending "
                    "to interval-start convention",
                    reason="Timestamps appear to use hour-ending labeling",
                    confidence=float(interval_result["confidence"]),
                )
            )

        # --- 5. Calendar-basis comparison --------------------------------
        calendar_tz = candidate_timezones[0] if candidate_timezones else "UTC"
        tz_scores: list[dict[str, Any]] = []

        if target_col and target_col in data.columns:
            calendar_result = self._compare_calendar_basis(
                ts_series, data[target_col], candidate_timezones
            )
            calendar_tz = calendar_result["best_timezone"]
            tz_scores = calendar_result["scores"]
        else:
            # No target — use market hint timezone if available
            if market_hint and market_hint.lower() in _MARKET_TIMEZONES:
                calendar_tz = _MARKET_TIMEZONES[market_hint.lower()][0]
            for tz in candidate_timezones:
                tz_scores.append({"timezone": tz, "score": 0.0, "notes": ["no target column"]})

        # --- 6. Holiday calendar recommendation --------------------------
        holiday_result = self._recommend_holidays(
            ts_series,
            data.get(target_col) if target_col else None,
            calendar_tz,
            holiday_calendar,
        )

        if holiday_result["recommended"]:
            recommendations.append(
                CalendarProfileRecommendation(
                    id="holiday-calendar",
                    priority="P2",
                    action=f"Use {holiday_result['calendar']} holiday calendar "
                    f"derived in {calendar_tz}",
                    reason=holiday_result["reason"],
                    confidence=holiday_result["confidence"],
                )
            )

        # --- 7. Aggregation recommendation -------------------------------
        agg_result = self._recommend_aggregation(
            cadence_result["inferred_cadence"], expected_cadence
        )

        if agg_result["recommended_rollup"] != "none":
            recommendations.append(
                CalendarProfileRecommendation(
                    id="aggregate-rollup",
                    priority="P1",
                    action=f"Aggregate data to {agg_result['target_cadence']} "
                    f"using {agg_result['recommended_rollup']}",
                    reason=agg_result["reason"],
                    confidence=agg_result["confidence"],
                )
            )

        # Calendar derivation recommendation
        if calendar_tz != "UTC":
            recommendations.append(
                CalendarProfileRecommendation(
                    id="calendar-local-tz",
                    priority="P1",
                    action=f"Derive calendar features (hour_of_day, day_of_week, is_holiday) "
                    f"in {calendar_tz}",
                    reason=f"Calendar features explain more target variance "
                    f"in {calendar_tz} than UTC",
                    confidence=0.8,
                )
            )

        # Storage recommendation — only when DST or non-UTC calendar tz is relevant
        if dst_suspected or calendar_tz != "UTC":
            recommendations.append(
                CalendarProfileRecommendation(
                    id="store-utc",
                    priority="P2",
                    action="Store timestamps in UTC for unambiguous serialization",
                    reason="UTC avoids DST ambiguity in storage",
                    confidence=0.9,
                )
            )

        # Build timezone analysis
        tz_confidence = 0.5
        if tz_scores:
            best_score = max((float(s["score"]) for s in tz_scores), default=0.0)
            if best_score > 0:
                tz_confidence = min(float(best_score), 1.0)

        timezone_analysis: dict[str, object] = {
            "candidate_timezones": tz_scores,
            "recommended_storage_timezone": "UTC",
            "recommended_calendar_timezone": calendar_tz,
            "dst_suspected": dst_suspected,
            "confidence": tz_confidence,
        }

        calendar_analysis: dict[str, object] = {
            "derive_hour_of_day_in": calendar_tz,
            "derive_day_of_week_in": calendar_tz,
            "derive_is_holiday_in": calendar_tz,
            "holiday_calendar": holiday_result.get("calendar", holiday_calendar),
            "confidence": tz_confidence,
            "notes": [],
        }

        aggregation_analysis: dict[str, object] = {
            "recommended_rollup": agg_result["recommended_rollup"],
            "target_cadence": agg_result["target_cadence"],
            "confidence": agg_result["confidence"],
            "notes": [agg_result["reason"]],
        }

        return TimeSeriesCalendarProfile(
            dataset_id=dataset_id,
            summary=summary,
            timezone_analysis=timezone_analysis,
            interval_analysis=interval_result,
            calendar_analysis=calendar_analysis,
            aggregation_analysis=aggregation_analysis,
            warnings=profile_warnings,
            recommendations=recommendations,
            stop=stop,
        )

    # -----------------------------------------------------------------------
    # Check 1: Timestamp parse + monotonicity
    # -----------------------------------------------------------------------

    def _check_timestamps(self, data: pd.DataFrame, timestamp_col: str) -> dict[str, Any]:
        """Parse timestamps and check for duplicates, monotonicity, mixed tz."""
        ts_warnings: list[dict[str, object]] = []
        stop = False

        raw = data[timestamp_col]
        ts = pd.to_datetime(raw, errors="coerce")
        n_total = len(ts)
        n_failed = int(ts.isna().sum() - raw.isna().sum())  # only count parse failures
        parse_ok = n_failed == 0

        if n_failed > 0:
            fail_rate = n_failed / max(n_total, 1)
            ts_warnings.append(
                {
                    "code": "PARSE_FAILURE",
                    "severity": "error",
                    "message": f"{n_failed}/{n_total} timestamps failed to parse ({fail_rate:.1%})",
                }
            )
            if fail_rate > 0.01:
                stop = True

        ts_clean = ts.dropna()
        monotonic = bool(ts_clean.is_monotonic_increasing)
        if not monotonic:
            ts_warnings.append(
                {
                    "code": "NOT_MONOTONIC",
                    "severity": "warning",
                    "message": "Timestamps are not monotonically increasing",
                }
            )

        dup_count = int(ts_clean.duplicated().sum())
        if dup_count > 0:
            stop = True
            ts_warnings.append(
                {
                    "code": "DUPLICATE_TIMESTAMPS",
                    "severity": "error",
                    "message": f"{dup_count} duplicate timestamp(s) detected",
                }
            )

        summary: dict[str, object] = {
            "row_count": n_total,
            "timestamp_start": str(ts_clean.min()) if len(ts_clean) > 0 else None,
            "timestamp_end": str(ts_clean.max()) if len(ts_clean) > 0 else None,
            "parse_ok": parse_ok,
            "monotonic": monotonic,
            "duplicate_timestamps": dup_count,
        }

        return {
            "series": ts,
            "summary": summary,
            "warnings": ts_warnings,
            "stop": stop,
        }

    # -----------------------------------------------------------------------
    # Check 2: Cadence inference
    # -----------------------------------------------------------------------

    def _infer_cadence(self, ts: pd.Series[Any]) -> dict[str, Any]:
        """Infer the dominant time interval and regularity."""
        cadence_warnings: list[dict[str, object]] = []
        stop = False

        ts_clean = ts.dropna().sort_values()
        if len(ts_clean) < 2:
            return {
                "inferred_cadence": "unknown",
                "cadence_regularity": 0.0,
                "gap_count": 0,
                "warnings": [
                    {
                        "code": "INSUFFICIENT_DATA",
                        "severity": "error",
                        "message": "Fewer than 2 valid timestamps",
                    }
                ],
                "stop": True,
            }

        diffs = ts_clean.diff().dropna()
        diff_seconds = diffs.dt.total_seconds()

        # Dominant interval = mode of rounded diffs
        median_seconds = float(diff_seconds.median())

        # Classify cadence
        cadence = "unknown"
        if 50 <= median_seconds <= 70:
            cadence = "1min"
        elif 250 <= median_seconds <= 350:
            cadence = "5min"
        elif 800 <= median_seconds <= 1000:
            cadence = "15min"
        elif 1700 <= median_seconds <= 1900:
            cadence = "30min"
        elif 3400 <= median_seconds <= 3700:
            cadence = "hourly"
        elif 84600 <= median_seconds <= 90000:
            cadence = "daily"
        elif 575000 <= median_seconds <= 635000:
            cadence = "weekly"

        # Regularity: fraction of diffs within 10% of median
        tolerance = max(median_seconds * 0.1, 1.0)
        n_regular = int(((diff_seconds - median_seconds).abs() <= tolerance).sum())
        regularity = n_regular / max(len(diff_seconds), 1)

        # Gap count: diffs > 1.5x median
        gap_threshold = median_seconds * 1.5
        gap_count = int((diff_seconds > gap_threshold).sum())
        if gap_count > 0:
            cadence_warnings.append(
                {
                    "code": "GAPS_DETECTED",
                    "severity": "warning",
                    "message": f"{gap_count} gap(s) detected (intervals > 1.5x median)",
                }
            )

        if regularity < 0.8 and cadence == "unknown":
            stop = True
            cadence_warnings.append(
                {
                    "code": "CADENCE_AMBIGUOUS",
                    "severity": "error",
                    "message": f"Cadence regularity {regularity:.2f} is below 0.8 "
                    "and no clear dominant interval found",
                }
            )

        return {
            "inferred_cadence": cadence,
            "cadence_regularity": round(regularity, 4),
            "gap_count": gap_count,
            "warnings": cadence_warnings,
            "stop": stop,
        }

    # -----------------------------------------------------------------------
    # Check 3: DST / timezone artifact detection
    # -----------------------------------------------------------------------

    def _detect_dst(self, ts: pd.Series[Any]) -> dict[str, Any]:
        """Look for 23-hour/25-hour day patterns suggesting DST transitions."""
        ts_clean = ts.dropna().sort_values()
        if len(ts_clean) < 48:
            return {"dst_suspected": False, "notes": ["insufficient data for DST detection"]}

        # Group by calendar date, count hours per day
        dates = ts_clean.dt.date
        hours_per_day = dates.value_counts().sort_index()

        # The median gives us the "expected" hours-per-day
        expected = int(hours_per_day.median())
        if expected <= 0:
            return {"dst_suspected": False, "notes": ["unexpected hour counts"]}

        short_days = int((hours_per_day == expected - 1).sum())
        long_days = int((hours_per_day == expected + 1).sum())

        dst_suspected = short_days > 0 or long_days > 0

        notes: list[str] = []
        if short_days > 0:
            notes.append(f"{short_days} day(s) with {expected - 1} rows (spring-forward?)")
        if long_days > 0:
            notes.append(f"{long_days} day(s) with {expected + 1} rows (fall-back?)")

        return {
            "dst_suspected": dst_suspected,
            "short_days": short_days,
            "long_days": long_days,
            "notes": notes,
        }

    # -----------------------------------------------------------------------
    # Check 4: Interval convention detection
    # -----------------------------------------------------------------------

    def _detect_interval_convention(self, ts: pd.Series[Any]) -> dict[str, Any]:
        """Heuristic: detect hour-ending vs interval-start labeling."""
        ts_clean = ts.dropna().sort_values()
        if len(ts_clean) < 24:
            return {
                "recommended_convention": "needs_review",
                "confidence": 0.0,
                "notes": ["insufficient data for interval convention detection"],
            }

        # Check if the first timestamp of the earliest date starts at hour 1
        # (classic hour-ending: the first interval of the day is labeled 01:00)
        earliest_date = ts_clean.dt.date.min()
        first_of_day = ts_clean[ts_clean.dt.date == earliest_date]
        first_hour_of_day = int(first_of_day.iloc[0].hour) if len(first_of_day) > 0 else 0
        first_minute = int(first_of_day.iloc[0].minute) if len(first_of_day) > 0 else 0

        # Heuristic: if first timestamp of day is 01:00 (not 00:00),
        # and data is hourly-ish, suspect hour-ending
        diffs = ts_clean.diff().dropna().dt.total_seconds()
        median_diff = float(diffs.median())
        is_hourly_ish = 3000 <= median_diff <= 3900

        if first_hour_of_day == 1 and first_minute == 0 and is_hourly_ish:
            # Check: do we ever see hour 0? If not, strong signal for hour-ending.
            has_hour_zero = bool((ts_clean.dt.hour == 0).any())
            if not has_hour_zero:
                return {
                    "recommended_convention": "shift_back_one_hour",
                    "confidence": 0.85,
                    "notes": [
                        "First hour of day is 01:00, hour 00 never appears. "
                        "Consistent with hour-ending convention."
                    ],
                }
            else:
                return {
                    "recommended_convention": "needs_review",
                    "confidence": 0.4,
                    "notes": [
                        "First timestamp starts at 01:00 but hour 0 also appears. "
                        "Ambiguous — may be hour-ending or simply offset data."
                    ],
                }

        return {
            "recommended_convention": "interval_start",
            "confidence": 0.8,
            "notes": ["Timestamps appear to use interval-start convention"],
        }

    # -----------------------------------------------------------------------
    # Check 5: Calendar-basis comparison
    # -----------------------------------------------------------------------

    def _compare_calendar_basis(
        self,
        ts: pd.Series[Any],
        target: pd.Series[Any],
        candidate_timezones: list[str],
    ) -> dict[str, Any]:
        """Compare variance explained by hour-of-day buckets across timezones.

        Uses three complementary signals:
        1. Global hour-of-day R2 (eta-squared) — weight 0.4.
        2. Within-hour homogeneity (1 - avg_within_std/overall_std) — weight 0.4.
        3. Midnight-alignment bonus: 0.2 if the first record in a candidate
           timezone falls on hour 0 while the raw (UTC) hour is non-zero.

        During DST transitions, UTC hour buckets conflate two different local
        hours (e.g. UTC hour 15 maps to 9am CST and 10am CDT), inflating
        within-group variance and lowering the homogeneity score.
        """
        ts_clean = ts.dropna()
        # Normalize to tz-naive UTC so tz_localize("UTC") never fails
        if ts_clean.dt.tz is not None:
            ts_clean = ts_clean.dt.tz_convert("UTC").dt.tz_localize(None)
        target_clean = target.loc[ts_clean.index].dropna()
        common_idx = ts_clean.index.intersection(target_clean.index)
        ts_aligned = ts_clean.loc[common_idx]
        target_aligned = target_clean.loc[common_idx].astype(float)

        if len(target_aligned) < 48:
            return {
                "best_timezone": candidate_timezones[0] if candidate_timezones else "UTC",
                "scores": [],
            }

        total_var = float(target_aligned.var())
        if total_var == 0:
            return {
                "best_timezone": candidate_timezones[0] if candidate_timezones else "UTC",
                "scores": [],
            }

        overall_std = float(target_aligned.std())

        # Data-start alignment heuristic: if the earliest timestamp's
        # hour corresponds to midnight (hour 0) in a candidate timezone,
        # that timezone likely matches the data's originating locale.
        first_utc_hour = int(ts_aligned.iloc[0].hour)

        scores: list[dict[str, object]] = []
        best_tz = candidate_timezones[0] if candidate_timezones else "UTC"
        best_combined = -1.0

        for tz in candidate_timezones:
            try:
                if tz == "UTC":
                    localized = ts_aligned
                else:
                    localized = ts_aligned.dt.tz_localize("UTC").dt.tz_convert(tz)
                hours = localized.dt.hour

                # Global R2
                group_means = target_aligned.groupby(hours).transform("mean")
                ss_between = float(((group_means - target_aligned.mean()) ** 2).sum())
                ss_total = float(((target_aligned - target_aligned.mean()) ** 2).sum())
                global_r2 = ss_between / max(ss_total, 1e-10)

                # Within-hour homogeneity: lower avg within-group std = better
                within_stds = target_aligned.groupby(hours).std()
                avg_within_std = float(within_stds.mean())
                homogeneity = max(
                    0.0,
                    min(1.0, 1.0 - avg_within_std / max(overall_std, 1e-10)),
                )

                # Midnight-alignment bonus: if the first record in this
                # timezone starts at hour 0 while the raw UTC hour is
                # non-zero, the data likely originates in this locale.
                first_local_hour = int(hours.iloc[0])
                midnight_aligned = first_local_hour == 0 and first_utc_hour != 0

                combined = 0.4 * global_r2 + 0.4 * homogeneity + (0.2 if midnight_aligned else 0.0)

                notes = [
                    f"Global hour R2={global_r2:.4f}",
                    f"Homogeneity={homogeneity:.4f}",
                    f"Combined={combined:.4f}",
                ]
                if midnight_aligned:
                    notes.append("Midnight-alignment bonus: data starts at local midnight")

                scores.append(
                    {
                        "timezone": tz,
                        "score": round(combined, 4),
                        "notes": notes,
                    }
                )

                if combined > best_combined:
                    best_combined = combined
                    best_tz = tz
            except Exception as exc:
                warnings.warn(
                    f"Timezone evaluation failed for {tz}: {exc}",
                    stacklevel=2,
                )
                scores.append(
                    {
                        "timezone": tz,
                        "score": 0.0,
                        "notes": [f"failed to evaluate: {exc}"],
                    }
                )

        return {"best_timezone": best_tz, "scores": scores}

    # -----------------------------------------------------------------------
    # Check 6: Holiday calendar recommendation
    # -----------------------------------------------------------------------

    def _recommend_holidays(
        self,
        ts: pd.Series[Any],
        target: pd.Series[Any] | None,
        calendar_tz: str,
        holiday_calendar: str,
    ) -> dict[str, Any]:
        """Check if holiday flags improve target variance explanation."""
        if target is None or holiday_calendar == "NONE":
            return {
                "recommended": holiday_calendar != "NONE",
                "calendar": holiday_calendar,
                "reason": "No target column to evaluate holiday effect"
                if target is None
                else "No holiday calendar configured",
                "confidence": 0.3,
            }

        ts_clean = ts.dropna()
        target_clean = target.loc[ts_clean.index].dropna()
        common_idx = ts_clean.index.intersection(target_clean.index)

        if len(common_idx) < 48:
            return {
                "recommended": True,
                "calendar": holiday_calendar,
                "reason": "Insufficient data for holiday evaluation; using market default",
                "confidence": 0.3,
            }

        ts_aligned = ts_clean.loc[common_idx]
        # Normalize to tz-naive UTC so tz_localize("UTC") never fails
        if ts_aligned.dt.tz is not None:
            ts_aligned = ts_aligned.dt.tz_convert("UTC").dt.tz_localize(None)
        target_aligned = target_clean.loc[common_idx].astype(float)

        # Build holiday flag
        try:
            if calendar_tz != "UTC":
                local_ts = ts_aligned.dt.tz_localize("UTC").dt.tz_convert(calendar_tz)
            else:
                local_ts = ts_aligned
            local_dates = local_ts.dt.date
        except Exception as exc:
            warnings.warn(
                f"Holiday tz conversion failed for {calendar_tz}: {exc}",
                stacklevel=2,
            )
            local_dates = ts_aligned.dt.date

        years = {d.year for d in local_dates if isinstance(d, date)}
        all_holidays: set[date] = set()
        for y in years:
            all_holidays |= _us_federal_holidays(y)

        is_holiday = pd.Series([d in all_holidays for d in local_dates], index=common_idx)

        holiday_count = int(is_holiday.sum())
        if holiday_count < 2:
            return {
                "recommended": True,
                "calendar": holiday_calendar,
                "reason": "Too few holidays in data range for reliable evaluation; "
                "using market default",
                "confidence": 0.3,
            }

        # Compare mean target on holidays vs non-holidays
        holiday_mean = float(target_aligned[is_holiday].mean())
        non_holiday_mean = float(target_aligned[~is_holiday].mean())
        overall_std = float(target_aligned.std())

        effect_size = abs(holiday_mean - non_holiday_mean) / max(overall_std, 1e-10)

        if effect_size > 0.2:
            return {
                "recommended": True,
                "calendar": holiday_calendar,
                "reason": f"Holiday effect detected (Cohen's d={effect_size:.2f}). "
                f"Holiday mean={holiday_mean:.1f}, "
                f"non-holiday mean={non_holiday_mean:.1f}",
                "confidence": min(0.5 + effect_size * 0.3, 0.95),
            }

        return {
            "recommended": True,
            "calendar": holiday_calendar,
            "reason": "Weak holiday effect but including as market convention",
            "confidence": 0.4,
        }

    # -----------------------------------------------------------------------
    # Check 7: Aggregation recommendation
    # -----------------------------------------------------------------------

    def _recommend_aggregation(
        self, inferred_cadence: str, expected_cadence: str | None
    ) -> dict[str, Any]:
        """Recommend rollup if current cadence is finer than expected.

        Note: defaults to ``"mean"`` aggregation, which is correct for
        instantaneous measurements (temperature, price, load-as-power).
        For cumulative quantities (energy = power x time) the caller
        should override with ``"sum"``.
        """
        if expected_cadence is None:
            return {
                "recommended_rollup": "none",
                "target_cadence": inferred_cadence,
                "reason": "No expected cadence specified",
                "confidence": 0.0,
            }

        sub_hourly = {"1min", "5min", "15min", "30min"}
        if inferred_cadence in sub_hourly and expected_cadence == "hourly":
            return {
                "recommended_rollup": "mean",
                "target_cadence": "hourly",
                "reason": f"Data is {inferred_cadence} but expected cadence is hourly. "
                "Mean aggregation recommended for instantaneous variables "
                "(temperature, price); use sum for cumulative quantities (energy).",
                "confidence": 0.8,
            }

        if inferred_cadence == "hourly" and expected_cadence == "daily":
            return {
                "recommended_rollup": "mean",
                "target_cadence": "daily",
                "reason": "Data is hourly but expected cadence is daily. "
                "Mean aggregation recommended for instantaneous variables; "
                "use sum for cumulative quantities.",
                "confidence": 0.7,
            }

        return {
            "recommended_rollup": "none",
            "target_cadence": expected_cadence,
            "reason": f"Cadence {inferred_cadence} matches or is coarser than "
            f"expected {expected_cadence}",
            "confidence": 0.9,
        }
