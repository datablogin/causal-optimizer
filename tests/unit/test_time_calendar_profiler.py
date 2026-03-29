"""Tests for TimeSeriesCalendarProfiler — TDD-first."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from causal_optimizer.diagnostics.time_calendar_profiler import (
    TimeSeriesCalendarProfile,
    TimeSeriesCalendarProfiler,
)


def _make_hourly_df(
    start: str = "2023-01-01",
    periods: int = 168,
    tz: str | None = None,
    freq: str = "h",
) -> pd.DataFrame:
    """Helper: create a clean hourly DataFrame."""
    ts = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "target_load": (
                100.0
                + 20.0 * np.sin(np.arange(periods) * 2 * np.pi / 24)
                + rng.normal(0, 2, periods)
            ),
        }
    )


class TestDuplicateTimestampsStop:
    """test_duplicate_timestamps_stop: DataFrame with duplicate timestamps -> stop=true."""

    def test_duplicate_timestamps_stop(self) -> None:
        df = _make_hourly_df(periods=48)
        # Inject duplicates
        dup_row = df.iloc[[5]].copy()
        df = (
            pd.concat([df, dup_row], ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert isinstance(profile, TimeSeriesCalendarProfile)
        assert profile.stop is True
        assert profile.summary["duplicate_timestamps"] > 0


class TestCadenceInferenceHourly:
    """Regular hourly data -> cadence='hourly', regularity >= 0.99."""

    def test_cadence_inference_hourly(self) -> None:
        df = _make_hourly_df(periods=168)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.summary["inferred_cadence"] == "hourly"
        assert profile.summary["cadence_regularity"] >= 0.99
        assert profile.stop is False


class TestDSTSuspicion:
    """test_dst_suspicion: Fixture with 23-hour and 25-hour days -> dst_suspected=true."""

    def test_dst_suspicion(self) -> None:
        # Build UTC data that, when viewed in US/Central, shows DST artifacts.
        # Spring-forward: 2023-03-12 in US/Central loses an hour (23-hour day in local).
        # Fall-back: 2023-11-05 in US/Central gains an hour (25-hour day in local).
        # We simulate UTC timestamps where the *count of rows per calendar day* varies.
        # Easier approach: create timestamps with explicit 23h and 25h days.
        timestamps = []
        base = datetime(2023, 3, 11, 0, 0, 0)
        # Day 1: normal 24 hours
        for h in range(24):
            timestamps.append(base + timedelta(hours=h))
        # Day 2: 23 hours (spring-forward)
        base2 = datetime(2023, 3, 12, 0, 0, 0)
        for h in range(23):
            timestamps.append(base2 + timedelta(hours=h))
        # Day 3: normal 24 hours
        base3 = base2 + timedelta(hours=23)
        for h in range(24):
            timestamps.append(base3 + timedelta(hours=h))

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "target_load": rng.normal(100, 10, len(timestamps)),
            }
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.timezone_analysis["dst_suspected"] is True


class TestERCOTLocalCalendarRecommendation:
    """test_ercot_local_calendar_recommendation: UTC data with stronger hourly pattern
    in US/Central than UTC -> recommends US/Central."""

    def test_ercot_local_calendar_recommendation(self) -> None:
        # Create data spanning a DST transition so UTC vs local hour groupings
        # produce different R2 values.  US/Central switches from CST (UTC-6) to
        # CDT (UTC-5) on 2023-03-12, so a period covering that transition breaks
        # the 1:1 mapping between UTC hours and local hours.
        local_tz = "US/Central"
        ts_local = pd.date_range("2023-02-01", "2023-04-30 23:00", freq="h", tz=local_tz)
        ts_utc = ts_local.tz_convert("UTC")
        n_hours = len(ts_utc)

        # Target with a peaked local-business-hours pattern (not a pure sine)
        # that is sharper in local time than any fixed UTC offset.
        local_hours = ts_local.hour.to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        # Peak during local hours 9-17, trough at night
        peak = np.where((local_hours >= 9) & (local_hours <= 17), 40.0, 0.0)
        target = 100.0 + peak + rng.normal(0, 3, n_hours)

        df = pd.DataFrame(
            {
                "timestamp": ts_utc.tz_localize(None),  # strip tz for raw UTC
                "target_load": target,
            }
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(
            df,
            timestamp_col="timestamp",
            target_col="target_load",
            candidate_timezones=["UTC", "US/Central"],
        )

        tz_analysis = profile.timezone_analysis
        assert tz_analysis["recommended_calendar_timezone"] == "US/Central"
        assert profile.calendar_analysis["derive_hour_of_day_in"] == "US/Central"


class TestHourEndingDetection:
    """test_hour_ending_detection: First timestamp 01:00 instead of 00:00 -> recommends shift."""

    def test_hour_ending_detection(self) -> None:
        # Hour-ending convention: labels are 01:00, 02:00, ..., 23:00
        # (hour 0 never appears).  We build multiple days of hourly data
        # starting at 01:00 with hour 0 explicitly excluded so the
        # profiler's "shift_back_one_hour" detection path fires.
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        hours = range(1, 24)  # 01:00 through 23:00 — no hour 0
        ts_list = sorted(
            pd.Timestamp(d.date()) + pd.Timedelta(hours=h)
            for d, h in itertools.product(dates, hours)
        )
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "timestamp": ts_list,
                "target_load": rng.normal(100, 10, len(ts_list)),
            }
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.interval_analysis["recommended_convention"] == "shift_back_one_hour"


class TestMarketHintERCOT:
    """test_market_hint_ercot: market_hint='ercot' -> US/Central in candidate timezones."""

    def test_market_hint_ercot(self) -> None:
        df = _make_hourly_df(periods=48)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp", market_hint="ercot")

        # US/Central should appear in candidate timezones
        candidate_tzs = [c["timezone"] for c in profile.timezone_analysis["candidate_timezones"]]
        assert "US/Central" in candidate_tzs


class TestCleanDataNoStop:
    """test_clean_data_no_stop: Clean hourly data with no issues -> stop=false."""

    def test_clean_data_no_stop(self) -> None:
        df = _make_hourly_df(periods=168)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.stop is False
        assert profile.summary["parse_ok"] is True
        assert profile.summary["monotonic"] is True
        assert profile.summary["duplicate_timestamps"] == 0


class TestSubHourlyAggregation:
    """test_sub_hourly_aggregation: 15-minute data -> recommends hourly mean rollup."""

    def test_sub_hourly_aggregation(self) -> None:
        ts = pd.date_range("2023-01-01", periods=96 * 4, freq="15min")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "target_load": rng.normal(100, 10, len(ts)),
            }
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(
            df,
            timestamp_col="timestamp",
            expected_cadence="hourly",
        )

        agg = profile.aggregation_analysis
        assert agg["recommended_rollup"] == "mean"
        assert agg["target_cadence"] == "hourly"


class TestParseFailures:
    """Unparseable timestamps produce warnings and stop when rate exceeds threshold."""

    def test_unparseable_timestamps_warn(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["2023-01-01 00:00", "NOT_A_DATE", "2023-01-01 02:00"]
                + [f"2023-01-01 {h:02d}:00" for h in range(3, 48)],
                "target_load": np.random.default_rng(42).normal(100, 10, 48),
            }
        )
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.summary["parse_ok"] is False
        warning_codes = [w["code"] for w in profile.warnings]
        assert "PARSE_FAILURE" in warning_codes


class TestGapDetection:
    """Large gaps in timestamps produce gap warnings."""

    def test_gap_detected(self) -> None:
        # Create hourly data with a 12-hour gap in the middle
        ts_part1 = pd.date_range("2023-01-01", periods=48, freq="h")
        ts_part2 = pd.date_range("2023-01-03 12:00", periods=48, freq="h")
        ts = ts_part1.append(ts_part2)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"timestamp": ts, "target_load": rng.normal(100, 10, len(ts))})

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        warning_codes = [w["code"] for w in profile.warnings]
        assert "GAPS_DETECTED" in warning_codes


class TestToDictAndHumanSummary:
    """Serialization via to_dict() and human_summary() round-trips correctly."""

    def test_to_dict_returns_dict(self) -> None:
        df = _make_hourly_df(periods=48)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        result = profile.to_dict()
        assert isinstance(result, dict)
        assert "dataset_id" in result
        assert "summary" in result
        assert "recommendations" in result

    def test_to_json_returns_string(self) -> None:
        df = _make_hourly_df(periods=48)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        import json

        result = profile.to_json()
        parsed = json.loads(result)
        assert parsed["dataset_id"] == "unknown"

    def test_human_summary_contains_key_info(self) -> None:
        df = _make_hourly_df(periods=48)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        text = profile.human_summary()
        assert "Time-Series Calendar Profile" in text
        assert "Rows:" in text
        assert "Cadence:" in text


class TestCalendarBasisTzAwareInput:
    """_compare_calendar_basis handles tz-aware input without raising."""

    def test_tz_aware_timestamps_handled(self) -> None:
        # Create tz-aware UTC timestamps
        ts = pd.date_range("2023-01-01", periods=168, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        local_hours = ts.tz_convert("US/Central").hour.to_numpy(dtype=float)
        peak = np.where((local_hours >= 9) & (local_hours <= 17), 40.0, 0.0)
        target = 100.0 + peak + rng.normal(0, 3, len(ts))

        df = pd.DataFrame({"timestamp": ts, "target_load": target})

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(
            df,
            timestamp_col="timestamp",
            target_col="target_load",
            candidate_timezones=["UTC", "US/Central"],
        )

        # Should not raise; should produce a valid profile
        assert isinstance(profile, TimeSeriesCalendarProfile)
        assert profile.timezone_analysis["recommended_calendar_timezone"] in {
            "UTC",
            "US/Central",
        }


class TestEmptyDataFrame:
    """Empty DataFrame produces a stop profile."""

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]")})
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.stop is True


class TestTargetColValidation:
    """Passing a target_col that does not exist raises ValueError."""

    def test_missing_target_col_raises(self) -> None:
        df = _make_hourly_df(periods=48)
        profiler = TimeSeriesCalendarProfiler()

        with pytest.raises(ValueError, match="target_col.*not found"):
            profiler.profile(df, timestamp_col="timestamp", target_col="nonexistent")


class TestWeeklyCadenceDetection:
    """Weekly data is classified as 'weekly' cadence."""

    def test_weekly_cadence(self) -> None:
        ts = pd.date_range("2023-01-01", periods=52, freq="W")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"timestamp": ts, "target_load": rng.normal(100, 10, len(ts))})

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        assert profile.summary["inferred_cadence"] == "weekly"


class TestMixedOffsetDSTNoCrash:
    """Mixed UTC offsets around DST transitions must not crash the profiler."""

    def test_mixed_offset_dst_no_crash(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2023-03-12 00:00:00-0600",
                    "2023-03-12 01:00:00-0600",
                    "2023-03-12 03:00:00-0500",
                    "2023-03-12 04:00:00-0500",
                ],
                "target_load": [100.0, 110.0, 120.0, 115.0],
            }
        )

        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        # Must return a structured result, not crash
        assert isinstance(profile, TimeSeriesCalendarProfile)
        # Must contain a warning about mixed offsets
        warning_codes = [w["code"] for w in profile.warnings]
        assert "MIXED_OFFSETS" in warning_codes
        # UTC normalization resolves the issue — should not stop
        assert profile.stop is False


class TestStoreUtcNotEmittedForPlainUTC:
    """store-utc recommendation should NOT appear when data is UTC-only with no DST."""

    def test_no_store_utc_for_plain_utc(self) -> None:
        df = _make_hourly_df(periods=168)
        profiler = TimeSeriesCalendarProfiler()
        profile = profiler.profile(df, timestamp_col="timestamp")

        rec_ids = [r.id for r in profile.recommendations]
        assert "store-utc" not in rec_ids
