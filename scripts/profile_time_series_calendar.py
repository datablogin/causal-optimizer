#!/usr/bin/env python3
"""CLI script for the TimeSeriesCalendarProfiler.

Usage:
    uv run python scripts/profile_time_series_calendar.py \
        --data-path path/to/data.parquet \
        --timestamp-col timestamp \
        --target-col target_load \
        --market-hint ercot

Outputs structured JSON to stdout (or --output file) and a human-readable
summary to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a time-series dataset for timestamp and calendar diagnostics."
    )
    parser.add_argument("--data-path", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name")
    parser.add_argument("--target-col", default=None, help="Target column name (optional)")
    parser.add_argument(
        "--candidate-timezones",
        nargs="*",
        default=None,
        help="Candidate timezones to evaluate",
    )
    parser.add_argument("--market-hint", default=None, help="Market hint (e.g. ercot, pjm)")
    parser.add_argument("--expected-cadence", default=None, help="Expected cadence (e.g. hourly)")
    parser.add_argument("--dataset-id", default=None, help="Dataset identifier")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: stdout)")
    args = parser.parse_args()

    # Lazy imports so --help is fast
    import pandas as pd

    from causal_optimizer.diagnostics.time_calendar_profiler import TimeSeriesCalendarProfiler

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix in (".csv", ".tsv"):
        sep = "\t" if data_path.suffix == ".tsv" else ","
        df = pd.read_csv(data_path, sep=sep)
    else:
        print(f"Error: unsupported file format: {data_path.suffix}", file=sys.stderr)
        sys.exit(1)

    dataset_id = args.dataset_id or data_path.stem

    profiler = TimeSeriesCalendarProfiler()
    profile = profiler.profile(
        data=df,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        candidate_timezones=args.candidate_timezones,
        market_hint=args.market_hint,
        expected_cadence=args.expected_cadence,
        dataset_id=dataset_id,
    )

    # Human-readable summary to stderr
    print(profile.human_summary(), file=sys.stderr)

    # Structured JSON
    json_str = profile.to_json(indent=2)
    if args.output:
        Path(args.output).write_text(json_str)
        print(f"\nJSON written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
