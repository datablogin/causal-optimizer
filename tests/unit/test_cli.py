"""Tests for the causal-optimizer CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from causal_optimizer.storage.sqlite import ExperimentStore
from causal_optimizer.types import SearchSpace, Variable, VariableType


def _make_search_space() -> SearchSpace:
    return SearchSpace(
        variables=[
            Variable(name="x", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
            Variable(name="y", variable_type=VariableType.CONTINUOUS, lower=-5.0, upper=5.0),
        ]
    )


def test_cli_list_empty_db(tmp_path: Any) -> None:
    """causal-optimizer list --db <path> exits 0 on empty DB."""
    db_path = str(tmp_path / "test.db")
    # Create the DB with a store so the tables exist
    store = ExperimentStore(db_path)
    del store

    result = subprocess.run(
        [sys.executable, "-m", "causal_optimizer", "list", "--db", db_path],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_cli_run_toy_adapter(tmp_path: Any) -> None:
    """Run 5 steps via CLI on a test adapter; assert DB has 5 rows."""
    db_path = str(tmp_path / "test.db")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_optimizer",
            "run",
            "--adapter",
            "tests.fixtures.toy_adapter:ToyAdapter",
            "--budget",
            "5",
            "--db",
            db_path,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Verify DB has 5 results
    store = ExperimentStore(db_path)
    experiments = store.list_experiments()
    assert len(experiments) == 1
    log = store.load_log(experiments[0]["id"])
    assert len(log.results) == 5


def test_cli_report_json(tmp_path: Any) -> None:
    """report --format json returns valid JSON with best_result key."""
    db_path = str(tmp_path / "test.db")
    # First run some experiments
    subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_optimizer",
            "run",
            "--adapter",
            "tests.fixtures.toy_adapter:ToyAdapter",
            "--budget",
            "5",
            "--db",
            db_path,
            "--id",
            "json-test",
        ],
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_optimizer",
            "report",
            "--id",
            "json-test",
            "--db",
            db_path,
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert "best_result" in data


def test_cli_resume_adds_steps(tmp_path: Any) -> None:
    """Run 3, resume 3, assert 6 total steps."""
    db_path = str(tmp_path / "test.db")

    # Run first 3
    subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_optimizer",
            "run",
            "--adapter",
            "tests.fixtures.toy_adapter:ToyAdapter",
            "--budget",
            "3",
            "--db",
            db_path,
            "--id",
            "resume-test",
        ],
        capture_output=True,
        text=True,
    )

    # Resume for 3 more
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal_optimizer",
            "resume",
            "--adapter",
            "tests.fixtures.toy_adapter:ToyAdapter",
            "--id",
            "resume-test",
            "--db",
            db_path,
            "--budget",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    store = ExperimentStore(db_path)
    log = store.load_log("resume-test")
    assert len(log.results) == 6
