"""SQLite-backed persistent store for experiment logs."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from causal_optimizer.types import ExperimentLog, ExperimentResult, SearchSpace

if TYPE_CHECKING:
    from pathlib import Path


class ExperimentStore:
    """SQLite-backed persistent store for experiment logs.

    Schema:
        experiments(id TEXT PRIMARY KEY, created_at TEXT, search_space_json TEXT)
        results(id TEXT, experiment_id TEXT, step INT, parameters_json TEXT,
                metrics_json TEXT, status TEXT, metadata_json TEXT, timestamp TEXT)
    """

    def __init__(self, path: str | Path) -> None:
        """Open or create a store at the given path. ':memory:' for in-memory."""
        db_path = str(path)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    search_space_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );
                """
            )

    def create_experiment(self, experiment_id: str, search_space: SearchSpace) -> None:
        """Create a new experiment. No-op if already exists."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO experiments (id, created_at, search_space_json) "
                "VALUES (?, ?, ?)",
                (
                    experiment_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(search_space.model_dump()),
                ),
            )
            self._conn.commit()

    def append_result(self, experiment_id: str, result: ExperimentResult, step: int) -> None:
        """Append an experiment result to the store."""
        data = result.model_dump()
        with self._lock:
            self._conn.execute(
                "INSERT INTO results "
                "(id, experiment_id, step, parameters_json, metrics_json, "
                "status, metadata_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["experiment_id"],
                    experiment_id,
                    step,
                    json.dumps(data["parameters"]),
                    json.dumps(data["metrics"]),
                    data["status"],
                    json.dumps(data["metadata"]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def load_log(self, experiment_id: str) -> ExperimentLog:
        """Load all results for an experiment as an ExperimentLog.

        Raises:
            KeyError: If the experiment does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Experiment {experiment_id!r} not found")

            rows = self._conn.execute(
                "SELECT id, parameters_json, metrics_json, status, metadata_json "
                "FROM results WHERE experiment_id = ? ORDER BY step",
                (experiment_id,),
            ).fetchall()

        results: list[ExperimentResult] = []
        for r in rows:
            results.append(
                ExperimentResult.model_validate(
                    {
                        "experiment_id": r["id"],
                        "parameters": json.loads(r["parameters_json"]),
                        "metrics": json.loads(r["metrics_json"]),
                        "status": r["status"],
                        "metadata": json.loads(r["metadata_json"]),
                    }
                )
            )
        return ExperimentLog(results=results)

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments with their metadata."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.id, e.created_at, COUNT(r.id) as step_count "
                "FROM experiments e "
                "LEFT JOIN results r ON e.id = r.experiment_id "
                "GROUP BY e.id ORDER BY e.created_at",
            ).fetchall()
        return [
            {"id": r["id"], "created_at": r["created_at"], "step_count": r["step_count"]}
            for r in rows
        ]

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and all its results."""
        with self._lock:
            self._conn.execute("DELETE FROM results WHERE experiment_id = ?", (experiment_id,))
            self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()
