"""SQLite-backed persistent store for experiment logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from causal_optimizer.types import ExperimentLog, ExperimentResult, SearchSpace


class ExperimentStore:
    """SQLite-backed persistent store for experiment logs.

    Schema:
        experiments(id TEXT PRIMARY KEY, created_at TEXT, search_space_json TEXT)
        results(id TEXT, experiment_id TEXT, step INT, parameters_json TEXT,
                metrics_json TEXT, status TEXT, metadata_json TEXT, timestamp TEXT)
    """

    def __init__(self, path: str | Path) -> None:
        """Open or create a store at the given path. ':memory:' for in-memory."""
        raise NotImplementedError

    def create_experiment(self, experiment_id: str, search_space: SearchSpace) -> None:
        raise NotImplementedError

    def append_result(self, experiment_id: str, result: ExperimentResult, step: int) -> None:
        raise NotImplementedError

    def load_log(self, experiment_id: str) -> ExperimentLog:
        raise NotImplementedError

    def list_experiments(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def delete_experiment(self, experiment_id: str) -> None:
        raise NotImplementedError
