"""Benchmark provenance metadata capture for reproducible artifacts.

Provides utilities to record the runtime environment, package versions,
dataset identity, and run parameters alongside benchmark JSON output.
"""

from __future__ import annotations


def collect_provenance(
    *,
    seeds: list[int],
    budgets: list[int],
    strategies: list[str],
    dataset_path: str | None = None,
) -> dict[str, object]:
    """Capture a provenance metadata block for a benchmark run.

    Args:
        seeds: RNG seeds used in the run.
        budgets: Experiment budgets used.
        strategies: Strategy names used.
        dataset_path: Optional path to the dataset file.

    Returns:
        A JSON-serializable dict with provenance metadata.
    """
    raise NotImplementedError


def dataset_hash(path: str) -> str | None:
    """Compute a SHA-256 hex digest of a file's contents.

    Args:
        path: Filesystem path to the file.

    Returns:
        64-character hex string, or ``None`` if the file does not exist.
    """
    raise NotImplementedError
